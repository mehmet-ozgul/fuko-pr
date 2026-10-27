"""Embeddings client: OpenAI-compatible (default: a local Ollama model) or Vertex AI.

``FUKO_EMBED_PROVIDER`` picks the wire protocol; everything above the
transport -- batching, truncation, the dimension probe, the store's re-embed on
a model change -- is shared. See :attr:`sidecar.config.Settings.embed_provider`.
"""

import httpx

from .config import settings


class EmbedError(RuntimeError):
    """Raised when an embeddings request fails or returns no data."""


class EmbedModelMismatch(EmbedError):
    """Raised when the endpoint does not serve the configured embedding model."""


def _fit(text: str) -> str:
    # Last line of defence, deliberately at the transport edge rather than at
    # each call site: every path that reaches an embedding endpoint goes
    # through here, so no future caller can reintroduce the 500 that a 12k-token
    # PR body used to cause. Truncating only what is *embedded* is safe -- the
    # full text is what gets stored and what a review reads back.
    return text[: settings.embed_max_chars]


#: Vertex ``task_type`` for the two sides of retrieval.
_DOCUMENT = "RETRIEVAL_DOCUMENT"
_QUERY = "RETRIEVAL_QUERY"


def _vertex_url() -> str:
    """The ``:predict`` URL for the configured Vertex model, project and location."""
    location = settings.embed_vertex_location
    host = (
        "aiplatform.googleapis.com"
        if location == "global"
        else f"{location}-aiplatform.googleapis.com"
    )
    return (
        f"https://{host}/v1/projects/{settings.embed_vertex_project}/locations/{location}"
        f"/publishers/google/models/{settings.embed_model}:predict"
    )


class Embedder:
    """Embed text via an OpenAI-compatible ``/embeddings`` endpoint or Vertex AI."""

    def __init__(self) -> None:
        """Configure the endpoint URL, auth, and HTTP client for the configured provider."""
        self._client = httpx.Client(timeout=120.0)
        self._dim: int | None = None
        self._vertex = settings.embed_provider == "vertex"
        self._headers = {"Content-Type": "application/json"}
        self._tokens = None
        if self._vertex:
            if not settings.embed_vertex_project:
                raise EmbedError("FUKO_EMBED_PROVIDER=vertex needs FUKO_EMBED_VERTEX_PROJECT")
            # Lazy: an OpenAI-compatible deployment never touches Google auth.
            from .reviewer.gcp_token import SourceTokenCache

            self._tokens = SourceTokenCache()
            self._url = _vertex_url()
        else:
            self._url = f"{settings.embed_base_url.rstrip('/')}/embeddings"
            if settings.embed_api_key:
                self._headers["Authorization"] = f"Bearer {settings.embed_api_key}"

    def _request(self, batch: list[str], task: str) -> tuple[dict, dict]:
        """Headers and JSON body for one batch, in the provider's dialect."""
        if not self._vertex:
            return self._headers, {"model": settings.embed_model, "input": batch}
        body: dict = {"instances": [{"content": t, "task_type": task} for t in batch]}
        if settings.embed_dimensions:
            body["parameters"] = {"outputDimensionality": settings.embed_dimensions}
        headers = {**self._headers, "Authorization": f"Bearer {self._tokens.bearer()}"}
        return headers, body

    def _vectors(self, payload: dict) -> list[list[float]]:
        """The vectors in one response, in input order, or raise ``EmbedError``."""
        if self._vertex:
            predictions = payload.get("predictions")
            if not predictions:
                raise EmbedError(f"empty 'predictions' in embeddings response from {self._url}")
            try:
                return [p["embeddings"]["values"] for p in predictions]
            except (KeyError, TypeError) as e:
                raise EmbedError(f"malformed prediction from {self._url}: {e}") from None
        data = payload.get("data")
        if not data:
            raise EmbedError(f"empty 'data' in embeddings response from {self._url}")
        return [d["embedding"] for d in data]

    def embed(self, texts: list[str], *, task: str = _DOCUMENT) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input.

        Each input is truncated to ``embed_max_chars`` before it is sent. An
        embedding endpoint rejects an over-long input outright rather than
        truncating it for you, and the model behind it has a fixed context, so
        the choice is between a shorter vector and no vector at all. ``task`` is
        the Vertex ``task_type`` and is ignored by OpenAI-compatible endpoints.
        """
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), settings.embed_batch_size):
            batch = [_fit(t) for t in texts[i : i + settings.embed_batch_size]]
            try:
                headers, body = self._request(batch, task)
                resp = self._client.post(self._url, headers=headers, json=body)
                resp.raise_for_status()
            except httpx.HTTPError as e:
                body = ""
                if isinstance(e, httpx.HTTPStatusError) and e.response is not None:
                    body = e.response.text[:1000]
                raise EmbedError(
                    f"embedding request to {self._url} failed: {e}\nbody: {body}"
                ) from e
            out.extend(self._vectors(resp.json()))
        return out

    def embed_one(self, text: str) -> list[float]:
        """Embed a single string as a *document*."""
        return self.embed([text])[0]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single string as a *query*, with the model's task instruction.

        Kept separate from :meth:`embed_one` rather than handled by a flag
        because the asymmetry is easy to get half-right: prefixing both sides,
        or neither, still returns 1024 well-formed dimensions and degrades
        retrieval quietly. One method per side makes each call site say which
        it means.
        """
        if self._vertex:
            # The task type IS the asymmetry here; adding the prefix as well
            # would embed the query somewhere the documents are not.
            return self.embed([text], task=_QUERY)[0]
        return self.embed([settings.embed_query_prefix + text])[0]

    def served_models(self) -> list[str] | None:
        """Return the model names the endpoint reports, or ``None`` if it will not say.

        ``None`` means *cannot verify*, which is deliberately different from
        *mismatch*: an endpoint that is down, does not implement ``/models``, or
        answers with something unparseable must not be able to stop the sidecar
        from starting.

        Two response shapes are accepted because both are in use here. OpenAI's
        is ``{"data": [{"id": ...}]}``; Ollama's is ``{"models": [{"name": ...}]}``.
        llama.cpp answers with both keys at once, but a stricter OpenAI endpoint
        sends only the first.

        Vertex answers ``None``. Its model names are versioned publisher models
        served under exactly the requested name, so the alias swap this guards
        against (#220) has no Vertex equivalent, and there is no list endpoint
        in the embeddings dialect to ask.
        """
        if self._vertex:
            return None
        try:
            resp = self._client.get(
                f"{settings.embed_base_url.rstrip('/')}/models",
                headers=self._headers,
                timeout=10.0,
            )
            resp.raise_for_status()
            body = resp.json()
        except (httpx.HTTPError, ValueError):
            return None
        if not isinstance(body, dict):
            return None
        names: list[str] = []
        for key, fields in (("data", ("id",)), ("models", ("name", "model"))):
            entries = body.get(key)
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                for field in fields:
                    value = entry.get(field)
                    if isinstance(value, str) and value:
                        names.append(value)
                        break
        return names or None

    def _is_served(self, names: list[str]) -> bool:
        """Whether ``names`` contains the configured model, allowing Ollama's tag.

        Ollama lists its models tag-qualified (``bge-m3:latest``) but resolves an
        *untagged* request name to that same ``:latest`` tag, so every Ollama
        deployment documented here -- which all pin the untagged ``bge-m3`` --
        is genuinely served by a list that only ever says ``bge-m3:latest``.

        Only ``:latest`` is accepted, and only when the configured name carries
        no tag of its own. Any other tag is a variant the endpoint would *not*
        resolve an untagged request to, so accepting it would widen the guard
        into the silent-wrong-vectors hole this exists to close (#220).
        """
        model = settings.embed_model
        if model in names:
            return True
        return ":" not in model and f"{model}:latest" in names

    def verify_model(self) -> None:
        """Raise if the endpoint definitively does not serve the configured model.

        The request's ``model`` field cannot be trusted to mean anything:
        llama-server ignores it and answers with whatever it has loaded, echoing
        the requested name back. That is not a hypothetical -- a server-side
        model swap under a constant alias silently mixed two embedding spaces
        for a day (see #220), invisible to both the dimension check and the
        ``meta.embed_model`` marker, because the marker records the name that
        was *asked for*.

        So ask the endpoint what it actually has. Silent when it will not say,
        loud only when it answers and the configured model is not in the list --
        where "in the list" is tag-aware for Ollama, see :meth:`_is_served`.

        Raises:
            EmbedModelMismatch: The endpoint serves models, none of them the
                configured one.
        """
        names = self.served_models()
        if names is None or self._is_served(names):
            return
        raise EmbedModelMismatch(
            f"{self._url.removesuffix('/embeddings')} does not serve "
            f"'{settings.embed_model}' -- it serves {', '.join(sorted(set(names)))}. "
            "Embedding requests would still succeed and return vectors from the "
            "wrong model, so this refuses to start instead. Point "
            "FUKO_EMBED_MODEL at what is actually served (and move "
            "FUKO_EMBED_QUERY_PREFIX with it)."
        )

    def probe_dim(self) -> int:
        """Return the embedding dimension reported by the configured model.

        Embeds a tiny probe string and caches the resulting vector length, so
        the pgvector column can be sized to whatever the model actually returns
        instead of a hard-coded guess.
        """
        if self._dim is None:
            self._dim = len(self.embed_one("dimension probe"))
        return self._dim


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Return the process-wide singleton ``Embedder``."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
