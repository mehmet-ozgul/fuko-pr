"""Unit tests for the embeddings client error handling (no network)."""

import httpx
import pytest

from sidecar.config import settings
from sidecar.embed import EmbedError, Embedder
from sidecar.reviewer import gcp_token
from sidecar.retrieve import _build_query


def test_embed_empty_returns_empty():
    assert Embedder().embed([]) == []


def test_embed_surfaces_error_body(monkeypatch):
    def fake_post(self, url, headers=None, json=None):
        request = httpx.Request("POST", url, headers=headers or {})
        response = httpx.Response(400, request=request, text='{"error":{"message":"bad model"}}')
        raise httpx.HTTPStatusError("400 Bad Request", request=request, response=response)

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    with pytest.raises(EmbedError) as exc:
        Embedder().embed(["hi"])
    assert "bad model" in str(exc.value)
    assert "400" in str(exc.value)


def test_embed_truncates_oversized_input(monkeypatch):
    sent: list[list[str]] = []

    def fake_post(self, url, headers=None, json=None):
        sent.append(json["input"])
        request = httpx.Request("POST", url)
        return httpx.Response(200, request=request, json={"data": [{"embedding": [0.0]}]})

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    monkeypatch.setattr(settings, "embed_max_chars", 100)
    Embedder().embed(["x" * 5000])
    assert sent == [["x" * 100]]


def test_embed_leaves_a_short_input_alone(monkeypatch):
    sent: list[list[str]] = []

    def fake_post(self, url, headers=None, json=None):
        sent.append(json["input"])
        request = httpx.Request("POST", url)
        return httpx.Response(200, request=request, json={"data": [{"embedding": [0.0]}]})

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    monkeypatch.setattr(settings, "embed_max_chars", 100)
    Embedder().embed(["short enough"])
    assert sent == [["short enough"]]


def test_embed_query_applies_the_instruction_prefix(monkeypatch):
    sent: list[list[str]] = []

    def fake_post(self, url, headers=None, json=None):
        sent.append(json["input"])
        request = httpx.Request("POST", url)
        return httpx.Response(200, request=request, json={"data": [{"embedding": [0.0]}]})

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    monkeypatch.setattr(settings, "embed_query_prefix", "Instruct: task\nQuery: ")
    Embedder().embed_query("why does retrieval fail")
    assert sent == [["Instruct: task\nQuery: why does retrieval fail"]]


def test_embed_one_stays_unprefixed(monkeypatch):
    # Documents must not carry the query instruction: prefixing both sides is
    # the failure this asymmetry exists to prevent.
    sent: list[list[str]] = []

    def fake_post(self, url, headers=None, json=None):
        sent.append(json["input"])
        request = httpx.Request("POST", url)
        return httpx.Response(200, request=request, json={"data": [{"embedding": [0.0]}]})

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    monkeypatch.setattr(settings, "embed_query_prefix", "Instruct: task\nQuery: ")
    Embedder().embed_one("a stored learning")
    assert sent == [["a stored learning"]]


def test_assembled_query_reaches_the_endpoint_with_its_files_block(monkeypatch):
    # The end-to-end property the three caps have to compose into: what _fit
    # actually posts still carries the "Changed files:" block, so the transport
    # backstop never fires on a query _build_query assembled.
    #
    # Driven through embed_query, not embed: the query instruction is prepended
    # inside embed_query and _fit cuts the COMBINED string, so a test that
    # posted the bare assembled query would assert a composition the production
    # path never performs -- and would keep passing while the real one lost the
    # tail of the files block.
    sent: list[list[str]] = []

    def fake_post(self, url, headers=None, json=None):
        sent.append(json["input"])
        request = httpx.Request("POST", url)
        return httpx.Response(200, request=request, json={"data": [{"embedding": [0.0]}]})

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    files = [f"packages/shared/src/module_{i:04d}/index.ts" for i in range(300)]
    q = _build_query(files, "log line\n" * 5000, None)
    Embedder().embed_query(q)
    assert sent == [[settings.embed_query_prefix + q]]
    assert "Changed files:\npackages/shared/src/module_0000/index.ts" in sent[0][0]
    # Nothing was cut: the last path posted is the last path assembled, whole.
    assert sent[0][0].endswith(q[-len("index.ts") - 60 :])


# --- Vertex AI (native :predict, process ADC) -------------------------------------


class _Tok:
    def __init__(self, env=None, *, client=None):
        pass

    def bearer(self):
        return "ADC-TOKEN"


@pytest.fixture
def vertex(monkeypatch):
    monkeypatch.setattr(settings, "embed_provider", "vertex")
    monkeypatch.setattr(settings, "embed_vertex_project", "proj")
    monkeypatch.setattr(settings, "embed_vertex_location", "global")
    monkeypatch.setattr(settings, "embed_model", "text-embedding-005")
    monkeypatch.setattr(settings, "embed_dimensions", 768)
    monkeypatch.setattr(settings, "embed_batch_size", 2)
    monkeypatch.setattr(gcp_token, "SourceTokenCache", _Tok)
    calls = []

    def fake_post(self, url, headers=None, json=None):
        calls.append({"url": url, "headers": headers, "json": json})
        preds = [
            {"embeddings": {"values": [float(i)] * 3}} for i, _ in enumerate(json["instances"])
        ]
        return httpx.Response(200, json={"predictions": preds}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    return calls


def test_vertex_documents_are_batched_predict_calls_with_adc(vertex):
    vectors = Embedder().embed(["a", "b", "c"])
    assert len(vectors) == 3
    assert len(vertex) == 2  # batch size 2
    first = vertex[0]
    assert first["url"] == (
        "https://aiplatform.googleapis.com/v1/projects/proj/locations/global"
        "/publishers/google/models/text-embedding-005:predict"
    )
    assert first["headers"]["Authorization"] == "Bearer ADC-TOKEN"
    assert first["json"]["instances"] == [
        {"content": "a", "task_type": "RETRIEVAL_DOCUMENT"},
        {"content": "b", "task_type": "RETRIEVAL_DOCUMENT"},
    ]
    assert first["json"]["parameters"] == {"outputDimensionality": 768}


def test_vertex_queries_use_the_task_type_and_never_the_prefix(vertex, monkeypatch):
    monkeypatch.setattr(settings, "embed_query_prefix", "Instruct: PREFIX\nQuery: ")
    Embedder().embed_query("how do we lock?")
    (call,) = vertex
    assert call["json"]["instances"] == [
        {"content": "how do we lock?", "task_type": "RETRIEVAL_QUERY"}
    ]


def test_vertex_regional_location_and_native_dimensions(vertex, monkeypatch):
    monkeypatch.setattr(settings, "embed_vertex_location", "us-central1")
    monkeypatch.setattr(settings, "embed_dimensions", None)
    Embedder().embed_one("x")
    (call,) = vertex
    assert call["url"].startswith("https://us-central1-aiplatform.googleapis.com/v1/projects/proj/")
    assert "parameters" not in call["json"]


def test_vertex_needs_a_project(monkeypatch):
    monkeypatch.setattr(settings, "embed_provider", "vertex")
    monkeypatch.setattr(settings, "embed_vertex_project", "")
    with pytest.raises(EmbedError, match="FUKO_EMBED_VERTEX_PROJECT"):
        Embedder()


@pytest.mark.parametrize(
    "payload, match",
    [({"predictions": []}, "empty 'predictions'"), ({"predictions": [{"nope": 1}]}, "malformed")],
)
def test_vertex_bad_responses_are_embed_errors(vertex, monkeypatch, payload, match):
    def fake_post(self, url, headers=None, json=None):
        return httpx.Response(200, json=payload, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    with pytest.raises(EmbedError, match=match):
        Embedder().embed(["a"])


def test_vertex_cannot_be_asked_what_it_serves(vertex):
    embedder = Embedder()
    assert embedder.served_models() is None
    embedder.verify_model()  # "cannot verify" never refuses to start
