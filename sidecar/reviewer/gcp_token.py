"""A short-lived, impersonated Google token for the Gemini CLI harness.

The Gemini CLI's Vertex path authenticates only through Application Default
Credentials -- a credentials file, or the GCE metadata server -- and ignores a
bare access token (``GOOGLE_CLOUD_ACCESS_TOKEN`` is Code Assist's). On a shared
CI runner that leaves two bad choices: a long-lived service-account key on
disk, or the VM's own metadata identity, which is usually far broader than a
reviewer needs.

This module is the third choice. :class:`ImpersonatedToken` mints a token for
a narrow service account through IAM Credentials ``generateAccessToken``, using
the runner's own credential as the source; :class:`MetadataStandIn` serves it
on a loopback port that speaks just enough of the metadata protocol for ADC,
and the harness is pointed at it with ``GCE_METADATA_HOST``. Measured against
Gemini CLI 0.65.0: with no credentials file, the CLI makes one availability
request (``/computeMetadata/v1/instance``) and one token request, then calls
Vertex with the served token.

Properties worth knowing:

* **Nothing lands on disk.** The minted token lives in this process and the
  stand-in's responses; the harness environment carries only the loopback
  address.
* **Unreachable from the agent.** The harness has no network tool, and the
  stand-in lives only for the run. It is reachable by other processes on the
  runner for that window, the same trust boundary as the runner itself; it
  answers only requests carrying ``Metadata-Flavor: Google`` (the header a
  browser cannot set cross-origin, which is why the real server requires it).
* **Bounded lifetime.** Tokens are minted for at most an hour and re-minted on
  demand when one is within :data:`REFRESH_MARGIN_S` of expiry, so a run longer
  than the first token's life keeps working.
* **The source is the runner's own identity**: the real metadata server, or a
  gcloud ``authorized_user`` ADC file (a laptop). A service-account KEY file as
  the source is refused -- signing it needs a crypto dependency, and a key on
  the runner is the thing this module exists to avoid.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

#: Scope the minted token carries; Vertex accepts nothing narrower.
SCOPE = "https://www.googleapis.com/auth/cloud-platform"

#: Longest lifetime ``generateAccessToken`` grants without an org-policy change.
MAX_LIFETIME_S = 3600

#: Re-mint when the current token has less than this left.
REFRESH_MARGIN_S = 300

_REAL_METADATA_HOST = "metadata.google.internal"
_IAM_CREDENTIALS = "https://iamcredentials.googleapis.com/v1"
_OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
_HTTP_TIMEOUT = 20.0


class GcpTokenError(RuntimeError):
    """The runner could not obtain or exchange a credential."""


@dataclass(frozen=True)
class AccessToken:
    """A bearer token and the monotonic time it stops being served."""

    value: str
    expires_at: float

    def seconds_left(self) -> int:
        """Whole seconds until expiry, never negative."""
        return max(0, int(self.expires_at - time.monotonic()))


def _adc_file(env: dict[str, str]) -> Path | None:
    """The ADC file the runner's own credential would come from, if any."""
    explicit = env.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if explicit:
        return Path(explicit)
    home = env.get("HOME")
    if home:
        default = Path(home) / ".config" / "gcloud" / "application_default_credentials.json"
        if default.is_file():
            return default
    return None


def source_token(env: dict[str, str], client: httpx.Client) -> str:
    """Return an access token for the RUNNER's own identity.

    An ADC file wins when there is one (explicit, else gcloud's default under
    ``HOME``), and only the ``authorized_user`` type is supported; otherwise
    the metadata server answers, at ``GCE_METADATA_HOST`` if the runner sets
    it. Every failure is a :class:`GcpTokenError` naming what was tried.
    """
    adc = _adc_file(env)
    if adc is not None:
        try:
            info = json.loads(adc.read_text())
        except (OSError, ValueError) as e:
            raise GcpTokenError(f"cannot read ADC file {adc}: {e}") from None
        kind = info.get("type") if isinstance(info, dict) else None
        if kind != "authorized_user":
            raise GcpTokenError(
                f"ADC file {adc} is of type {kind!r}; only 'authorized_user' (a gcloud "
                f"login) or the metadata server can be the impersonation source"
            )
        try:
            resp = client.post(
                _OAUTH_TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "client_id": info["client_id"],
                    "client_secret": info["client_secret"],
                    "refresh_token": info["refresh_token"],
                },
            )
        except KeyError as e:
            raise GcpTokenError(f"ADC file {adc} lacks {e}") from None
        except httpx.HTTPError as e:
            raise GcpTokenError(f"refreshing the ADC login failed: {e}") from None
        return _token_from(resp, "refreshing the ADC login")
    host = env.get("GCE_METADATA_HOST", "").strip() or _REAL_METADATA_HOST
    url = f"http://{host}/computeMetadata/v1/instance/service-accounts/default/token"
    try:
        resp = client.get(url, headers={"Metadata-Flavor": "Google"})
    except httpx.HTTPError as e:
        raise GcpTokenError(
            f"no ADC file and the metadata server at {host} is unreachable: {e}"
        ) from None
    return _token_from(resp, "the metadata server")


def _token_from(resp: httpx.Response, what: str) -> str:
    """Pull ``access_token`` out of an OAuth-shaped response, or raise."""
    if resp.status_code != 200:
        raise GcpTokenError(f"{what} answered {resp.status_code}: {resp.text[:200]}")
    try:
        token = resp.json()["access_token"]
    except (ValueError, KeyError, TypeError):
        raise GcpTokenError(f"{what} answered without an access_token") from None
    if not isinstance(token, str) or not token:
        raise GcpTokenError(f"{what} answered with an empty access_token")
    return token


class ImpersonatedToken:
    """Mints and caches a token for ``service_account``; thread-safe.

    ``env`` is the environment the runner's own credential is found in (the
    AMBIENT one, before the harness strips it). ``client`` is injectable for
    tests; the default is a private client with bounded timeouts.
    """

    def __init__(
        self,
        service_account: str,
        env: dict[str, str] | None = None,
        *,
        lifetime_s: int = MAX_LIFETIME_S,
        client: httpx.Client | None = None,
    ) -> None:
        """Bind the target account; nothing is fetched until :meth:`current`."""
        self.service_account = service_account
        self._env = dict(os.environ if env is None else env)
        self._lifetime = max(60, min(lifetime_s, MAX_LIFETIME_S))
        self._client = client or httpx.Client(timeout=_HTTP_TIMEOUT)
        self._lock = threading.Lock()
        self._token: AccessToken | None = None

    def current(self) -> AccessToken:
        """The cached token, re-minted first if it is near expiry or absent."""
        with self._lock:
            if self._token is None or self._token.seconds_left() < REFRESH_MARGIN_S:
                self._token = self._mint()
            return self._token

    def _mint(self) -> AccessToken:
        source = source_token(self._env, self._client)
        url = (
            f"{_IAM_CREDENTIALS}/projects/-/serviceAccounts/"
            f"{self.service_account}:generateAccessToken"
        )
        try:
            resp = self._client.post(
                url,
                headers={"Authorization": f"Bearer {source}"},
                json={"scope": [SCOPE], "lifetime": f"{self._lifetime}s"},
            )
        except httpx.HTTPError as e:
            raise GcpTokenError(f"generateAccessToken failed: {e}") from None
        if resp.status_code != 200:
            # A 403 here is almost always the missing grant, so name it.
            raise GcpTokenError(
                f"impersonating {self.service_account} failed ({resp.status_code}); the "
                f"runner's identity needs roles/iam.serviceAccountTokenCreator on it: "
                f"{resp.text[:200]}"
            )
        try:
            value = resp.json()["accessToken"]
        except (ValueError, KeyError, TypeError):
            raise GcpTokenError("generateAccessToken answered without an accessToken") from None
        # Our clock, not the server's expireTime: a skewed runner clock must not
        # serve a token past its real life. Minted now, so now + lifetime is an
        # upper bound.
        return AccessToken(value=value, expires_at=time.monotonic() + self._lifetime)

    def close(self) -> None:
        """Release the HTTP client."""
        self._client.close()


def _handler(token: ImpersonatedToken, project: str):
    class _MetadataHandler(BaseHTTPRequestHandler):
        """Just enough of the GCE metadata protocol for ADC; nothing else."""

        def do_GET(self) -> None:  # noqa: N802 - the stdlib's spelling
            if self.headers.get("Metadata-Flavor") != "Google":
                self._reply(403, "missing Metadata-Flavor header")
                return
            path = self.path.split("?", 1)[0].rstrip("/")
            if path.endswith("/instance/service-accounts/default/token"):
                try:
                    current = token.current()
                except GcpTokenError as e:
                    # Surfaces in the CLI's own auth error, which the driver
                    # classifies as auth rather than throttle.
                    self._reply(500, str(e))
                    return
                body = {
                    "access_token": current.value,
                    "expires_in": current.seconds_left(),
                    "token_type": "Bearer",
                }
                self._reply(200, json.dumps(body), "application/json")
            elif path.endswith("/instance/service-accounts/default/email"):
                self._reply(200, token.service_account)
            elif path.endswith("/project/project-id"):
                self._reply(200, project)
            elif path in ("/computeMetadata/v1", "/computeMetadata/v1/instance", ""):
                self._reply(200, "")
            else:
                self._reply(404, "not served")

        def _reply(self, status: int, body: str, ctype: str = "application/text") -> None:
            data = body.encode()
            self.send_response(status)
            self.send_header("Metadata-Flavor", "Google")
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args) -> None:
            pass  # access logs would print token paths into CI output for nothing

    return _MetadataHandler


class MetadataStandIn:
    """A loopback metadata server for one run; use as a context manager.

    ``host`` is the ``GCE_METADATA_HOST`` value (``127.0.0.1:<port>``), valid
    between ``__enter__`` and ``__exit__``. The port is chosen by the kernel.
    """

    def __init__(self, token: ImpersonatedToken, project: str) -> None:
        """Prepare a stand-in serving ``token`` for ``project``; not yet listening."""
        self._token = token
        self._project = project
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def host(self) -> str:
        """``127.0.0.1:<port>`` while running."""
        if self._server is None:
            raise RuntimeError("stand-in is not running")
        return f"127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self) -> MetadataStandIn:
        """Start listening on a kernel-chosen loopback port."""
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(self._token, self._project))
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        """Stop listening; the port is released before this returns."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
