"""Unit tests for the impersonated-token minting and the loopback metadata stand-in."""

import json

import httpx
import pytest

from sidecar.reviewer import gcp_token as gt

SA = "ai-reviewer@proj.iam.gserviceaccount.com"


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _adc(tmp_path, **info):
    path = tmp_path / "adc.json"
    path.write_text(json.dumps(info))
    return {"GOOGLE_APPLICATION_CREDENTIALS": str(path)}


# --- the runner's own credential ------------------------------------------------


def test_an_authorized_user_adc_file_is_refreshed(tmp_path):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = request.content.decode()
        return httpx.Response(200, json={"access_token": "SRC"})

    env = _adc(
        tmp_path, type="authorized_user", client_id="c", client_secret="s", refresh_token="r"
    )
    assert gt.source_token(env, _client(handler)) == "SRC"
    assert seen["url"] == "https://oauth2.googleapis.com/token"
    assert "grant_type=refresh_token" in seen["body"] and "refresh_token=r" in seen["body"]


def test_the_gcloud_default_adc_file_under_home_is_found(tmp_path):
    adc = tmp_path / ".config" / "gcloud" / "application_default_credentials.json"
    adc.parent.mkdir(parents=True)
    adc.write_text(
        json.dumps(
            {
                "type": "authorized_user",
                "client_id": "c",
                "client_secret": "s",
                "refresh_token": "r",
            }
        )
    )
    client = _client(lambda r: httpx.Response(200, json={"access_token": "HOME-SRC"}))
    assert gt.source_token({"HOME": str(tmp_path)}, client) == "HOME-SRC"


def test_a_key_file_is_refused_as_the_source(tmp_path):
    env = _adc(tmp_path, type="service_account", private_key="k")
    with pytest.raises(gt.GcpTokenError, match="'service_account'"):
        gt.source_token(env, _client(lambda r: httpx.Response(500)))


@pytest.mark.parametrize("content", ["not json", "[]"])
def test_an_unreadable_or_odd_adc_file_says_so(tmp_path, content):
    path = tmp_path / "adc.json"
    path.write_text(content)
    with pytest.raises(gt.GcpTokenError):
        gt.source_token({"GOOGLE_APPLICATION_CREDENTIALS": str(path)}, _client(lambda r: None))


def test_an_incomplete_authorized_user_file_names_the_missing_field(tmp_path):
    env = _adc(tmp_path, type="authorized_user", client_id="c")
    with pytest.raises(gt.GcpTokenError, match="client_secret"):
        gt.source_token(env, _client(lambda r: None))


def test_without_an_adc_file_the_metadata_server_answers(tmp_path):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["flavor"] = request.headers.get("Metadata-Flavor")
        return httpx.Response(200, json={"access_token": "VM"})

    env = {"HOME": str(tmp_path), "GCE_METADATA_HOST": "10.0.0.9:8080"}
    assert gt.source_token(env, _client(handler)) == "VM"
    assert seen["url"].startswith("http://10.0.0.9:8080/computeMetadata/v1/instance/")
    assert seen["flavor"] == "Google"


def test_an_unreachable_metadata_server_is_named(tmp_path):
    def handler(request):
        raise httpx.ConnectError("nope")

    with pytest.raises(gt.GcpTokenError, match="metadata.google.internal"):
        gt.source_token({"HOME": str(tmp_path)}, _client(handler))


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, text="denied"),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"access_token": ""}),
    ],
)
def test_a_bad_source_answer_is_an_error_not_a_token(tmp_path, response):
    with pytest.raises(gt.GcpTokenError):
        gt.source_token({"HOME": str(tmp_path)}, _client(lambda r: response))


# --- impersonation --------------------------------------------------------------


def _iam_client(calls, status=200, body=None):
    def handler(request):
        if "metadata" in str(request.url) or request.url.host == "127.0.0.9":
            return httpx.Response(200, json={"access_token": "SRC"})
        calls.append(request)
        payload = {"accessToken": f"MINTED-{len(calls)}"} if body is None else body
        return httpx.Response(status, json=payload)

    return _client(handler)


def test_mint_exchanges_the_source_for_a_scoped_bounded_token(tmp_path):
    calls = []
    env = {"HOME": str(tmp_path), "GCE_METADATA_HOST": "127.0.0.9"}
    token = gt.ImpersonatedToken(SA, env, lifetime_s=99999, client=_iam_client(calls))
    current = token.current()
    assert current.value == "MINTED-1"
    request = calls[0]
    assert str(request.url).endswith(f"/serviceAccounts/{SA}:generateAccessToken")
    assert request.headers["Authorization"] == "Bearer SRC"
    assert json.loads(request.content) == {"scope": [gt.SCOPE], "lifetime": "3600s"}
    assert 3590 <= current.seconds_left() <= 3600
    token.close()


def test_the_token_is_cached_until_it_nears_expiry(tmp_path, monkeypatch):
    calls = []
    clock = [1000.0]
    monkeypatch.setattr(gt.time, "monotonic", lambda: clock[0])
    env = {"HOME": str(tmp_path), "GCE_METADATA_HOST": "127.0.0.9"}
    token = gt.ImpersonatedToken(SA, env, lifetime_s=600, client=_iam_client(calls))
    assert token.current().value == "MINTED-1"
    clock[0] += 200  # 400 s left: still served
    assert token.current().value == "MINTED-1"
    clock[0] += 150  # 250 s left: inside the refresh margin
    assert token.current().value == "MINTED-2"
    assert len(calls) == 2


def test_a_denied_impersonation_names_the_missing_grant(tmp_path):
    env = {"HOME": str(tmp_path), "GCE_METADATA_HOST": "127.0.0.9"}
    token = gt.ImpersonatedToken(SA, env, client=_iam_client([], status=403, body={"error": {}}))
    with pytest.raises(gt.GcpTokenError, match="serviceAccountTokenCreator"):
        token.current()


def test_an_answer_without_a_token_is_an_error(tmp_path):
    env = {"HOME": str(tmp_path), "GCE_METADATA_HOST": "127.0.0.9"}
    token = gt.ImpersonatedToken(SA, env, client=_iam_client([], body={"expireTime": "x"}))
    with pytest.raises(gt.GcpTokenError, match="accessToken"):
        token.current()


def test_a_transport_failure_while_minting_is_an_error(tmp_path):
    def handler(request):
        if request.url.host == "127.0.0.9":
            return httpx.Response(200, json={"access_token": "SRC"})
        raise httpx.ReadTimeout("slow")

    env = {"HOME": str(tmp_path), "GCE_METADATA_HOST": "127.0.0.9"}
    token = gt.ImpersonatedToken(SA, env, client=_client(handler))
    with pytest.raises(gt.GcpTokenError, match="generateAccessToken"):
        token.current()


# --- the stand-in, on a real loopback socket ------------------------------------


class _FixedToken:
    service_account = SA

    def __init__(self, fail=False):
        self.fail = fail

    def current(self):
        if self.fail:
            raise gt.GcpTokenError("no grant")
        return gt.AccessToken(value="SERVED", expires_at=gt.time.monotonic() + 1200)


def _get(host, path, flavor=True):
    headers = {"Metadata-Flavor": "Google"} if flavor else {}
    return httpx.get(f"http://{host}{path}", headers=headers, timeout=5)


def test_the_stand_in_speaks_just_enough_metadata():
    with gt.MetadataStandIn(_FixedToken(), "proj") as stand_in:
        host = stand_in.host
        assert host.startswith("127.0.0.1:")
        token = _get(host, "/computeMetadata/v1/instance/service-accounts/default/token?scopes=x")
        assert token.status_code == 200
        assert token.headers["Metadata-Flavor"] == "Google"
        assert token.json()["access_token"] == "SERVED"
        assert 1190 <= token.json()["expires_in"] <= 1200
        assert _get(host, "/computeMetadata/v1/project/project-id").text == "proj"
        email = _get(host, "/computeMetadata/v1/instance/service-accounts/default/email")
        assert email.text == SA
        assert _get(host, "/computeMetadata/v1/instance").status_code == 200
        assert _get(host, "/computeMetadata/v1/instance/attributes/ssh-keys").status_code == 404
        unflavored = _get(
            host, "/computeMetadata/v1/instance/service-accounts/default/token", False
        )
        assert unflavored.status_code == 403
    with pytest.raises(httpx.ConnectError):
        _get(host, "/computeMetadata/v1/instance")
    with pytest.raises(RuntimeError, match="not running"):
        stand_in.host


def test_a_failed_mint_reaches_the_cli_as_a_server_error():
    with gt.MetadataStandIn(_FixedToken(fail=True), "proj") as stand_in:
        resp = _get(stand_in.host, "/computeMetadata/v1/instance/service-accounts/default/token")
        assert resp.status_code == 500
        assert "no grant" in resp.text
