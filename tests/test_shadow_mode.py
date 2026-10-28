"""Shadow mode (`[review] post = false`) and the head each run reviewed."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from sidecar import main, run_metrics, runner
from sidecar.backends import agentic as agentic_mod
from sidecar.backends.agentic import AgenticBackend, _PendingReview
from sidecar.backends.base import InvokeResult, PRRef
from sidecar.fukoconfig import ModelConfig, ReviewConfig, ReviewModel
from sidecar.reviewer.prompt import AgenticFinding

PR = PRRef(repo="o/r", number=9, url="https://github.com/o/r/pull/9")
_TOKEN = "test-token"
# Captured at import: an autouse fixture stubs `runner._record_run` per test.
_REAL_RECORD_RUN = runner._record_run


def _no_http(*_a, **_k):
    raise AssertionError("shadow mode must not write to GitHub")


def test_shadow_mode_is_accepted_for_the_agentic_backend():
    assert ReviewConfig(backend="agentic", post=False).post is False


def test_shadow_mode_is_rejected_for_pr_agent():
    """PR-Agent posts from inside its container, so post = false could not hold."""
    with pytest.raises(ValidationError, match="post = false needs the agentic backend"):
        ReviewConfig(post=False)


def test_shadow_mode_is_rejected_when_one_seat_runs_pr_agent():
    with pytest.raises(ValidationError, match=r"\['pr-agent'\]"):
        ReviewConfig(
            backend="agentic",
            post=False,
            models=[
                ReviewModel(provider="vertex", name="a", role="active"),
                ReviewModel(provider="p", name="b", role="active", backend="pr-agent"),
            ],
        )


def test_posting_stays_the_default():
    assert ReviewConfig().post is True
    assert AgenticBackend().post is True


def test_shadow_normalize_returns_signals_without_posting(monkeypatch, capsys):
    backend = AgenticBackend(ReviewConfig(backend="agentic", post=False))
    backend._pending[(PR.url, "claude-x", agentic_mod._identity("tok"))] = _PendingReview(
        findings=[
            AgenticFinding(file="src/app.py", line=4, severity="high", title="leak", body="b1"),
            AgenticFinding(file="docs/other.md", title="stale doc", body="b2"),
        ],
        summary="s",
        head_sha="beef",
        diff_files=frozenset({"src/app.py"}),
    )
    monkeypatch.setattr(backend, "_post_review", _no_http)
    signals = backend.normalize_output(PR, "anthropic/claude-x", token="tok", api_url="https://x")
    assert [s.title for s in signals] == ["leak", "stale doc"]
    assert "shadow mode: 2 findings" in capsys.readouterr().err
    # Claimed like a posted review, so a second normalize finds nothing to post.
    assert backend._pending == {}


def test_shadow_branch_header_is_not_posted(monkeypatch):
    monkeypatch.setattr(runner.httpx, "post", _no_http)
    got = runner._post_branch_header(PR, "tok", "https://api.github.com", "vertex/a", post=False)
    assert got == (None, None, None)


class _Backend:
    """Just enough of a backend for `_run_pool`: one successful invoke."""

    name = "agentic"

    def __init__(self, head_sha):
        self.head_sha = head_sha

    def build_env(self, *_a, **_k):
        return {}

    def invoke(self, *_a, **_k):
        return InvokeResult(returncode=0, head_sha=self.head_sha)

    def normalize_output(self, *_a, **_k):
        return []


@pytest.mark.parametrize("head_sha", ["0123abc", None])
def test_run_pool_records_the_head_the_backend_reviewed(monkeypatch, head_sha):
    seen = {}
    monkeypatch.setattr(runner, "_record_run", lambda *a, **k: seen.update(k))
    runner._run_pool(
        _Backend(head_sha),
        PR,
        "",
        {},
        ReviewConfig(backend="agentic"),
        [ModelConfig(provider="vertex", name="gemini-x")],
        set(),
        None,
    )
    assert seen["head_sha"] == head_sha


def test_record_run_posts_the_head(monkeypatch):
    monkeypatch.setenv("FUKO_URL", "http://fuko.internal:8000")
    posted = {}

    class _Resp:
        def raise_for_status(self):
            return None

    monkeypatch.setattr(
        runner.httpx, "post", lambda url, **kw: (posted.update(kw["json"]), _Resp())[1]
    )
    _REAL_RECORD_RUN(
        PR,
        ModelConfig(provider="vertex", name="gemini-x"),
        slot=None,
        duration_s=1.0,
        attempts=1,
        outcome="ok",
        findings=0,
        detail="",
        head_sha="0123abc",
    )
    assert posted["head_sha"] == "0123abc"


class _Conn:
    def __init__(self, rows=()):
        self.statements = []
        self._rows = list(rows)

    def execute(self, sql, params=()):
        self.statements.append((" ".join(sql.split()), tuple(params)))
        return self

    def fetchall(self):
        return self._rows


def _pg(monkeypatch, rows=()):
    import contextlib

    import sidecar.db

    monkeypatch.setattr(run_metrics.settings, "database_url", "postgres://x")
    conn = _Conn(rows)

    @contextlib.contextmanager
    def fake_db(*_a, **_k):
        yield conn

    monkeypatch.setattr(sidecar.db, "db", fake_db)
    return conn


def test_record_writes_the_head_last(monkeypatch):
    conn = _pg(monkeypatch)
    run_metrics.record("o/r", 9, "vertex", "gemini-x", head_sha="0123abc")
    sql, params = conn.statements[0]
    assert sql.startswith("INSERT INTO review_runs") and "transcript_key, head_sha)" in sql
    assert sql.count("%s") == len(params)
    assert params[-1] == "0123abc"


def test_recent_runs_narrows_to_one_pr_and_returns_the_head(monkeypatch):
    from datetime import datetime, timezone

    started = datetime(2026, 10, 28, tzinfo=timezone.utc)
    row = ("o/r", 9, "vertex", "gemini-x", None, started, 900.0, 1, "ok", 2, "agentic", "beef")
    conn = _pg(monkeypatch, rows=[row])
    (run,) = run_metrics.recent_runs(repo="o/r", pr=9, limit=500)
    sql, params = conn.statements[0]
    assert "WHERE repo = %s AND pr = %s" in sql
    assert params == ("o/r", 9, 200)
    assert run["pr"] == 9 and run["head_sha"] == "beef"


def _client(monkeypatch):
    monkeypatch.setattr(main.settings, "auth_token", _TOKEN)
    return TestClient(main.app, headers={"Authorization": f"Bearer {_TOKEN}"})


def test_metrics_run_endpoint_carries_the_head(monkeypatch):
    monkeypatch.setattr(main.settings, "database_url", "")
    seen = {}
    monkeypatch.setattr(run_metrics, "record", lambda *a, **kw: seen.update(kw))
    body = {"repo": "o/r", "pr": 9, "provider": "p", "model": "m", "head_sha": "0123abc"}
    assert _client(monkeypatch).post("/metrics/run", json=body).status_code == 200
    assert seen["head_sha"] == "0123abc"


def test_metrics_run_endpoint_rejects_a_head_that_is_not_a_sha(monkeypatch):
    body = {"repo": "o/r", "pr": 9, "provider": "p", "model": "m", "head_sha": "main"}
    assert _client(monkeypatch).post("/metrics/run", json=body).status_code == 422


def test_metrics_runs_endpoint(monkeypatch):
    rows = [
        {
            "repo": "o/r",
            "pr": 9,
            "provider": "vertex",
            "model": "gemini-x",
            "slot": None,
            "started_at": "2026-10-28T00:00:00+00:00",
            "duration_s": 900.0,
            "attempts": 1,
            "outcome": "ok",
            "findings": 2,
            "backend": "agentic",
            "head_sha": "beef",
        }
    ]
    seen = {}

    def fake_recent(repo=None, limit=50, pr=None):
        seen.update(repo=repo, limit=limit, pr=pr)
        return rows

    monkeypatch.setattr(run_metrics, "recent_runs", fake_recent)
    resp = _client(monkeypatch).get("/metrics/runs", params={"repo": "o/r", "pr": 9})
    assert resp.status_code == 200
    assert resp.json() == {"runs": rows}
    assert seen == {"repo": "o/r", "limit": 50, "pr": 9}


def test_metrics_runs_endpoint_needs_the_token(monkeypatch):
    monkeypatch.setattr(main.settings, "auth_token", _TOKEN)
    assert TestClient(main.app).get("/metrics/runs", params={"repo": "o/r"}).status_code == 401


def test_migration_014_adds_a_nullable_head_column():
    sql = (
        Path(__file__).resolve().parent.parent / "migrations" / "014_review_run_head_sha.sql"
    ).read_text(encoding="utf-8")
    stripped = "\n".join(ln for ln in sql.splitlines() if not ln.lstrip().startswith("--"))
    stmts = [" ".join(s.split()) for s in stripped.split(";") if s.strip()]
    assert stmts[0] == "ALTER TABLE review_runs ADD COLUMN IF NOT EXISTS head_sha TEXT"
    assert stmts[1].startswith("CREATE INDEX IF NOT EXISTS review_runs_repo_pr_idx")
