"""The two-part review comment (``[review] comment_style = "two-part"``)."""

import json
import re

import pytest
from pydantic import ValidationError

from sidecar.backends import agentic as agentic_mod
from sidecar.backends.agentic import AgenticBackend, _PendingReview
from sidecar.backends.base import PRRef
from sidecar.fukoconfig import ReviewConfig, ReviewModel
from sidecar.reviewer import comment
from sidecar.reviewer.prompt import AgenticFinding
from sidecar.signals import ReviewSignal, extract_markers, make_id

PR = PRRef(repo="o/r", number=9, url="https://github.com/o/r/pull/9")
HEADER = "## fuko agentic review — `vertex_ai/gemini-x`"
FACTS = comment.RunFacts(
    model="vertex_ai/gemini-x",
    head_sha="beefcafe1234",
    run_id="20261028T1-abc",
    round=2,
    examined=7,
)


def _finding(
    title, severity="medium", line=4, confidence="high", file="src/app.py", body="b", **kw
):
    return AgenticFinding(
        file=file, line=line, severity=severity, confidence=confidence, title=title, body=body, **kw
    )


def _pair(f):
    signal = ReviewSignal(
        id=make_id(PR.url, f.file, str(f.line or 0), f.title),
        file=f.file,
        line=f.line,
        severity=f.severity,
        severity_source="declared",
        category=f.category,
        title=f.title,
        body=f.body,
        backend="agentic",
        model="vertex_ai/gemini-x",
    )
    return f, signal


def _render(findings, anchorable=lambda f: True, **kw):
    return comment.render_two_part(
        [_pair(f) for f in findings],
        pr=9,
        header=HEADER,
        facts=kw.pop("facts", FACTS),
        human_findings=kw.pop("human_findings", 3),
        anchorable=anchorable,
    )


def _json_block(body):
    match = re.search(r"(`{3,})json\n(.*?)\n\1\n", body, re.S)
    assert match, body
    return json.loads(match.group(2))


def test_disclosure_comes_first_and_the_header_stays_for_the_read_back():
    body = _render([_finding("a")]).body
    first = body.splitlines()[0]
    assert first.startswith("> 🤖 **Automated review**")
    assert (
        "`vertex_ai/gemini-x`" in first and "run `20261028T1-abc`" in first and "round 2" in first
    )
    assert "does not approve or block" in first
    assert HEADER in body


def test_findings_rank_by_severity_then_confidence():
    findings = [
        _finding("low", "low"),
        _finding("high-but-unsure", "high", confidence="medium"),
        _finding("crit", "critical"),
        _finding("high-sure", "high"),
    ]
    ranked = [f.title for f, _ in comment.rank([_pair(f) for f in findings])]
    assert ranked == ["crit", "high-sure", "high-but-unsure", "low"]


def test_only_the_shown_findings_go_inline_and_every_finding_keeps_its_marker():
    findings = [_finding(f"f{i}", line=i + 1) for i in range(5)]
    rendered = _render(findings)
    assert [f.title for f, _ in rendered.inline] == ["f0", "f1", "f2"]
    assert "2 more findings of lower rank" in rendered.body
    posted = rendered.body + "".join(c["body"] for _, c in rendered.inline)
    recovered = {s.id for s in extract_markers(posted)}
    assert recovered == {_pair(f)[1].id for f in findings}


def test_an_unanchorable_shown_finding_stays_in_the_body_with_its_marker():
    f = _finding("outside the diff", line=None)
    rendered = _render([f], anchorable=lambda _: False)
    assert rendered.inline == []
    assert "`src/app.py` — **outside the diff**" in rendered.body
    assert {s.id for s in extract_markers(rendered.body)} == {_pair(f)[1].id}


def test_the_assistant_section_carries_every_finding_as_finding_v1():
    findings = [_finding("crit", "critical", evidence="read x.go:1-9"), _finding("info", "info")]
    records = _json_block(_render(findings, human_findings=1).body)
    assert [r["claim"] for r in records] == ["crit", "info"]
    first = records[0]
    assert first["schema"] == "finding/v1"
    assert first["severity"] == "blocking" and first["severity_raw"] == "critical"
    assert records[1]["severity"] == "nit"
    assert first["fingerprint"] == _pair(findings[0])[1].id
    assert (first["pr"], first["head_sha"], first["round"]) == (9, "beefcafe1234", 2)
    assert first["evidence"] == "read x.go:1-9"
    assert "end_line" not in first  # not produced is absent, not null


def test_no_findings_says_so_and_has_no_assistant_section():
    body = _render([]).body
    assert "**Verdict:** no material findings." in body
    assert "<details>" not in body
    assert "read-only review of `beefcafe12`" in body
    assert "7 code regions were read" in body


def test_model_text_cannot_mention_anyone():
    f = AgenticFinding(
        file="a.py", line=1, title="ask @alice", body="cc @org/team", evidence="@bob"
    )
    rendered = _render([f])
    posted = rendered.body + rendered.inline[0][1]["body"]
    assert re.search(r"@[A-Za-z]", posted) is None


def test_backticks_in_a_finding_cannot_close_the_json_fence():
    f = _finding("x", body="see ````` and ``` here")
    records = _json_block(_render([f]).body)
    assert records[0]["detail"] == "see ````` and ``` here"


def _backend(**config):
    backend = AgenticBackend(ReviewConfig(backend="agentic", comment_style="two-part", **config))
    backend._pending[(PR.url, "claude-x", agentic_mod._identity("tok"))] = _PendingReview(
        findings=[_finding(f"f{i}", line=i + 1) for i in range(4)],
        summary="s",
        head_sha="beefcafe1234",
        diff_files=frozenset({"src/app.py"}),
        run_id="20261028T1-abc",
        round=1,
        examined=3,
    )
    return backend


def test_backend_posts_the_rendered_body_and_three_inline_comments(monkeypatch):
    backend = _backend()
    seen = {}

    def fake_post_review(pr, token, api_url, stash, inline, overflow, label="", body=None):
        seen.update(inline=inline, overflow=overflow, body=body)
        return True

    monkeypatch.setattr(backend, "_post_review", fake_post_review)
    signals = backend.normalize_output(PR, "anthropic/claude-x", token="tok", api_url="https://x")
    assert len(signals) == 4
    assert len(seen["inline"]) == 3
    assert seen["body"].startswith("> 🤖 **Automated review**")
    assert "run `20261028T1-abc`" in seen["body"]


def test_shadow_preview_prints_the_comment_instead_of_posting(monkeypatch, capsys):
    backend = _backend(post=False, shadow_preview=True)
    monkeypatch.setattr(backend, "_post_review", lambda *a, **k: pytest.fail("posted"))
    backend.normalize_output(PR, "anthropic/claude-x", token="tok", api_url="https://x")
    err = capsys.readouterr().err
    assert "fuko: shadow| > 🤖 **Automated review**" in err
    assert "fuko: shadow| [inline src/app.py:1]" in err


def test_two_part_is_refused_for_a_pr_agent_seat():
    with pytest.raises(ValidationError, match="comment_style = 'two-part' needs the agentic"):
        ReviewConfig(comment_style="two-part")
    with pytest.raises(ValidationError, match="comment_style"):
        ReviewConfig(
            backend="agentic",
            comment_style="two-part",
            models=[ReviewModel(provider="p", name="m", role="active", backend="pr-agent")],
        )


def test_an_inline_finding_is_one_line_in_the_body_and_detailed_inline():
    f = _finding("leak", body="the full explanation")
    rendered = _render([f])
    human = rendered.body.split("<details>")[0]
    assert "**leak** (details inline)" in human
    assert "the full explanation" not in human
    assert "the full explanation" in rendered.inline[0][1]["body"]


def test_no_examined_regions_is_not_stated_as_zero():
    facts = comment.RunFacts(model="m", head_sha="beefcafe1234", examined=0)
    body = _render([], facts=facts).body
    assert "Nothing was built, run or tested." in body
    assert "code regions" not in body
