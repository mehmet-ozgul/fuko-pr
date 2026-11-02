"""What other reviewers already raised, rendered for the prompt."""

import httpx

from sidecar.reviewer import peers
from sidecar.signals import ReviewSignal, encode_marker

_BUGBOT = {
    "id": 1,
    "path": "app/billing.py",
    "line": None,
    "html_url": "https://github.com/o/r/pull/9#discussion_r1",
    "user": {"login": "cursor[bot]", "type": "Bot"},
    "body": (
        "### Discount applied twice\n\n**High Severity**\n\n"
        "<!-- DESCRIPTION START -->\nrenew() applies it again.\n<!-- DESCRIPTION END -->\n\n"
        "<!-- BUGBOT_BUG_ID: 0f1e2d3c-aaaa -->\n\n"
        "<!-- LOCATIONS START\napp/billing.py#L48-L55\nLOCATIONS END -->"
    ),
}


def _human(body, **kw):
    return {
        "id": kw.pop("id", 2),
        "path": "app/pricing.py",
        "line": 10,
        "user": {"login": "randy", "type": "User"},
        "body": body,
        **kw,
    }


def _fuko_marker():
    return encode_marker(ReviewSignal(id="fk1", title="t", body="b", backend="agentic", model="m"))


def test_bot_findings_and_people_are_listed():
    text = peers.render_already_raised([_BUGBOT, _human("This leaks the lock on error.")], [])
    lines = text.splitlines()
    assert lines[0] == (
        "- [bugbot · high] app/billing.py:48 (outdated) — Discount applied twice: "
        "renew() applies it again."
    )
    assert lines[1] == "- [randy] app/pricing.py:10 — This leaks the lock on error."


def test_replies_and_fukos_own_comments_are_left_out():
    reply = _human("agreed", id=3, in_reply_to_id=2)
    ours = _human(f"**finding**\n\n{_fuko_marker()}", id=4)
    our_review = {
        "user": {"login": "avoxi-swdev", "type": "User"},
        "state": "COMMENTED",
        "body": "> 🤖 Automated review\n\n## fuko agentic review — `m`\n\nno findings",
    }
    assert peers.render_already_raised([reply, ours], [our_review]) == ""


def test_a_persons_review_body_is_listed_and_a_bots_is_not():
    randy = {"login": "randy", "type": "User"}
    reviews = [
        {"user": randy, "state": "CHANGES_REQUESTED", "body": "Fix X."},
        {"user": {"login": "cursor[bot]", "type": "Bot"}, "state": "COMMENTED", "body": "found 1"},
        {"user": randy, "state": "APPROVED", "body": ""},
    ]
    assert (
        peers.render_already_raised([], reviews) == "- [randy · review changes_requested]\n  Fix X."
    )


def test_pr_comments_from_people_are_listed_with_their_lines():
    review = {
        "id": 20,
        "user": {"login": "charlie", "type": "User"},
        "body": (
            "**Note: written by an AI reviewer.**\n\n\n\n"
            "**Verdict:** approve\n\n### fast-follow — event PT"
        ),
    }
    command = {"id": 21, "user": {"login": "randy", "type": "User"}, "body": "/ai-review"}
    bot = {"id": 22, "user": {"login": "cursor[bot]", "type": "Bot"}, "body": "status"}
    ours = {"id": 23, "user": {"login": "svc", "type": "User"}, "body": f"x\n{_fuko_marker()}"}
    text = peers.render_already_raised([], [], [review, command, bot, ours])
    assert text == (
        "- [charlie · PR comment]\n"
        "  **Note: written by an AI reviewer.**\n\n"
        "  **Verdict:** approve\n\n"
        "  ### fast-follow — event PT"
    )


def test_a_long_pr_comment_is_clipped(monkeypatch):
    monkeypatch.setattr(peers, "MAX_LONG_TEXT", 50)
    comment = {"id": 30, "user": {"login": "randy", "type": "User"}, "body": "y" * 200}
    body = peers.render_already_raised([], [], [comment]).splitlines()[1]
    assert body == "  " + "y" * 49 + "…"


def _mock_github(monkeypatch, handler):
    real = httpx.Client
    monkeypatch.setattr(
        peers.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )


def test_fetch_drops_pr_comments_from_the_posting_account(monkeypatch):
    def handler(request):
        if request.url.path == "/user":
            return httpx.Response(200, json={"id": 7, "login": "svc"})
        if request.url.path.endswith("/issues/9/comments"):
            return httpx.Response(
                200,
                json=[
                    {"id": 1, "user": {"id": 7, "login": "svc"}, "body": "Code Coverage Report"},
                    {"id": 2, "user": {"id": 8, "login": "charlie"}, "body": "a review"},
                ],
            )
        return httpx.Response(200, json=[])

    _mock_github(monkeypatch, handler)
    inline, reviews, issue = peers.fetch_peer_activity("o/r", 9, token="t", api_url="https://api.x")
    assert (inline, reviews) == ([], [])
    assert [c["id"] for c in issue] == [2]


def test_fetch_keeps_pr_comments_when_the_posting_account_is_unknown(monkeypatch):
    def handler(request):
        if request.url.path == "/user":
            return httpx.Response(403, json={"message": "Resource not accessible by integration"})
        if request.url.path.endswith("/issues/9/comments"):
            return httpx.Response(200, json=[{"id": 2, "user": {"id": 8}, "body": "a review"}])
        return httpx.Response(200, json=[])

    _mock_github(monkeypatch, handler)
    _, _, issue = peers.fetch_peer_activity("o/r", 9, token="t", api_url="https://api.x")
    assert [c["id"] for c in issue] == [2]


def test_long_text_is_clipped_and_the_list_is_capped(monkeypatch):
    monkeypatch.setattr(peers, "MAX_ENTRIES", 2)
    comments = [_human("x" * 1000, id=10 + n) for n in range(4)]
    lines = peers.render_already_raised(comments, []).splitlines()
    assert len(lines) == 3
    assert lines[-1] == "- (2 more not shown)"
    assert lines[0].endswith("…") and len(lines[0]) < peers.MAX_TEXT + 60
