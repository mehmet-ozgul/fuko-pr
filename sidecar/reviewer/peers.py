"""What other reviewers already raised on a PR, rendered for the review prompt.

A review that re-raises what Bugbot or a teammate already said costs the author
a second read for nothing, and one that misses that a raised finding is fixed at
the current head leaves a stale claim standing. The reviewer is therefore shown
the findings already on the PR -- external bots through their normalizers, and
top-level comments from people, PR (issue) comments included -- and asked to
settle each one instead of repeating it. A review a teammate's AI assistant
posted from their account arrives as a PR comment, and it is the one most
worth settling: it often raised the same finding this round would.

fuko's own earlier findings are left out: the findings ledger carries them into
the prompt already, with ids the round must cite. Everything rendered here is
contributor-reachable text and goes into the prompt fenced, as data.
"""

from __future__ import annotations

import re

import httpx

from ..normalizers import collect_review_signals, collect_signals
from ..signals import extract_markers
from .checkout import _api_headers

#: Reviewers whose findings come back through :mod:`sidecar.normalizers`.
EXTERNAL_BACKENDS = frozenset({"bugbot", "coderabbit", "copilot", "pr-agent"})
#: fuko's own review body opens with this; one with no findings carries no marker.
FUKO_REVIEW_HEADER = "## fuko agentic review"
#: Entries rendered, and characters kept from each entry's text. A PR comment or
#: review body is a whole review, findings past its first lines, so it keeps more.
MAX_ENTRIES = 40
MAX_TEXT = 400
MAX_LONG_TEXT = 3000


def _paginated(client: httpx.Client, url: str) -> list[dict]:
    out: list[dict] = []
    page = 1
    while True:
        resp = client.get(url, params={"page": page, "per_page": 100})
        resp.raise_for_status()
        batch = resp.json()
        out.extend(batch)
        if len(batch) < 100:
            return out
        page += 1


def _own_user_id(client: httpx.Client, api: str) -> int | None:
    """The id of the account fuko posts as, or None when ``GET /user`` won't say.

    App installation tokens get a 403 here; their comments are bot-typed and
    filtered as bots anyway.
    """
    try:
        resp = client.get(f"{api}/user")
        resp.raise_for_status()
        return resp.json().get("id")
    except (httpx.HTTPError, ValueError):
        return None


def fetch_peer_activity(
    repo: str, number: int, *, token: str, api_url: str
) -> tuple[list[dict], list[dict], list[dict]]:
    """Return the PR's inline review comments, submitted reviews and PR comments.

    PR comments written by the account fuko posts as are dropped here. In a
    deployment that posts through a shared service user, that account also posts
    the CI reports (coverage, security scans) and fuko's own run notes, none of
    which is a finding to settle.
    """
    api = api_url.rstrip("/")
    base = f"{api}/repos/{repo}"
    with httpx.Client(timeout=30.0, headers=_api_headers(token)) as client:
        inline = _paginated(client, f"{base}/pulls/{number}/comments")
        reviews = _paginated(client, f"{base}/pulls/{number}/reviews")
        issue = _paginated(client, f"{base}/issues/{number}/comments")
        own = _own_user_id(client, api)
    if own is not None:
        issue = [c for c in issue if (c.get("user") or {}).get("id") != own]
    return inline, reviews, issue


def _is_bot(item: dict) -> bool:
    user = item.get("user") or {}
    return user.get("type") == "Bot" or (user.get("login") or "").endswith("[bot]")


def _is_fuko(body: str) -> bool:
    return bool(extract_markers(body)) or FUKO_REVIEW_HEADER in body


def _clip(text: str) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= MAX_TEXT else flat[: MAX_TEXT - 1] + "…"


def _clip_long(text: str) -> str:
    """Keep a whole review's line structure, indented under its entry."""
    lines = [line.rstrip() for line in (text or "").strip().splitlines()]
    kept = [line for n, line in enumerate(lines) if line or (n and lines[n - 1])]
    body = "\n".join(kept)
    if len(body) > MAX_LONG_TEXT:
        body = body[: MAX_LONG_TEXT - 1] + "…"
    return "\n".join("  " + line if line else "" for line in body.splitlines())


def _detail(body: str, title: str) -> str:
    """The body without the lines that only repeat the title or a severity tag."""
    kept = [
        line
        for line in (body or "").splitlines()
        if title not in line and not re.fullmatch(r"\s*(#+.*|\*\*\w+ Severity\*\*)\s*", line)
    ]
    return "\n".join(kept)


def _where(path: str | None, line: int | None) -> str:
    if not path:
        return "(no file)"
    return f"{path}:{line}" if line else path


def render_already_raised(
    inline: list[dict], reviews: list[dict], issue_comments: list[dict] | None = None
) -> str:
    """Render other reviewers' findings and people's comments, one entry each.

    Bot findings come from the normalizers, so their severity is the vendor's.
    A comment from a person is any top-level inline comment, non-empty review
    body or PR comment that is neither a bot's nor fuko's own (a marker or fuko's
    review header): fuko posts under a user token in some deployments, so the
    author type alone does not tell. A PR comment that is a command
    (``/ai-review``) carries no finding and is left out.
    Review bodies and PR comments keep their lines, up to :data:`MAX_LONG_TEXT`.
    An inline comment GitHub no longer anchors (``line`` null) is marked
    outdated: the code under it changed since it was written.
    """
    entries: list[str] = []
    signals = collect_signals(inline) + collect_review_signals(reviews)
    by_url = {c.get("html_url"): c for c in inline}
    for signal in signals:
        if signal.backend not in EXTERNAL_BACKENDS:
            continue
        source = by_url.get(signal.thread_url) or {}
        outdated = " (outdated)" if source and source.get("line") is None else ""
        entries.append(
            f"- [{signal.backend} · {signal.severity}] {_where(signal.file, signal.line)}"
            f"{outdated} — {_clip(signal.title)}: {_clip(_detail(signal.body, signal.title))}"
        )
    for comment in inline:
        if comment.get("in_reply_to_id") or _is_bot(comment) or _is_fuko(comment.get("body") or ""):
            continue
        login = (comment.get("user") or {}).get("login", "?")
        outdated = " (outdated)" if comment.get("line") is None else ""
        line = comment.get("line") or comment.get("original_line")
        entries.append(
            f"- [{login}] {_where(comment.get('path'), line)}{outdated} — "
            f"{_clip(comment.get('body') or '')}"
        )
    for review in reviews:
        body = (review.get("body") or "").strip()
        if not body or _is_bot(review) or _is_fuko(body):
            continue
        login = (review.get("user") or {}).get("login", "?")
        state = review.get("state", "").lower()
        entries.append(f"- [{login} · review {state}]\n{_clip_long(body)}")
    for comment in issue_comments or []:
        body = (comment.get("body") or "").strip()
        if not body or body.startswith("/") or _is_bot(comment) or _is_fuko(body):
            continue
        login = (comment.get("user") or {}).get("login", "?")
        entries.append(f"- [{login} · PR comment]\n{_clip_long(body)}")
    if len(entries) > MAX_ENTRIES:
        cut = len(entries) - MAX_ENTRIES
        entries = entries[:MAX_ENTRIES] + [f"- ({cut} more not shown)"]
    return "\n".join(entries)
