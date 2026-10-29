"""Unit tests for the Gemini CLI review harness."""

import json
import os
import re
import stat
from pathlib import Path

import pytest

from sidecar.reviewer import gemini_harness as gh
from sidecar.reviewer.harness import HarnessNotAvailableError, usage_tokens
from sidecar.throttle import TIMEOUT_RETURNCODE

STATS = {
    "total_tokens": 16237,
    "input_tokens": 14609,
    "output_tokens": 210,
    "cached": 3925,
    "input": 10684,
    "duration_ms": 14339,
    "tool_calls": 1,
}


def _lines(*events) -> list[str]:
    return [json.dumps(e) + "\n" for e in events]


def _fold(*events):
    seen = []
    outcome = gh._consume_gemini_stream(_lines(*events), lambda *a: seen.append(a))
    return outcome, seen


# --- the stream fold ------------------------------------------------------------


def test_final_text_is_the_assistant_text_after_the_last_tool_call():
    outcome, seen = _fold(
        {"type": "init", "model": "gemini-x"},
        {"type": "message", "role": "user", "content": "the prompt"},
        {"type": "message", "role": "assistant", "content": "Let me read ", "delta": True},
        {"type": "message", "role": "assistant", "content": "the file.", "delta": True},
        {"type": "tool_use", "tool_name": "read_file", "parameters": {"file_path": "/r/a.go"}},
        {"type": "tool_result", "status": "success", "output": ""},
        {"type": "message", "role": "assistant", "content": '{"summary": ', "delta": True},
        {"type": "message", "role": "assistant", "content": '"ok"}', "delta": True},
        {"type": "result", "status": "success", "stats": STATS},
    )
    assert outcome.text == '{"summary": "ok"}'
    assert outcome.saw_result is True
    assert outcome.subtype == "success"
    assert seen == [(1, "read_file", "/r/a.go")]


def test_usage_maps_uncached_input_and_keeps_the_raw_stats():
    outcome, _ = _fold({"type": "result", "status": "success", "stats": STATS})
    assert usage_tokens(outcome.usage) == {
        "input_tokens": 10684,
        # 210 reported plus 1418 of thinking (16237 - 14609 - 210).
        "output_tokens": 1628,
        "cache_read_tokens": 3925,
        "cache_write_tokens": None,
    }
    assert outcome.usage["gemini"] == STATS
    # Neither is reported by the CLI; "not measured" is never a zero.
    assert outcome.cost_usd is None
    assert outcome.turns is None


def test_turns_count_model_responses_and_parallel_calls_share_one():
    outcome, seen = _fold(
        {"type": "init", "model": "gemini-x"},
        {"type": "message", "role": "user", "content": "the prompt"},
        # Turn 1: two calls in parallel.
        {"type": "tool_use", "tool_name": "read_file", "parameters": {"file_path": "/r/a.go"}},
        {"type": "tool_use", "tool_name": "grep_search", "parameters": {"pattern": "X"}},
        {"type": "tool_result", "status": "success", "output": ""},
        {"type": "tool_result", "status": "success", "output": ""},
        # Turn 2: narration, then one call.
        {"type": "message", "role": "assistant", "content": "now ", "delta": True},
        {"type": "tool_use", "tool_name": "glob", "parameters": {"pattern": "*.go"}},
        {"type": "tool_result", "status": "success", "output": ""},
        # Turn 3: the answer.
        {"type": "message", "role": "assistant", "content": "{}", "delta": True},
        {"type": "result", "status": "success", "stats": STATS},
    )
    assert outcome.turns == 3
    assert len(seen) == 3


def test_settings_pin_every_helper_alias_to_the_seat_model():
    s = gh.harness_settings(max_turns=42, model="gemini-3.8-flash")
    aliases = s["modelConfigs"]["customAliases"]
    assert set(aliases) == {*gh.HELPER_ALIASES_ON_BASE, *gh.HELPER_ALIASES_PLAIN}
    for name in gh.HELPER_ALIASES_ON_BASE:
        assert aliases[name] == {"extends": "base", "modelConfig": {"model": "gemini-3.8-flash"}}
    for name in gh.HELPER_ALIASES_PLAIN:
        assert aliases[name] == {"modelConfig": {"model": "gemini-3.8-flash"}}
    assert "modelConfigs" not in gh.harness_settings(max_turns=42)


def test_output_tokens_include_the_thinking_vertex_bills_as_output():
    # Measured on a real review: total exceeds input + output by the thinking.
    stats = {
        "total_tokens": 9135366,
        "input_tokens": 9061210,
        "output_tokens": 9098,
        "cached": 8368910,
        "input": 692300,
    }
    outcome, _ = _fold({"type": "result", "status": "success", "stats": stats})
    assert usage_tokens(outcome.usage)["output_tokens"] == 9098 + 65058
    assert outcome.usage["gemini"] == stats


def test_output_tokens_never_shrink_on_an_inconsistent_total():
    stats = {"total_tokens": 10, "input_tokens": 20, "output_tokens": 5, "input": 20}
    outcome, _ = _fold({"type": "result", "status": "success", "stats": stats})
    assert usage_tokens(outcome.usage)["output_tokens"] == 5


def test_garbled_stats_degrade_per_field():
    outcome, _ = _fold(
        {"type": "result", "status": "success", "stats": {"input": -1, "output_tokens": True}}
    )
    assert usage_tokens(outcome.usage)["input_tokens"] is None
    assert usage_tokens(outcome.usage)["output_tokens"] is None
    outcome, _ = _fold({"type": "result", "status": "success", "stats": "nope"})
    assert outcome.usage is None


def test_a_failed_result_surfaces_its_error_as_the_text():
    outcome, _ = _fold(
        {"type": "message", "role": "assistant", "content": "partial", "delta": True},
        {
            "type": "result",
            "status": "error",
            "error": {"type": "unknown", "message": "[API Error: 403 PERMISSION_DENIED]"},
        },
    )
    assert outcome.subtype == "error"
    assert outcome.text == "[API Error: 403 PERMISSION_DENIED]"


def test_the_fold_tolerates_noise_and_a_missing_result():
    outcome, seen = _fold(
        {"type": "tool_use", "tool_name": "glob", "parameters": "not a dict"},
        {"type": "message", "role": "assistant", "content": "answer"},
    )
    lines = ["\n", "not json\n", "[1, 2]\n"] + _lines({"type": "unknown"})
    noisy = gh._consume_gemini_stream(lines, lambda *a: None)
    assert noisy.text == "" and noisy.saw_result is False
    assert outcome.text == "answer"
    assert outcome.saw_result is False
    assert seen == [(1, "glob", "")]


# --- environment and settings ---------------------------------------------------


def test_harness_environment_strips_ambient_routing_and_sets_vertex():
    base = {
        "PATH": "/bin",
        "HOME": "/home/r",
        "GOOGLE_APPLICATION_CREDENTIALS": "/secrets/adc.json",
        "GCE_METADATA_HOST": "127.0.0.1:8181",
        "GOOGLE_CLOUD_PROJECT": "someone-elses",
        "GOOGLE_CLOUD_LOCATION": "us-central1",
        "GOOGLE_GEMINI_BASE_URL": "https://evil.example",
        "GOOGLE_API_KEY": "k",
        "GEMINI_API_KEY": "k",
        "GEMINI_SYSTEM_MD": "/tmp/evil.md",
        "GEMINI_CLI_HOME": "/home/r",
    }
    env = gh.harness_environment(base, project="p1", location="global")
    assert env == {
        "PATH": "/bin",
        "HOME": "/home/r",
        "GOOGLE_APPLICATION_CREDENTIALS": "/secrets/adc.json",
        "GCE_METADATA_HOST": "127.0.0.1:8181",
        "GOOGLE_GENAI_USE_VERTEXAI": "true",
        "GOOGLE_CLOUD_PROJECT": "p1",
        "GOOGLE_CLOUD_LOCATION": "global",
    }


def test_vertex_endpoint_spells_global_without_a_region_prefix():
    assert gh.vertex_endpoint("p", "global") == (
        "https://aiplatform.googleapis.com/v1/projects/p/locations/global"
    )
    assert gh.vertex_endpoint("p", "us-east5") == (
        "https://us-east5-aiplatform.googleapis.com/v1/projects/p/locations/us-east5"
    )


def test_settings_pin_the_tool_registry_and_switch_off_the_load_vectors():
    s = gh.harness_settings(max_turns=42)
    assert s["tools"] == {"core": list(gh.GEMINI_TOOLS)}
    assert "run_shell_command" not in s["tools"]["core"]
    assert s["model"]["maxSessionTurns"] == 42
    assert s["hooksConfig"] == {"enabled": False}
    assert s["mcpServers"] == {}
    assert s["skills"] == {"enabled": False}
    assert s["advanced"]["ignoreLocalEnv"] is True
    assert s["telemetry"]["enabled"] is False
    assert s["privacy"]["usageStatisticsEnabled"] is False
    assert s["context"]["fileFiltering"] == {"enableRecursiveFileSearch": False}


def test_private_home_finds_the_laptop_adc_file_under_the_real_home(tmp_path, monkeypatch):
    monkeypatch.delenv(gh.ENV_SYSTEM_SETTINGS, raising=False)
    adc = tmp_path / ".config" / "gcloud" / "application_default_credentials.json"
    adc.parent.mkdir(parents=True)
    adc.write_text("{}")
    home, env = gh._private_home({"HOME": str(tmp_path)}, max_turns=5)
    try:
        assert env["GOOGLE_APPLICATION_CREDENTIALS"] == str(adc)
        assert env["HOME"] == env["GEMINI_CLI_HOME"] == str(home)
        assert env["TMPDIR"] == str(home / "tmp")
        assert "GEMINI_CLI_SYSTEM_SETTINGS_PATH" not in env
        settings = json.loads((home / ".gemini" / "settings.json").read_text())
        assert settings["model"]["maxSessionTurns"] == 5
    finally:
        import shutil

        shutil.rmtree(home)


def test_private_home_keeps_an_explicit_adc_and_forwards_system_settings(tmp_path, monkeypatch):
    monkeypatch.setenv(gh.ENV_SYSTEM_SETTINGS, " /etc/gemini-cli/settings.json ")
    home, env = gh._private_home(
        {"HOME": str(tmp_path), "GOOGLE_APPLICATION_CREDENTIALS": "/x/adc.json"}, max_turns=5
    )
    try:
        assert env["GOOGLE_APPLICATION_CREDENTIALS"] == "/x/adc.json"
        assert env["GEMINI_CLI_SYSTEM_SETTINGS_PATH"] == "/etc/gemini-cli/settings.json"
    finally:
        import shutil

        shutil.rmtree(home)


def test_no_adc_file_and_no_home_leaves_credentials_to_adc_itself(tmp_path):
    assert gh._default_adc_file(None) is None
    assert gh._default_adc_file(str(tmp_path)) is None


# --- run_review against a fake CLI ----------------------------------------------

FAKE_CLI = r"""#!/usr/bin/env python3
import json, os, sys, time
prompt = sys.stdin.read()
record = {
    "argv": sys.argv[1:],
    "prompt": prompt,
    "cwd": os.getcwd(),
    "env": {k: os.environ.get(k) for k in (
        "HOME", "GEMINI_CLI_HOME", "TMPDIR", "GOOGLE_CLOUD_PROJECT")},
    "settings": json.load(
        open(os.path.join(os.environ["GEMINI_CLI_HOME"], ".gemini", "settings.json"))
    ),
}
json.dump(record, open(os.environ["FAKE_RECORD"], "w"))
mode = os.environ.get("FAKE_MODE", "ok")
if mode == "sleep":
    time.sleep(30)
def out(e):
    print(json.dumps(e), flush=True)
out({"type": "init", "model": "gemini-x"})
out({"type": "tool_use", "tool_name": "read_file", "parameters": {"file_path": "a.go"}})
out({"type": "message", "role": "assistant", "content": "FINAL", "delta": True})
if mode == "error-exit-0":
    out({"type": "result", "status": "error", "error": {"message": "quota RESOURCE_EXHAUSTED"}})
    sys.exit(0)
out({"type": "result", "status": "success", "stats": {"input": 3, "output_tokens": 2, "cached": 1}})
"""


@pytest.fixture
def fake_cli(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    cli = bindir / "gemini"
    cli.write_text(FAKE_CLI)
    cli.chmod(cli.stat().st_mode | stat.S_IEXEC)
    record = tmp_path / "record.json"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()

    def run(mode="ok", timeout=60, prompt="THE PROMPT"):
        env = {
            "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
            "HOME": str(tmp_path),
            "FAKE_RECORD": str(record),
            "FAKE_MODE": mode,
            "GOOGLE_CLOUD_PROJECT": "p1",
        }
        result = gh.run_review(
            prompt, repo, cwd=cwd, model="gemini-x", env=env, timeout=timeout, max_turns=7
        )
        seen = json.loads(record.read_text()) if record.exists() else None
        return result, seen

    run.repo = repo
    run.cwd = cwd
    return run


def test_run_review_spawns_the_hardened_command(fake_cli):
    result, seen = fake_cli()

    assert result.returncode == 0
    assert result.text == "FINAL"
    assert usage_tokens(result.usage)["input_tokens"] == 3
    # Over stdin, never argv, with this harness's tool guidance after it.
    assert seen["prompt"] == f"THE PROMPT\n\n{gh.TOOL_GUIDANCE}"
    argv = seen["argv"]
    assert argv[:2] == ["-p", ""]
    assert argv[argv.index("--model") + 1] == "gemini-x"
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert argv[argv.index("--approval-mode") + 1] == "default"
    assert argv[argv.index("--include-directories") + 1] == str(fake_cli.repo)
    assert "--skip-trust" in argv
    assert "--yolo" not in argv
    assert Path(seen["cwd"]).resolve() == fake_cli.cwd.resolve()
    # A private home, OUTSIDE the readable cwd, removed after the run.
    home = Path(seen["env"]["HOME"])
    assert seen["env"]["GEMINI_CLI_HOME"] == str(home)
    assert seen["env"]["TMPDIR"] == str(home / "tmp")
    assert fake_cli.cwd not in home.parents
    assert not home.exists()
    assert seen["settings"]["model"]["maxSessionTurns"] == 7
    aliases = seen["settings"]["modelConfigs"]["customAliases"]
    assert {a["modelConfig"]["model"] for a in aliases.values()} == {"gemini-x"}
    assert result.turns == 1


def test_run_review_logs_the_fixed_prompt_size(fake_cli, capsys):
    fake_cli()
    expected = len(f"THE PROMPT\n\n{gh.TOOL_GUIDANCE}")
    assert f"prompt {expected} chars" in capsys.readouterr().err


def test_a_failed_terminal_event_is_a_failure_even_on_exit_0(fake_cli):
    result, _ = fake_cli(mode="error-exit-0")
    assert result.returncode == 1
    assert "RESOURCE_EXHAUSTED" in result.text


def test_a_hung_cli_is_killed_and_reads_as_a_timeout(fake_cli):
    result, seen = fake_cli(mode="sleep", timeout=1)
    assert result.timed_out is True
    assert result.returncode == TIMEOUT_RETURNCODE
    assert result.text == ""
    assert not Path(seen["env"]["HOME"]).exists()


def test_a_missing_cli_says_what_to_install(tmp_path):
    with pytest.raises(HarnessNotAvailableError, match="gemini-cli"):
        gh.run_review(
            "p", tmp_path, cwd=tmp_path, model="m", env={"PATH": str(tmp_path)}, timeout=5
        )


def test_a_metadata_host_outranks_the_laptop_adc_file(tmp_path, monkeypatch):
    """The driver's stand-in chose the credential; a found ADC file would override it."""
    monkeypatch.delenv(gh.ENV_SYSTEM_SETTINGS, raising=False)
    adc = tmp_path / ".config" / "gcloud" / "application_default_credentials.json"
    adc.parent.mkdir(parents=True)
    adc.write_text("{}")
    home, env = gh._private_home(
        {"HOME": str(tmp_path), "GCE_METADATA_HOST": "127.0.0.1:9"}, max_turns=5
    )
    try:
        assert "GOOGLE_APPLICATION_CREDENTIALS" not in env
        assert env["GCE_METADATA_HOST"] == "127.0.0.1:9"
    finally:
        import shutil

        shutil.rmtree(home)


def test_every_at_is_escaped_so_the_cli_never_expands_one():
    diff = "@@ -1,2 +1,3 @@ func\n+// see @mod/CLAUDE.md, a@b and \\@kept\n"
    escaped = gh.escape_at(diff)
    # The CLI's own pattern: an @ not preceded by a backslash is a command.
    assert re.search(r"(?<!\\)@", escaped) is None
    assert escaped.replace("\\@", "@") == diff.replace("\\@", "@")


def test_the_models_copy_of_an_escape_is_undone_in_prose_and_json():
    assert gh.unescape_at("cc \\@team at \\@@ -1 +1 \\@@") == "cc @team at @@ -1 +1 @@"
    raw = '{"title": "x \\@y", "body": "a\\\\@b"}'
    # \@ alone is not even valid JSON; the even run is an escaped backslash.
    assert json.loads(gh.unescape_at(raw)) == {"title": "x @y", "body": "a\\@b"}


def test_run_review_sends_the_prompt_escaped_and_returns_plain_text(fake_cli, monkeypatch):
    monkeypatch.setattr(gh, "unescape_at", lambda text: f"<{text}>")
    result, seen = fake_cli(prompt="@@ -1 +1 @@ see @mod/CLAUDE.md")
    assert seen["prompt"].startswith("\\@\\@ -1 +1 \\@\\@ see \\@mod/CLAUDE.md\n\n")
    assert result.text == "<FINAL>"
