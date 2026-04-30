from __future__ import annotations

"""Validator-first wrapper for fool.py.

This module is intentionally additive: it imports the current King agent from
``fool.py`` and patches its prompt/tool layer before delegating to the original
entrypoint.  It is designed for miner evaluation where the agent receives only a
task description + repository and must submit a git diff before a hard deadline.

Run this file instead of ``fool.py`` when you want the 40-minute
validator-first behavior.
"""

import inspect
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import fool as king


# ---------------------------------------------------------------------------
# Deadline and phase policy
# ---------------------------------------------------------------------------

class DeadlineManager:
    """Small wall-clock deadline helper shared by prompt and tools."""

    def __init__(self, total_seconds: int | None = None) -> None:
        self.total_seconds = int(total_seconds or os.getenv("AGENT_TIMEOUT", "2400"))
        self.started_at = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at

    @property
    def remaining(self) -> float:
        return max(0.0, self.total_seconds - self.elapsed)

    @property
    def phase(self) -> str:
        elapsed = self.elapsed
        if elapsed < 3 * 60:
            return "triage"
        if elapsed < 10 * 60:
            return "localize"
        if elapsed < 24 * 60:
            return "patch"
        if elapsed < 31 * 60:
            return "validate"
        if elapsed < 36 * 60:
            return "adversarial_review"
        if elapsed < 39 * 60:
            return "repair_or_simplify"
        return "finish_now"

    def instruction(self) -> str:
        phase = self.phase
        remaining = int(self.remaining)
        if remaining <= 90:
            return (
                "PANIC FINISH: less than 90 seconds remain. Do not explore or run more tests. "
                "If the current patch is plausible and non-empty, call finish now."
            )
        return {
            "triage": "Identify repo type, likely commands, and task class. Do not edit yet unless the fix is trivial.",
            "localize": "Find the public API and minimal files to change. State likely hidden-validator checks before editing.",
            "patch": "Apply the smallest general fix. Avoid benchmark-specific hacks and unrelated behavior changes.",
            "validate": "Run fast checks only: compile/import, targeted tests, then full tests only if cheap.",
            "adversarial_review": "Act as the hidden validator. Find edge cases/backward-compat risks in the current diff.",
            "repair_or_simplify": "Repair only high-confidence validator risks. Prefer simplification over broad rewrites.",
            "finish_now": "Finish immediately with the best current patch. Do not inspect more files.",
        }[phase]


_DEADLINE = DeadlineManager()
_CHECKPOINT = Path("/tmp/ridges_validator_first_checkpoint.patch")


VALIDATOR_FIRST_PROMPT = textwrap.dedent(
    """

    # Validator-First 40-Minute Strategy

    You are evaluated only by hidden validator tests after you submit a git diff.
    You do not know those tests. Optimize for general correctness, public API
    behavior, backward compatibility, and minimal unrelated change.

    Follow this phase-driven plan instead of wandering:
    1. 0-3 min: triage repo type, task type, likely test commands.
    2. 3-10 min: localize relevant public APIs and files. Do not edit until localized.
    3. 10-24 min: make the smallest general patch that solves the task semantics.
    4. 24-31 min: run fast available checks. Do not burn time on long full suites.
    5. 31-36 min: use hidden_validator_review on the current diff; infer hidden tests.
    6. 36-39 min: repair only high-confidence hidden-test risks; otherwise keep patch.
    7. Last 90 sec: stop exploring and finish with the best current patch.

    Before editing, write a short localization note in next_thought:
    - likely changed files
    - public API involved
    - likely hidden validator checks

    Before finish, verify:
    - current diff is non-empty unless the task truly needs no code change
    - no tests were modified
    - no unrelated files were changed
    - the fix is semantic, not overfit to visible wording
    - old API behavior still works unless the task explicitly requires breaking it

    Use the added tools when helpful:
    - deadline_status: see remaining time and current phase
    - repo_triage: detect language/framework/test commands quickly
    - save_checkpoint: save the current patch before risky repair
    - restore_checkpoint: rollback to last good patch
    - diff_summary: inspect changed files and diff size
    - hidden_validator_review: generate likely hidden tests and patch risks
    - run_fast_checks: run cheap validation commands under a strict cap
    """
)


# ---------------------------------------------------------------------------
# Utilities used by tools
# ---------------------------------------------------------------------------

def _run(command: str, timeout: int = 30) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            ["bash", "-lc", command],
            text=True,
            capture_output=True,
            timeout=max(1, timeout),
        )
        out = ""
        if proc.stdout:
            out += "STDOUT:\n" + proc.stdout
        if proc.stderr:
            out += "STDERR:\n" + proc.stderr
        return proc.returncode, out.strip()
    except subprocess.TimeoutExpired:
        return 124, f"Command timed out after {timeout}s: {command}"
    except Exception as exc:  # defensive tool boundary
        return 1, f"Command failed to execute: {exc}"


def _truncate(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[...truncated by validator-first wrapper...]"


def _changed_files() -> list[str]:
    code, out = _run("git diff --name-only", timeout=10)
    if code != 0:
        return []
    return [line.strip() for line in out.splitlines() if line.strip() and not line.startswith("STDOUT:")]


def _detect_test_commands() -> list[str]:
    commands: list[str] = []
    cwd = Path.cwd()

    if any((cwd / p).exists() for p in ["pyproject.toml", "pytest.ini", "setup.py", "setup.cfg"]):
        commands.append("python -m compileall .")
        commands.append("python -m pytest -q")
    elif any(cwd.glob("**/*.py")):
        commands.append("python -m compileall .")

    if (cwd / "package.json").exists():
        commands.extend(["npm test -- --runInBand", "npm test"])
    if (cwd / "go.mod").exists():
        commands.append("go test ./...")
    if (cwd / "Cargo.toml").exists():
        commands.append("cargo test")
    if (cwd / "pom.xml").exists():
        commands.append("mvn test -q")
    if (cwd / "build.gradle").exists() or (cwd / "build.gradle.kts").exists():
        commands.append("./gradlew test")
    if (cwd / "composer.json").exists():
        commands.extend(["composer test", "vendor/bin/phpunit"])
    if (cwd / "Gemfile").exists():
        commands.extend(["bundle exec rake test", "bundle exec rspec"])
    if (cwd / "Makefile").exists():
        commands.append("make test")

    # Preserve order, remove duplicates.
    seen: set[str] = set()
    unique: list[str] = []
    for cmd in commands:
        if cmd not in seen:
            unique.append(cmd)
            seen.add(cmd)
    return unique


# ---------------------------------------------------------------------------
# Monkey-patched tools
# ---------------------------------------------------------------------------

def deadline_status(self: Any) -> Any:
    """
    Show remaining wall-clock budget and the required phase behavior.
    Arguments: none
    """
    return king.ToolResult(
        observation=(
            f"elapsed={int(_DEADLINE.elapsed)}s remaining={int(_DEADLINE.remaining)}s "
            f"phase={_DEADLINE.phase}\nInstruction: {_DEADLINE.instruction()}"
        ),
        is_error=False,
    )


def repo_triage(self: Any) -> Any:
    """
    Quickly detect repo type, important files, test directories, and likely validation commands.
    Arguments: none
    """
    commands = [
        "pwd",
        "find . -maxdepth 3 -type f \\\n          \( -name 'pyproject.toml' -o -name 'setup.py' -o -name 'setup.cfg' -o -name 'pytest.ini' \\\n          -o -name 'package.json' -o -name 'go.mod' -o -name 'Cargo.toml' -o -name 'pom.xml' \\\n          -o -name 'build.gradle' -o -name 'build.gradle.kts' -o -name 'composer.json' -o -name 'Gemfile' \\\n          -o -name 'Makefile' \) | sort",
        "find . -maxdepth 4 -type d \\\n          \( -name tests -o -name test -o -name spec -o -name __tests__ \) | sort | head -80",
        "find . -maxdepth 4 -type f | sed 's/.*\\.//' | sort | uniq -c | sort -nr | head -25",
    ]
    parts: list[str] = []
    for cmd in commands:
        code, out = _run(cmd, timeout=20)
        parts.append(f"$ {cmd}\nexit={code}\n{out}")
    detected = _detect_test_commands()
    parts.append("Likely validation commands, cheapest first:\n" + "\n".join(f"- {c}" for c in detected))
    parts.append(f"Deadline phase: {_DEADLINE.phase}; {_DEADLINE.instruction()}")
    return king.ToolResult(observation=_truncate("\n\n".join(parts)), is_error=False)


def save_checkpoint(self: Any) -> Any:
    """
    Save the current git diff to a temporary checkpoint before risky edits.
    Arguments: none
    """
    code, out = _run(f"git diff --no-color > {_CHECKPOINT}", timeout=20)
    if code != 0:
        return king.ToolResult(observation=f"ERROR: failed to save checkpoint.\n{out}", is_error=True)
    size = _CHECKPOINT.stat().st_size if _CHECKPOINT.exists() else 0
    return king.ToolResult(observation=f"Saved checkpoint to {_CHECKPOINT} ({size} bytes).", is_error=False)


def restore_checkpoint(self: Any) -> Any:
    """
    Restore the last saved git diff checkpoint after a bad repair attempt.
    Arguments: none
    """
    if not _CHECKPOINT.exists():
        return king.ToolResult(observation="ERROR: no checkpoint exists.", is_error=True)
    code, out = _run(f"git reset --hard && git clean -fd && git apply {_CHECKPOINT}", timeout=60)
    if code != 0:
        return king.ToolResult(observation=f"ERROR: failed to restore checkpoint.\n{out}", is_error=True)
    return king.ToolResult(observation="Restored checkpoint patch.", is_error=False)


def diff_summary(self: Any) -> Any:
    """
    Summarize the current patch: files changed, stat, and first part of diff.
    Arguments: none
    """
    code1, names = _run("git diff --name-only", timeout=15)
    code2, stat = _run("git diff --stat --no-color", timeout=15)
    code3, diff = _run("git diff --no-color", timeout=20)
    is_error = code1 != 0 or code2 != 0 or code3 != 0
    obs = f"Changed files:\n{names}\n\nStat:\n{stat}\n\nDiff preview:\n{diff}"
    return king.ToolResult(observation=_truncate(obs), is_error=is_error)


def hidden_validator_review(self: Any) -> Any:
    """
    Deterministically review current diff as a hidden SWE-bench/Polyglot validator would.
    Arguments: none
    """
    files = _changed_files()
    code_stat, stat = _run("git diff --stat --no-color", timeout=15)
    code_diff, diff = _run("git diff --no-color", timeout=25)
    test_files = [f for f in files if any(part in f.lower() for part in ["test", "spec", "__tests__"])]

    risks: list[str] = []
    if not files:
        risks.append("No files changed. Hidden validators will almost certainly fail unless task required no-op.")
    if test_files:
        risks.append(f"Patch modifies test/spec files: {test_files}. Usually invalid for SWE/Polyglot validation.")
    if len(files) > 6:
        risks.append("Many files changed. Hidden tests may fail from unrelated behavior changes; consider simplifying.")
    if "except Exception" in diff or "except:" in diff:
        risks.append("Broad exception handling appears in diff. Ensure it does not hide real errors.")
    if "TODO" in diff or "pass  #" in diff or "NotImplemented" in diff:
        risks.append("Patch may contain placeholder behavior.")
    if any(s in diff.lower() for s in ["swe-bench", "polyglot", "benchmark", "validator"]):
        risks.append("Patch mentions benchmark/validator strings. Hidden tests may treat this as overfitting.")
    if not risks:
        risks.append("No obvious structural risks found. Focus on semantic edge cases and backward compatibility.")

    likely_tests = [
        "Regression exactly described by the task, through the public API rather than private helpers.",
        "Backward-compatible old call patterns/import paths still work unless explicitly changed.",
        "Empty/None/boundary inputs around the changed function or CLI path.",
        "Multiple affected files/modules behave consistently, not just the first located case.",
        "Error messages/types remain compatible where users may assert them.",
    ]

    probes: list[str] = []
    if any(f.endswith(".py") for f in files):
        probes.extend([
            "python -m compileall .",
            "python - <<'PY'\n# Import the changed public module(s) and exercise the task-described API.\nPY",
        ])
    if any(f.endswith((".js", ".ts", ".tsx", ".jsx")) for f in files):
        probes.append("npm test -- --runInBand  # or a targeted package test if available")
    if any(f.endswith(".go") for f in files):
        probes.append("go test ./...")
    if any(f.endswith(".rs") for f in files):
        probes.append("cargo test")

    obs = "\n".join(
        [
            f"Deadline phase: {_DEADLINE.phase}; remaining={int(_DEADLINE.remaining)}s",
            "Changed files:",
            *(f"- {f}" for f in files),
            "",
            "Diff stat:",
            stat,
            "",
            "Likely hidden-validator tests:",
            *(f"- {t}" for t in likely_tests),
            "",
            "Patch risks:",
            *(f"- {r}" for r in risks),
            "",
            "Suggested cheap probes:",
            *(f"- {p}" for p in probes),
            "",
            "Decision rule: repair only high-confidence semantic risks. If time is low and patch is plausible, finish.",
        ]
    )
    return king.ToolResult(observation=_truncate(obs), is_error=bool(test_files or not files))


def run_fast_checks(self: Any, max_seconds: int = 240) -> Any:
    """
    Run cheap validation commands under a strict total cap. Stops before deadline gets dangerous.
    Arguments:
        max_seconds: maximum total seconds to spend, default 240
    """
    max_seconds = max(15, min(int(max_seconds), 360))
    if _DEADLINE.remaining < 150:
        return king.ToolResult(
            observation="Skipped fast checks: less than 150 seconds remain; finish or make only tiny repair.",
            is_error=False,
        )

    commands = _detect_test_commands()
    if not commands:
        return king.ToolResult(observation="No standard validation commands detected.", is_error=False)

    started = time.time()
    parts: list[str] = []
    failed = False
    for cmd in commands:
        remaining_budget = max_seconds - int(time.time() - started)
        if remaining_budget <= 10 or _DEADLINE.remaining < 120:
            parts.append("Stopping checks to preserve finish time.")
            break
        # Compile/import checks and targeted commands are valuable; full tests get capped.
        timeout = min(90, remaining_budget)
        if cmd in {"python -m pytest -q", "npm test", "go test ./...", "cargo test", "mvn test -q", "make test"}:
            timeout = min(timeout, 120)
        code, out = _run(cmd, timeout=timeout)
        parts.append(f"$ {cmd}\nexit={code}\n{out}")
        if code != 0:
            failed = True
            # Stop on first concrete failure; the agent should repair instead of running more.
            break
    return king.ToolResult(observation=_truncate("\n\n".join(parts)), is_error=failed)


for _tool in [
    deadline_status,
    repo_triage,
    save_checkpoint,
    restore_checkpoint,
    diff_summary,
    hidden_validator_review,
    run_fast_checks,
]:
    _tool._is_tool = True  # type: ignore[attr-defined]
    setattr(king.ToolManager, _tool.__name__, _tool)


# ---------------------------------------------------------------------------
# Prompt patching and entrypoint delegation
# ---------------------------------------------------------------------------

def install_validator_first_policy() -> None:
    """Patch King agent globals before an agent instance is created."""
    if VALIDATOR_FIRST_PROMPT not in king.SYSTEM_PROMPT:
        king.SYSTEM_PROMPT = king.SYSTEM_PROMPT + VALIDATOR_FIRST_PROMPT

    panic = (
        "\nDeadline reminder: call deadline_status when uncertain. "
        "In the last 90 seconds, do not explore; finish with the best current patch.\n"
    )
    if panic not in king.STOP_INSTRUCTION:
        king.STOP_INSTRUCTION = king.STOP_INSTRUCTION + panic

    # Avoid 400-step wandering. Keep env override, but make default phase-driven.
    try:
        king.MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "180"))
    except Exception:
        king.MAX_STEPS = 180


install_validator_first_policy()


def _call_original_entrypoint() -> Any:
    """Delegate to common entrypoints without assuming fool.py's exact shape."""
    for name in ["main", "run", "cli"]:
        obj = getattr(king, name, None)
        if callable(obj):
            return obj()

    # Fallback for class-based agents.
    for name, obj in inspect.getmembers(king, inspect.isclass):
        if name.lower() in {"agent", "codingagent", "codingassistant"} or "agent" in name.lower():
            try:
                instance = obj()
            except TypeError:
                continue
            for method_name in ["run", "main", "solve", "__call__"]:
                method = getattr(instance, method_name, None)
                if callable(method):
                    return method()

    raise RuntimeError(
        "Could not find an executable entrypoint in fool.py. Import this module before constructing "
        "the original agent, or add an explicit call to fool_validator_first.install_validator_first_policy()."
    )


if __name__ == "__main__":
    _call_original_entrypoint()
