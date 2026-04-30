from __future__ import annotations

"""Standalone validator-first coding agent for Ridges miner submissions.

This file does not import ``fool.py``.  It keeps the useful mechanics from the
King-style agent—strict tool-call parsing, sandbox/direct inference support,
robust tool execution, truncation, git-diff finishing—and adds a phase-driven
40-minute hidden-validator strategy.

Input:
- first CLI argument, or
- stdin, or
- TASK / PROMPT / PROBLEM_STATEMENT environment variable.

Output:
- final git diff patch is printed to stdout when the agent finishes.
"""

import inspect
import json
import logging
import os
import random
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4

import requests


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "2400"))
MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "180"))
CONTEXT_COMPACTION_THRESHOLD = int(os.getenv("AGENT_CONTEXT_COMPACTION_THRESHOLD", "24"))
LLM_CONNECT_TIMEOUT = int(os.getenv("LLM_CONNECT_TIMEOUT", "30"))
LLM_READ_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "120"))
TOOL_OUTPUT_MAX_CHARS = int(os.getenv("AGENT_TOOL_OUTPUT_MAX_CHARS", "50000"))
TOOL_OUTPUT_MAX_LINE_LENGTH = int(os.getenv("AGENT_TOOL_OUTPUT_MAX_LINE", "2000"))
BASH_TIMEOUT_DEFAULT = 60
BASH_TIMEOUT_MAX = 300
RETRY_INITIAL_WAIT = 0.3
RETRY_MAX_WAIT = 5.0
RETRY_JITTER = 0.5
RETRYABLE_HTTP_CODES = (429, 500, 502, 503, 504)

GLM_5_MODEL = os.getenv("AGENT_MODEL", "zai-org/GLM-5-FP8")
KIMI_MODEL = "moonshotai/Kimi-K2.5"
MINIMAX_MODEL = "MiniMaxAI/MiniMax-M2.5"

_GATEWAY_TO_OPENROUTER_MODEL: Dict[str, str] = {
    "deepseek-ai/DeepSeek-R1-0528": "deepseek/deepseek-r1-0528",
    "zai-org/GLM-4.6": "z-ai/glm-4.6",
    "zai-org/GLM-4.6-FP8": "z-ai/glm-4.6",
    "zai-org/GLM-4.7": "z-ai/glm-4.7",
    "zai-org/GLM-4.7-FP8": "z-ai/glm-4.7",
    "zai-org/GLM-5-FP8": "z-ai/glm-5",
    "Qwen/Qwen3-Coder-Next": "qwen/qwen3-coder-next",
    "Qwen/Qwen3.5-397B-A17B": "qwen/qwen3.5-397b-a17b",
    "moonshotai/Kimi-K2.5": "moonshotai/kimi-k2.5",
    "MiniMaxAI/MiniMax-M2.5": "minimax/minimax-m2.5",
}

logger = logging.getLogger("validator_first_agent")
logger.setLevel(logging.DEBUG)
for h in list(logger.handlers):
    logger.removeHandler(h)
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


# ============================================================================
# DEADLINE POLICY
# ============================================================================

class DeadlineManager:
    def __init__(self, total_seconds: int = DEFAULT_TIMEOUT) -> None:
        self.total_seconds = total_seconds
        self.started_at = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at

    @property
    def remaining(self) -> float:
        return max(0.0, self.total_seconds - self.elapsed)

    @property
    def phase(self) -> str:
        e = self.elapsed
        if e < 3 * 60:
            return "triage"
        if e < 10 * 60:
            return "localize"
        if e < 24 * 60:
            return "patch"
        if e < 31 * 60:
            return "validate"
        if e < 36 * 60:
            return "adversarial_review"
        if e < 39 * 60:
            return "repair_or_simplify"
        return "finish_now"

    def instruction(self) -> str:
        if self.remaining <= 90:
            return (
                "PANIC FINISH: less than 90 seconds remain. Do not inspect files or run tests. "
                "If the current patch is plausible and non-empty, call finish now."
            )
        return {
            "triage": "Identify repo type, task type, likely validation commands. Avoid edits unless trivial.",
            "localize": "Find public API and minimal files. State likely hidden-validator checks before editing.",
            "patch": "Make the smallest semantic fix. Avoid benchmark-specific hacks and unrelated changes.",
            "validate": "Run cheap compile/import/targeted checks. Do not burn time on long full suites.",
            "adversarial_review": "Act as hidden validator. Review current diff for edge cases and compatibility risks.",
            "repair_or_simplify": "Repair only high-confidence hidden-test risks. Prefer simplification over rewrites.",
            "finish_now": "Finish immediately with the best current patch.",
        }[self.phase]


DEADLINE = DeadlineManager()
CHECKPOINT = Path("/tmp/ridges_validator_first_checkpoint.patch")


# ============================================================================
# PROMPTS
# ============================================================================

FORMAT_PROMPT = textwrap.dedent(
    """
    Response format requirements:
    Generate exactly these three fields in this order:
    1. next_thought: concise reasoning about what to do next
    2. next_tool_name: one of bash, think, str_replace_edit, repo_triage, deadline_status,
       save_checkpoint, restore_checkpoint, diff_summary, hidden_validator_review,
       run_fast_checks, finish
    3. next_tool_args: valid JSON object

    Example:
    next_thought: I need to inspect the package metadata and tests before editing.
    next_tool_name: repo_triage
    next_tool_args: {}

    Do not generate observation. The system will provide observations.
    """
)

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a validator-first Coding Assistant. Your job is to modify the current
    repository so the final git diff passes hidden validators. You are evaluated
    only by the patch you submit, not by explanations.

    Core workflow:
    1. Understand the task from the prompt.
    2. Triage repository type and likely test commands.
    3. Localize the public API and exact files before editing.
    4. Make minimal, semantic, backward-compatible edits.
    5. Run cheap validation checks.
    6. Review the patch as a hidden validator.
    7. Repair only high-confidence risks.
    8. Finish before timeout.

    Critical rules:
    - Never modify tests or test fixtures unless the task explicitly requires it.
    - Do not create permanent files unless required by the task.
    - No internet access; do not try to install missing dependencies.
    - Preserve backward compatibility unless the task explicitly asks for a break.
    - Avoid hacks that mention benchmark, validator, SWE-bench, Polyglot, or task wording.
    - Prefer a small correct public behavior fix over a broad rewrite.
    - Before editing, include in next_thought: likely changed files, public API, likely hidden tests.
    - Before finish, use diff_summary or hidden_validator_review unless panic time remains.

    40-minute phase policy:
    - 0-3 min: triage.
    - 3-10 min: localization.
    - 10-24 min: patch.
    - 24-31 min: cheap validation.
    - 31-36 min: hidden-validator review.
    - 36-39 min: targeted repair/simplify.
    - last 90 sec: finish with the best current patch.

    Tools:
    {tools_docs}

    {format_prompt}
    """
)

STOP_INSTRUCTION = textwrap.dedent(
    """
    Generate only one tool action: next_thought, next_tool_name, next_tool_args.
    Do not include observation.
    """
)


# ============================================================================
# NETWORK & INFERENCE
# ============================================================================

def _resolve_model_for_local(name: str) -> str:
    return _GATEWAY_TO_OPENROUTER_MODEL.get(name, name)


def _use_direct_openai_inference() -> bool:
    if (os.getenv("SANDBOX_PROXY_URL") or "").strip():
        return False
    return bool(os.getenv("RIDGES_INFERENCE_API_KEY") and os.getenv("RIDGES_INFERENCE_BASE_URL"))


def _sandbox_proxy_base() -> str:
    return ((os.getenv("SANDBOX_PROXY_URL") or "").strip() or "http://sandbox_proxy").rstrip("/")


def _retry_sleep(attempt: int) -> None:
    wait = min(RETRY_INITIAL_WAIT * (2 ** attempt) + random.uniform(0, RETRY_JITTER), RETRY_MAX_WAIT)
    logger.info("Retrying in %.2fs", wait)
    time.sleep(wait)


def _is_retryable_request_error(exc: BaseException) -> bool:
    if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True
    if isinstance(exc, requests.exceptions.RequestException) and hasattr(exc, "response"):
        resp = getattr(exc, "response", None)
        return resp is not None and getattr(resp, "status_code", None) in RETRYABLE_HTTP_CODES
    return False


class Network:
    @classmethod
    def make_request(cls, messages: list, model: str, temperature: float = 0.0) -> Tuple[str, int]:
        timeout = (LLM_CONNECT_TIMEOUT, LLM_READ_TIMEOUT)
        direct = _use_direct_openai_inference()
        if direct:
            api_key = os.environ.get("RIDGES_INFERENCE_API_KEY")
            base_url = os.environ.get("RIDGES_INFERENCE_BASE_URL")
            if not api_key or not base_url:
                raise RuntimeError("Set SANDBOX_PROXY_URL or RIDGES_INFERENCE_API_KEY + RIDGES_INFERENCE_BASE_URL")
            url = f"{base_url.rstrip('/')}/chat/completions"
            resolved = _resolve_model_for_local(model)
            request_data = {"model": resolved, "messages": messages, "temperature": temperature}
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        else:
            url = f"{_sandbox_proxy_base()}/api/inference"
            request_data = {
                "evaluation_run_id": os.getenv("EVALUATION_RUN_ID", str(uuid4())),
                "messages": messages,
                "temperature": temperature,
                "model": model,
            }
            headers = {"Content-Type": "application/json"}

        wait = 1.0
        last_response: requests.Response | None = None
        for attempt in range(5):
            response = requests.post(url, json=request_data, headers=headers, timeout=timeout)
            last_response = response
            if response.status_code == 429 and attempt < 4:
                retry_after = response.headers.get("Retry-After")
                try:
                    time.sleep(float(retry_after) if retry_after else wait)
                except ValueError:
                    time.sleep(wait)
                wait = min(wait * 2, 60.0)
                continue
            response.raise_for_status()
            data = response.json()
            if direct:
                msg = (data.get("choices") or [{}])[0].get("message") or {}
                content = (msg.get("content") or "").strip()
            else:
                content = data.get("content") or ""
            return content, response.status_code
        if last_response is not None:
            last_response.raise_for_status()
        raise RuntimeError("LLM inference request failed")

    @classmethod
    def is_valid_response(cls, raw_text: str) -> tuple[bool, str]:
        if not raw_text or not raw_text.strip():
            return False, "Empty response"
        if "<|reserved_token_" in raw_text:
            return False, "Reserved token present"
        if "next_tool_args:" not in raw_text:
            return False, "Missing next_tool_args"
        return True, ""

    @classmethod
    def parse_response(cls, text_resp: str) -> tuple[str | None, str | None, dict | None, str | None]:
        text_resp = text_resp.strip().split("observation:")[0].strip()
        text_resp = re.sub(r"['\"]*(next_thought)['\"]*\s*:", "next_thought:", text_resp)
        text_resp = re.sub(r"['\"]*(next_tool_name)['\"]*\s*:", "next_tool_name:", text_resp)
        text_resp = re.sub(r"['\"]*(next_tool_args)['\"]*\s*:", "next_tool_args:", text_resp)
        if "next_thought:" not in text_resp:
            return None, None, None, "Invalid response: next_thought not found"
        if "next_tool_name:" not in text_resp:
            return None, None, None, "Invalid response: next_tool_name not found"
        if "next_tool_args:" not in text_resp:
            return None, None, None, "Invalid response: next_tool_args not found"
        try:
            thought = text_resp.split("next_thought:", 1)[1].split("next_tool_name:", 1)[0].strip()
            tool = text_resp.split("next_tool_name:", 1)[1].split("next_tool_args:", 1)[0].strip().strip("'\"")
            args_raw = text_resp.split("next_tool_args:", 1)[1].strip().replace("```json", "").replace("```", "").strip()
            if args_raw.startswith("{"):
                depth = 0
                end = -1
                in_str = False
                escape = False
                quote = ""
                for i, c in enumerate(args_raw):
                    if in_str:
                        if escape:
                            escape = False
                        elif c == "\\":
                            escape = True
                        elif c == quote:
                            in_str = False
                        continue
                    if c in "'\"":
                        in_str = True
                        quote = c
                    elif c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            end = i
                            break
                if end >= 0:
                    args_raw = args_raw[: end + 1]
            args = json.loads(args_raw) if args_raw else {}
        except (JSONDecodeError, IndexError) as e:
            return None, None, None, f"Error parsing response: {e}"
        allowed = {
            "bash", "think", "str_replace_edit", "repo_triage", "deadline_status", "save_checkpoint",
            "restore_checkpoint", "diff_summary", "hidden_validator_review", "run_fast_checks", "finish",
        }
        if tool not in allowed:
            return None, None, None, f"Invalid tool '{tool}'. Use one of: {', '.join(sorted(allowed))}"
        return thought, tool, args, None

    @classmethod
    def inference(cls, messages: List[Dict[str, Any]], model: str, temperature: float = 0.0, max_retries: int = 3) -> tuple:
        last_exception: BaseException | None = None
        for attempt in range(max_retries):
            try:
                raw_text, _ = cls.make_request(messages, model=model, temperature=temperature)
                valid, error = cls.is_valid_response(raw_text)
                if not valid:
                    if attempt < max_retries - 1:
                        messages.append({"role": "user", "content": f"observation: {error}. Follow the required format."})
                        _retry_sleep(attempt)
                        continue
                    raise RuntimeError(error)
                thought, tool, args, parse_error = cls.parse_response(raw_text)
                if parse_error:
                    if attempt < max_retries - 1:
                        messages.append({"role": "assistant", "content": raw_text})
                        messages.append({"role": "user", "content": f"observation: {parse_error}"})
                        _retry_sleep(attempt)
                        continue
                    raise RuntimeError(parse_error)
                return thought, tool, args, raw_text
            except requests.exceptions.HTTPError as e:
                last_exception = e
                if e.response is not None and e.response.status_code in RETRYABLE_HTTP_CODES and attempt < max_retries - 1:
                    _retry_sleep(attempt)
                    continue
                raise
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    _retry_sleep(attempt)
                    continue
                raise
            except Exception as e:
                last_exception = e
                if _is_retryable_request_error(e) and attempt < max_retries - 1:
                    _retry_sleep(attempt)
                    continue
                raise
        raise RuntimeError("Max retries exceeded") from last_exception


# ============================================================================
# TOOL RESULT AND TRUNCATION
# ============================================================================

def _truncate_line(line: str, max_length: int, marker: str = "...") -> str:
    if len(line) <= max_length:
        return line
    m = re.search(r"[\r\n]+$", line)
    suffix = m.group(0) if m else ""
    end = marker + suffix
    return line[: max(0, max_length - len(end))] + end


def truncate_tool_output(text: str, max_chars: int = TOOL_OUTPUT_MAX_CHARS, max_line_length: int = TOOL_OUTPUT_MAX_LINE_LENGTH) -> str:
    if not text:
        return text
    out: list[str] = []
    total = 0
    for line in text.splitlines(keepends=True):
        if max_line_length and len(line) > max_line_length:
            line = _truncate_line(line, max_line_length, "[...truncated]")
        if total + len(line) > max_chars:
            out.append("[...truncated]\n")
            break
        out.append(line)
        total += len(line)
    return "".join(out)


@dataclass
class ToolResult:
    observation: str
    is_error: bool = False


class ToolResultBuilder:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def write(self, text: str) -> None:
        self.parts.append(text)

    def ok(self, message: str = "", output: str | None = None) -> ToolResult:
        if output:
            self.write(output)
        body = truncate_tool_output("".join(self.parts))
        if message:
            return ToolResult((message.rstrip(".") + ".") + ("\n\n" + body if body else ""), False)
        return ToolResult(body or "Done.", False)

    def error(self, message: str, output: str | None = None) -> ToolResult:
        if output:
            self.write(output)
        body = truncate_tool_output("".join(self.parts))
        return ToolResult("ERROR: " + message.rstrip(".") + "." + ("\n\n" + body if body else ""), True)


# ============================================================================
# TOOL MANAGER
# ============================================================================

def tool(fn):
    fn._is_tool = True
    return fn


class ToolManager:
    def __init__(self) -> None:
        self.TOOL_LIST: dict[str, dict] = {}
        self.checkpoint = ""
        self._register_tools()

    def _register_tools(self) -> None:
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if getattr(method, "_is_tool", False):
                self.TOOL_LIST[name] = self._create_tool_schema(method)

    def _create_tool_schema(self, method) -> dict:
        doc = inspect.getdoc(method) or ""
        sig = inspect.signature(method)
        properties = {}
        required = []
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
            annotation = param.annotation
            type_str = str(annotation)
            param_type = "integer" if "int" in type_str else "boolean" if "bool" in type_str else "string"
            properties[param_name] = {"type": param_type, "description": f"Parameter {param_name}"}
        return {"name": method.__name__, "description": doc.split("Arguments:")[0].strip(), "input_schema": {"type": "object", "properties": properties, "required": required}}

    def get_tool_docs(self) -> str:
        return "\n".join(json.dumps(t, ensure_ascii=False) for t in self.TOOL_LIST.values())

    def run_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> ToolResult:
        if tool_name not in self.TOOL_LIST:
            return ToolResult(f"ERROR: Tool '{tool_name}' not found. Available: {', '.join(sorted(self.TOOL_LIST))}.", True)
        missing = [k for k in self.TOOL_LIST[tool_name]["input_schema"].get("required", []) if k not in tool_args or tool_args[k] is None]
        if missing:
            return ToolResult(f"ERROR: Missing required argument(s): {', '.join(missing)}.", True)
        try:
            raw = getattr(self, tool_name)(**tool_args) if tool_args else getattr(self, tool_name)()
        except Exception as e:
            logger.exception("Tool %s failed", tool_name)
            return ToolResult(f"ERROR: {e}.", True)
        if isinstance(raw, ToolResult):
            return ToolResult(truncate_tool_output(raw.observation), raw.is_error)
        text = truncate_tool_output(str(raw))
        return ToolResult(text, text.strip().upper().startswith("ERROR:"))

    @staticmethod
    def _run(command: str, timeout: int = 30) -> tuple[int, str]:
        try:
            proc = subprocess.run(["bash", "-lc", command], capture_output=True, text=True, timeout=max(1, timeout), cwd=os.getcwd())
            out = ""
            if proc.stdout:
                out += "STDOUT:\n" + proc.stdout
                if not proc.stdout.endswith("\n"):
                    out += "\n"
            if proc.stderr:
                out += "STDERR:\n" + proc.stderr
                if not proc.stderr.endswith("\n"):
                    out += "\n"
            return proc.returncode, out
        except subprocess.TimeoutExpired:
            return 124, f"Command killed by timeout ({timeout}s).\n"
        except Exception as e:
            return 1, f"Error executing command: {e}\n"

    @tool
    def bash(self, command: str, timeout: int = BASH_TIMEOUT_DEFAULT) -> ToolResult:
        """Execute a bash command to inspect, edit indirectly, or test the repository."""
        if not command.strip():
            return ToolResult("ERROR: Command cannot be empty.", True)
        timeout = max(1, min(int(timeout), BASH_TIMEOUT_MAX))
        code, out = self._run(command, timeout)
        if code != 0:
            return ToolResult(f"ERROR: Command failed with exit code {code}.\n\n{out}", True)
        return ToolResult("Command executed successfully.\n\n" + out, False)

    @tool
    def think(self, thought: str) -> ToolResult:
        """Record planning or reasoning without changing the repository."""
        logger.info("[THINK] %s", thought[:500])
        return ToolResult("Thought logged. Continue.", False)

    @tool
    def str_replace_edit(self, file_path: str, old_str: str, new_str: str) -> ToolResult:
        """Replace exact unique text in an existing file."""
        path = Path(file_path)
        if not path.exists():
            return ToolResult(f"ERROR: File '{file_path}' does not exist.", True)
        if old_str == new_str:
            return ToolResult("ERROR: old_str and new_str are identical.", True)
        data = path.read_text(encoding="utf-8")
        count = data.count(old_str)
        if count == 0:
            return ToolResult(f"ERROR: old_str not found in '{file_path}'.", True)
        if count > 1:
            return ToolResult(f"ERROR: old_str found {count} times in '{file_path}'. Use a unique match.", True)
        path.write_text(data.replace(old_str, new_str, 1), encoding="utf-8")
        return ToolResult(f"Successfully replaced text in '{file_path}'.", False)

    @tool
    def deadline_status(self) -> ToolResult:
        """Show current wall-clock budget and phase instruction."""
        return ToolResult(f"elapsed={int(DEADLINE.elapsed)}s remaining={int(DEADLINE.remaining)}s phase={DEADLINE.phase}\nInstruction: {DEADLINE.instruction()}", False)

    @tool
    def repo_triage(self) -> ToolResult:
        """Detect repo type, test dirs, language mix, and likely validation commands."""
        commands = [
            "pwd",
            "find . -maxdepth 3 -type f \( -name 'pyproject.toml' -o -name 'setup.py' -o -name 'setup.cfg' -o -name 'pytest.ini' -o -name 'package.json' -o -name 'go.mod' -o -name 'Cargo.toml' -o -name 'pom.xml' -o -name 'build.gradle' -o -name 'build.gradle.kts' -o -name 'composer.json' -o -name 'Gemfile' -o -name 'Makefile' \) | sort",
            "find . -maxdepth 4 -type d \( -name tests -o -name test -o -name spec -o -name __tests__ \) | sort | head -80",
            "find . -maxdepth 4 -type f | sed 's/.*\\.//' | sort | uniq -c | sort -nr | head -25",
        ]
        parts = []
        for cmd in commands:
            code, out = self._run(cmd, 20)
            parts.append(f"$ {cmd}\nexit={code}\n{out}")
        detected = self._detect_test_commands()
        parts.append("Likely validation commands:\n" + "\n".join(f"- {c}" for c in detected))
        parts.append(f"Deadline: phase={DEADLINE.phase}, remaining={int(DEADLINE.remaining)}s. {DEADLINE.instruction()}")
        return ToolResult("\n\n".join(parts), False)

    def _detect_test_commands(self) -> list[str]:
        cwd = Path.cwd()
        cmds: list[str] = []
        if any((cwd / p).exists() for p in ["pyproject.toml", "pytest.ini", "setup.py", "setup.cfg"]):
            cmds += ["python -m compileall .", "python -m pytest -q"]
        elif any(cwd.glob("**/*.py")):
            cmds.append("python -m compileall .")
        if (cwd / "package.json").exists():
            cmds += ["npm test -- --runInBand", "npm test"]
        if (cwd / "go.mod").exists():
            cmds.append("go test ./...")
        if (cwd / "Cargo.toml").exists():
            cmds.append("cargo test")
        if (cwd / "pom.xml").exists():
            cmds.append("mvn test -q")
        if (cwd / "build.gradle").exists() or (cwd / "build.gradle.kts").exists():
            cmds.append("./gradlew test")
        if (cwd / "composer.json").exists():
            cmds += ["composer test", "vendor/bin/phpunit"]
        if (cwd / "Gemfile").exists():
            cmds += ["bundle exec rake test", "bundle exec rspec"]
        if (cwd / "Makefile").exists():
            cmds.append("make test")
        unique = []
        seen = set()
        for c in cmds:
            if c not in seen:
                unique.append(c)
                seen.add(c)
        return unique

    @tool
    def save_checkpoint(self) -> ToolResult:
        """Save the current git diff to a temporary patch checkpoint."""
        code, out = self._run(f"git diff --no-color > {CHECKPOINT}", 20)
        if code != 0:
            return ToolResult(f"ERROR: failed to save checkpoint.\n{out}", True)
        return ToolResult(f"Saved checkpoint to {CHECKPOINT} ({CHECKPOINT.stat().st_size if CHECKPOINT.exists() else 0} bytes).", False)

    @tool
    def restore_checkpoint(self) -> ToolResult:
        """Restore the last saved checkpoint patch."""
        if not CHECKPOINT.exists():
            return ToolResult("ERROR: no checkpoint exists.", True)
        code, out = self._run(f"git reset --hard && git clean -fd && git apply {CHECKPOINT}", 60)
        if code != 0:
            return ToolResult(f"ERROR: failed to restore checkpoint.\n{out}", True)
        return ToolResult("Restored checkpoint patch.", False)

    @tool
    def diff_summary(self) -> ToolResult:
        """Summarize changed files, diff stat, and diff preview."""
        _, names = self._run("git diff --name-only", 15)
        _, stat = self._run("git diff --stat --no-color", 15)
        _, diff = self._run("git diff --no-color", 25)
        return ToolResult(f"Changed files:\n{names}\n\nStat:\n{stat}\n\nDiff preview:\n{diff}", False)

    def _changed_files(self) -> list[str]:
        code, out = self._run("git diff --name-only", 10)
        if code != 0:
            return []
        return [x.strip() for x in out.splitlines() if x.strip() and not x.startswith("STDOUT:")]

    @tool
    def hidden_validator_review(self) -> ToolResult:
        """Review current diff like a hidden SWE-bench/Polyglot validator."""
        files = self._changed_files()
        _, stat = self._run("git diff --stat --no-color", 15)
        _, diff = self._run("git diff --no-color", 25)
        test_files = [f for f in files if any(p in f.lower() for p in ["test", "spec", "__tests__"])]
        risks: list[str] = []
        if not files:
            risks.append("No files changed; validators likely fail unless task required no-op.")
        if test_files:
            risks.append(f"Patch modifies tests/specs: {test_files}. Usually invalid.")
        if len(files) > 6:
            risks.append("Many files changed; risk of unrelated behavior regression.")
        if "except Exception" in diff or "except:" in diff:
            risks.append("Broad exception handling appears; ensure real errors are not hidden.")
        if any(x in diff for x in ["TODO", "NotImplemented", "pass  #"]):
            risks.append("Patch may contain placeholder behavior.")
        if any(s in diff.lower() for s in ["swe-bench", "polyglot", "benchmark", "validator"]):
            risks.append("Patch mentions benchmark/validator terms; likely overfit.")
        if not risks:
            risks.append("No obvious structural risks. Focus on semantic edge cases and compatibility.")
        likely_tests = [
            "Regression described by task through public API.",
            "Old import/call patterns still work unless explicitly changed.",
            "None/empty/boundary inputs around changed behavior.",
            "Consistent behavior across all affected modules/files.",
            "Expected error type/message compatibility.",
        ]
        probes = []
        if any(f.endswith(".py") for f in files):
            probes += ["python -m compileall .", "python - <<'PY'\n# import changed public APIs and exercise task behavior\nPY"]
        if any(f.endswith((".js", ".ts", ".tsx", ".jsx")) for f in files):
            probes.append("npm test -- --runInBand")
        if any(f.endswith(".go") for f in files):
            probes.append("go test ./...")
        if any(f.endswith(".rs") for f in files):
            probes.append("cargo test")
        obs = "\n".join([
            f"Deadline phase={DEADLINE.phase}, remaining={int(DEADLINE.remaining)}s",
            "Changed files:", *(f"- {f}" for f in files), "",
            "Diff stat:", stat, "",
            "Likely hidden tests:", *(f"- {t}" for t in likely_tests), "",
            "Patch risks:", *(f"- {r}" for r in risks), "",
            "Suggested cheap probes:", *(f"- {p}" for p in probes), "",
            "Repair only high-confidence risks. If time is low and patch is plausible, finish.",
        ])
        return ToolResult(obs, bool(test_files or not files))

    @tool
    def run_fast_checks(self, max_seconds: int = 240) -> ToolResult:
        """Run cheap validation commands under a strict total cap."""
        max_seconds = max(15, min(int(max_seconds), 360))
        if DEADLINE.remaining < 150:
            return ToolResult("Skipped checks: less than 150 seconds remain; finish or tiny repair only.", False)
        cmds = self._detect_test_commands()
        if not cmds:
            return ToolResult("No standard validation commands detected.", False)
        started = time.time()
        parts = []
        failed = False
        for cmd in cmds:
            remaining_budget = max_seconds - int(time.time() - started)
            if remaining_budget <= 10 or DEADLINE.remaining < 120:
                parts.append("Stopping checks to preserve finish time.")
                break
            timeout = min(90, remaining_budget)
            if cmd in {"python -m pytest -q", "npm test", "go test ./...", "cargo test", "mvn test -q", "make test"}:
                timeout = min(timeout, 120)
            code, out = self._run(cmd, timeout)
            parts.append(f"$ {cmd}\nexit={code}\n{out}")
            if code != 0:
                failed = True
                break
        return ToolResult("\n\n".join(parts), failed)

    @tool
    def finish(self, summary: str = "") -> ToolResult:
        """Complete the task and capture final git diff patch."""
        self.checkpoint = self.get_final_git_patch()
        return ToolResult("finish", False)

    def get_final_git_patch(self) -> str:
        try:
            subprocess.run(["git", "add", "-A"], capture_output=True, text=True, timeout=30, check=False)
            diff = subprocess.run(["git", "diff", "--cached", "--no-color"], capture_output=True, text=True, timeout=30, check=True)
            return diff.stdout or ""
        except Exception as e:
            logger.exception("Error generating final patch")
            return f"Error generating git patch: {e}"


# ============================================================================
# CONVERSATION STATE
# ============================================================================

class ChainOfThought:
    @dataclass
    class Action:
        thought: str
        tool_name: str
        tool_args: dict
        observation: str
        is_error: bool = False

    def __init__(self, task: str) -> None:
        self.task = task
        self.actions: list[ChainOfThought.Action] = []
        self.summary = ""

    def add_action(self, thought: str, tool_name: str, tool_args: dict, observation: str, is_error: bool) -> None:
        self.actions.append(self.Action(thought, tool_name, tool_args, observation, is_error))
        if len(self.actions) > CONTEXT_COMPACTION_THRESHOLD:
            self._compact()

    def _compact(self) -> None:
        keep = self.actions[-10:]
        old = self.actions[:-10]
        lines = [self.summary] if self.summary else []
        lines.append("Compacted prior actions:")
        for a in old[-20:]:
            obs = a.observation.replace("\n", " ")[:500]
            lines.append(f"- {a.tool_name}({a.tool_args}) -> {'ERROR' if a.is_error else 'OK'}: {obs}")
        self.summary = "\n".join(lines)[-12000:]
        self.actions = keep

    def messages(self, tool_docs: str) -> list[dict]:
        system = SYSTEM_PROMPT.format(tools_docs=tool_docs, format_prompt=FORMAT_PROMPT)
        system += f"\nCurrent deadline phase: {DEADLINE.phase}. {DEADLINE.instruction()}\n"
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": "Task:\n" + self.task}]
        if self.summary:
            msgs.append({"role": "user", "content": "Memory summary:\n" + self.summary})
        for a in self.actions:
            msgs.append({"role": "assistant", "content": f"next_thought: {a.thought}\nnext_tool_name: {a.tool_name}\nnext_tool_args: {json.dumps(a.tool_args, ensure_ascii=False)}"})
            msgs.append({"role": "user", "content": "observation: " + a.observation})
        msgs.append({"role": "user", "content": f"Deadline status: remaining={int(DEADLINE.remaining)}s phase={DEADLINE.phase}. {DEADLINE.instruction()}\n{STOP_INSTRUCTION}"})
        return msgs


# ============================================================================
# AGENT LOOP
# ============================================================================

class ValidatorFirstAgent:
    def __init__(self, task: str, model: str = GLM_5_MODEL) -> None:
        self.task = task
        self.model = model
        self.tools = ToolManager()
        self.cot = ChainOfThought(task)

    def run(self) -> str:
        logger.info("Starting validator-first agent with model=%s", self.model)
        for step in range(MAX_STEPS):
            if DEADLINE.remaining <= 30:
                logger.warning("Deadline nearly exhausted; forcing final patch.")
                return self.tools.get_final_git_patch()
            try:
                thought, tool_name, tool_args, _raw = Network.inference(self.cot.messages(self.tools.get_tool_docs()), self.model, temperature=0.0)
            except Exception as e:
                logger.exception("Inference failed")
                # Fail soft: return any current patch instead of timing out with nothing.
                return self.tools.get_final_git_patch()
            assert thought is not None and tool_name is not None and tool_args is not None
            if DEADLINE.remaining <= 90 and tool_name not in {"finish", "diff_summary"}:
                tool_name, tool_args = "finish", {"summary": "panic finish before timeout"}
                thought = "Panic finish: less than 90 seconds remain, so submit the best current patch."
            logger.info("Step %d/%d: %s", step + 1, MAX_STEPS, tool_name)
            result = self.tools.run_tool(tool_name, tool_args)
            self.cot.add_action(thought, tool_name, tool_args, result.observation, result.is_error)
            if tool_name == "finish" and not result.is_error:
                return self.tools.checkpoint
        logger.warning("Max steps reached; returning current patch.")
        return self.tools.get_final_git_patch()


# ============================================================================
# CLI
# ============================================================================

def read_task() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()
    if not sys.stdin.isatty():
        data = sys.stdin.read().strip()
        if data:
            return data
    for key in ["TASK", "PROMPT", "PROBLEM_STATEMENT", "ISSUE"]:
        val = os.getenv(key)
        if val:
            return val.strip()
    raise SystemExit("No task provided. Pass task as argv, stdin, or TASK/PROMPT/PROBLEM_STATEMENT env var.")


def main() -> None:
    task = read_task()
    patch = ValidatorFirstAgent(task).run()
    sys.stdout.write(patch)
    if patch and not patch.endswith("\n"):
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
