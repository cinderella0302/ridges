from __future__ import annotations

import json
import os
import random
import re
import subprocess
import time
from typing import Any, Dict, Optional
from uuid import uuid4

import requests

# ---------------------------------------------------------------------------
# Environment bootstrap (same as the original agent.py)
# ---------------------------------------------------------------------------

RUN_ID = os.getenv("RUN_ID")
if not RUN_ID:
    print("[AGENT] WARNING: RUN_ID is not set")

AGENT_TIMEOUT = os.getenv("AGENT_TIMEOUT")
AGENT_TIMEOUT_SEC = float(AGENT_TIMEOUT) if AGENT_TIMEOUT else None

LLM_CONNECT_TIMEOUT = int(os.getenv("LLM_CONNECT_TIMEOUT", "30"))
LLM_READ_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "120"))

# Gateway whitelist name → OpenRouter (OpenAI-compatible) model id for local/direct calls.
# Keep aligned with inference_gateway/providers/openrouter.py WHITELISTED_OPENROUTER_* .
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

_DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
_GATEWAY_TO_OPENROUTER_EMBEDDING: Dict[str, str] = {
    _DEFAULT_EMBEDDING_MODEL: "qwen/qwen3-embedding-8b",
}

_evaluation_run_id_cached: Optional[str] = None


def _resolve_model_for_local(name: str) -> str:
    """Map gateway model name to provider model id (e.g. OpenRouter)."""
    return _GATEWAY_TO_OPENROUTER_MODEL.get(name, name)


def _resolve_embedding_for_local(name: str) -> str:
    return _GATEWAY_TO_OPENROUTER_EMBEDDING.get(name, name)


def _use_direct_openai_inference() -> bool:
    """True when SANDBOX_PROXY_URL is unset/empty and local credentials are set."""
    if (os.getenv("SANDBOX_PROXY_URL") or "").strip():
        return False
    return bool(os.getenv("RIDGES_INFERENCE_API_KEY") and os.getenv("RIDGES_INFERENCE_BASE_URL"))


def _sandbox_proxy_base() -> str:
    """Ridges sandbox proxy base URL (production / container default)."""
    return ((os.getenv("SANDBOX_PROXY_URL") or "").strip() or "http://sandbox_proxy").rstrip("/")


def _evaluation_run_id() -> str:
    """Stable id for gateway requests (EVALUATION_RUN_ID, RUN_ID, or one generated UUID)."""
    global _evaluation_run_id_cached
    if _evaluation_run_id_cached is not None:
        return _evaluation_run_id_cached
    e = (os.getenv("EVALUATION_RUN_ID") or os.getenv("RUN_ID") or "").strip()
    _evaluation_run_id_cached = e if e else str(uuid4())
    return _evaluation_run_id_cached


def _inference_env_configured() -> bool:
    if (os.getenv("SANDBOX_PROXY_URL") or "").strip():
        return True
    return _use_direct_openai_inference()


if not _inference_env_configured():
    print(
        "[AGENT] WARNING: No inference route configured. "
        "Set SANDBOX_PROXY_URL (production) or RIDGES_INFERENCE_API_KEY + RIDGES_INFERENCE_BASE_URL (local)."
    )

# ---------------------------------------------------------------------------
# Inference & Embedding APIs
# ---------------------------------------------------------------------------


def _retry_sleep_after_rate_limit(attempt: int) -> None:
    wait = min(0.3 * (2**attempt) + random.uniform(0, 0.5), 5.0)
    time.sleep(wait)


def inference(model, temperature, messages):
    """Chat completion: sandbox proxy (production) or OpenAI-compatible API (local).

    Gateway expects ``evaluation_run_id`` and returns JSON ``content`` (not legacy ``run_id`` + raw body).
    """
    timeout = (LLM_CONNECT_TIMEOUT, LLM_READ_TIMEOUT)
    direct = _use_direct_openai_inference()

    if direct:
        api_key = os.environ.get("RIDGES_INFERENCE_API_KEY")
        base_url = os.environ.get("RIDGES_INFERENCE_BASE_URL")
        if not api_key or not base_url:
            print("[AGENT] inference(): Local inference missing RIDGES_INFERENCE_API_KEY or RIDGES_INFERENCE_BASE_URL")
            return None
        url = f"{base_url.rstrip('/')}/chat/completions"
        resolved = _resolve_model_for_local(model)
        payload: dict[str, Any] = {
            "model": resolved,
            "messages": messages,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        print(
            f"[AGENT] inference(): direct endpoint model={resolved} (from {model}), "
            f"temperature={temperature}, {len(messages)} messages"
        )
    else:
        proxy_base = _sandbox_proxy_base()
        url = f"{proxy_base}/api/inference"
        payload = {
            "evaluation_run_id": _evaluation_run_id(),
            "model": model,
            "temperature": temperature,
            "messages": messages,
        }
        headers = {"Content-Type": "application/json"}
        print(
            f"[AGENT] inference(): proxy {proxy_base} model={model}, "
            f"temperature={temperature}, {len(messages)} messages"
        )

    wait = 1.0
    max_wait = 60.0
    for attempt in range(5):
        try:
            response = requests.post(url, json=payload, timeout=timeout, headers=headers)
            if response.status_code == 429 and attempt < 4:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        time.sleep(float(retry_after))
                    except ValueError:
                        time.sleep(wait)
                else:
                    time.sleep(wait)
                wait = min(wait * 2, max_wait)
                print(f"[AGENT] inference(): HTTP 429, retrying (attempt {attempt + 2}/5)...")
                continue
            if response.status_code != 200:
                print(
                    f"[AGENT] inference(): Inference failed with status {response.status_code}: "
                    f"{response.text[:800]}"
                )
                return None
            data = response.json()
            if direct:
                message = (data.get("choices") or [{}])[0].get("message") or {}
                result = (message.get("content") or "").strip()
            else:
                result = (data.get("content") or "").strip()
            print(f"[AGENT] inference(): Inference response: {len(result)} characters")
            return result or None

        except requests.exceptions.Timeout as e:
            print(f"[AGENT] inference(): Request timeout: {e}")
            if attempt < 4:
                _retry_sleep_after_rate_limit(attempt)
                continue
            return None
        except requests.exceptions.ConnectionError as e:
            print(f"[AGENT] inference(): Connection error: {e}")
            if attempt < 4:
                _retry_sleep_after_rate_limit(attempt)
                continue
            return None
        except (ValueError, json.JSONDecodeError) as e:
            print(f"[AGENT] inference(): Invalid JSON in response: {e}")
            return None
        except Exception as e:
            print(f"[AGENT] inference(): Inference request failed: {e}")
            return None

    return None


def embedding(input):
    """Embeddings: sandbox proxy or OpenAI-compatible ``/embeddings`` (local)."""
    timeout = (LLM_CONNECT_TIMEOUT, min(LLM_READ_TIMEOUT, 120))
    model = os.getenv("RIDGES_EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL)
    direct = _use_direct_openai_inference()

    try:
        if direct:
            api_key = os.environ.get("RIDGES_INFERENCE_API_KEY")
            base_url = os.environ.get("RIDGES_INFERENCE_BASE_URL")
            if not api_key or not base_url:
                print("[AGENT] embedding(): Local mode missing RIDGES_INFERENCE_API_KEY or RIDGES_INFERENCE_BASE_URL")
                return None
            url = f"{base_url.rstrip('/')}/embeddings"
            resolved = _resolve_embedding_for_local(model)
            payload = {"model": resolved, "input": input}
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            print(f"[AGENT] embedding(): direct endpoint model={resolved}")
            response = requests.post(url, json=payload, timeout=timeout, headers=headers)
            if response.status_code != 200:
                print(f"[AGENT] embedding(): failed status {response.status_code}: {response.text[:500]}")
                return None
            data = response.json()
            row = (data.get("data") or [{}])[0]
            result = row.get("embedding")
            if not isinstance(result, list):
                print("[AGENT] embedding(): unexpected response shape")
                return None
            print(f"[AGENT] embedding(): Embedding response: {len(result)} dimensions")
            return result

        proxy_base = _sandbox_proxy_base()
        payload = {
            "evaluation_run_id": _evaluation_run_id(),
            "model": model,
            "input": input,
        }
        print(f"[AGENT] embedding(): proxy {proxy_base} model={model}")
        response = requests.post(
            f"{proxy_base}/api/embedding",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
        if response.status_code != 200:
            print(f"[AGENT] embedding(): Embedding failed with status {response.status_code}: {response.text[:500]}")
            return None
        data = response.json()
        result = data.get("embedding")
        if not isinstance(result, list):
            print("[AGENT] embedding(): unexpected response shape (expected JSON embedding array)")
            return None
        print(f"[AGENT] embedding(): Embedding response: {len(result)} dimensions")
        return result

    except Exception as e:
        print(f"[AGENT] embedding(): Embedding request failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Agent Configuration
# ---------------------------------------------------------------------------

# The model to use for the coding agent — Ridges provides this through the
# sandbox proxy.  Kimi-K2 is the default from the original agent, but the
# config can be overridden via the RIDGES_AGENT_MODEL env var.
DEFAULT_MODEL = os.getenv("RIDGES_AGENT_MODEL", "MiniMaxAI/MiniMax-M2.5")

# The model to use for quick classification / routing tasks (cheaper, faster)
FAST_MODEL = os.getenv("RIDGES_AGENT_FAST_MODEL", "Qwen/Qwen3-Coder-Next")


class AgentConfig:
    """Runtime configuration for the coding agent.

    NOTE: Intentionally NOT a @dataclass because the Ridges miner runtime
    (ridges_miner_runtime.py) loads agent.py via importlib.util dynamically,
    and the @dataclass decorator fails when the module is not in sys.modules.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        fast_model: str = FAST_MODEL,
        temperature: float = 0.1,
        max_steps: int = 400,
        max_output_chars: int = 8000,
        max_head_tail_chars: int = 4000,
        max_conversation_chars: int = 120000,
        max_inference_retries: int = 3,
        inference_retry_delay: float = 5.0,
        command_timeout: int = 120,
        working_dir: Optional[str] = None,
    ):
        self.model = model
        self.fast_model = fast_model
        self.temperature = temperature
        self.max_steps = max_steps
        self.max_output_chars = max_output_chars
        self.max_head_tail_chars = max_head_tail_chars
        self.max_conversation_chars = max_conversation_chars
        self.max_inference_retries = max_inference_retries
        self.inference_retry_delay = inference_retry_delay
        self.command_timeout = command_timeout
        self.working_dir = working_dir


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------


MINI_ACTION_REGEX_RSWEA = re.compile(r"```rswea_bash_command\s*\n(.*?)\n```", re.DOTALL)
MINI_ACTION_REGEX_BASH = re.compile(r"```bash\s*\n(.*?)\n\s*```", re.DOTALL)

# Match mini_textbased.yaml observation truncation (10000 / 5000 / 5000).
MINI_OBSERVATION_FULL_MAX = 10000
MINI_OBSERVATION_HEAD = 5000
MINI_OBSERVATION_TAIL = 5000

SYSTEM_PROMPT = """\
You are a helpful assistant that can interact with a computer.

Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).
Include a THOUGHT section before your command where you explain your reasoning process.
Format your response as shown in:

THOUGHT: Your reasoning and analysis here. Explain why you want to perform the action.

```rswea_bash_command
your_command_here
```

Failure to follow these rules will cause your response to be rejected.

Every action runs in a fresh shell: directory and environment changes do not persist. Prefix actions with \
`VAR=value cd /path && ...` when you need a specific working directory or variables.

When you are done fixing the issue, submit a unified diff of your changes as your single action using:

```rswea_bash_command
echo SUBMIT_PATCH && git -c color.ui=false -c core.pager=cat diff HEAD
```

For new untracked files that `git diff HEAD` alone would miss, extend the command with listing and cat of untracked files as needed. Do not output SUBMIT_PATCH until you are confident the fix is correct and tested.
"""


def _instance_prompt_mini(problem_statement: str, working_dir: str) -> str:
    return f"""Please solve this issue:

{problem_statement}

You can execute bash commands and edit files to implement the necessary changes.

## Recommended Workflow

This workflow should be done step-by-step so that you can iterate on your changes and any possible problems.

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust
6. Submit your changes: use exactly one action whose command starts with `echo SUBMIT_PATCH` and includes \
`git -c color.ui=false -c core.pager=cat diff HEAD` as described in the system message. Do not combine submission with unrelated commands.

## Important Rules

1. Every response must contain exactly one action
2. The action must be enclosed in triple backticks with the `rswea_bash_command` label (or `bash` if you cannot use the label)
3. Directory or environment variable changes are not persistent. Every action is executed in a new subshell.

Environment: sandbox (each command runs in a fresh shell; use `cd ... &&` when needed).

## Formatting your response

Here is an example of a correct response:

THOUGHT: I need to understand the structure of the repository first. Let me check what files are in the current directory to get a better understanding of the codebase.

```rswea_bash_command
ls -la
```

## Useful command examples

### Create a new file:

```rswea_bash_command
cat <<'EOF' > newfile.py
import numpy as np
hello = "world"
print(hello)
EOF
```

### Edit files with sed:

```rswea_bash_command
# Replace all occurrences
sed -i 's/old_string/new_string/g' filename.py

# Replace only first occurrence
sed -i 's/old_string/new_string/' filename.py

# Replace first occurrence on line 1
sed -i '1s/old_string/new_string/' filename.py

# Replace all occurrences in lines 1-10
sed -i '1,10s/old_string/new_string/g' filename.py
```

### View file content:

```rswea_bash_command
nl -ba filename.py | sed -n '10,20p'
```

### Any other command you want to run

```rswea_bash_command
anything
```

Working directory: {working_dir}
"""


def format_mini_format_error(n_actions: int) -> str:
    """Same guidance shape as ridges-agent format_error_template (textbased)."""
    return f"""Format error:

Expected exactly 1 action, found {n_actions}.

Please always provide EXACTLY ONE action in triple backticks, found {n_actions} actions.

Please format your action in triple backticks as shown.

THOUGHT: Here are some thoughts about why you want to perform the action.

```rswea_bash_command
<action>
```

If you have completed your assignment, consult the first message about how to
submit your solution (you will not be able to continue working on this task after that)."""


FORMAT_ERROR_MESSAGE = format_mini_format_error(0)

SUBMISSION_SENTINEL = "SUBMIT_PATCH"


# ---------------------------------------------------------------------------
# Shell Executor
# ---------------------------------------------------------------------------


class ShellExecutor:
    """Execute bash commands in the sandbox and capture output."""

    def __init__(self, working_dir: str | None = None, timeout: int = 120):
        self.working_dir = working_dir or os.getcwd()
        self.timeout = timeout

    def execute(self, command: str) -> dict[str, Any]:
        """Run a bash command and return structured output.

        Returns:
            dict with keys: stdout, stderr, returncode, timed_out
        """
        try:
            result = subprocess.run(
                ["bash", "-c", command],
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, "TERM": "dumb"},
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Command timed out after {self.timeout} seconds",
                "returncode": -1,
                "timed_out": True,
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": f"Execution error: {type(e).__name__}: {e}",
                "returncode": -1,
                "timed_out": False,
            }


def normalize_patch_text(patch: str) -> str:
    """Strip ANSI color sequences and normalize newlines for ``git apply``."""
    if not patch:
        return ""
    out = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", patch)
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    return out.strip("\n") + ("\n" if out.strip() else "")


def authoritative_worktree_patch(executor: "ShellExecutor") -> str:
    """Unified diff from repo state (HEAD vs worktree), not captured shell transcript.

    Avoids relying on ``echo SUBMIT_PATCH && git diff`` stdout, which can pick up
    color codes, prompts, or truncation. Matches SWE-bench expectation: patch applies to HEAD.
    """
    diff = executor.execute("git -c color.ui=false -c core.pager=cat diff HEAD")
    parts: list[str] = []
    if diff.get("returncode") == 0 and (diff.get("stdout") or "").strip():
        parts.append(diff["stdout"].rstrip("\n"))
    untracked = executor.execute(
        "git ls-files --others --exclude-standard | while read -r f; do "
        'test -f "$f" || continue; echo "--- /dev/null"; echo "+++ b/$f"; cat "$f"; echo; done'
    )
    if untracked.get("returncode") == 0 and (untracked.get("stdout") or "").strip():
        parts.append(untracked["stdout"].rstrip("\n"))
    merged = "\n".join(parts).strip("\n")
    if not merged:
        return ""
    return merged + "\n"


# ---------------------------------------------------------------------------
# Action Parser — extracts bash commands from LLM responses
# ---------------------------------------------------------------------------


def count_mini_actions(response: str) -> int:
    """How many action blocks the mini-style parsers see (mswea takes precedence over bash)."""
    ms = MINI_ACTION_REGEX_RSWEA.findall(response)
    if ms:
        return len(ms)
    return len(MINI_ACTION_REGEX_BASH.findall(response))


def parse_bash_command(response: str) -> str | None:
    """Extract exactly one shell command (ridges-agent text-based protocol).

    A single ```rswea_bash_command ... ``` block, or exactly one ```bash ... ```.
    Zero or multiple matching blocks → None (format error).
    """
    ms = [a.strip() for a in MINI_ACTION_REGEX_RSWEA.findall(response)]
    if len(ms) == 1:
        return ms[0]
    if len(ms) > 1:
        return None

    bs = [a.strip() for a in MINI_ACTION_REGEX_BASH.findall(response)]
    if len(bs) == 1:
        return bs[0]
    if len(bs) > 1:
        return None
    return None


def check_submission(command: str, output: str) -> str | None:
    """Check if a command result contains a patch submission.

    Returns the patch text if a submission is detected, None otherwise.
    """
    if SUBMISSION_SENTINEL not in command:
        return None

    # The output after the sentinel is the patch
    sentinel_idx = output.find(SUBMISSION_SENTINEL)
    if sentinel_idx == -1:
        return None

    patch = output[sentinel_idx + len(SUBMISSION_SENTINEL) :].strip()
    return patch if patch else None


# ---------------------------------------------------------------------------
# Conversation Manager — manages message history with context window control
# ---------------------------------------------------------------------------


class ConversationManager:
    """Manage LLM conversation history with truncation and context window control."""

    def __init__(self, max_chars: int = 120000):
        self.messages: list[dict[str, str]] = []
        self.max_chars = max_chars

    def add(self, role: str, content: str) -> None:
        """Add a message to the conversation."""
        self.messages.append({"role": role, "content": content})
        self._trim_if_needed()

    def get_messages(self) -> list[dict[str, str]]:
        """Return messages suitable for the LLM API (no metadata keys)."""
        # Always keep system message + first user message (task)
        # Then trim from the oldest assistant/user pairs
        return list(self.messages)

    def _trim_if_needed(self) -> None:
        """Trim conversation history if it exceeds max_chars.

        Strategy: keep system message, first user message, and the most recent
        N messages that fit within the budget. When trimming, add a summary
        note so the LLM knows context was removed.
        """
        total_chars = sum(len(m.get("content", "")) for m in self.messages)

        if total_chars <= self.max_chars:
            return

        # Never trim system message (index 0) or first user message (index 1)
        # Trim from index 2 onward, keeping the most recent messages
        if len(self.messages) <= 3:
            return  # Can't trim further

        # Calculate how much we need to remove
        excess = total_chars - self.max_chars

        # Remove messages from the middle (after first user, before recent context)
        # Keep the last ~6 messages (3 turns) and the first 2 messages
        min_keep_head = 2  # system + first user
        min_keep_tail = 6  # last 3 turns

        if len(self.messages) <= min_keep_head + min_keep_tail:
            return  # Already at minimum

        # Find how many middle messages to remove
        head = self.messages[:min_keep_head]
        tail = self.messages[-min_keep_tail:]
        middle = self.messages[min_keep_head:-min_keep_tail]

        # Remove oldest middle messages until we're under budget
        trimmed_middle = list(middle)
        while trimmed_middle and excess > 0:
            removed = trimmed_middle.pop(0)
            excess -= len(removed.get("content", ""))

        # Add a context note
        context_note = {
            "role": "user",
            "content": (
                "[System note: Some earlier conversation history was trimmed to fit the context window. "
                "The original task and your most recent actions are preserved. "
                "Continue working on the problem as before.]"
            ),
        }

        self.messages = head + trimmed_middle + [context_note] + tail

    def total_chars(self) -> int:
        return sum(len(m.get("content", "")) for m in self.messages)


# ---------------------------------------------------------------------------
# Output Formatter — truncates long command outputs for context management
# ---------------------------------------------------------------------------


def format_observation(
    output: dict[str, Any],
    max_chars: int = 8000,
    max_head_tail: int = 4000,
) -> str:
    """Format command execution output for the LLM, with truncation.

    If output is short enough, show it in full. Otherwise, show the head
    and tail with a truncation notice.
    """
    stdout = output.get("stdout", "")
    stderr = output.get("stderr", "")
    returncode = output.get("returncode", 0)
    timed_out = output.get("timed_out", False)

    parts = []

    if timed_out:
        parts.append("<timed_out>Command timed out</timed_out>")

    parts.append(f"<returncode>{returncode}</returncode>")

    # Format stdout
    if stdout:
        if len(stdout) <= max_chars:
            parts.append(f"<stdout>\n{stdout}\n</stdout>")
        else:
            head = stdout[:max_head_tail]
            tail = stdout[-max_head_tail:]
            elided = len(stdout) - 2 * max_head_tail
            parts.append(
                f"<stdout>\n{head}\n"
                f"... [{elided} characters elided] ...\n"
                f"{tail}\n</stdout>"
            )

    # Format stderr (always show in full if small, truncate if large)
    if stderr:
        if len(stderr) <= max_chars:
            parts.append(f"<stderr>\n{stderr}\n</stderr>")
        else:
            head = stderr[:max_head_tail]
            tail = stderr[-max_head_tail:]
            elided = len(stderr) - 2 * max_head_tail
            parts.append(
                f"<stderr>\n{head}\n"
                f"... [{elided} characters elided] ...\n"
                f"{tail}\n</stderr>"
            )

    return "\n".join(parts)


def shell_output_to_mini_dict(output: dict[str, Any]) -> dict[str, Any]:
    """Map ShellExecutor output to mini LocalEnvironment-style dict for observation templates."""
    stdout = output.get("stdout") or ""
    stderr = output.get("stderr") or ""
    parts: list[str] = []
    if stdout.strip():
        parts.append(stdout.rstrip("\n"))
    if stderr.strip():
        parts.append(stderr.rstrip("\n"))
    combined = "\n".join(parts)
    if combined and not combined.endswith("\n"):
        combined += "\n"
    exc = ""
    if output.get("timed_out"):
        exc = (stderr or "Command timed out.").strip()
    elif output.get("returncode", 0) == -1 and stderr.strip():
        exc = stderr.strip()
    return {
        "output": combined,
        "returncode": output.get("returncode", 0),
        "exception_info": exc,
    }


def format_mini_observation(output: dict[str, Any]) -> str:
    """Observation text aligned with mini_textbased.yaml ``observation_template``."""
    mini = shell_output_to_mini_dict(output)
    lines: list[str] = []
    ei = (mini.get("exception_info") or "").strip()
    if ei:
        lines.append(ei)
    lines.append(str(mini.get("returncode", 0)))
    body = mini.get("output") or ""
    if len(body) < MINI_OBSERVATION_FULL_MAX:
        lines.append("")
        lines.append(body)
    else:
        elided = len(body) - MINI_OBSERVATION_FULL_MAX
        lines.append("")
        lines.append(
            "The output of your last command was too long.\n"
            "Please try a different command that produces less output.\n"
            "If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.\n"
            "If you're using grep or find and it produced too much output, you can use a more selective search pattern.\n"
            "If you really need to see something from the full command's output, you can redirect output to a file "
            "and then search in that file."
        )
        lines.append("")
        lines.append(body[:MINI_OBSERVATION_HEAD])
        lines.append("")
        lines.append(f"{elided} characters elided")
        lines.append("")
        lines.append(body[-MINI_OBSERVATION_TAIL:])
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Patch Validator — validates the generated diff before returning
# ---------------------------------------------------------------------------


def validate_patch(patch: str) -> bool:
    """Basic validation that a string looks like a valid unified diff.

    Checks for:
    - At least one diff hunk header (@@ ... @@)
    - At least one file header (--- / +++ or diff --git)
    - Non-empty content
    """
    if not patch or not patch.strip():
        return False

    # Must have at least one hunk header
    if not re.search(r"@@ -\d+(,\d+)? \+\d+(,\d+)? @@", patch):
        # Also accept new-file-only patches (no @@ headers for empty files)
        if "--- /dev/null" not in patch and "+++ b/" not in patch:
            return False

    return True


def validate_patch_with_git(patch: str, working_dir: str) -> bool:
    """Validate a patch by applying it with git apply --check on the current tree.

    Prefer :func:`validate_patch_applies_cleanly` for SWE-bench style runs: the working
    tree often already contains the patch, so checking without stashing is misleading.
    """
    try:
        result = subprocess.run(
            ["git", "apply", "--check"],
            input=patch,
            capture_output=True,
            text=True,
            cwd=working_dir,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _working_dir_is_git_repo(working_dir: str) -> bool:
    return bool(working_dir) and os.path.isdir(os.path.join(working_dir, ".git"))


def validate_patch_applies_cleanly(patch: str, working_dir: str) -> bool:
    """True if ``git apply --check`` succeeds against HEAD (same as Harbor/validator).

    Stashes the working tree (including untracked) so the check runs on a clean tree,
    then restores. If ``stash pop`` fails, the patch is treated as invalid (unsafe to ship).
    """
    if not working_dir or not os.path.isdir(working_dir):
        return False
    if not validate_patch(patch):
        return False
    if not _working_dir_is_git_repo(working_dir):
        return validate_patch_with_git(patch, working_dir)

    stashed = False
    apply_ok = False
    try:
        stash = subprocess.run(
            ["git", "-C", working_dir, "stash", "push", "-u", "-m", "ridges_patch_validate", "-q"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if stash.returncode == 0:
            stashed = True
        elif stash.returncode == 1:
            err = (stash.stderr or "").lower()
            if "no local changes to save" not in err and "nothing to stash" not in err:
                print(f"[AGENT] stash before patch check failed: {stash.stderr}")
                return False
        else:
            print(f"[AGENT] stash before patch check failed (exit {stash.returncode}): {stash.stderr}")
            return False

        check = subprocess.run(
            ["git", "-C", working_dir, "apply", "--check"],
            input=patch,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if check.returncode != 0:
            err = (check.stderr or check.stdout or "").strip()
            if err:
                print(f"[AGENT] git apply --check: {err[:800]}")
            return False
        apply_ok = True
    except subprocess.TimeoutExpired:
        print("[AGENT] patch validation timed out")
        return False
    except Exception as e:
        print(f"[AGENT] patch validation error: {e}")
        return False
    finally:
        if stashed:
            pop = subprocess.run(
                ["git", "-C", working_dir, "stash", "pop", "-q"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if pop.returncode != 0:
                print(f"[AGENT] stash pop after patch check failed: {pop.stderr}")
                apply_ok = False

    return apply_ok


def reset_worktree_to_head_for_harbor(working_dir: str) -> None:
    """Restore a clean tracked tree at HEAD after patch validation.

    Harbor runs ``git apply --check`` on the task repo *after* ``agent_main``
    returns, while the worktree usually still contains the edited files. In that
    state the working tree already matches the patch's ``+`` side, so
    ``git apply --check`` fails with ``patch does not apply`` even when the diff
    is correct (see ``trial_dir/agent/git-apply-check.log``).

    In-process validation uses ``git stash`` / ``git apply --check`` / ``stash pop``,
    which ends dirty again. This reset matches the clean preimage our validation
    actually checked.
    """
    if not working_dir or not os.path.isdir(working_dir):
        return
    if not _working_dir_is_git_repo(working_dir):
        return
    try:
        r = subprocess.run(
            ["git", "-C", working_dir, "reset", "--hard", "HEAD"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if r.returncode != 0:
            print(f"[AGENT] WARNING: git reset --hard HEAD failed ({r.returncode}): {r.stderr}")
        else:
            print("[AGENT] Reset worktree to HEAD after validated patch (Harbor apply-check compatibility)")
    except Exception as e:
        print(f"[AGENT] WARNING: git reset --hard HEAD error: {e}")


# ---------------------------------------------------------------------------
# The Coding Agent
# ---------------------------------------------------------------------------


class CodingAgent:
    """LLM + bash loop modeled on ridges-agent (text-based actions, linear messages).

    Each turn: query the model, parse exactly one fenced action (rswea_bash_command
    or bash), run it in a fresh shell, append the mini-style observation.
    Exit when a valid SUBMIT_PATCH diff is produced, step budget is exhausted, or
    time runs out.
    """

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self.executor = ShellExecutor(
            working_dir=self.config.working_dir,
            timeout=self.config.command_timeout,
        )
        self.conversation = ConversationManager(max_chars=self.config.max_conversation_chars)
        self.step_count = 0
        self.start_time: float = 0
        self.files_modified: set[str] = set()

    def _detect_working_dir(self) -> str:
        """Auto-detect the git repo working directory."""
        # Try the current directory first
        cwd = os.getcwd()
        git_dir = os.path.join(cwd, ".git")
        if os.path.isdir(git_dir):
            return cwd

        # Walk up to find a git repo
        path = cwd
        while path != "/":
            parent = os.path.dirname(path)
            if parent == path:
                break
            if os.path.isdir(os.path.join(parent, ".git")):
                return parent
            path = parent

        return cwd

    def _build_initial_messages(self, problem_statement: str) -> None:
        """Build the initial system and user messages for the conversation."""
        working_dir = self.config.working_dir or self._detect_working_dir()

        # System message
        self.conversation.add("system", SYSTEM_PROMPT)

        self.conversation.add("user", _instance_prompt_mini(problem_statement, working_dir))

    def _call_inference(self, messages: list[dict[str, str]]) -> str | None:
        """Call the LLM with retry logic.

        Returns the LLM response text, or None if all retries fail.
        """
        for attempt in range(self.config.max_inference_retries):
            if attempt > 0:
                delay = self.config.inference_retry_delay * (2 ** (attempt - 1))
                print(f"[AGENT] Retrying inference (attempt {attempt + 1}/{self.config.max_inference_retries}) "
                      f"after {delay:.1f}s delay...")
                time.sleep(delay)

            response = inference(self.config.model, self.config.temperature, messages)
            if response is not None:
                return response

        print("[AGENT] All inference retries exhausted")
        return None

    def _check_timeout(self) -> bool:
        """Check if the agent has exceeded its time budget."""
        if AGENT_TIMEOUT_SEC is None:
            return False
        elapsed = time.time() - self.start_time
        # Leave a 30-second buffer for final diff generation
        return elapsed > (AGENT_TIMEOUT_SEC - 30)

    def _collect_patch_emergency(self) -> str:
        """Generate a patch from any uncommitted changes as a last resort.

        Called when the agent runs out of steps or time before a formal
        submission. Returns whatever diff is available, even if incomplete.
        """
        try:
            patch = normalize_patch_text(authoritative_worktree_patch(self.executor))
            if patch.strip():
                return patch
        except Exception as e:
            print(f"[AGENT] Emergency patch collection failed: {e}")

        return ""

    def run(self, problem_statement: str) -> str:
        """Run the coding agent on a problem and return a unified diff.

        This is the core agentic loop.

        Args:
            problem_statement: The description of the bug/feature to implement.

        Returns:
            A unified diff string (may be empty if no solution found).
        """
        self.start_time = time.time()
        self.step_count = 0

        # Resolve working directory
        if not self.config.working_dir:
            self.config.working_dir = self._detect_working_dir()
            self.executor.working_dir = self.config.working_dir

        print(f"[AGENT] Starting CodingAgent in {self.config.working_dir}")
        print(f"[AGENT] Model: {self.config.model}, Temperature: {self.config.temperature}")
        print(f"[AGENT] Max steps: {self.config.max_steps}")

        # Do not commit on top of an existing clone (SWE-bench / Harbor): the submitted
        # ``git diff`` must be relative to the task baseline HEAD, not a new commit.
        if _working_dir_is_git_repo(self.config.working_dir):
            print("[AGENT] Existing git repository — skipping git init / baseline commit")
        else:
            self.executor.execute("git init 2>/dev/null || true")
            self.executor.execute("git add -A 2>/dev/null || true")
            self.executor.execute("git commit -m 'initial state' --allow-empty 2>/dev/null || true")

        # Build initial conversation
        self._build_initial_messages(problem_statement)

        # --- Main agentic loop ---
        consecutive_format_errors = 0
        max_consecutive_format_errors = 3

        while self.step_count < self.config.max_steps:
            self.step_count += 1

            # Check timeout
            if self._check_timeout():
                print(f"[AGENT] Timeout reached at step {self.step_count}")
                break

            print(f"[AGENT] === Step {self.step_count}/{self.config.max_steps} ===")

            # THINK: Call the LLM
            messages = self.conversation.get_messages()
            response = self._call_inference(messages)

            if response is None:
                print("[AGENT] LLM returned no response, retrying...")
                self.conversation.add(
                    "user",
                    "The inference call failed. Please try again with a different command.",
                )
                continue

            # Add the assistant response to conversation
            self.conversation.add("assistant", response)

            # ACT: Parse the bash command
            command = parse_bash_command(response)

            if command is None:
                consecutive_format_errors += 1
                print(f"[AGENT] No valid command found (format error #{consecutive_format_errors})")

                if consecutive_format_errors >= max_consecutive_format_errors:
                    print("[AGENT] Too many consecutive format errors, attempting emergency patch")
                    break

                n_act = count_mini_actions(response)
                self.conversation.add("user", format_mini_format_error(n_act))
                continue

            # Reset format error counter on successful parse
            consecutive_format_errors = 0

            # Check for submission before executing
            if SUBMISSION_SENTINEL in command:
                print("[AGENT] Submission detected in command, executing to get patch...")

                # Execute the submission command to get the diff
                output = self.executor.execute(command)
                self.conversation.add("user", format_mini_observation(output))

                # Extract the patch
                full_output = output.get("stdout", "")
                if output.get("stderr"):
                    full_output += "\n" + output["stderr"]

                extracted = check_submission(command, full_output)
                auth = normalize_patch_text(authoritative_worktree_patch(self.executor))
                patch = auth if auth.strip() else normalize_patch_text(extracted or "")

                if patch and validate_patch_applies_cleanly(patch, self.config.working_dir):
                    print(f"[AGENT] Valid patch received ({len(patch)} chars)")
                    return patch

                if patch:
                    print(f"[AGENT] Patch received but does not pass git apply --check ({len(patch)} chars)")
                    # Try to fix the patch or get a better one
                    self.conversation.add(
                        "user",
                        "The patch you submitted fails `git apply --check` against the repository baseline "
                        "(wrong line numbers, missing context, or mixed with unrelated edits). "
                        "Re-read the current files from disk, make minimal edits, then run "
                        "`git diff` again and resubmit. Do not rely on remembered line numbers.",
                    )
                    continue

                # Sentinel was in command but no patch in output
                print("[AGENT] Submission sentinel found but no patch in output")
                self.conversation.add(
                    "user",
                    "The submission command ran but no patch was produced in the output. "
                    "Make sure your files are saved and tracked by git, then try again.",
                )
                continue

            # Execute the regular command
            print(f"[AGENT] Executing: {command[:200]}{'...' if len(command) > 200 else ''}")
            output = self.executor.execute(command)

            # Track modified files
            if output.get("returncode") == 0:
                # Check if the command might have modified files
                modifying_commands = (
                    "sed", "echo >", "cat >", "tee", "patch", "mv", "cp",
                    "python -c", "pip install", "npm", "touch", "chmod",
                    "truncate", "dd", "install",
                )
                if any(cmd_part in command for cmd_part in modifying_commands):
                    # Quick check what changed
                    diff_result = self.executor.execute("git diff --name-only 2>/dev/null")
                    if diff_result["returncode"] == 0 and diff_result["stdout"].strip():
                        for filename in diff_result["stdout"].strip().splitlines():
                            self.files_modified.add(filename)

            self.conversation.add("user", format_mini_observation(output))

            # Log step summary
            rc = output.get("returncode", -1)
            out_len = len(output.get("stdout", "")) + len(output.get("stderr", ""))
            print(f"[AGENT] Step {self.step_count} complete: returncode={rc}, output={out_len} chars, "
                  f"conversation={self.conversation.total_chars()} chars")

        # --- Loop ended without submission ---
        print(f"[AGENT] Loop ended at step {self.step_count}/{self.config.max_steps}")

        # Try to collect a patch from any uncommitted changes
        patch = self._collect_patch_emergency()
        if patch and validate_patch_applies_cleanly(patch, self.config.working_dir):
            print(f"[AGENT] Emergency patch collected ({len(patch)} chars)")
            return patch

        print("[AGENT] No valid patch could be generated")
        return ""


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def create_agent(problem_statement: str, config: AgentConfig | None = None) -> CodingAgent:
    """Return the coding agent (ridges-agent-style single-phase loop)."""
    _ = problem_statement  # reserved for future routing
    cfg = config or AgentConfig()
    print("[AGENT] Selected: CodingAgent (ridges-agent workflow)")
    return CodingAgent(config=cfg)


# ---------------------------------------------------------------------------
# Main Entry Point — the Ridges miner contract
# ---------------------------------------------------------------------------


def agent_main(input):
    """Entry point for the Ridges miner.

    Args:
        input: dict with at least a 'problem_statement' key containing
               the task description (from instruction.md).

    Returns:
        A unified diff string (the patch), or an empty string on failure.
    """
    print("[AGENT] Entered agent_main()")

    problem_statement = input.get("problem_statement", "") if isinstance(input, dict) else str(input)

    if not problem_statement:
        print("[AGENT] ERROR: Empty problem statement")
        return ""

    print(f"[AGENT] Problem statement: {len(problem_statement)} characters")
    print(f"[AGENT] Problem preview: {problem_statement[:300]}...")

    # Create and run the agent
    config = AgentConfig()
    agent = create_agent(problem_statement, config)

    try:
        patch = agent.run(problem_statement)
    except Exception as e:
        print(f"[AGENT] Agent crashed: {type(e).__name__}: {e}")
        # Last-resort emergency patch
        try:
            patch = agent._collect_patch_emergency()
        except Exception:
            patch = ""

    if not patch or not patch.strip():
        print("[AGENT] WARNING: Returning empty patch")
        return ""

    patch = normalize_patch_text(patch)

    wd = getattr(agent, "config", None) and agent.config.working_dir
    if not wd:
        wd = os.getcwd()
    if not validate_patch_applies_cleanly(patch, wd):
        print("[AGENT] WARNING: Final patch failed git apply --check; returning empty patch")
        return ""

    # Harbor runs ``git apply --check`` on a tree that still has our edits; reset
    # so the preimage matches what we validated (see reset_worktree_to_head_for_harbor).
    reset_worktree_to_head_for_harbor(wd)

    print(f"[AGENT] Returning patch: {len(patch)} characters")
    print(f"[AGENT] Patch preview:\n{patch[:500]}...")

    return patch
