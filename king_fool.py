from __future__ import annotations
import json
import os
import random
import requests
import subprocess
import sys
import textwrap
import time
from typing import Any, Dict, List, Tuple
from json import JSONDecodeError
import re
import inspect
import logging
import threading
import fnmatch
from uuid import uuid4

# Retry config: exponential backoff + jitter
RETRY_INITIAL_WAIT = 0.3
RETRY_MAX_WAIT = 5.0
RETRY_JITTER = 0.5
RETRYABLE_HTTP_CODES = (429, 500, 502, 503, 504)

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "2400"))
# LLM HTTP timeouts: (connect, read) for requests.post — matches agent_wonder / gateway patterns.
LLM_CONNECT_TIMEOUT = int(os.getenv("LLM_CONNECT_TIMEOUT", "30"))
LLM_READ_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "120"))
GLM_5_MODEL = "zai-org/GLM-5-FP8"
KIMI_MODEL = "moonshotai/Kimi-K2.5"
MINIMAX_MODEL = "MiniMaxAI/MiniMax-M2.5"
MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "400"))
# When action count exceeds this, older turns are compacted into a summary
CONTEXT_COMPACTION_THRESHOLD = int(os.getenv("AGENT_CONTEXT_COMPACTION_THRESHOLD", "20"))

# Tool output limits (truncate long tool output to fit context)
TOOL_OUTPUT_MAX_CHARS = int(os.getenv("AGENT_TOOL_OUTPUT_MAX_CHARS", "50000"))
TOOL_OUTPUT_MAX_LINE_LENGTH = int(os.getenv("AGENT_TOOL_OUTPUT_MAX_LINE", "2000"))
BASH_TIMEOUT_DEFAULT = 60
BASH_TIMEOUT_MAX = 5 * 60  # 300s
READ_FILE_MAX_CHARS = int(os.getenv("AGENT_READ_FILE_MAX_CHARS", "120000"))
GREP_MAX_MATCHES = int(os.getenv("AGENT_GREP_MAX_MATCHES", "200"))
WATCHDOG_ABORT_FRACTION = float(os.getenv("AGENT_WATCHDOG_ABORT_FRACTION", "0.92"))
NO_EDIT_WARNING_STEP = int(os.getenv("AGENT_NO_EDIT_WARNING_STEP", "6"))
REPEAT_ACTION_THRESHOLD = int(os.getenv("AGENT_REPEAT_ACTION_THRESHOLD", "2"))
AGENT_MODELS = [MINIMAX_MODEL, KIMI_MODEL, GLM_5_MODEL]

# Gateway whitelist `name` → OpenRouter model id. Production proxy maps server-side; local
# direct calls need the provider id. Keep aligned with
# inference_gateway/providers/openrouter.py::WHITELISTED_OPENROUTER_INFERENCE_MODELS.
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


def _resolve_model_for_local(name: str) -> str:
    """Map gateway model name to OpenRouter (or other OpenAI-compatible) model id."""
    return _GATEWAY_TO_OPENROUTER_MODEL.get(name, name)


def _use_direct_openai_inference() -> bool:
    """True when SANDBOX_PROXY_URL is unset/empty and local credentials are configured."""
    if (os.getenv("SANDBOX_PROXY_URL") or "").strip():
        return False
    return bool(os.getenv("RIDGES_INFERENCE_API_KEY") and os.getenv("RIDGES_INFERENCE_BASE_URL"))


def _sandbox_proxy_base() -> str:
    """Ridges sandbox proxy URL when not using direct inference."""
    return ((os.getenv("SANDBOX_PROXY_URL") or "").strip() or "http://sandbox_proxy").rstrip("/")


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

for h in list(logger.handlers):
    logger.removeHandler(h)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# ============================================================================
# PROMPTS
# ============================================================================

FORMAT_PROMPT = textwrap.dedent("""
**📝 Response Format Requirements**

You must generate EXACTLY three fields in this order:
1. `next_thought`: Your detailed reasoning about what to do next
2. `next_tool_name`: The exact tool name to use (bash, read_file, grep_search, think, finish, or str_replace_edit)
3. `next_tool_args`: Valid JSON object with tool parameters

Example:
next_thought: I need to check if the file exists first
next_tool_name: bash
next_tool_args: {"command": "ls -la abcd.py"}

DO NOT generate `observation:` - it will be provided by the system.
""")

STOP_INSTRUCTION = textwrap.dedent("""
# 🎨 
DO NOT generate `observation:` in your response. It will be provided by user for you.
Generate only SINGLE triplet of `next_thought`, `next_tool_name`, `next_tool_args` in your response.
""")

SYSTEM_PROMPT = textwrap.dedent("""
You are a Coding Assistant. Your task is to fix or implement changes in the codebase.

## Workflow (follow in order when appropriate):
1. **Understand** — Read the problem statement and clarify scope.
2. **Locate** — Use bash (find, grep, ls, cat) to discover relevant files and structure.
3. **Read** — Prefer read_file and grep_search for deterministic, bounded context.
4. **Edit** — Make minimal, precise edits with str_replace_edit (exact match, unique old_str).
5. **Test** — Run tests or commands to verify (bash).
6. **Finish** — Call finish only when done and tested.

## Priorities:
- **Current task state**: What you are working on right now.
- **Errors & solutions**: Note errors and how you resolved them.
- **Minimal edits**: Prefer small, targeted changes; keep backward compatibility.

## Guidelines:
- Use think to plan or reason before multi-step actions.
- For str_replace_edit: old_str must appear exactly once; copy from file to avoid typos.
- If you see "[output omitted for brevity]" or compacted summary, rely on the most recent observations.
- Test thoroughly before calling finish.

## Critical Rules:
- **Never create new files** (including test files, config files, etc.) - only modify existing files
- **Never modify test files** - fix your code implementation instead
- **Sandbox environment**: No internet access. If you encounter missing dependencies, do not try to install them. Use only what's already available.
- **Backward compatibility**: Code must remain backward compatible unless the problem statement says otherwise.
- **Multi-file fixes**: Search across the codebase and apply consistent changes to all relevant files before re-running tests.
- **Handle specific cases first**: When fixing validation, type checking, or inference logic, handle the specific problematic case before modifying general logic. This ensures the fix addresses the root cause without breaking other cases.
- When making API changes (renaming parameters, changing signatures, modifying behavior), maintain backward compatibility by supporting both old and new usage patterns. Keep deprecated features functional with warnings, add validation to prevent conflicting usage, and ensure existing code continues to work without modification.
- Test that code using the old API still functions correctly. Breaking changes that remove or disable existing functionality will cause test failures and break user code.
- Return a valid git diff patch as final output.


You have access to the following tools:
{tools_docs}

{format_prompt}
""")

# ============================================================================
# NETWORK & INFERENCE
# ============================================================================

def _retry_sleep(attempt: int) -> None:
    """Exponential backoff with jitter."""
    wait = min(
        RETRY_INITIAL_WAIT * (2 ** attempt) + random.uniform(0, RETRY_JITTER),
        RETRY_MAX_WAIT,
    )
    logger.info("Retrying in %.2fs (attempt %d)", wait, attempt + 1)
    time.sleep(wait)


def _is_retryable_request_error(exc: BaseException) -> bool:
    """True if we should retry (connection, timeout, 429, 5xx)."""
    if isinstance(exc, requests.exceptions.Timeout):
        return True
    if isinstance(exc, requests.exceptions.ConnectionError):
        return True
    if isinstance(exc, requests.exceptions.RequestException) and hasattr(exc, "response"):
        resp = getattr(exc, "response", None)
        if resp is not None and getattr(resp, "status_code", None) in RETRYABLE_HTTP_CODES:
            return True
    return False


class Network:
    @classmethod
    def make_request(cls, messages: list, model: str, temperature: float = 0.0) -> Tuple[str, int]:
        """POST chat inference: sandbox proxy (production) or OpenAI-compatible API (local).

        Mirrors submissions/4_29/agent_wonder.py: branch on ``SANDBOX_PROXY_URL`` vs
        ``RIDGES_INFERENCE_API_KEY`` + ``RIDGES_INFERENCE_BASE_URL``.

        Returns (content, status_code). Raises on non-2xx for non-retryable codes.
        """
        timeout = (LLM_CONNECT_TIMEOUT, LLM_READ_TIMEOUT)
        direct = _use_direct_openai_inference()

        if direct:
            api_key = os.environ.get("RIDGES_INFERENCE_API_KEY")
            base_url = os.environ.get("RIDGES_INFERENCE_BASE_URL")
            if not api_key or not base_url:
                raise RuntimeError(
                    "Local inference is not configured. Set SANDBOX_PROXY_URL (production) or "
                    "RIDGES_INFERENCE_API_KEY + RIDGES_INFERENCE_BASE_URL (local)."
                )
            url = f"{base_url.rstrip('/')}/chat/completions"
            resolved = _resolve_model_for_local(model)
            request_data = {
                "model": resolved,
                "messages": messages,
                "temperature": temperature,
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            logger.info("LLM request: direct endpoint, model=%s", resolved)
        else:
            proxy_base = _sandbox_proxy_base()
            url = f"{proxy_base}/api/inference"
            request_data = {
                "evaluation_run_id": os.getenv("EVALUATION_RUN_ID", str(uuid4())),
                "messages": messages,
                "temperature": temperature,
                "model": model,
            }
            headers = {"Content-Type": "application/json"}
            logger.debug("LLM request: sandbox proxy %s, model=%s", proxy_base, model)

        wait = 1.0
        max_wait = 60.0
        last_response: requests.Response | None = None
        for attempt in range(5):
            response = requests.post(url, json=request_data, timeout=timeout, headers=headers)
            last_response = response
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
                continue
            response.raise_for_status()
            response_json = response.json()
            if direct:
                message = (response_json.get("choices") or [{}])[0].get("message") or {}
                content = (message.get("content") or "").strip()
            else:
                content = response_json.get("content") or ""
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
        # Require at least next_tool_args to be present (often ends with } but allow trailing text)
        if "next_tool_args:" not in raw_text:
            return False, "Missing next_tool_args"
        return True, None

    @classmethod
    def parse_response(cls, text_resp: str) -> tuple[str, str, dict, str]:
        """Parse LLM response into thought, tool name, and args."""
        error_msg = None
        text_resp = text_resp.strip().split("observation:")[0].strip()
        
        # Clean up the response
        text_resp = re.sub(r"['\"]*(next_thought)['\"]*([ ]*):", r"next_thought:", text_resp)
        text_resp = re.sub(r"['\"]*(next_tool_name)['\"]*([ ]*):", r"next_tool_name:", text_resp)
        text_resp = re.sub(r"['\"]*(next_tool_args)['\"]*([ ]*):", r"next_tool_args:", text_resp)
        
        # Check if all required fields are present
        if "next_thought:" not in text_resp:
            return None, None, None, "Invalid response: next_thought not found"
        if "next_tool_name:" not in text_resp:
            return None, None, None, "Invalid response: next_tool_name not found"
        if "next_tool_args:" not in text_resp:
            return None, None, None, "Invalid response: next_tool_args not found"
        
        # Extract fields
        try:
            next_thought = text_resp.split("next_thought:")[1].split("next_tool_name:")[0].strip()
            next_tool_name = (
                text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip().strip('"').strip("'")
            )
            next_tool_args_raw = text_resp.split("next_tool_args:")[1].strip()

            # Parse args: allow trailing text after JSON object (robustness)
            next_tool_args_raw = next_tool_args_raw.replace("```json", "").replace("```", "").strip()
            # Extract first complete {...} if there is trailing content
            if next_tool_args_raw.startswith("{"):
                depth = 0
                end = -1
                for idx, c in enumerate(next_tool_args_raw):
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            end = idx
                            break
                if end >= 0:
                    next_tool_args_raw = next_tool_args_raw[: end + 1]
            next_tool_args = json.loads(next_tool_args_raw) if next_tool_args_raw else {}

        except (JSONDecodeError, IndexError) as e:
            return None, None, None, f"Error parsing response: {str(e)}"

        # Validate tool name is allowed
        allowed_tools = {"bash", "read_file", "grep_search", "think", "str_replace_edit", "finish"}
        if next_tool_name not in allowed_tools:
            return None, None, None, f"Invalid tool '{next_tool_name}'. Use one of: {', '.join(sorted(allowed_tools))}"

        return next_thought, next_tool_name, next_tool_args, error_msg

    @classmethod
    def inference(
        cls,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> tuple:
        """Make inference request with retry: exponential backoff, retry only on retryable errors."""
        last_exception = None
        for attempt in range(max_retries):
            try:
                raw_text, _ = cls.make_request(messages, model=model, temperature=temperature)
                is_valid, error_msg = cls.is_valid_response(raw_text)

                if not is_valid:
                    if attempt < max_retries - 1:
                        logger.warning("Invalid response: %s, retrying...", error_msg)
                        _retry_sleep(attempt)
                        continue
                    raise Exception(f"Invalid response: {error_msg}")

                next_thought, next_tool_name, next_tool_args, parse_error = cls.parse_response(raw_text)

                if parse_error:
                    if attempt < max_retries - 1:
                        logger.warning("Parse error: %s, retrying...", parse_error)
                        messages.append({"role": "assistant", "content": raw_text})
                        messages.append({"role": "user", "content": f"observation: {parse_error}"})
                        _retry_sleep(attempt)
                        continue
                    raise Exception(parse_error)

                return next_thought, next_tool_name, next_tool_args, raw_text

            except requests.exceptions.HTTPError as e:
                last_exception = e
                if e.response is not None and e.response.status_code in RETRYABLE_HTTP_CODES:
                    if attempt < max_retries - 1:
                        logger.warning("HTTP %s, retrying...", e.response.status_code)
                        _retry_sleep(attempt)
                        continue
                raise
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning("Request error (timeout/connection): %s, retrying...", e)
                    _retry_sleep(attempt)
                    continue
                raise
            except Exception as e:
                last_exception = e
                if _is_retryable_request_error(e) and attempt < max_retries - 1:
                    logger.warning("Inference error: %s, retrying...", e)
                    _retry_sleep(attempt)
                    continue
                raise

        raise Exception("Max retries exceeded") from last_exception


# ============================================================================
# TOOL RESULT (structured observation + truncation)
# ============================================================================

def _truncate_line(line: str, max_length: int, marker: str = "...") -> str:
    """Truncate a single line; preserve line break at end."""
    if len(line) <= max_length:
        return line
    m = re.search(r"[\r\n]+$", line)
    linebreak = m.group(0) if m else ""
    end = marker + linebreak
    max_length = max(max_length, len(end))
    return line[: max_length - len(end)] + end


def truncate_tool_output(
    text: str,
    max_chars: int = TOOL_OUTPUT_MAX_CHARS,
    max_line_length: int | None = TOOL_OUTPUT_MAX_LINE_LENGTH,
    marker: str = "[...truncated]",
) -> str:
    """Limit tool output size for context."""
    if not text or len(text) <= max_chars and (max_line_length is None or not any(len(line) > max_line_length for line in text.splitlines())):
        return text
    lines = text.splitlines(keepends=True)
    out: List[str] = []
    n_chars = 0
    for line in lines:
        if n_chars >= max_chars:
            out.append(marker + "\n")
            break
        original = line
        if max_line_length is not None and len(line) > max_line_length:
            line = _truncate_line(line, max_line_length, marker)
        if n_chars + len(line) > max_chars:
            line = line[: max_chars - n_chars - len(marker) - 1] + marker + "\n"
            out.append(line)
            break
        out.append(line)
        n_chars += len(line)
    return "".join(out)


class ToolResult:
    """Structured tool result for the agent loop."""

    __slots__ = ("observation", "is_error")

    def __init__(self, observation: str, is_error: bool = False):
        self.observation = observation
        self.is_error = is_error


class ToolResultBuilder:
    """Build tool output with optional truncation, then produce ToolResult."""

    def __init__(
        self,
        max_chars: int = TOOL_OUTPUT_MAX_CHARS,
        max_line_length: int | None = TOOL_OUTPUT_MAX_LINE_LENGTH,
    ):
        self.max_chars = max_chars
        self.max_line_length = max_line_length
        self._buffer: List[str] = []
        self._n_chars = 0
        self._truncated = False
        self._marker = "[...truncated]"

    def write(self, text: str) -> None:
        """Append text; respect max_chars and max_line_length."""
        if self._n_chars >= self.max_chars:
            return
        for line in text.splitlines(keepends=True):
            if self._n_chars >= self.max_chars:
                self._buffer.append(self._marker + "\n")
                self._truncated = True
                return
            orig = line
            if self.max_line_length is not None and len(line) > self.max_line_length:
                line = _truncate_line(line, self.max_line_length, self._marker)
                self._truncated = True
            if self._n_chars + len(line) > self.max_chars:
                line = line[: self.max_chars - self._n_chars - len(self._marker) - 1] + self._marker + "\n"
                self._truncated = True
            self._buffer.append(line)
            self._n_chars += len(line)
            if self._n_chars >= self.max_chars:
                return

    def ok(self, message: str = "", *, output: str | None = None) -> ToolResult:
        """Build success result. If output is given, it is written first then message."""
        if output:
            self.write(output)
        text = "".join(self._buffer)
        if message:
            msg = message if message.endswith(".") else message + "."
            if self._truncated:
                msg += " Output is truncated to fit in the message."
            text = msg + ("\n\n" + text if text else "")
        elif not text:
            text = "Done."
        return ToolResult(observation=text, is_error=False)

    def error(self, message: str, *, output: str | None = None) -> ToolResult:
        """Build error result (ERROR: prefix so model sees failure)."""
        if output:
            self.write(output)
        text = "".join(self._buffer)
        msg = message if message.endswith(".") else message + "."
        if self._truncated:
            msg += " Output is truncated to fit in the message."
        full = "ERROR: " + msg + ("\n\n" + text if text else "")
        return ToolResult(observation=full, is_error=True)


# ============================================================================
# TOOL MANAGER
# ============================================================================

class ToolManager:
    """Unified tool manager: schema generation, tool dispatch, and run_tool with truncation."""
    
    def __init__(self):
        self.TOOL_LIST = {}
        self.checkpoint = ""
        self._register_tools()
    
    def _register_tools(self):
        """Register all available tools."""
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, '_is_tool'):
                self.TOOL_LIST[name] = self._create_tool_schema(method)
    
    def _create_tool_schema(self, method) -> dict:
        """Create JSON schema for a tool."""
        doc = inspect.getdoc(method) or ""
        sig = inspect.signature(method)
        
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            # Extract parameter description from docstring
            pattern = f"{param_name}:([^\\n]+)"
            match = re.search(pattern, doc)
            param_desc = match.group(1).strip() if match else f"Parameter {param_name}"
            
            # Determine if required
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
            
            # Get type
            annotation = param.annotation
            if annotation != inspect.Parameter.empty:
                type_str = str(annotation)
                if 'str' in type_str:
                    param_type = "string"
                elif 'int' in type_str:
                    param_type = "integer"
                elif 'bool' in type_str:
                    param_type = "boolean"
                else:
                    param_type = "string"
            else:
                param_type = "string"
            
            properties[param_name] = {
                "type": param_type,
                "description": param_desc
            }
        
        # Extract main description (before Arguments:)
        description = doc.split("Arguments:")[0].strip()
        
        return {
            "name": method.__name__,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    
    def get_tool_docs(self) -> str:
        """Get documentation for all tools."""
        return '\n\n'.join([json.dumps(tool, ensure_ascii=False) for tool in self.TOOL_LIST.values()])
    
    def get_tool(self, tool_name: str):
        """Get a tool by name."""
        if tool_name not in self.TOOL_LIST:
            raise Exception(f"Tool '{tool_name}' not found. Available tools: {', '.join(self.TOOL_LIST.keys())}")
        return getattr(self, tool_name)

    def run_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool by name with given args.
        Validates args, catches exceptions, truncates output, and returns structured ToolResult.
        """
        if tool_name not in self.TOOL_LIST:
            return ToolResult(
                observation=f"ERROR: Tool '{tool_name}' not found. Available: {', '.join(sorted(self.TOOL_LIST.keys()))}.",
                is_error=True,
            )
        schema = self.TOOL_LIST[tool_name]
        required = schema.get("input_schema", {}).get("required", [])
        missing = [k for k in required if k not in tool_args or tool_args[k] is None]
        if missing:
            return ToolResult(
                observation=f"ERROR: Missing required argument(s): {', '.join(missing)}.",
                is_error=True,
            )
        try:
            tool = self.get_tool(tool_name)
            raw = tool(**tool_args) if tool_args else tool()
        except Exception as e:
            logger.exception("Tool %s failed", tool_name)
            return ToolResult(observation=f"ERROR: {str(e)}.", is_error=True)
        if isinstance(raw, ToolResult):
            obs = truncate_tool_output(raw.observation)
            return ToolResult(observation=obs, is_error=raw.is_error)
        if isinstance(raw, str):
            is_error = raw.strip().upper().startswith("ERROR:")
            obs = truncate_tool_output(raw)
            return ToolResult(observation=obs, is_error=is_error)
        return ToolResult(observation=str(raw), is_error=False)

    # ========================================================================
    # TOOL DEFINITIONS (clear success/error, optional timeout)
    # ========================================================================

    def bash(self, command: str, timeout: int = BASH_TIMEOUT_DEFAULT) -> ToolResult:
        """
        Execute bash commands in the terminal. Use this to explore the codebase, run tests, or execute any shell commands.
        Arguments:
            command: the bash command to execute (e.g., "ls -la", "cat file.txt", "node test.js", "python test.py")
            timeout: timeout in seconds (default 60, max 300). Commands that run longer will be killed.
        """
        builder = ToolResultBuilder()
        if not (command or "").strip():
            return builder.error("Command cannot be empty.", output="")
        timeout = max(1, min(int(timeout), BASH_TIMEOUT_MAX))
        try:
            result = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd(),
            )
            if result.stdout:
                builder.write("STDOUT:\n")
                builder.write(result.stdout)
                if not result.stdout.endswith("\n"):
                    builder.write("\n")
            if result.stderr:
                builder.write("STDERR:\n")
                builder.write(result.stderr)
                if not result.stderr.endswith("\n"):
                    builder.write("\n")
            if result.returncode != 0:
                builder.write(f"Command exited with code {result.returncode}\n")
                return builder.error(f"Command failed with exit code {result.returncode}.", output="")
            return builder.ok("Command executed successfully.")
        except subprocess.TimeoutExpired:
            return builder.error(f"Command killed by timeout ({timeout}s).", output="")
        except Exception as e:
            return builder.error(f"Error executing command: {e}.", output="")

    bash._is_tool = True

    def read_file(self, file_path: str, start_line: int = 1, end_line: int = 200) -> ToolResult:
        """
        Read a file with line numbers. Use this instead of shell cat for deterministic context windows.
        Arguments:
            file_path: path to file to read
            start_line: starting line number (1-indexed)
            end_line: ending line number (inclusive)
        """
        if not os.path.exists(file_path):
            return ToolResult(observation=f"ERROR: File '{file_path}' does not exist.", is_error=True)
        if start_line < 1:
            start_line = 1
        if end_line < start_line:
            end_line = start_line
        end_line = min(end_line, start_line + 2000)
        builder = ToolResultBuilder(max_chars=READ_FILE_MAX_CHARS)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            total = len(lines)
            if total == 0:
                return builder.ok(f"Read {file_path} (empty file).")
            if start_line > total:
                return builder.error(
                    f"Start line {start_line} exceeds total lines {total} in '{file_path}'."
                )
            slice_lines = lines[start_line - 1 : min(end_line, total)]
            for idx, line in enumerate(slice_lines, start=start_line):
                builder.write(f"{idx:6d}|{line}")
            return builder.ok(
                f"Read '{file_path}' lines {start_line}-{min(end_line, total)} of {total}.",
                output=""
            )
        except Exception as e:
            return builder.error(f"Failed to read file: {e}.")

    read_file._is_tool = True

    def grep_search(self, pattern: str, root_path: str = ".", glob: str = "*", max_matches: int = GREP_MAX_MATCHES) -> ToolResult:
        """
        Search text in files recursively. Prefer this over shell grep for bounded results.
        Arguments:
            pattern: regex pattern to search for
            root_path: root directory to search
            glob: file glob filter, e.g. *.py or *.{py,md}
            max_matches: max number of matches to return
        """
        if not pattern:
            return ToolResult(observation="ERROR: pattern cannot be empty.", is_error=True)
        if not os.path.exists(root_path):
            return ToolResult(observation=f"ERROR: root_path '{root_path}' does not exist.", is_error=True)
        builder = ToolResultBuilder(max_chars=TOOL_OUTPUT_MAX_CHARS)
        max_matches = max(1, min(int(max_matches), 1000))
        matched = 0
        scanned_files = 0
        flags = re.MULTILINE
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return builder.error(f"Invalid regex pattern: {e}.")
        for dirpath, dirnames, filenames in os.walk(root_path):
            dirnames[:] = [d for d in dirnames if d != ".git"]
            for filename in filenames:
                if not fnmatch.fnmatch(filename, glob):
                    continue
                path = os.path.join(dirpath, filename)
                scanned_files += 1
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f, start=1):
                            if regex.search(line):
                                builder.write(f"{path}:{i}: {line}")
                                matched += 1
                                if matched >= max_matches:
                                    return builder.ok(
                                        f"Search completed early: {matched} matches in {scanned_files} files.",
                                        output=""
                                    )
                except Exception:
                    continue
        if matched == 0:
            return builder.ok(f"No matches for pattern in {scanned_files} files.")
        return builder.ok(f"Search completed: {matched} matches in {scanned_files} files.", output="")

    grep_search._is_tool = True

    def think(self, thought: str) -> ToolResult:
        """
        Use this tool to organize your thoughts, plan your approach, or reason about the problem.
        Arguments:
            thought: your detailed thought process, analysis, or plan
        """
        logger.info("[THINK] %s", thought[:200] + "..." if len(thought) > 200 else thought)
        return ToolResult(observation="Thought logged. Continue with your implementation.", is_error=False)

    think._is_tool = True

    def str_replace_edit(self, file_path: str, old_str: str, new_str: str) -> ToolResult:
        """
        Replace text in a file. The old_str must match exactly and appear only once in the file.
        Arguments:
            file_path: path to the file to edit
            old_str: exact text to find and replace (must be unique in the file)
            new_str: text to replace with
        """
        if not os.path.exists(file_path):
            return ToolResult(observation=f"ERROR: File '{file_path}' does not exist.", is_error=True)
        if old_str == new_str:
            return ToolResult(observation="ERROR: old_str and new_str are identical. No change needed.", is_error=True)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            count = content.count(old_str)
            if count == 0:
                return ToolResult(
                    observation=f"ERROR: old_str not found in '{file_path}'. Provide the exact text to replace.",
                    is_error=True,
                )
            if count > 1:
                return ToolResult(
                    observation=f"ERROR: old_str found {count} times in '{file_path}'. Use a string that appears only once.",
                    is_error=True,
                )
            new_content = content.replace(old_str, new_str, 1)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            return ToolResult(observation=f"Successfully replaced text in '{file_path}'.", is_error=False)
        except Exception as e:
            return ToolResult(observation=f"ERROR: {str(e)}.", is_error=True)

    str_replace_edit._is_tool = True

    def finish(self, summary: str = "") -> ToolResult:
        """
        Signal that you have completed the task. Call this only when you're confident the work is done and tested.
        Arguments:
            summary: optional summary of what was accomplished
        """
        self.checkpoint = self.get_final_git_patch()
        return ToolResult(observation="finish", is_error=False)

    finish._is_tool = True
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def get_final_git_patch(self) -> str:
        """Generate git patch of all changes."""
        try:
            # Stage modified/untracked files
            subprocess.run(
                ["git", "ls-files", "-m", "-o", "--exclude-standard"],
                capture_output=True, text=True, timeout=30, check=True
            )
            
            # Add all modified and new files
            subprocess.run(
                ["git", "add", "-A"],
                capture_output=False, text=True, timeout=30, check=False
            )
            
            # Generate diff
            diff = subprocess.run(
                ["git", "diff", "--cached", "--no-color"],
                capture_output=True, text=True, timeout=30, check=True
            )
            
            return diff.stdout or ""
        except Exception as e:
            logger.exception("Error generating git patch")
            return f"Error generating git patch: {e}"

# ============================================================================
# CHAIN OF THOUGHT
# ============================================================================

class ChainOfThought:
    """Manages the conversation history and observations."""

    class Action:
        def __init__(self, thought: str, tool_name: str, tool_args: dict, observation: str, is_error: bool = False):
            self.thought = thought
            self.tool_name = tool_name
            self.tool_args = tool_args
            self.observation = observation
            self.is_error = is_error

    def __init__(self, max_recent_observations: int = 5, compaction_threshold: int = CONTEXT_COMPACTION_THRESHOLD):
        self.actions: List[ChainOfThought.Action] = []
        self.max_recent_observations = max_recent_observations
        self.compaction_threshold = compaction_threshold

    def add_action(self, action: Action) -> None:
        """Add an action to the history."""
        self.actions.append(action)

    def last_action_signature(self) -> str:
        """Stable signature of the latest action for repeat detection."""
        if not self.actions:
            return ""
        a = self.actions[-1]
        return f"{a.tool_name}|{json.dumps(a.tool_args, sort_keys=True)}"

    def _summarize_actions(self, actions: List[Action]) -> str:
        """Summarize a list of actions into a compact line."""
        lines = []
        for i, a in enumerate(actions, 1):
            obs_preview = (a.observation or "").strip()
            if len(obs_preview) > 120:
                obs_preview = obs_preview[:117] + "..."
            if a.is_error:
                lines.append(f"{i}) {a.tool_name}({json.dumps(a.tool_args)}): Error — {obs_preview}")
            else:
                lines.append(f"{i}) {a.tool_name}({json.dumps(a.tool_args)}): {obs_preview}")
        return "\n".join(lines)

    def to_messages(self) -> List[Dict[str, Any]]:
        """Convert to message format for LLM. Compacts old turns when over threshold."""
        messages = []
        n = len(self.actions)
        if n == 0:
            return messages

        # compaction: when we have many actions, summarize oldest into one user message
        if n > self.compaction_threshold:
            keep_full = self.max_recent_observations
            to_compact = self.actions[: n - keep_full]
            recent = self.actions[n - keep_full :]
            summary = self._summarize_actions(to_compact)
            summary_user = (
                "observation: [Previous steps compacted for context limit. Focus on recent steps below.]\n"
                "Summary of earlier steps:\n" + summary
            )
            messages.append({"role": "assistant", "content": "[earlier turn omitted]"})
            messages.append({"role": "user", "content": summary_user})
            actions_to_emit = recent
        else:
            actions_to_emit = self.actions

        for i, action in enumerate(actions_to_emit):
            assistant_msg = (
                f"next_thought: {action.thought}\n"
                f"next_tool_name: {action.tool_name}\n"
                f"next_tool_args: {json.dumps(action.tool_args)}"
            )
            omit = actions_to_emit is self.actions and i < len(actions_to_emit) - self.max_recent_observations
            user_msg = "observation: [output omitted for brevity]" if omit else f"observation: {action.observation}"
            messages.append({"role": "assistant", "content": assistant_msg})
            messages.append({"role": "user", "content": user_msg})

        return messages

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def _initialize_workflow_runtime(
    problem_statement: str,
    model: str,
    timeout: int,
    max_steps: int,
) -> Dict[str, Any]:
    tool_manager = ToolManager()
    cot = ChainOfThought()
    system_prompt = SYSTEM_PROMPT.format(
        tools_docs=tool_manager.get_tool_docs(),
        format_prompt=FORMAT_PROMPT,
    )
    return {
        "problem_statement": problem_statement,
        "model": model,
        "timeout": timeout,
        "max_steps": max_steps,
        "tool_manager": tool_manager,
        "cot": cot,
        "system_prompt": system_prompt,
        "start_time": time.time(),
        "abort_event": threading.Event(),
        "watchdog_timer": None,
        "model_index": 0,
        "inference_failures": 0,
        "repeat_actions": 0,
        "last_action_sig": "",
    }


def _install_watchdog(runtime: Dict[str, Any]) -> None:
    """Set abort event before timeout so loop can return partial patch."""
    timeout = runtime["timeout"]
    if timeout <= 0:
        return
    fire_at = max(1.0, timeout * WATCHDOG_ABORT_FRACTION)
    event = runtime["abort_event"]

    def _fire() -> None:
        logger.warning("Watchdog fired at %.0fs. Forcing graceful finish.", fire_at)
        event.set()

    timer = threading.Timer(fire_at, _fire)
    timer.daemon = True
    timer.start()
    runtime["watchdog_timer"] = timer


def _cancel_watchdog(runtime: Dict[str, Any]) -> None:
    timer = runtime.get("watchdog_timer")
    if timer is not None:
        try:
            timer.cancel()
        except Exception:
            pass


def _has_uncommitted_diff() -> bool:
    """True when working tree has changes worth patching."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.stdout.strip():
            return True
        staged = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return bool(staged.stdout.strip())
    except Exception:
        return False


def _build_workflow_messages(runtime: Dict[str, Any], elapsed: float) -> List[Dict[str, Any]]:
    messages = [
        {"role": "system", "content": runtime["system_prompt"]},
        {"role": "user", "content": f"# Problem Statement\n\n{runtime['problem_statement']}"},
    ]
    messages.extend(runtime["cot"].to_messages())
    messages.append({"role": "system", "content": STOP_INSTRUCTION})

    if len(runtime["cot"].actions) >= NO_EDIT_WARNING_STEP and not _has_uncommitted_diff():
        messages.append({
            "role": "user",
            "content": (
                "⚠️ You have not made any code edits yet. "
                "Make a concrete file change this turn using str_replace_edit."
            ),
        })

    if elapsed > runtime["timeout"] * 0.8:
        messages.append({
            "role": "user",
            "content": (
                f"⚠️ WARNING: Running low on time! "
                f"Only {runtime['timeout'] - elapsed:.0f}s remaining. Finish soon!"
            ),
        })
    return messages


def _request_next_action(
    runtime: Dict[str, Any],
    messages: List[Dict[str, Any]],
) -> Tuple[str, str, Dict[str, Any]] | None:
    model = AGENT_MODELS[runtime["model_index"] % len(AGENT_MODELS)]
    temperature = 0.1 if runtime["repeat_actions"] < REPEAT_ACTION_THRESHOLD else 0.4
    max_retries = 1 if (time.time() - runtime["start_time"]) > runtime["timeout"] * 0.7 else 3
    try:
        next_thought, next_tool_name, next_tool_args, _ = Network.inference(
            messages,
            model=model,
            temperature=temperature,
            max_retries=max_retries,
        )
        runtime["inference_failures"] = 0
        logger.info("Model: %s, Tool: %s, Args: %s", model, next_tool_name, next_tool_args)
        return next_thought, next_tool_name, next_tool_args or {}
    except Exception as e:
        runtime["inference_failures"] += 1
        if runtime["inference_failures"] >= 2:
            runtime["model_index"] = (runtime["model_index"] + 1) % len(AGENT_MODELS)
            runtime["inference_failures"] = 0
            logger.warning("Switching model to %s after repeated inference errors", AGENT_MODELS[runtime["model_index"]])
        logger.error("Inference error: %s", e)
        runtime["cot"].add_action(ChainOfThought.Action(
            thought="Inference failed",
            tool_name="",
            tool_args={},
            observation=f"Error: {str(e)}",
            is_error=True,
        ))
        return None


def _execute_agent_action(
    runtime: Dict[str, Any],
    thought: str,
    tool_name: str,
    tool_args: Dict[str, Any],
) -> bool:
    tool_result = runtime["tool_manager"].run_tool(tool_name, tool_args)
    observation = tool_result.observation
    logger.info("Observation: %s...", observation[:200] if len(observation) > 200 else observation)

    runtime["cot"].add_action(ChainOfThought.Action(
        thought=thought,
        tool_name=tool_name,
        tool_args=tool_args,
        observation=observation,
        is_error=tool_result.is_error,
    ))
    return tool_name == "finish"


def execute_workflow(
    problem_statement: str,
    model: str = MINIMAX_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
    max_steps: int = MAX_STEPS
) -> str:
    """
    Execute the agent workflow.
    
    Args:
        problem_statement: The task description
        model: Model to use for inference
        timeout: Maximum time in seconds
        max_steps: Maximum number of steps
    
    Returns:
        Git patch with the changes
    """

    runtime = _initialize_workflow_runtime(
        problem_statement=problem_statement,
        model=model,
        timeout=timeout,
        max_steps=max_steps,
    )
    _install_watchdog(runtime)

    try:
        for step in range(runtime["max_steps"]):
            elapsed = time.time() - runtime["start_time"]
            logger.info("Step %d/%d, Elapsed: %.1fs", step + 1, runtime["max_steps"], elapsed)

            if runtime["abort_event"].is_set():
                logger.warning("Watchdog abort requested")
                break
            if elapsed > runtime["timeout"]:
                logger.warning("Timeout reached")
                break

            messages = _build_workflow_messages(runtime, elapsed)
            next_action = _request_next_action(runtime, messages)
            if next_action is None:
                continue

            next_thought, next_tool_name, next_tool_args = next_action
            current_sig = f"{next_tool_name}|{json.dumps(next_tool_args or {}, sort_keys=True)}"
            if current_sig == runtime["last_action_sig"]:
                runtime["repeat_actions"] += 1
            else:
                runtime["repeat_actions"] = 0
            runtime["last_action_sig"] = current_sig

            if _execute_agent_action(runtime, next_thought, next_tool_name, next_tool_args):
                logger.info("Workflow finished successfully")
                if not runtime["tool_manager"].checkpoint:
                    runtime["tool_manager"].checkpoint = runtime["tool_manager"].get_final_git_patch()
                return runtime["tool_manager"].checkpoint
    finally:
        _cancel_watchdog(runtime)

    logger.warning("Workflow ended without explicit finish")
    if not runtime["tool_manager"].checkpoint:
        runtime["tool_manager"].checkpoint = runtime["tool_manager"].get_final_git_patch()
    return runtime["tool_manager"].checkpoint

# ============================================================================
# ENTRY POINT
# ============================================================================

def ensure_git_initialized():
    """Initialize git repository if needed."""
    try:
        if not os.path.exists(".git"):
            logger.info("Initializing git repository...")
            subprocess.run(["git", "init"], check=True)
            subprocess.run(["git", "config", "user.email", "agent@sandbox.local"], check=True)
            subprocess.run(["git", "config", "user.name", "sandbox_agent"], check=True)
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", "Initial commit"], check=False)
            logger.info("Git initialization completed")
    except Exception as e:
        logger.error(f"Error initializing git: {e}")


def _prepare_repository(repo_dir: str) -> None:
    """Switch to repository directory and ensure git is ready."""
    absolute_repo_dir = os.path.abspath(repo_dir)
    if os.path.exists(absolute_repo_dir):
        os.chdir(absolute_repo_dir)
        sys.path.insert(0, absolute_repo_dir)
    ensure_git_initialized()


def _cleanup_repository() -> None:
    """Return repository to clean state for next run."""
    subprocess.run(["git", "reset", "--hard"], check=False)


def _run_workflow_with_defaults(problem_statement: str) -> str:
    return execute_workflow(
        problem_statement=problem_statement,
        timeout=DEFAULT_TIMEOUT - 50,
    )


def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo") -> Dict[str, str]:
    """
    Main entry point for the agent.
    
    Args:
        input_dict: Dictionary with 'problem_statement' and optional 'task_type'
        repo_dir: Directory to work in
    
    Returns:
        Dict with "patch" key containing git diff
    """
    problem_statement = input_dict.get("problem_statement", "")

    if not problem_statement:
        raise ValueError("problem_statement is required")

    _prepare_repository(repo_dir)

    try:
        patch = _run_workflow_with_defaults(problem_statement)
    except Exception as e:
        logger.exception("Workflow failed")
        patch = ""
    finally:
        _cleanup_repository()
    logger.info(f"Patch: {patch}")
    return {"patch": patch or ""}
