
from __future__ import annotations
import os
import re
import sys
import json
import time
import random
import difflib
import inspect
import logging
import tempfile
import requests
import textwrap
import traceback
import threading
import subprocess
from pathlib import Path
from json import JSONDecodeError
from typing import Any, Dict, List, Optional
from enum import Enum
from uuid import uuid4
from pydantic import BaseModel
from dataclasses import dataclass, field

try:
    from tree_sitter import Parser
    from tree_sitter_language_pack import get_language
except ImportError:
    Parser = None
    get_language = None


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for _h in list(logger.handlers):
    logger.removeHandler(_h)

_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setLevel(logging.DEBUG)
_stream_handler.setFormatter(_formatter)
logger.addHandler(_stream_handler)

# ── Global state ──────────────────────────────────────────────────────────────
run_id = None
agent_start_time = None
_current_tool_manager = None
total_inferenced_chars = 0
individual_inferenced_chars = 0

# ── Watchdog (SN66-port): fires at 92% of AGENT_TIMEOUT, commits partial diff ─
_watchdog_abort = threading.Event()  # set → all LLM calls must bail out
_watchdog_timer: threading.Timer | None = None

HARD_ABORT_FRACTION = 0.92  # 92% of budget


def _watchdog_fire() -> None:
    """Called by timer at 92% of AGENT_TIMEOUT. Sets abort flag."""
    global _current_tool_manager
    logger.warning(
        f"⚠️ [WATCHDOG] HARD_ABORT fired at {HARD_ABORT_FRACTION*100:.0f}% of {AGENT_TIMEOUT}s budget — "
        "cancelling in-flight LLM calls and committing partial diff"
    )
    _watchdog_abort.set()
    # Best-effort: flush partial patch right now so caller can return it
    if _current_tool_manager is not None:
        try:
            patch = _current_tool_manager.get_final_git_patch()
            if patch and not patch.startswith("Error"):
                logger.info(f"[WATCHDOG] Partial diff captured ({len(patch)} chars)")
        except Exception:
            pass


def install_watchdog(timeout_secs: int) -> None:
    """Install the watchdog timer. Call once after agent_start_time is set."""
    global _watchdog_timer
    if timeout_secs <= 0:
        return
    fire_at = timeout_secs * HARD_ABORT_FRACTION
    _watchdog_timer = threading.Timer(fire_at, _watchdog_fire)
    _watchdog_timer.daemon = True
    _watchdog_timer.start()
    logger.info(f"🐕 [WATCHDOG] Installed — fires at {fire_at:.0f}s ({HARD_ABORT_FRACTION*100:.0f}% of {timeout_secs}s)")

class TimeBudget:
    """Distribute wall-clock budget across agent stages to prevent silent zero-scores."""
    def __init__(self, total_secs: int = 1400):
        self.start = time.time()
        self.total = total_secs
    def remaining(self) -> float:
        return self.total - (time.time() - self.start)
    def stage_timeout(self, fraction: float = 0.3) -> int:
        return max(15, int(self.remaining() * fraction))
    def is_expired(self) -> bool:
        return self.remaining() < 45
    def __repr__(self) -> str:
        return f"TimeBudget({self.remaining():.0f}s remaining)"

class InvestigationMemory:
    """Structured scratchpad injected into every LLM turn to prevent re-investigation."""
    def __init__(self):
        self.confirmed_facts = []
        self.rejected_theories = []
        self.files_examined = []
        self.call_sites_found = []

    def to_prompt_block(self):
        if not any([self.confirmed_facts, self.rejected_theories, self.files_examined]):
            return ""
        parts = ["<investigation_state>"]
        if self.files_examined:
            parts.append(f"Files already read: {', '.join(self.files_examined[-10:])}")
        if self.confirmed_facts:
            parts.append(f"Confirmed facts: {'; '.join(self.confirmed_facts[-5:])}")
        if self.rejected_theories:
            parts.append(f"Rejected theories: {'; '.join(self.rejected_theories[-3:])}")
        parts.append("</investigation_state>")
        return "\n".join(parts)


PROBLEM_TYPE_CREATE = "CREATE"
PROBLEM_TYPE_FIX = "FIX"

class Model(BaseModel):
    name: str
    timeout: int  # per-model request timeout in seconds

QWEN_MODEL_NAME   = Model(name="Qwen/Qwen3-Coder-Next",     timeout=100)
KIMI_MODEL_NAME   = Model(name="moonshotai/Kimi-K2.5",       timeout=60)
GLM47_MODEL_NAME  = Model(name="zai-org/GLM-4.7",            timeout=120)
MINIMAX_MODEL_NAME= Model(name="MiniMaxAI/MiniMax-M2.5",     timeout=80)

AGENT_MODELS: List[Model] = [KIMI_MODEL_NAME, QWEN_MODEL_NAME, GLM47_MODEL_NAME, MINIMAX_MODEL_NAME]

DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
AGENT_TIMEOUT     = int(os.getenv("AGENT_TIMEOUT", "2100"))  # total budget
INNER_TIMEOUT     = AGENT_TIMEOUT - 100                       # leave 100s buffer
EVALUATION_RUN_ID = os.getenv("EVALUATION_RUN_ID", "")
LOCAL_CHUTES_KEY = (
    os.getenv("CHUTES_ACCESS_TOKEN") or
    os.getenv("RIDGES_CHUTES_API_KEY") or
    os.getenv("RIDGES_INFERENCE_API_KEY") or ""  # Ridges chutes provider passes this
)
LOCAL_CHUTES_URL = (
    (os.getenv("RIDGES_INFERENCE_BASE_URL") or "https://llm.chutes.ai/v1").rstrip("/") + "/chat/completions"
)
_USE_LOCAL_CHUTES: bool = not bool(os.getenv("SANDBOX_PROXY_URL", "").strip())

MAX_FIX_TASK_STEPS                   = 200
LATEST_OBSERVATIONS_TO_KEEP          = 15
MAX_SUMMARY_RANGES                   = 6
SUMMARIZE_BATCH_SIZE                 = 5
REJECT_OBSERVATION_TOKEN_THRESHOLD   = 50_000
SAVE_OBSERVATION_TO_FILE_TOKEN_THRESHOLD = 5_000

DO_NOT_REPEAT_TOOL_CALLS = textwrap.dedent(
    """
You are making same mistakes.
Your previous response:
{previous_response}

**Critical**:
1. Notice what you are going to do.
2. Find the reason the same mistake is repeated.
3. Don't make the same mistakes any more and make a real progress.
"""
)

PROBLEM_DECOMPOSITION_PROMPT = textwrap.dedent("""
You are an expert software debugging analyst. Analyze the bug report and extract structured information.

Extract the following from the problem statement:

1. **Problem Summary**: Brief description of the issue type in your own words

2. **Key Entities**: Extract identifiers mentioned (file paths, function names, class names, error messages, etc.)

3. **Behavior**:
   - Expected: What should happen
   - Actual: What actually happens
   - Trigger: Conditions that cause the issue

4. **Success Criteria**: What would indicate a successful fix

5. **Investigation Starting Points**: 3-5 specific places to start looking (files, search terms, code areas)

6. **Initial Hypotheses**: 2-4 plausible root cause theories with:
   - Specific description
   - Likelihood score (0.0-1.0)
   - What would confirm or reject it

Respond in JSON:
```json
{
    "problem_summary": "brief description",
    "key_entities": {
        "files": [],
        "functions": [],
        "classes": [],
        "error_messages": [],
        "other": []
    },
    "behavior": {
        "expected": "",
        "actual": "",
        "trigger": ""
    },
    "success_criteria": [],
    "investigation_starting_points": [
        {"location": "", "reason": ""}
    ],
    "initial_hypotheses": [
        {
            "description": "",
            "likelihood": 0.5,
            "confirming_evidence": "",
            "rejecting_evidence": ""
        }
    ]
}
```
""")

VERSION_COMPATIBILITY_FIX = """
import sys, pytest, collections, collections.abc, urllib3.exceptions, _pytest.pytester, numpy;
collections.Mapping = collections.abc.Mapping;
collections.MutableMapping = collections.abc.MutableMapping;
collections.MutableSet = collections.abc.MutableSet;
collections.Sequence = collections.abc.Sequence;
collections.Callable = collections.abc.Callable;
collections.Iterable = collections.abc.Iterable;
collections.Iterator = collections.abc.Iterator;
urllib3.exceptions.SNIMissingWarning = urllib3.exceptions.DependencyWarning;
pytest.RemovedInPytest4Warning = DeprecationWarning;
_pytest.pytester.Testdir = _pytest.pytester.Pytester;
numpy.PINF = numpy.inf;
numpy.unicode_ = numpy.str_;
numpy.bytes_ = numpy.bytes_;
numpy.float_ = numpy.float64;
numpy.string_ = numpy.bytes_;
numpy.NaN = numpy.nan;
"""

STOP_INSTRUCTION = textwrap.dedent(
    """
    # 🎯 RESPONSE REQUIREMENTS
    - DO NOT generate `observation:` - it will be provided by the system
    - You can make MULTIPLE tool calls in one response using tool_call_1, tool_call_2, tool_call_3, etc.
    - For efficiency: Batch related operations together (e.g., edit + test in ONE response)
    - Format: next_thought: ... followed by one or more tool_call_N blocks
    """
)

FORMAT_PROMPT_FIX = textwrap.dedent(
    """
    **CRITICAL: You can make MULTIPLE tool calls in ONE response for efficiency!**
    ## Response Formats
    ### Format 1: Multiple Tool Calls (RECOMMENDED for efficiency)
    next_thought: [Your detailed reasoning]
    tool_call_1:
        tool_name: [exact tool name]
        tool_args: {valid JSON}
    tool_call_2:
        tool_name: [exact tool name]
        tool_args: {valid JSON}
    ### Format 2: Single Tool Call (Legacy, less efficient)
    next_thought: [Your detailed reasoning]
    next_tool_name: [exact tool name]
    next_tool_args: {valid JSON}
    ## Critical Rules
    - Use multiple tool_call_N when possible (tool_call_1, tool_call_2, tool_call_3, ...)
    - After any edit: MUST include test in same response
    - All JSON must be properly formatted with quotes
    - Tool names must match exactly (case-sensitive)
"""
)

FIX_TASK_SYSTEM_PROMPT = textwrap.dedent(
    """
Role: You are a senior bug-fix engineer working on an open-source repository.

You will be tasked to fix an issue from this repository.

Your thinking should be thorough and so it's fine if it's very long. You should think step by step before and after each action you decide to take.

You already have everything you need to solve this problem in the repository, even without internet connection.

Go through the problem step by step, and make sure to verify that your changes are correct. NEVER GIVE UP without having solved the problem, and when you say you are going to make a tool call, make sure you ACTUALLY make the tool call, instead of ending your turn.

THE PROBLEM CAN DEFINITELY BE SOLVED WITHOUT THE INTERNET.

Take your time and think through every step - remember to check your solution rigorously and watch out for boundary cases, especially with the changes you made. Your solution must be perfect. If not, continue working on it. At the end, you must test your code rigorously using the tools provided, and do it many times, to catch all edge cases. If it is not robust, iterate more and make it perfect. Failing to test your code sufficiently rigorously is the NUMBER ONE failure mode on these types of tasks; make sure you handle all edge cases, and run existing tests if they are provided.

You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully.

# Workflow

## High-Level Problem Solving Strategy

1. Understand the problem deeply. Carefully read the issue and think critically about what is required.
2. Investigate the codebase. Explore relevant files, search for key functions, and gather context.
3. Develop a clear, step-by-step plan. Break down the fix into manageable, incremental steps.
4. Implement the fix incrementally. Make small, testable code changes.
5. Debug as needed. Use debugging techniques to isolate and resolve issues.
6. Test frequently. Run tests after each change to verify correctness.
7. Iterate until the root cause is fixed and all tests pass.
8. Reflect and validate comprehensively. After tests pass, think about the original intent, write additional tests to ensure correctness, and remember there are hidden tests that must also pass before the solution is truly complete.

## 1. Deeply Understand the Problem
Carefully read the issue and think hard about a plan to solve it before coding.

## 2. Codebase Investigation
**CRITICAL: Find working examples first, then identify what's broken.**

- Search for key terms from the issue throughout the codebase
- Find similar functionality that WORKS correctly - this is your template
- Study how working code accomplishes what you need
- Locate the broken code using same keywords
- Look beyond surface symptoms - search in domains, helpers, utilities, base classes
- Trace to where mechanisms are actually DEFINED, not just where they're called
- Find the ROOT files where functionality is implemented

**Trace from final output backwards to root cause:**
- Start with working feature's final output, trace backwards to find generator
- Start with broken feature's final output, trace backwards to find what's missing or different
- Compare the paths: where do they diverge?
- Don't stop at the first file you find - keep tracing back to where the behavior originates

## 3. Root Cause Verification
**Before implementing any fix, verify you understand the root cause.**

## 4. Develop a Detailed Plan
Outline a specific, simple, and verifiable sequence of steps.

## 5. Making Code Changes
**Copy patterns from working code. Make minimal focused changes.**

## 6. Debugging
**CRITICAL: Fix root cause, not symptoms. Search broadly across the repository.**

## 7. Testing
- Run tests frequently using the available testing tools.
- If tests fail, analyze failures and revise your patch.

## 8. Final Reflection and Additional Testing
- Reflect carefully on the original intent.
- Think about potential edge cases or scenarios.
- Be aware that there are additional hidden tests that must also pass.

# Tool Documentation
You have access to the following tools:-
{tools_docs}

# Tool Usage Guidelines
- Use appropriate tools to gather context before making changes.
- Use exact values provided by the user (especially in quotes).
- Use `grep_search` to find all occurrences of an issue before fixing.

# Critical Requirements
- Fix must be backward compatible unless stated otherwise.
- Ensure changes are exhaustive and don't break other functionality.
- Don't edit test files directly - use the dedicated test generation tool when needed.
- Don't create new files unless absolutely necessary.

## Battle-Tested Execution Rules

**1. Edit failure recovery:** If `apply_code_edit` fails twice on the same file after re-reading → use `create_new_file(file_path, full_content, overwrite=True)` to replace the entire file. Never attempt a 3rd edit with the same approach.

**2. Sibling propagation:** After each edit, `grep_search` the edited symbol across the repo. If found in callers/importers that need the same change, edit them too.

**3. Criteria completion guard:** Before `finish`, count: (a) distinct requirements in the problem, (b) files successfully edited. If (b) < (a), keep working.

**4. Zero-edit prevention:** After 5+ tool calls with no file edit → stop planning, immediately use `apply_code_edit` on the most likely source file. If apply_code_edit fails, use `create_new_file(file_path, full_content, overwrite=True)` to replace it entirely. An imperfect edit passes some tests. An empty diff passes ZERO tests. Before ANY file edit, call `finish_root_cause_analysis(root_cause, files_to_edit)` to confirm root cause. This gate auto-unlocks at step 5.

**5. Breadth over depth:** When multiple files need changes, one edit per file first. Touching 4/5 files partially outscores perfecting 1/5 completely.

Here is the problem statement:
{problem_statement}

# Response Format Requirements (authoritative)
{format_prompt}
"""
)

FIX_TASK_INSTANCE_PROMPT_TEMPLATE = textwrap.dedent(
    """
Now let's start.

Here is the problem statement:
```
{problem_statement}
```
"""
)

CREATE_TASK_SYSTEM_PROMPT = textwrap.dedent(
    """
Role: You are a senior engineer implementing NEW functionality in an open-source repository.

You will be tasked to CREATE new code (new file, new function, new feature) that satisfies requirements.

Unlike bug fixing, you are writing from scratch. Focus on:
1. Understand the expected interface from tests and imports
2. Write COMPLETE implementations — no `pass`, no stubs, no NotImplementedError
3. Match exact function signatures and return types
4. Handle all edge cases mentioned in the problem

# Tool Documentation
{tools_docs}

# Critical Rules
- Write complete, working code from the start
- Run tests early to validate your interface understanding
- An imperfect but complete implementation scores > an empty diff

{format_prompt}
"""
)

CREATE_TASK_INSTANCE_PROMPT_TEMPLATE = FIX_TASK_INSTANCE_PROMPT_TEMPLATE

_codeparse_util_language_cache: dict = {}


def Exchange_Model(current_model) -> Model:
    """Rotate to a different model, avoiding the current one."""
    # Handle both Model objects and strings
    current_name = current_model.name if isinstance(current_model, Model) else current_model
    other = [m for m in AGENT_MODELS if m.name != current_name]
    if other:
        return random.choice(other)
    return AGENT_MODELS[0]


def _probe_model_health(model: Model, proxy_url: str, eval_run_id: str) -> bool:
    """Send a 1-token ping to check if a model is alive. Returns True if healthy."""
    try:
        url = f"{proxy_url.rstrip('/')}/api/inference"
        payload = {
            "evaluation_run_id": eval_run_id if eval_run_id else str(uuid4()),
            "model": model.name,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": "ping"}],
            "tool_mode": "none",
            "tools": [],
        }
        resp = requests.post(url, json=payload, timeout=(45, 45), headers={"Content-Type": "application/json"})
        return resp.status_code < 500
    except Exception:
        return False


def _build_healthy_model_list(eval_run_id: str) -> List[Model]:
    """Probe all models at startup, return list ordered by health (healthy first)."""
    healthy, dead = [], []
    for model in AGENT_MODELS:
        if _probe_model_health(model, DEFAULT_PROXY_URL, eval_run_id):
            healthy.append(model)
            logger.info(f"✅ Model probe: {model.name} healthy")
        else:
            dead.append(model)
            logger.warning(f"❌ Model probe: {model.name} unreachable — moving to back")
    return healthy + dead  # healthy models first, dead ones as fallback


class FileOperationsUtil:
    def __init__(self, new_files_created: list):
        self.new_files_created = new_files_created
        self.file_system_manager = None
        self.search_manager = None

    def save(self, file_path: str, content: str) -> str:
        with open(file_path, "w") as f:
            f.write(content)
        self.new_files_created.append(file_path)
        return f"File {file_path} saved successfully"

    def get_file_content(
        self,
        file_path: str,
        search_start_line: int = None,
        search_end_line: int = None,
        search_term: str = None,
        limit: int = 1000,
        add_line_numbers: bool = False,
    ) -> str:
        search_callback = lambda fp, st: self.search_manager.search_in_file(fp, st)
        return self.file_system_manager.get_file_content(
            file_path=file_path,
            search_start_line=search_start_line,
            search_end_line=search_end_line,
            search_term=search_term,
            limit=limit,
            add_line_numbers=add_line_numbers,
            search_in_file_callback=search_callback,
        )

    def set_managers(self, file_system_manager, search_manager):
        self.file_system_manager = file_system_manager
        self.search_manager = search_manager


class COT:
    def __init__(self, latest_observations_to_keep=15, summarize_batch_size=5):
        self.thoughts = []
        self.latest_observations_to_keep = latest_observations_to_keep
        self.repeated_thoughts = 0
        self.summarize_batch_size = summarize_batch_size
        self.summaries = {}
        self.summarized_ranges = []

    def _summarize_messages_batch(self, start_idx, end_idx):
        if start_idx >= end_idx or end_idx > len(self.thoughts):
            return None
        conversation_parts = []
        for i in range(start_idx, end_idx):
            thought = self.thoughts[i]
            if getattr(thought, "is_deleted", False):
                continue
            assistant_part = (
                f"next_thought: {thought.next_thought}\n"
                f"next_tool_name: {thought.next_tool_name}\n"
                f"next_tool_args: {thought.next_tool_args}\n"
            )
            obs = thought.observation
            if isinstance(obs, (list, tuple)):
                try:
                    obs_render = json.dumps(list(obs), ensure_ascii=False)
                except Exception:
                    obs_render = str(obs)
            else:
                obs_render = str(obs) if obs else ""
            if len(obs_render) > 40000:
                obs_render = obs_render[:40000] + "... [truncated for summarization]"
            user_part = f"observation: {obs_render}"
            conversation_parts.append(
                {
                    "assistant": assistant_part,
                    "user": user_part,
                    "is_error": getattr(thought, "is_error", False),
                }
            )
        if not conversation_parts:
            return None
        conv_lines = []
        for idx, part in enumerate(conversation_parts, 1):
            conv_lines.append(f"\n--- Step {idx} ---")
            conv_lines.append(f"Assistant: {part['assistant']}")
            user_obs = part["user"]
            if len(user_obs) > 40000:
                user_obs = user_obs[:40000] + "... [truncated]"
            conv_lines.append(f"User: {user_obs}")
            if part.get("is_error"):
                conv_lines.append("[Error occurred]")
        conversation_text = "\n".join(conv_lines)
        summarization_prompt = textwrap.dedent(
            f"""
            You are summarizing a conversation history between an AI agent and its environment.
            Summarize the following conversation steps concisely, focusing on:
            1. Key actions taken (tools used, files modified, tests run)
            2. Important findings or errors encountered
            3. Progress made toward solving the problem
            4. Critical decisions or changes in approach
            Keep the summary concise (2-4 sentences per step) but preserve important details.
            Conversation to summarize:
            {conversation_text}
            Provide a concise summary:
        """
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes conversation history concisely."},
            {"role": "user", "content": summarization_prompt},
        ]
        current_model = QWEN_MODEL_NAME
        for _ in range(10):
            try:
                response, _ = Network.make_request(messages, model=current_model, temperature=0.0, where="summarize_messages_batch")
                return response.strip()
            except Exception:
                current_model = Exchange_Model(current_model)
                time.sleep(1)
        return None

    def _check_and_summarize_if_needed(self):
        total_thoughts = len(self.thoughts)
        cutoff_idx = total_thoughts - self.latest_observations_to_keep
        if cutoff_idx < self.summarize_batch_size:
            return
        unsummarized = 0
        for s, e in sorted(self.summarized_ranges):
            if s <= unsummarized < e:
                unsummarized = e
            elif s > unsummarized:
                break
        if unsummarized >= cutoff_idx:
            return
        summarize_start = unsummarized
        summarize_end = min(summarize_start + self.summarize_batch_size, cutoff_idx)
        batch_size = summarize_end - summarize_start
        if batch_size >= self.summarize_batch_size:
            range_key = (summarize_start, summarize_end)
            if range_key not in self.summaries:
                summary = self._summarize_messages_batch(summarize_start, summarize_end)
                if summary:
                    self.summaries[range_key] = summary
                    self.summarized_ranges.append(range_key)
                    self.summarized_ranges.sort()

    def add_action(self, action):
        self.thoughts.append(action)
        if len(self.thoughts) >= self.latest_observations_to_keep + self.summarize_batch_size:
            self._check_and_summarize_if_needed()
        return True

    def pop_action(self):
        return self.thoughts.pop()

    def to_str(self):
        messages = []
        last_summary_range = None
        allowed_ranges = set(self.summarized_ranges[-MAX_SUMMARY_RANGES:]) if self.summarized_ranges else set()
        total = len(self.thoughts)
        keep_last = self.latest_observations_to_keep
        for i, thought in enumerate(self.thoughts):
            if getattr(thought, "is_deleted", False):
                continue
            recent = i >= total - keep_last
            if not recent:
                summary = self._get_summary_for_index(i)
                if summary:
                    found_range = False
                    for (start, end), _ in self.summaries.items():
                        if start <= i < end:
                            cur_range = (start, end)
                            if cur_range not in allowed_ranges:
                                found_range = True
                                break
                            if cur_range != last_summary_range:
                                messages.append(
                                    {"role": "system", "content": f"[Summarized conversation history (steps {start+1} to {end}):\n{summary}\n]"}
                                )
                                last_summary_range = cur_range
                            found_range = True
                            break
                    if found_range:
                        continue
                assistant_str = (
                    f"next_thought:{thought.next_thought}\n"
                    f"next_tool_name:{thought.next_tool_name}\n"
                    f"next_tool_args:{thought.next_tool_args}"
                )
                obs = thought.observation
                if isinstance(obs, (list, tuple)):
                    try:
                        obs_render = json.dumps(list(obs), ensure_ascii=False)
                    except Exception:
                        obs_render = str(obs)
                else:
                    obs_render = str(obs) if obs else ""
                user_str = f"observation: {obs_render}"
                messages.append({"role": "assistant", "content": assistant_str})
                messages.append({"role": "user", "content": user_str})
            else:
                if thought.is_error is None or i == total - 1:
                    assistant_str = (
                        f"next_thought:{thought.next_thought}\n"
                        f"next_tool_name:{thought.next_tool_name}\n"
                        f"next_tool_args:{thought.next_tool_args}"
                    )
                    obs = thought.observation
                    if isinstance(obs, (list, tuple)):
                        try:
                            obs_render = json.dumps(list(obs), ensure_ascii=False)
                        except Exception:
                            obs_render = str(obs)
                    else:
                        obs_render = str(obs)
                    user_str = f"observation: {obs_render}"
                else:
                    if self.thoughts[-1].is_error is None and thought.is_error is not None:
                        assistant_str = (
                            f"next_thought:{thought.next_thought}\n"
                            f"next_tool_name:{thought.next_tool_name}\n"
                            f"next_tool_args:{thought.next_tool_args}"
                        )
                        obs = thought.observation
                        if obs is None:
                            obs_len = 0
                        elif isinstance(obs, (list, tuple)):
                            obs_len = len(obs)
                        else:
                            obs_len = len(str(obs).splitlines())
                        user_str = f"observation: error occurred. detailed output omitted ({obs_len}) lines\n"
                    else:
                        assistant_str = (
                            f"next_thought:{thought.next_thought}\n"
                            f"next_tool_name:{thought.next_tool_name}\n"
                            f"next_tool_args:{thought.next_tool_args}"
                        )
                        obs = thought.observation
                        if isinstance(obs, (list, tuple)):
                            try:
                                obs_render = json.dumps(list(obs), ensure_ascii=False)
                            except Exception:
                                obs_render = str(obs)
                        else:
                            obs_render = str(obs)
                        user_str = f"observation: {obs_render}"
                messages.append({"role": "assistant", "content": assistant_str})
                messages.append({"role": "user", "content": user_str})
        return messages

    def _get_summary_for_index(self, idx):
        for (start, end), summary in self.summaries.items():
            if start <= idx < end:
                return summary
        return None

    def count_repeated_thoughts(self) -> int:
        if len(self.thoughts) < 2:
            return 0
        last_thought = self.thoughts[-1]
        last_tool_name = last_thought.next_tool_name
        last_tool_args = last_thought.next_tool_args
        count = 0
        for i in range(len(self.thoughts) - 1, -1, -1):
            thought = self.thoughts[i]
            if thought.next_tool_name == last_tool_name and thought.next_tool_args == last_tool_args:
                count += 1
            else:
                break
        return max(0, count - 1)

    def is_thought_repeated(self):
        if len(self.thoughts) < 2:
            self.repeated_thoughts = 0
            return False
        last = self.thoughts[-1]
        prev = self.thoughts[-2]
        if last.next_tool_name == prev.next_tool_name and last.next_tool_args == prev.next_tool_args:
            self.repeated_thoughts += 1
            return True
        self.repeated_thoughts = 0
        return False

    class Action:
        def __init__(
            self,
            next_thought: str,
            next_tool_name: str,
            next_tool_args: dict,
            observation,
            is_error: bool = False,
            raw_response: str = None,
            total_attempts: int = 0,
            inference_error_counter: dict = None,
            request_data: list = None,
        ):
            self.next_thought = next_thought
            self.next_tool_name = next_tool_name
            self.next_tool_args = next_tool_args
            self.observation = ";".join(observation) if isinstance(observation, list) else observation
            self.is_error = is_error
            self.raw_response = raw_response
            self.total_attempts = total_attempts
            self.inference_error_counter = inference_error_counter
            self.request_data = request_data
            self.is_deleted = False


class Utils:
    @classmethod
    def count_tokens(cls, messages) -> int:
        if isinstance(messages, list):
            text = " ".join(str(m.get("content", "") if isinstance(m, dict) else m) for m in messages)
        else:
            text = messages
        tokens = re.findall(r"\w+|[^\w\s]|\s+", text)
        count = 0
        for token in tokens:
            if token.isspace():
                continue
            elif len(token) == 1:
                count += 1
            else:
                count += max(1, (len(token) + 2) // 3)
        return count

    @classmethod
    def limit_strings(cls, strings: str, n=1000) -> str:
        strings_list = strings.split("\n")
        if len(strings_list) > n:
            return "\n".join(strings_list[:n]) + "\n..." + f"({len(strings_list)-n} more lines)"
        return strings

    @classmethod
    def load_json(cls, json_string: str) -> dict:
        try:
            return json.loads(json_string)
        except Exception:
            try:
                return eval(json_string)
            except Exception:
                fixed_json = Network.fix_json_string_with_llm(json_string)
                if fixed_json:
                    return fixed_json
                raise JSONDecodeError("Invalid JSON", json_string, 0)

class ProblemDecomposer:
    def __init__(self):
        self.decomposition_cache = {}

    def decompose(self, problem_statement: str) -> dict:
        cache_key = hash(problem_statement[:500])
        if cache_key in self.decomposition_cache:
            return self.decomposition_cache[cache_key]
        truncated_problem = problem_statement
        if len(problem_statement) > 8000:
            truncated_problem = problem_statement[:4000] + "\n\n[...truncated...]\n\n" + problem_statement[-4000:]
        messages = [
            {"role": "system", "content": PROBLEM_DECOMPOSITION_PROMPT},
            {"role": "user", "content": f"Analyze this problem:\n\n{truncated_problem}"}
        ]
        result = self._default_decomposition()
        for attempt in range(3):
            try:
                response, _ = Network.make_request(messages, model=QWEN_MODEL_NAME, temperature=0.0, where="problem_decompose")
                parsed = self._parse_response(response)
                if parsed:
                    result = parsed
                    break
            except Exception:
                time.sleep(1)
        self.decomposition_cache[cache_key] = result
        return result

    def _parse_response(self, response: str) -> dict | None:
        try:
            return json.loads(response)
        except Exception:
            try:
                json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                brace_start = response.find("{")
                if brace_start >= 0:
                    return json.loads(response[brace_start:])
            except Exception:
                pass
        return None

    def _default_decomposition(self) -> dict:
        return {
            "problem_summary": "",
            "key_entities": {"files": [], "functions": [], "classes": [], "error_messages": [], "other": []},
            "behavior": {"expected": "Not specified", "actual": "Not specified", "trigger": "Not specified"},
            "success_criteria": [],
            "investigation_starting_points": [],
            "initial_hypotheses": []
        }

    def format_for_prompt(self, decomposition: dict) -> str:
        sections = []
        if decomposition.get("problem_summary"):
            sections.append(f"**Problem Summary**: {decomposition['problem_summary']}")
        entities = decomposition.get("key_entities", {})
        entity_parts = []
        if entities.get("files"):
            entity_parts.append(f"  - Files: {', '.join(entities['files'][:5])}")
        if entities.get("functions"):
            entity_parts.append(f"  - Functions: {', '.join(entities['functions'][:5])}")
        if entities.get("classes"):
            entity_parts.append(f"  - Classes: {', '.join(entities['classes'][:5])}")
        if entities.get("error_messages"):
            for msg in entities["error_messages"][:2]:
                entity_parts.append(f"  - Error: `{msg[:100]}`")
        if entity_parts:
            sections.append("**Key Entities**:\n" + "\n".join(entity_parts))
        behavior = decomposition.get("behavior", {})
        if behavior.get("expected") != "Not specified" or behavior.get("actual") != "Not specified":
            sections.append(
                f"**Behavior**:\n"
                f"  - Expected: {behavior.get('expected', 'N/A')}\n"
                f"  - Actual: {behavior.get('actual', 'N/A')}\n"
                f"  - Trigger: {behavior.get('trigger', 'N/A')}"
            )
        if decomposition.get("success_criteria"):
            criteria = "\n".join(f"  - {c}" for c in decomposition["success_criteria"][:3])
            sections.append(f"**Success Criteria**:\n{criteria}")
        if decomposition.get("investigation_starting_points"):
            points = []
            for point in decomposition["investigation_starting_points"][:4]:
                if isinstance(point, dict):
                    points.append(f"  - {point.get('location', 'N/A')}: {point.get('reason', '')}")
                else:
                    points.append(f"  - {point}")
            sections.append(f"**Suggested Starting Points**:\n" + "\n".join(points))
        if decomposition.get("initial_hypotheses"):
            hyp_parts = []
            for i, hyp in enumerate(decomposition["initial_hypotheses"][:4], 1):
                if isinstance(hyp, dict):
                    likelihood = hyp.get("likelihood", 0.5)
                    desc = hyp.get("description", "N/A")
                    hyp_parts.append(f"  {i}. [{likelihood:.0%}] {desc}")
                else:
                    hyp_parts.append(f"  {i}. {hyp}")
            sections.append(f"**Initial Hypotheses** (ranked by likelihood):\n" + "\n".join(hyp_parts))
        return "\n\n".join(sections)


_problem_decomposer = ProblemDecomposer()

class SolutionVerifier:
    """
    ALL FIX tasks must verify before finish is accepted.
    Score = 100% test pass rate. SolutionVerifier is the key quality gate.
    """
    def __init__(self, cot: "COT" = None, problem_statement: str = None):
        self.cot = cot
        self.problem_statement = problem_statement

    def verify_solution(self) -> str:
        conversation_history = self.cot.to_str() if self.cot else "No conversation history available"
        problem_statement = self.problem_statement or "No problem statement available"

        regression_check_prompt = textwrap.dedent(
            """
            You are a rigorous QA reviewer checking if an agent has properly fixed BOTH the original bug AND all regressions before finishing.

            **PROBLEM STATEMENT (Original Bug Description)**:

            {problem_statement}

            **Your job**: Analyze the agent's conversation history and verify TWO critical conditions:

            1. **NO REGRESSIONS INTRODUCED** - All tests that were passing before changes are still passing
            2. **ORIGINAL BUG IS FIXED** - The hidden tests that were originally failing are now passing

            **CRITICAL FAILURE PATTERNS TO DETECT**:

            1. **Selective Test Running** - Agent ran only 1-2 specific test cases instead of the full test suite
            2. **Ignoring Test Failures** - Agent saw test failures but didn't fix them
            3. **Rationalization** - Agent explained away failures as "unrelated" or "acceptable"
            4. **No Full Suite Run** - Agent never ran the full test suite for the affected module

            **WHAT CONSTITUTES COMPLETE SUCCESS (BOTH CONDITIONS REQUIRED)**:

            ✅ **CONDITION 1: ORIGINAL BUG IS FIXED**
               - The hidden tests mentioned in the problem statement are now passing
               - Agent verified the fix with actual test runs (not just theory)

            ✅ **CONDITION 2: NO REGRESSIONS**
               - Agent ran the FULL test suite (or at minimum, the full test class) for affected modules
               - The FINAL test run before calling finish showed ALL tests passing

            **YOUR RESPONSE FORMAT**:
            - **IF BOTH CONDITIONS MET**: Return exactly "REGRESSION_AND_BUG_CHECK_PASSED" followed by brief explanation
            - **IF ANY ISSUES FOUND**: Return detailed feedback explaining what must be fixed

            **CONVERSATION HISTORY TO ANALYZE**:

            {conversation_history}

            **YOUR RESPONSE**:
        """
        ).strip()

        messages = [
            {"role": "system", "content": "You are a rigorous QA reviewer checking for proper regression testing. Be strict and thorough."},
            {
                "role": "user",
                "content": regression_check_prompt.format(
                    problem_statement=problem_statement,
                    conversation_history=conversation_history,
                ),
            },
        ]

        retry = 0
        selected_model = QWEN_MODEL_NAME
        max_retries = 10
        while retry < max_retries:
            try:
                review_result, _ = Network.make_request(messages, model=selected_model, temperature=0.0, where="verify_solution")
                return review_result.strip()
            except Exception as e:
                logger.warning(f"verify_solution attempt {retry+1}: {e}")
            retry += 1
            if retry > 5:
                selected_model = Exchange_Model(selected_model)
            time.sleep(0.5)
        return "⚠️ Regression verification failed after max retries. Please manually verify all tests pass."



class SearchManager:
    def search_in_file(self, file_path: str, search_term: str) -> str:
        def extract_matches(filepath, term, max_output_lines=1000):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    lines = f.read().splitlines()
            except Exception as e:
                return f"Error reading '{filepath}': {e}"
            match_lines = [i + 1 for i, line in enumerate(lines) if term in line]
            if not match_lines:
                return f"'{term}' not found in file '{filepath}'"
            context = 20
            seen = set()
            chunks = []
            for ln in match_lines:
                start = max(1, ln - context)
                end = min(len(lines), ln + context)
                rkey = (start, end)
                if rkey in seen:
                    continue
                seen.add(rkey)
                chunk = lines[start - 1 : end]
                chunks.append(f"(lines {start}-{end}):\n" + "\n".join(chunk))
            return Utils.limit_strings("\n\n".join(chunks), n=max_output_lines)

        output = extract_matches(file_path, search_term)
        if Utils.count_tokens(output) > 3000:
            return "Search results are too long. Please refine your search term into more specific terms."
        return output

    def search_in_all_files(self, grep_search_command: str) -> str:
        cmd = grep_search_command.lstrip()
        if not cmd.startswith("grep"):
            return f"Error: Invalid command. Expected a grep command but got: '{grep_search_command}'"
        try:
            result = subprocess.run(["bash", "-c", grep_search_command], capture_output=True, text=True, timeout=45)
        except Exception as e:
            return f"Error: Failed to execute grep command: {e}"
        if result.returncode > 1:
            error_msg = result.stderr.strip() or "Unknown error"
            return f"Error: Grep command failed with return code {result.returncode}: {error_msg}"
        output = result.stdout
        if not output.strip():
            return "No matches found for pattern in codebase."
        if Utils.count_tokens(output) > 3000:
            return "Search results are too long. Please refine your search term into more specific terms."
        return output


class TestManager:
    def __init__(self, runner_hint=None, runner_mode_hint=None, file_ops=None):
        self.runner_hint = runner_hint
        self.runner_mode_hint = runner_mode_hint
        self.file_ops = file_ops

    def run_code(self, content: str, file_path: str, generated_test_files: list, run_command: list) -> str:
        if file_path.endswith((".py", ".pyw", ".pyx", ".pyi", ".pxd", ".pxi", ".pyz")):
            content = VERSION_COMPATIBILITY_FIX + "\n\n" + content
        file_exists = os.path.exists(file_path) and os.path.isfile(file_path)
        self.file_ops.save(file_path, content)
        if file_path not in generated_test_files and not file_exists:
            generated_test_files.append(file_path)
        try:
            result = subprocess.run(run_command, capture_output=True, text=True, check=False, timeout=60)
            if result.returncode != 0:
                return f"Error running code: {result.stderr}"
            return f"{result.stdout}\n"
        except Exception as e:
            return f"Error: {e}"


class CodeParseUtil:
    def __init__(self):
        self._parsers = {}

    def check_language(self, source: str, file_path: str | None = None) -> str | None:
        """Detect language from file extension (fast) or LLM fallback."""
        global _codeparse_util_language_cache
        if not source or not source.strip():
            return None
        if file_path:
            abs_path = os.path.abspath(file_path)
            if abs_path in _codeparse_util_language_cache:
                return _codeparse_util_language_cache[abs_path]
            # Fast path: extension heuristic
            ext_map = {
                ".py": "python", ".js": "javascript", ".ts": "typescript",
                ".java": "java", ".c": "c", ".cpp": "cpp", ".go": "go",
                ".rs": "rust", ".rb": "ruby", ".php": "php", ".cs": "csharp",
                ".kt": "kotlin", ".swift": "swift", ".sh": "bash",
            }
            ext = Path(file_path).suffix.lower()
            if ext in ext_map:
                lang = ext_map[ext]
                _codeparse_util_language_cache[abs_path] = lang
                return lang

        # LLM fallback
        stripped_source = source.strip()
        sample = (
            stripped_source
            if len(stripped_source) <= 1000
            else f"{stripped_source[:500]}\n\n... [middle content omitted] ...\n\n{stripped_source[-500:]}"
        )
        prompt = f"""Detect the programming language of the following code sample.
        Return ONLY the language name in lowercase.
        If you cannot determine the language, return "unknown".
        Code sample:
        ```
        {sample}
        ```
        Return ONLY the language name in lowercase, no other text or explanation."""
        retry = 0
        messages = [{"role": "user", "content": prompt}]
        models_to_try = [QWEN_MODEL_NAME, KIMI_MODEL_NAME]
        while retry < 3:
            try:
                result, _ = Network.make_request(
                    messages=messages, model=models_to_try[retry % len(models_to_try)], attempt=1, temperature=0.0, where="check_language"
                )
                cleaned = result.strip().lower().removeprefix("```").removesuffix("```").strip().strip('"').strip("'").strip()
                if cleaned and " " not in cleaned and cleaned.isalpha():
                    detected = cleaned if cleaned != "unknown" else None
                    if file_path:
                        _codeparse_util_language_cache[os.path.abspath(file_path)] = detected
                    return detected
            except Exception:
                pass
            retry += 1
            time.sleep(1)
        return None

    def _is_identifier_node(self, node) -> bool:
        return "identifier" in node.type.lower()

    def _get_parser(self, language: str):
        if Parser is None or get_language is None:
            return None
        if language not in self._parsers:
            try:
                lang_obj = get_language(language)
                if lang_obj is None:
                    return None
                parser = Parser(lang_obj)
                self._parsers[language] = parser
            except Exception:
                return None
        return self._parsers[language]

    def get_function_body(self, file_path: str, function_name: str, add_line_numbers: bool = False) -> str:
        if not function_name or not os.path.exists(file_path):
            return ""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
        except Exception:
            return ""
        if not source or Parser is None:
            return ""
        try:
            source_bytes = bytes(source, "utf8")
            source_lines = source.splitlines()
            language = self.check_language(source, file_path=file_path)
            if not language:
                return ""
            parser = self._get_parser(language)
            if parser is None:
                return ""
            tree = parser.parse(source_bytes)
            target_qualified = function_name
            target_simple = function_name.split(".")[-1]
            func_info = self._find_specific_function(tree.root_node, source_lines, target_qualified, target_simple)
            if func_info is None:
                return ""
            start_idx = func_info["start_line"] - 1
            end_idx = func_info["end_line"] - 1
            if 0 <= start_idx < len(source_lines) and 0 <= end_idx < len(source_lines):
                body_lines = source_lines[start_idx : end_idx + 1]
                if add_line_numbers:
                    return "\n".join(f"{start_idx + i + 1}| {line}" for i, line in enumerate(body_lines))
                return "\n".join(body_lines)
        except Exception as e:
            logger.warning(f"Error finding function {function_name} in {file_path}: {e}")
        return ""

    def _classify_node_type(self, node):
        node_type_str = node.type.lower()
        if "function" in node_type_str or "method" in node_type_str:
            for i, child in enumerate(node.children):
                if self._is_identifier_node(child):
                    return ("function", i)
            return ("function", None)
        elif "class" in node_type_str:
            for i, child in enumerate(node.children):
                if self._is_identifier_node(child):
                    return ("class", i)
            return ("class", None)
        return ("other", None)

    def _find_specific_function(self, node, source_lines, target_qualified, target_simple, class_name="", parent_node=None):
        if not node.children:
            return None
        node_type, name_child_index = self._classify_node_type(node)

        if node_type == "class":
            name = None
            if name_child_index is not None and name_child_index < len(node.children):
                name_child = node.children[name_child_index]
                ns, ne = name_child.start_point, name_child.end_point
                if ns[0] < len(source_lines):
                    line = source_lines[ns[0]]
                    name = line[ns[1] : ne[1]].strip() if ns[0] == ne[0] else line[ns[1]:].strip()
            if name:
                new_class_name = f"{class_name}.{name}" if class_name else name
                for child in node.children:
                    result = self._find_specific_function(child, source_lines, target_qualified, target_simple, new_class_name, node)
                    if result is not None:
                        return result
        elif node_type == "function":
            internal_name = None
            if name_child_index is not None and name_child_index < len(node.children):
                nc = node.children[name_child_index]
                ns, ne = nc.start_point, nc.end_point
                if ns[0] < len(source_lines):
                    line = source_lines[ns[0]]
                    internal_name = line[ns[1] : ne[1]].strip() if ns[0] == ne[0] else line[ns[1]:].strip()
            name = internal_name
            if name:
                qualified_name = f"{class_name}.{name}" if class_name else name
                is_qualified_target = "." in target_qualified
                if qualified_name == target_qualified or (not is_qualified_target and name == target_simple):
                    at_start = node.start_point[0]
                    for i in range(at_start - 1, -1, -1):
                        if i < len(source_lines) and source_lines[i].strip().startswith("@"):
                            at_start = i
                        elif i < len(source_lines) and source_lines[i].strip():
                            break
                    return {"start_line": at_start + 1, "end_line": node.end_point[0] + 1}
            for child in node.children:
                result = self._find_specific_function(child, source_lines, target_qualified, target_simple, class_name, node)
                if result is not None:
                    return result
        for child in node.children:
            result = self._find_specific_function(child, source_lines, target_qualified, target_simple, class_name, node)
            if result is not None:
                return result
        return None

class CodeEditManager:
    def __init__(self, file_ops: "FileOperationsUtil" = None):
        self.file_ops = file_ops

    def apply_code_edit(self, file_path: str, search: str, replace: str) -> str:
        def find_most_similar_content(original_content: str, search_string: str, max_results: int = 3):
            lines = original_content.split("\n")
            search_lines = search_string.split("\n")
            target_chunk_size = max(3, len(search_lines))
            chunks = []
            for i in range(len(lines) - target_chunk_size + 1):
                chunk_lines = lines[i : i + target_chunk_size]
                chunk_content = "\n".join(chunk_lines).strip()
                if chunk_content:
                    chunks.append((f"Lines {i+1}-{i+target_chunk_size}: ...", chunk_content))
            similarities = []
            for chunk_desc, chunk_content in chunks:
                ratio = difflib.SequenceMatcher(None, search_string.strip(), chunk_content).ratio()
                if ratio > 0.3:
                    similarities.append((ratio, chunk_desc, chunk_content))
            similarities.sort(key=lambda x: x[0], reverse=True)
            return [(ratio, f"{desc}\n{content}") for ratio, desc, content in similarities[:max_results]]

        if search == replace:
            return "ERROR: search and replace are the same. Please provide a different search and replace."
        if not os.path.exists(file_path):
            return f"Error: file '{file_path}' does not exist."
        original = self.file_ops.get_file_content(file_path, limit=-1)
        match_count = original.count(search)
        if match_count == 0:
            similar_matches = find_most_similar_content(original, search, 1)
            error_msg = f"Error: search string not found in file {file_path}."
            if similar_matches:
                error_msg += "\n\nMost similar snippet found:"
                for i, (ratio, content) in enumerate(similar_matches, 1):
                    similarity_pct = int(ratio * 100)
                    error_msg += f"\n\n{i}. Similarity: {similarity_pct}%\n{content}"
            else:
                error_msg += " No similar content found. Please check the file content and provide the exact code."
            return error_msg
        elif match_count == 1:
            new_content = original.replace(search, replace)
            try:
                self.file_ops.save(file_path, new_content)
                replace_pos = new_content.find(replace)
                if replace_pos != -1:
                    lines = new_content.split("\n")
                    chars_so_far = 0
                    replace_line_start = 0
                    for i, line in enumerate(lines):
                        if chars_so_far + len(line) >= replace_pos:
                            replace_line_start = i
                            break
                        chars_so_far += len(line) + 1
                    replace_lines_count = replace.count("\n") + 1
                    replace_line_end = replace_line_start + replace_lines_count - 1
                    start_line = max(0, replace_line_start - 10)
                    end_line = min(len(lines), replace_line_start + 10)
                    context_lines = []
                    for i in range(start_line, end_line):
                        line_num = i + 1
                        prefix = ">>> " if replace_line_start <= i <= replace_line_end else "    "
                        context_lines.append(f"{prefix}{line_num:4}| {lines[i]}")
                    context = "\n".join(context_lines)
                    return f"ok, code edit applied successfully. Here is the edited code (lines {start_line+1}-{end_line}):\n\n{context}"
                return "ok, code edit applied successfully"
            except Exception as e:
                return f"Error: syntax error in file {file_path}. {str(e)}"
        else:
            return f"Error: search string found {match_count} times in file '{file_path}'.\nPlease reformulate your search and replace to apply only one change."


class FileSystemManager:
    def get_file_content(
        self,
        file_path: str,
        search_start_line: int = None,
        search_end_line: int = None,
        search_term: str = None,
        limit: int = 1000,
        add_line_numbers: bool = False,
        search_in_file_callback=None,
    ) -> str:
        def add_line_numbers_to_content(content: str, start_line: int = 1) -> str:
            lines = content.splitlines()
            return "\n".join(f"{start_line + i:6}|{line}" for i, line in enumerate(lines))

        if search_term and search_in_file_callback:
            return search_in_file_callback(file_path, search_term)
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            if search_start_line is not None or search_end_line is not None:
                lines = f.readlines()
                start_idx = max(0, (search_start_line or 1) - 1)
                end_idx = min(len(lines), search_end_line or len(lines))
                content = "".join(lines[start_idx:end_idx])
                if add_line_numbers:
                    result = add_line_numbers_to_content(content, start_idx + 1)
                else:
                    result = content
            else:
                content = f.read()
                if add_line_numbers:
                    result = add_line_numbers_to_content(content, 1)
                else:
                    result = content
        return Utils.limit_strings(result, n=limit) if limit != -1 else result

    def list_directory_structure(self, directory_path: str, max_depth: int = 0) -> str:
        if not os.path.exists(directory_path):
            return f"Error: Directory '{directory_path}' does not exist."
        if not os.path.isdir(directory_path):
            return f"Error: '{directory_path}' is not a directory."
        ignore = {".git", "__pycache__", ".pytest_cache", "node_modules", ".tox", ".venv", "venv", ".eggs"}

        def tree(path: str, prefix: str = "", depth: int = 0, current_max_depth: int = 0):
            if depth > current_max_depth:
                return []
            try:
                items = sorted(os.listdir(path))
            except (PermissionError, OSError) as e:
                return [f"{prefix}[Error reading directory: {e}]"]
            dirs = [i for i in items if os.path.isdir(os.path.join(path, i)) and not i.startswith(".") and i not in ignore and not i.endswith(".egg-info")]
            files = [i for i in items if os.path.isfile(os.path.join(path, i)) and not i.startswith(".")]
            lines = []
            for idx, d in enumerate(dirs):
                is_last = (idx == len(dirs) - 1) and not files
                branch = "└── " if is_last else "├── "
                new_prefix = prefix + ("    " if is_last else "│   ")
                lines.append(f"{prefix}{branch}{d}/")
                lines.extend(tree(os.path.join(path, d), new_prefix, depth + 1, current_max_depth))
            for idx, fn in enumerate(files):
                is_last = idx == len(files) - 1
                branch = "└── " if is_last else "├── "
                lines.append(f"{prefix}{branch}{fn}")
            return lines

        MAX_TOKENS = 3000
        current_depth = max_depth
        while current_depth >= 0:
            entries = tree(directory_path, "", 0, current_depth)
            result = f"Directory structure (depth={current_depth}):\n{directory_path}/\n" + "\n".join(entries)
            if Utils.count_tokens(result) <= MAX_TOKENS:
                return result
            if current_depth == 0:
                return result + f"\n\n[Warning: Result exceeds token limit. Consider using a more specific directory_path.]"
            current_depth -= 1
        entries = tree(directory_path, "", 0, 0)
        return f"Directory structure (depth=0):\n{directory_path}/\n" + "\n".join(entries)


class Network:
    @classmethod
    def make_request(
        cls,
        messages: list,
        model: Model,
        attempt: int = 0,
        temperature: float = 0.0,
        tool_mode: str = "none",
        tool_docs: list = [],
        timeout: int = None,
        where: str = "none",
    ) -> tuple[str, list]:
        global run_id, agent_start_time, total_inferenced_chars, individual_inferenced_chars
        messages_str = json.dumps(messages, ensure_ascii=False)
        individual_inferenced_chars = len(messages_str)
        total_inferenced_chars += individual_inferenced_chars

        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
        attempts = max(1, attempt or 1)
        model_obj = model if isinstance(model, Model) else QWEN_MODEL_NAME
        model_name = model_obj.name

        if timeout is None:
            timeout = model_obj.timeout

        request_data = {
            "evaluation_run_id": run_id if run_id else str(uuid4()),
            "messages": messages,
            "temperature": temperature,
            "model": model_name,
            "tool_mode": tool_mode,
            "tools": tool_docs,
        }
        headers = {"Content-Type": "application/json"}

        wait = 1
        max_wait = 60

        for i in range(max(1, attempts)):
            if _watchdog_abort.is_set():
                raise RuntimeError("WATCHDOG: aborting LLM call — budget limit reached")
            try:
                logger.debug(f"⏳ [{where}] {model_name} timeout={timeout}s")
                resp = requests.post(url, json=request_data, timeout=(30, timeout), headers=headers)

                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    wait_time = int(retry_after) if retry_after else wait
                    time.sleep(wait_time)
                    wait = min(wait * 2, max_wait)
                    continue

                resp.raise_for_status()
                try:
                    resp_json = resp.json()
                except JSONDecodeError as e:
                    if i >= attempts - 1:
                        raise ValueError(f"HTTP ERROR: Invalid JSON response for model {model_name}: {e}")
                    continue
                try:
                    raw_text = resp_json["content"]
                    tool_calls = resp_json["tool_calls"]
                except Exception:
                    raise RuntimeError(f"HTTP ERROR: Response Parse Error for model {model_name}")
                if (tool_mode == "none" and not raw_text) or (tool_mode != "none" and not tool_calls):
                    raise RuntimeError(f"HTTP ERROR: NO RESPONSE FOUND for model {model_name}")
                return raw_text, tool_calls
            except requests.exceptions.Timeout:
                if i >= attempts - 1:
                    raise RuntimeError(f"HTTP ERROR: Request timeout for model {model_name}")
                model_name = Exchange_Model(model_obj).name
                time.sleep(1)
            except requests.exceptions.ConnectionError as e:
                global _USE_LOCAL_CHUTES
                if LOCAL_CHUTES_KEY and not _USE_LOCAL_CHUTES:
                    logger.info("[T68] Proxy unreachable, switching to direct Chutes fallback")
                    _USE_LOCAL_CHUTES = True
                if _USE_LOCAL_CHUTES and LOCAL_CHUTES_KEY:
                    try:
                        # Chutes uses model names with -TEE suffix
                        chutes_model = model_name if model_name.endswith("-TEE") else model_name + "-TEE"
                        chutes_payload = {"model": chutes_model, "messages": messages,
                                          "temperature": temperature, "max_tokens": 4096}
                        if tool_mode != "none" and tool_docs:
                            chutes_payload["tools"] = tool_docs
                            chutes_payload["tool_choice"] = "auto"
                        chutes_resp = requests.post(
                            LOCAL_CHUTES_URL, json=chutes_payload,
                            headers={"Authorization": f"Bearer {LOCAL_CHUTES_KEY}",
                                     "Content-Type": "application/json"},
                            timeout=(30, timeout or 120)
                        )
                        chutes_resp.raise_for_status()
                        chutes_data = chutes_resp.json()
                        choice = chutes_data.get("choices", [{}])[0]
                        msg = choice.get("message", {})
                        raw_text = msg.get("content") or msg.get("reasoning_content", "") or ""
                        raw_tool_calls = msg.get("tool_calls")
                        return raw_text, raw_tool_calls
                    except Exception as chutes_err:
                        logger.warning(f"[T68] Chutes direct also failed: {chutes_err}")
                if i >= attempts - 1:
                    raise RuntimeError(f"HTTP ERROR: Connection error for model {model_name}: {e}")
                time.sleep(1)
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response is not None else "unknown"
                if status_code == 504:
                    if i >= attempts - 1:
                        raise RuntimeError(f"HTTP ERROR 504: Gateway Timeout for model {model_name}: {e}")
                    time.sleep(1)
                    continue
                if status_code == 429:
                    if i >= attempts - 1:
                        raise RuntimeError(f"HTTP ERROR 429: Too many requests for model {model_name}: {e}")
                    model_name = Exchange_Model(model_obj).name
                    time.sleep(1)
                    continue
                if i >= attempts - 1:
                    raise RuntimeError(f"HTTP ERROR {status_code} for model {model_name}: {e}")
                time.sleep(1)
            except requests.exceptions.RequestException as e:
                if i >= attempts - 1:
                    raise RuntimeError(f"HTTP ERROR: Request failed for model {model_name}: {e}")
                time.sleep(1)
        raise RuntimeError(f"HTTP ERROR: Failed to get response for model {model_name} after {attempts} attempts")

    @classmethod
    def inference(cls, messages: list, model, run_id: str = None, temperature: float = 0.0) -> tuple:
        models = model if isinstance(model, list) else [model]
        cleaned_msgs = [
            {"role": m["role"], "content": m.get("content", "")}
            for m in messages
            if m.get("role") in {"system", "user", "assistant", "tool"} and (m.get("role") != "assistant" or m.get("content", "").strip())
        ]
        if not cleaned_msgs:
            raise RuntimeError("No valid messages to send to proxy.")
        result = cls._request_next_action_with_retry(cleaned_msgs, models=models, temperature=temperature)
        return result

    @classmethod
    def get_cost_usage(cls) -> dict:
        global run_id
        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/usage?evaluation_run_id={run_id if run_id else str(uuid4())}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            usage_info = response.json()
            if isinstance(usage_info, dict):
                return usage_info
        except Exception:
            pass
        return {"used_cost_usd": 0, "max_cost_usd": float("inf")}

    @classmethod
    def _extract_balanced_braces(cls, text: str, start_pos: int) -> str | None:
        if start_pos >= len(text):
            return None
        brace_count = 0
        in_string = False
        escape_next = False
        start = -1
        for i in range(start_pos, len(text)):
            c = text[i]
            if escape_next:
                escape_next = False
                continue
            if c == "\\":
                escape_next = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if not in_string:
                if c == "{":
                    if start == -1:
                        start = i
                    brace_count += 1
                elif c == "}":
                    brace_count -= 1
                    if brace_count == 0 and start != -1:
                        return text[start : i + 1]
        return None

    @classmethod
    def _extract_tool_call_from_block(cls, block: str) -> dict | None:
        tool_name_match = re.search(r"tool_name\s*:\s*([^\s]+)", block, re.IGNORECASE)
        if not tool_name_match:
            return None
        tool_name = tool_name_match.group(1).strip("\"'")
        args_match = re.search(r"tool_args\s*:\s*[\n\r\s]*\{", block, re.IGNORECASE | re.DOTALL)
        if not args_match:
            return None
        args_start = args_match.end() - 1
        json_str = cls._extract_balanced_braces(block, args_start)
        if json_str:
            try:
                tool_args = json.loads(json_str)
                return {"tool_name": tool_name, "tool_args": tool_args}
            except json.JSONDecodeError:
                try:
                    tool_args = json.loads(json_str.replace("'", '"'))
                    return {"tool_name": tool_name, "tool_args": tool_args}
                except Exception:
                    pass
        return None

    @classmethod
    def _request_next_action_with_retry(
        cls,
        messages: dict,
        models: list,
        max_retries: int = 3,
        temperature: float = 0.0,
    ) -> tuple:
        raw_text = None
        error_counter = cls.get_error_counter()
        next_thought = next_tool_name = next_tool_args = None
        total_attempts = 0
        current_model_idx = 0
        used_model = models[0] if models else QWEN_MODEL_NAME
        for attempt in range(max_retries):
            try:
                total_attempts += 1
                current_model = models[min(current_model_idx, len(models) - 1)]
                used_model = current_model
                raw_text, _ = cls.make_request(messages, model=current_model, temperature=temperature, where="request_next_action")
                is_valid, error_msg = cls.is_valid_response(raw_text)
                if not is_valid:
                    raise Exception(error_msg)
                next_thought, next_tool_name, next_tool_args, error_msg = cls.parse_response(raw_text)
                if error_msg:
                    raise Exception(error_msg)
                break
            except Exception as e:
                error_body = str(e)
                is_504 = "504" in error_body or "Gateway Timeout" in error_body
                if is_504 and current_model_idx < len(models) - 1:
                    current_model_idx += 1
                    time.sleep(3)
                    continue
                if attempt < max_retries - 1:
                    matched = False
                    for key in ["RATE_LIMIT_EXCEEDED", "RESERVED_TOKEN_PRESENT", "EMPTY_RESPONSE", "TIMEOUT", "Invalid JSON", "Invalid response"]:
                        if key in error_body:
                            attr_name = key if key in cls.ErrorType.__members__ else "INVALID_RESPONSE_FORMAT"
                            error_counter[attr_name] = error_counter.get(attr_name, 0) + 1
                            matched = True
                            break
                    if not matched:
                        error_counter[cls.ErrorType.UNKNOWN.name] = error_counter.get(cls.ErrorType.UNKNOWN.name, 0) + 1
                    skip_http = any(
                        x in error_body
                        for x in ["HTTP ERROR", "RATE_LIMIT_EXCEEDED", "EMPTY_RESPONSE", "TIMEOUT", "NETWORK_ERROR", "HTTP ERROR 429", "INCOMPLETE_RESPONSE"]
                    )
                    if not skip_http:
                        messages.append({"role": "assistant", "content": raw_text})
                        messages.append({"role": "user", "content": "observation: " + error_body})
                    time.sleep(3)
                    continue
                else:
                    error_counter[cls.ErrorType.TIMEOUT.name] = error_counter.get(cls.ErrorType.TIMEOUT.name, 0) + 1
                    raise RuntimeError(error_body)
        return (
            next_thought,
            next_tool_name,
            next_tool_args,
            raw_text,
            total_attempts,
            error_counter,
            messages,
            used_model,
        )

    @classmethod
    def parse_malformed_json(cls, arguments: list, json_string: str) -> dict | str:
        pattern = r",\s*".join(rf'"{k}": (.*)' for k in arguments)
        match = re.search(pattern, json_string)
        if not match:
            return f"Error: {json_string} can not match pattern {pattern}"
        return {k: match.group(i + 1).strip().strip('"').replace("\\n", "\n") for i, k in enumerate(arguments)}

    @classmethod
    def fix_json_string_with_llm(cls, json_string: str, attempt: int = 0) -> dict:
        messages = [
            {"role": "system", "content": "Fix the json string sent by the user. Reply only with the json string and nothing else."},
            {"role": "user", "content": json_string},
        ]
        selected_model = QWEN_MODEL_NAME
        retry = 0
        response = None
        while retry < 5:
            try:
                response, _ = cls.make_request(messages, model=selected_model, where="fix_json")
                break
            except Exception:
                retry += 1
                selected_model = Exchange_Model(selected_model)
                time.sleep(1)
        if response is None:
            return None
        try:
            response = response.replace("```json", "").strip("```")
            return json.loads(response)
        except Exception:
            return None

    @classmethod
    def sanitise_text_resp(cls, text_resp: str) -> str:
        text_resp = re.sub(r"['\"]*next_thought['\"]*:", "next_thought:", text_resp)
        text_resp = re.sub(r"['\"]*next_tool_name['\"]*:", "next_tool_name:", text_resp)
        text_resp = re.sub(r"['\"]*next_tool_args['\"]*:", "next_tool_args:", text_resp)
        text_resp = re.sub(r"['\"]*observation['\"]*:", "observation:", text_resp)
        text_resp = re.sub(r"['\"]*tool_call_['\"]*", "tool_call_", text_resp)
        if (
            "next_thought" not in text_resp
            and "next_tool_name:" in text_resp
            and "next_tool_args:" in text_resp
            and text_resp.find("next_tool_name:") < text_resp.find("next_tool_args:")
            and text_resp.find("next_tool_name:") > 10
        ):
            text_resp = "next_thought: " + text_resp
        return text_resp

    @classmethod
    def parse_next_tool_args(cls, tool_name, next_tool_args: str):
        next_tool_args = next_tool_args.replace("```json", "").strip("```")
        try:
            return Utils.load_json(next_tool_args.strip())
        except JSONDecodeError:
            try:
                schema_tool_name = tool_name[0] if isinstance(tool_name, list) and tool_name else tool_name
                return cls.parse_malformed_json(
                    ToolManager.get_tool_args_for_tool(schema_tool_name, required_only=True),
                    next_tool_args,
                )
            except Exception:
                raise Exception(f"Invalid JSON: {next_tool_args}")

    @classmethod
    def parse_response(cls, text_resp: str) -> tuple:
        error_msg = None
        text_resp = text_resp.strip()
        if "observation:" in text_resp.lower():
            text_resp = re.split(r"observation\s*:", text_resp, flags=re.IGNORECASE)[0].strip()
        text_resp = cls.sanitise_text_resp(text_resp)
        if "Infrastructure is at maximum capacity" in text_resp:
            return None, None, None, "HTTP ERROR Maximum Capacity"
        if "No instances available" in text_resp:
            return None, None, None, "HTTP ERROR NO INSTANCES AVAILABLE"
        next_thought = None
        for pat in [
            r"next_thought\s*:\s*(.*?)(?=\n(?:tool_call_|next_tool_name:|$))",
            r"next_thought\s*:\s*(.*?)(?=\ntool_call_)",
            r"next_thought\s*:\s*(.*?)(?=\nnext_tool_name:)",
            r"next_thought\s*:\s*(.*)",
        ]:
            match = re.search(pat, text_resp, re.DOTALL | re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                if candidate and len(candidate) > 2:
                    next_thought = candidate
                    break
        if not next_thought:
            next_thought = "Processing request"

        tool_call_matches = list(re.finditer(r"tool_call_(\d+)\s*:", text_resp, re.IGNORECASE))
        if tool_call_matches:
            tool_calls = []
            for i, match in enumerate(tool_call_matches):
                start = match.end()
                end = tool_call_matches[i + 1].start() if i + 1 < len(tool_call_matches) else len(text_resp)
                block = text_resp[start:end].strip()
                call = cls._extract_tool_call_from_block(block)
                if call:
                    tool_calls.append(call)
            if not tool_calls:
                return next_thought, None, None, "Multi-tool format detected but no valid tool calls extracted"
            tool_names = [c["tool_name"] for c in tool_calls]
            tool_args_list = [c["tool_args"] for c in tool_calls]
            if len(tool_names) == 1:
                return next_thought, tool_names[0], tool_args_list[0], error_msg
            return next_thought, tool_names, tool_args_list, error_msg

        if "next_tool_name:" in text_resp and "next_tool_args:" in text_resp:
            name_idx = text_resp.find("next_tool_name:")
            args_idx = text_resp.find("next_tool_args:")
            if text_resp.find("next_thought:") < name_idx < args_idx:
                next_tool_name_raw = text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip()
                next_tool_args_raw = text_resp.split("next_tool_args:")[1].strip().split("next_thought:")[0].strip()
                try:
                    if next_tool_name_raw.startswith("["):
                        next_tool_names = Utils.load_json(next_tool_name_raw)
                    else:
                        next_tool_names = [next_tool_name_raw]
                    parsed_args = cls.parse_next_tool_args(next_tool_names, next_tool_args_raw)
                    next_tool_args_list = parsed_args if isinstance(parsed_args, list) else [parsed_args for _ in next_tool_names]
                    if len(next_tool_names) == 1:
                        return next_thought, next_tool_names[0], next_tool_args_list[0], error_msg
                    return next_thought, next_tool_names, next_tool_args_list, error_msg
                except (JSONDecodeError, Exception) as e:
                    error_msg = f"Invalid JSON in tool args: {str(e)}"
                    return next_thought, None, None, error_msg

        if "next_thought:" not in text_resp:
            error_msg = "Invalid response. next_thought not found"
        elif "next_tool_name:" not in text_resp and "tool_call_" not in text_resp:
            error_msg = "Invalid response. No tool calls found"
        elif "next_tool_args:" not in text_resp and "tool_call_" not in text_resp:
            error_msg = "Invalid response. next_tool_args not found"
        else:
            error_msg = "Invalid response format. Could not parse tool calls."
        return next_thought, None, None, error_msg

    @classmethod
    def get_error_counter(cls) -> dict:
        return {k: 0 for k in cls.ErrorType.__members__}

    @classmethod
    def is_valid_response(cls, raw_text: str) -> tuple[bool, str | None]:
        if isinstance(raw_text, dict) and raw_text.get("error"):
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        if not raw_text:
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        if "<|reserved_token_" in raw_text:
            return False, cls.ErrorType.RESERVED_TOKEN_PRESENT.name
        stripped = raw_text.strip()
        lower = raw_text.lower()
        has_next_thought = "next_thought" in lower
        has_next_tool_name = "next_tool_name" in lower
        has_next_tool_args = "next_tool_args" in lower
        # valid_ending check removed — causes false positives when model appends trailing text
        # Any response with all 3 required fields is valid regardless of ending
        if "Infrastructure is at maximum capacity" in raw_text:
            return False, "HTTP ERROR Maximum Capacity"
        if "API request failed with status 429" in raw_text:
            return False, cls.ErrorType.RATE_LIMIT_EXCEEDED.name
        if "Read timed out" in raw_text:
            return False, cls.ErrorType.TIMEOUT.name
        return True, None

    @classmethod
    def is_http_response(cls, raw_text: str):
        if "API request failed with status 429" in raw_text:
            return False, cls.ErrorType.RATE_LIMIT_EXCEEDED.name
        if "Read timed out" in raw_text:
            return False, cls.ErrorType.TIMEOUT.name
        if "HTTP ERROR: Request failed for model" in raw_text:
            return False, cls.ErrorType.NETWORK_ERROR.name
        return True, None

    class ErrorType(Enum):
        EMPTY_RESPONSE = 1
        RESERVED_TOKEN_PRESENT = 2
        RATE_LIMIT_EXCEEDED = 3
        INVALID_RESPONSE_FORMAT = 4
        TIMEOUT = 5
        UNKNOWN = 6
        NETWORK_ERROR = 7
        AUTHENTICATION_ERROR = 8
        RESOURCE_EXHAUSTED = 9
        INCOMPLETE_RESPONSE = 10


class ToolManager:
    TOOL_LIST = {}

    def get_tool_docs(self) -> str:
        return "\n\n".join([json.dumps(tool_metadata, ensure_ascii=False) for _, tool_metadata in self.TOOL_LIST.items()])

    def __init__(self, **kwargs):
        pass

    @classmethod
    def tool_parsing(cls, fn):
        name = fn.__name__
        doc_fn = fn.__doc__ or ""
        doc = doc_fn.split("Arguments:")[0]
        output_description = doc_fn.split("Output:")
        if len(output_description) > 1:
            output_description = "Output: " + output_description[1].strip()
            doc = doc + "\n\n" + output_description
        sig = inspect.signature(fn)
        properties = {}
        required = []
        for param in sig.parameters.values():
            if param.name == "self":
                continue
            if param.default is param.empty and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                required.append(param.name)
            type_hint = str(param.annotation) if param.annotation != param.empty else "string"
            param_description = re.search(f"{param.name}:([^\n]+)", doc_fn)
            if param_description:
                param_description = param_description.group(1)
            else:
                param_description = f"Parameter {param.name}"
            if ("list" in type_hint.lower()) and ("str" in type_hint):
                properties[param.name] = {"type": "array", "items": {"type": "string"}, "description": param_description}
                continue
            elif "str" in type_hint:
                json_type = "string"
            elif "int" in type_hint:
                json_type = "integer"
            elif "float" in type_hint:
                json_type = "number"
            elif "bool" in type_hint:
                json_type = "boolean"
            else:
                json_type = "string"
            properties[param.name] = {"type": json_type, "description": param_description}
        parameters = {"type": "object", "properties": properties, "required": required}
        tool_schemas = {"name": name, "description": doc.strip(), "input_schema": parameters}
        return tool_schemas

    def tool(fn):
        def wrapper(self, *args, **kwargs):
            self.tool_invocations[fn.__name__] = self.tool_invocations.get(fn.__name__, 0) + 1
            try:
                return fn(self, *args, **kwargs)
            except ToolManager.Error as e:
                if fn.__name__ not in self.tool_failure:
                    self.tool_failure[fn.__name__] = {j: 0 for j in ToolManager.Error.ErrorType.__members__}
                self.tool_failure[fn.__name__][e.error_type] += 1
                return e.message

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        wrapper.__signature__ = inspect.signature(fn)
        wrapper.__annotations__ = fn.__annotations__.copy()
        wrapper.is_tool = True
        return wrapper

    def get_tool(self, tool_name: str):
        if tool_name not in self.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found. Available: {', '.join(self.TOOL_LIST.keys())}"
        tool_method = getattr(self, tool_name, None)
        if tool_method is None or not callable(tool_method):
            return f"Error: tool '{tool_name}' does not exist."
        return tool_method

    @classmethod
    def get_tool_args_for_tool(cls, tool_name: str, required_only: bool = False) -> list:
        if tool_name not in cls.TOOL_LIST:
            return []
        if not required_only:
            return list(cls.TOOL_LIST[tool_name]["input_schema"]["properties"].keys())
        return cls.TOOL_LIST[tool_name]["input_schema"]["required"]

    @classmethod
    def get_final_git_patch(cls) -> str:
        try:
            exclude = {"src/agent.py", "src/agent_runner.py", "agent.py"}
            try:
                for _p in getattr(cls, "generated_test_files", []):
                    if os.path.exists(_p) and os.path.isfile(_p):
                        exclude.add(os.path.relpath(_p))
            except Exception:
                pass
            modified = subprocess.run(
                ["git", "ls-files", "-m", "--exclude-standard"], capture_output=True, text=True, timeout=30, check=False
            ).stdout.splitlines()
            untracked = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"], capture_output=True, text=True, timeout=30, check=False
            ).stdout.splitlines()
            to_add = [f for f in (modified + untracked) if f not in exclude]
            if to_add:
                subprocess.run(["git", "add", "--"] + to_add, check=False, timeout=30)
            diff = subprocess.run(
                ["git", "diff", "--cached", "--no-color", "--unified=3"], capture_output=True, text=True, timeout=30, check=False
            )
            return diff.stdout or ""
        except Exception as e:
            return f"Error generating git patch: {e}"

    @classmethod
    def get_modified_files_list(cls) -> list:
        try:
            exclude = {"src/agent.py", "src/agent_runner.py"}
            result = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=M", "HEAD"],
                capture_output=True, text=True, timeout=30, check=False,
            )
            if result.returncode != 0:
                return []
            modified_files = [f.strip() for f in result.stdout.splitlines() if f.strip()]
            modified_files = [f for f in modified_files if f not in exclude]
            final_list = []
            for file_path in modified_files:
                check_result = subprocess.run(
                    ["git", "ls-tree", "--name-only", "HEAD", file_path], capture_output=True, text=True, timeout=10
                )
                if check_result.returncode == 0 and check_result.stdout.strip():
                    final_list.append(file_path)
            return final_list
        except Exception:
            return []

    class Error(Exception):
        class ErrorType(Enum):
            SYNTAX_ERROR = 1
            RUNTIME_ERROR = 2
            TIMEOUT = 3
            FILE_NOT_FOUND = 4
            SEARCH_TERM_NOT_FOUND = 5
            UNKNOWN = 6
            THIRD_PARTY_DEPENDENCIES = 7
            MULTIPLE_SEARCH_RESULTS_FOUND = 8
            BUG_REPORT_REQUIRED = 9
            INVALID_RESPONSE_FORMAT = 10
            INVALID_TOOL_NAME = 11
            INVALID_FILE_PATH = 12
            INVALID_TOOL_CALL = 13
            IMPORT_ERROR = 14

        def __init__(self, error_type, message: str):
            self.error_type = error_type
            self.message = message


class FixTaskToolManager(ToolManager):
    def __init__(
        self,
        available_tools: Optional[list] = None,
        runner_hint: str = None,
        runner_mode_hint: str = None,
        initial_checkpoint=None,
        problem_statement: str = None,
        should_review: bool = True,
        is_fix_task: bool = False,
        initial_structure: str = None,
        function_behaviours: dict = None,
        cot: "COT" = None,
    ):
        self.new_files_created = []
        self.available_tools = available_tools or []
        self.runner_hint = runner_hint
        self.runner_mode_hint = runner_mode_hint
        self.generated_test_files = []
        self.initial_checkpoint = initial_checkpoint
        self.observation_dir = ".observation"
        self.problem_statement = problem_statement
        self.initial_structure = initial_structure
        self.repo_dir = "."
        self.saved_observation_counter = 0
        self.is_fix_task = is_fix_task
        self.strategy_counter = 0
        self.strategies = []
        self.is_reviewed = True  # skip review gate for speed
        self.file_by_file_reviewed = True
        os.makedirs(self.observation_dir, exist_ok=True)
        self.file_ops = FileOperationsUtil(new_files_created=self.new_files_created)
        self.search_manager = SearchManager()
        self.file_system_manager = FileSystemManager()
        self.test_manager = TestManager(runner_hint=runner_hint, runner_mode_hint=runner_mode_hint, file_ops=self.file_ops)
        self.code_edit_manager = CodeEditManager(file_ops=self.file_ops)
        self.code_parser = CodeParseUtil()
        self.file_ops.set_managers(self.file_system_manager, self.search_manager)
        self.TOOL_LIST = {}
        for cls in self.__class__.__mro__:
            for name, attr in cls.__dict__.items():
                if getattr(attr, "is_tool", False) and name not in self.TOOL_LIST:
                    if self.available_tools and name not in self.available_tools:
                        continue
                    self.TOOL_LIST[name] = self.__class__.tool_parsing(attr)
        self.tool_failure = {k: {j: 0 for j in self.Error.ErrorType.__members__} for k in self.TOOL_LIST.keys()}
        self.tool_invocations = {k: 0 for k in self.TOOL_LIST.keys()}
        self.finish_called_count = 0
        self.hypothesis_counter = 0
        self.hypotheses: List[Dict] = []
        self._current_step = 0
        self._cot_snapshot_cache = []
        self.validated_num = 0
        self._test_call_count = 0
        self._pending_run_tests_confirmation = False
        self._last_run_tests_step = None
        self._last_run_tests_passed = None
        self._last_edit_step = None
        self._edit_count = 0
        self._last_blocked_edit_step = None
        self._blocked_edit_count = 0
        self._last_blocked_edit_message = None
        self._last_test_output = ""  # [v3]
        self._rca_complete = False
        self._rca_data = {}
        self.cot = cot
        self.solution_verifier = SolutionVerifier(cot=cot, problem_statement=problem_statement) if cot else None
        self.problem_decomposition = None
        self.fix_strategy = {}
        self.boundary_proofs = {}
        self._last_pre_edit_warning_step = None
        self._pre_edit_warning_count = 0
        self._last_pre_edit_warning_message = None
        self.files_to_fix = []

    def get_final_git_patch(self) -> str:
        try:
            exclude = {"src/agent.py", "src/agent_runner.py", "agent.py"}
            try:
                for _p in getattr(self, "generated_test_files", []):
                    if os.path.exists(_p) and os.path.isfile(_p):
                        exclude.add(os.path.relpath(_p))
            except Exception:
                pass
            modified = subprocess.run(
                ["git", "ls-files", "-m", "--exclude-standard"], capture_output=True, text=True, timeout=30, check=False
            ).stdout.splitlines()
            untracked = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"], capture_output=True, text=True, timeout=30, check=False
            ).stdout.splitlines()
            to_add = [f for f in (modified + untracked) if f not in exclude]
            if to_add:
                subprocess.run(["git", "add", "--"] + to_add, check=False, timeout=30)
            diff = subprocess.run(
                ["git", "diff", "--cached", "--no-color", "--unified=3"], capture_output=True, text=True, timeout=30, check=False
            )
            return diff.stdout or ""
        except Exception as e:
            return f"Error generating git patch: {e}"

    def _save_large_observation(self, observation: str, tool_name: str) -> tuple:
        self.saved_observation_counter += 1
        filename = f"observation_{self.saved_observation_counter}_{tool_name}_{int(time.time())}.txt"
        if not os.path.exists(self.observation_dir):
            os.makedirs(self.observation_dir, exist_ok=True)
        file_path = os.path.join(self.observation_dir, filename)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(observation)
            line_count = observation.count("\n") + 1 if observation else 0
            return file_path, line_count
        except Exception as e:
            return f"Error: Failed to save observation: {e}", -1

    def _summarize_test_output(self, test_output: str) -> str:
        prompt = textwrap.dedent(
            f"""
            Summarize this test execution output. Focus on:
            1. Total tests run, passed, and failed counts
            2. List ALL failed test cases with their exact names
            3. For each failure: exact important short error message, location (file:line), and root cause
            4. Critical error traces
            Keep all specific error details sufficient for debugging.
            Test Output:
            {test_output}
            Provide a concise but complete summary:"""
        )
        messages = [{"role": "user", "content": prompt}]
        retry = 0
        selected_model = QWEN_MODEL_NAME
        while retry < 10:
            try:
                summary, _ = Network.make_request(messages=messages, model=selected_model, where="summarize_test_output")
                return f"[TEST OUTPUT SUMMARIZED]\n\n{summary}"
            except Exception:
                pass
            retry += 1
            if retry > 5:
                selected_model = Exchange_Model(selected_model)
            time.sleep(0.5)
        lines = test_output.split("\n")
        if len(lines) > 200:
            return f"[TEST OUTPUT TRUNCATED]\n\n" + "\n".join(lines[:100]) + f"\n\n... ({len(lines)-200} lines omitted) ...\n\n" + "\n".join(lines[-100:])
        return test_output

    def validate_patch_application(self) -> str:
        """Quick patch validation."""
        try:
            subprocess.run(["git", "reset", "HEAD"], capture_output=True, text=True, timeout=10, check=False)
            status_result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=30, check=False)
            if status_result.returncode != 0 or not status_result.stdout.strip():
                return "Patch validation skipped: No modified files to validate"
            return "Patch validation passed"
        except Exception as e:
            return f"Patch validation skipped: {e}"

    @ToolManager.tool
    def apply_code_edit(self, file_path: str, search: str, replace: str) -> str:
        """
        Performs targeted text replacement within source files.
        Arguments:
            file_path: target file for modification
            search: exact text pattern to locate and replace
            replace: new text content to substitute
        Output:
            operation status - success confirmation or detailed error with guidance
        """
        result = self.code_edit_manager.apply_code_edit(file_path=file_path, search=search, replace=replace)
        try:
            if isinstance(result, str) and "ok, code edit applied successfully" in result.lower():
                self._last_edit_step = self._current_step
                self._edit_count += 1
        except Exception:
            pass
        return result

    @ToolManager.tool
    def modify_test_case(self, file_path: str, search: str, replace: str) -> str:
        """
        Modifies test files or test cases when they are incorrect or need correction.
        Arguments:
            file_path: path to the test file that needs modification
            search: exact text pattern in the test file to locate and replace
            replace: corrected test case code to substitute
        Output:
            Operation status - success confirmation or detailed error with guidance
        """
        return self.code_edit_manager.apply_code_edit(file_path=file_path, search=search, replace=replace)

    @ToolManager.tool
    def run_code(self, content: str, file_path: str, run_command: List[str]) -> str:
        """
        Runs any code. Saves the code at the given file_path and then runs it.
        Arguments:
            content: text code to write in file
            file_path: path of the file to save the code in
            run_command: command to run the file (i.e., ["python", "file.py"] or ["node", "file.js"] etc)
        """
        return self.test_manager.run_code(
            content=content,
            file_path=file_path,
            generated_test_files=self.generated_test_files,
            run_command=run_command,
        )

    @ToolManager.tool
    def run_tests(self, command: List[str], timeout: int = 120) -> str:
        """
        Runs tests with strict timeout.
        Arguments:
            command: list of command line arguments
            timeout: timeout in seconds (default: 120 — use 120 or higher for real test suites like pytest)
        Output:
            Standard output or error output of the command.
        """
        if self.is_fix_task and self._test_call_count == 0 and not self._pending_run_tests_confirmation:
            self._test_call_count += 1
            self._pending_run_tests_confirmation = True
            return textwrap.dedent(
                f"""
            ⚠️  VERIFICATION WORKFLOW DISCOVERY CHECK ⚠️

            You are about to run tests for the first time with command: {' '.join(command)}

            Before proceeding, you MUST confirm you have completed the mandatory discovery steps:
            ✓ Examined repository root structure for verification entry scripts?
            ✓ Inspected project documentation for test execution instructions?
            ✓ Determined the canonical execution path with proper priority?
            ✓ Checked whether `tests/runtests.py` exists (Django projects) — if yes, use `python tests/runtests.py --verbosity=2` NOT bare pytest (which fails with ImproperlyConfigured)?

            If you HAVE completed discovery and verified this is the correct command:
            - Call run_tests again with the same command to proceed

            This confirmation only appears once. Subsequent run_tests calls will execute immediately.
            """
            ).strip()

        if self._pending_run_tests_confirmation:
            self._pending_run_tests_confirmation = False

        try:
            preface_lines = []
            if self.is_fix_task:
                try:
                    if self._last_blocked_edit_step is not None and (
                        self._last_edit_step is None or self._last_blocked_edit_step > self._last_edit_step
                    ):
                        preface_lines.append(
                            "⚠️ NOTE: Your most recent code edit attempt was blocked. "
                            "This test run will execute against the last successfully applied code state."
                        )
                    if self._last_run_tests_step is not None and (self._last_edit_step is None or self._last_edit_step <= self._last_run_tests_step):
                        preface_lines.append(
                            "ℹ️ NOTE: No new successful code edits have been applied since the last test run."
                        )
                except Exception:
                    pass
            # [v3] Django auto-detect: redirect bare pytest → tests/runtests.py
            def _maybe_rewrite_django_command(cmd):
                """If repo has tests/runtests.py and command looks like bare pytest, use Django runner."""
                if not cmd:
                    return cmd
                is_pytest = (
                    cmd[0] in ('pytest', 'py.test') or
                    (len(cmd) >= 2 and cmd[0] == 'python' and cmd[1] == '-m' and 'pytest' in cmd) or
                    (len(cmd) >= 3 and cmd[0:3] == ['python', '-m', 'pytest'])
                )
                if not is_pytest:
                    return cmd
                runtests = os.path.join(os.getcwd(), 'tests', 'runtests.py')
                if not os.path.exists(runtests):
                    return cmd
                # Sniff DJANGO_SETTINGS_MODULE from setup.cfg / tox.ini
                settings = None
                for cfg in ['setup.cfg', 'tox.ini']:
                    cfg_path = os.path.join(os.getcwd(), cfg)
                    if os.path.exists(cfg_path):
                        with open(cfg_path) as f:
                            for line in f:
                                if 'DJANGO_SETTINGS_MODULE' in line and '=' in line:
                                    settings = line.split('=', 1)[1].strip()
                                    break
                    if settings:
                        break
                if settings:
                    os.environ.setdefault('DJANGO_SETTINGS_MODULE', settings)
                return ['python', runtests, '--verbosity=2']
            _cmd = _maybe_rewrite_django_command(command)
            result = subprocess.run(_cmd, capture_output=True, text=True, timeout=timeout)
            test_output = result.stdout + result.stderr
            try:
                self._last_run_tests_step = self._current_step
                self._last_run_tests_passed = result.returncode == 0
                self._last_test_output = test_output  # [v3]
            except Exception:
                pass
            token_count = Utils.count_tokens(test_output)
            if token_count > SAVE_OBSERVATION_TO_FILE_TOKEN_THRESHOLD:
                test_output = self._summarize_test_output(test_output)
            if preface_lines:
                return "\n".join(preface_lines).strip() + "\n\n" + test_output
            return test_output
        except subprocess.TimeoutExpired:
            return "Test run timed out."
        except Exception as e:
            return f"Test execution error: {e}"

    @ToolManager.tool
    def get_file_content(self, file_path: str, search_start_line: int = None, search_end_line: int = None, search_term: str = None) -> str:
        """
        Retrieves file contents with optional filtering based on search term and line numbers
        Arguments:
            file_path: filesystem path to target file.
            search_start_line: optional start line number to begin extraction (1-indexed)
            search_end_line: optional end line number to end extraction (1-indexed)
            search_term: optional text pattern to filter matching lines
        """
        return self.file_ops.get_file_content(
            file_path,
            search_start_line,
            search_end_line,
            search_term,
            add_line_numbers=True,
            limit=1000,
        )

    @ToolManager.tool
    def list_directory_structure(self, directory_path: str = ".", max_depth: int = 1) -> str:
        """
        Lists the directory structure of the repository
        Arguments:
            directory_path: the directory path to list (default: ".")
            max_depth: maximum depth to traverse (default: 1)
        """
        return self.file_system_manager.list_directory_structure(directory_path=directory_path, max_depth=max_depth)

    @ToolManager.tool
    def grep_search(self, grep_search_command: str) -> str:
        """
        Performs grep search on a single file or across multiple files in the codebase
        Arguments:
            grep_search_command: grep search command to locate (e.g., "grep <your grep command>").
        Output:
            locations where pattern was found with file paths and line numbers
        """
        return self.search_manager.search_in_all_files(grep_search_command)

    @ToolManager.tool
    def search_in_file(self, file_path: str, search_term: str) -> str:
        """
        Locates text patterns within a specific file
        Arguments:
            file_path: target file for pattern matching.
            search_term: text pattern to find
        Output:
            matching locations with line numbers, or error description
        """
        return self.search_manager.search_in_file(file_path=file_path, search_term=search_term)

    @ToolManager.tool
    def get_function_body(self, file_path: str, function_name: str) -> str:
        """
        Retrieves the complete body of a function from a file, including decorators.
        Arguments:
            file_path: filesystem path to target file.
            function_name: name of the function to retrieve
        Output:
            The complete function body including decorators, or empty string if not found.
        """
        if not hasattr(self, "code_parser"):
            self.code_parser = CodeParseUtil()
        return self.code_parser.get_function_body(file_path, function_name, add_line_numbers=True)

    @ToolManager.tool
    def find_symbol_references(self, symbol_identifier: str) -> str:
        """
        Discovers all code locations where a specific function, class, method, or variable is referenced.
        Arguments:
            symbol_identifier: exact name of the function, class, method, or variable to locate
        Output:
            comprehensive listing of files and line numbers with surrounding context
        """
        try:
            cmd = f"grep -rn --binary-files=without-match '{symbol_identifier}' . | head -100"
            result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=30)
            refs = result.stdout.strip()
            if not refs:
                return f"No references discovered for symbol '{symbol_identifier}' in the codebase."
            lines = refs.split("\n")
            if len(lines) > 50:
                return f"Found {len(lines)} references for '{symbol_identifier}' (showing first 50):\n\n" + "\n".join(lines[:50])
            return f"References for '{symbol_identifier}' ({len(lines)} found):\n{refs}"
        except subprocess.TimeoutExpired:
            return f"Search timeout for symbol '{symbol_identifier}'. Try a more specific identifier."
        except Exception as e:
            return f"Error locating symbol references: {str(e)}"

    @ToolManager.tool
    def think(self, thought: str) -> str:
        """Use the tool to think about something. It will not make any changes to the repository.
        Arguments:
            thought: Your thoughts.
        Output:
            Confirmation that the thought has been logged.
        """
        return "ok"

    @ToolManager.tool
    def create_new_file(self, file_path: str, content: str, overwrite: bool = False) -> str:
        """
        Creates a new file with the specified content.
        Arguments:
            file_path: Path where the new file should be created.
            content: The content to write into the file.
            overwrite: If True, will overwrite the file if it exists.
        Output:
            Status message indicating success or error.
        """
        if os.path.exists(file_path) and not overwrite:
            return f"Error: File '{file_path}' already exists. Set overwrite=True to overwrite."
        try:
            parent_dir = os.path.dirname(file_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            if hasattr(self, "file_ops") and hasattr(self.file_ops, "new_files_created"):
                self.file_ops.new_files_created.append(file_path)
            return f"File '{file_path}' created successfully."
        except Exception as e:
            return f"Error creating file '{file_path}': {e}"

    @ToolManager.tool
    def generate_test_cases_from_root_cause(self, root_cause_code: str, file_path: str = None, function_name: str = None) -> str:
        """
        Generates comprehensive test cases based on the problem statement and the identified root cause code section.
        Arguments:
            root_cause_code: The code section identified as the root cause of the issue (required)
            file_path: Optional file path where the root cause code is located
            function_name: Optional function name where the root cause code is located
        Output:
            A structured markdown document containing test cases
        """
        if not self.problem_statement:
            return "Error: Problem statement not available."
        TEST_CASE_PROMPT = textwrap.dedent(
            """
            You are an expert test case generator. Generate comprehensive test cases based on a problem statement and root cause code section.
            Cover: bug reproduction, edge cases, boundary conditions.
            Format as structured markdown. Be specific and actionable.
            """
        )
        root_cause_context = root_cause_code
        if file_path:
            root_cause_context += f"\n\nFile: {file_path}"
        if function_name:
            root_cause_context += f"\n\nFunction: {function_name}"
        retry = 0
        selected_model = QWEN_MODEL_NAME
        while retry < 10:
            try:
                messages = [
                    {"role": "system", "content": TEST_CASE_PROMPT},
                    {"role": "user", "content": f"Problem Statement:\n{self.problem_statement}\n\nRoot Cause Code:\n{root_cause_context}\n\nGenerate comprehensive test cases."},
                ]
                test_cases, _ = Network.make_request(messages, model=selected_model, attempt=1, temperature=0.0, where="generate_test_cases")
                self.generated_test_cases = test_cases
                return f"Test cases generated successfully.\n\n{test_cases}"
            except Exception as e:
                retry += 1
                if retry > 5:
                    selected_model = Exchange_Model(selected_model)
                time.sleep(1)
        return "Error: Failed to generate test cases"

    @ToolManager.tool
    def analyze_edge_cases(self, file_contents: Dict[str, str], target_identifier: str = None) -> str:
        """
        Analyzes provided code to identify edge cases, boundary conditions, and implicit requirements.
        Arguments:
            file_contents: dictionary mapping file paths/identifiers to their content
            target_identifier: Optional identifier to focus analysis
        Output:
            Comprehensive analysis identifying edge cases and implicit requirements
        """
        if not file_contents or not isinstance(file_contents, dict):
            return "Error: file_contents must be a dictionary mapping file identifiers to content."
        context_parts = []
        for file_id, content in list(file_contents.items())[:10]:
            if not content or not isinstance(content, str):
                continue
            if len(content) > 50000:
                content = content[:50000] + "\n... [content truncated]"
            context_parts.append(f"=== {file_id} ===\n{content}")
        if not context_parts:
            return "Error: No valid file content provided."
        messages = [
            {"role": "system", "content": "You are an expert code analyzer specializing in identifying edge cases, implicit requirements, and specification mismatches."},
            {"role": "user", "content": f"Target: {target_identifier or 'General analysis'}\n\nContent:\n" + "\n".join(context_parts)},
        ]
        retry = 0
        selected_model = QWEN_MODEL_NAME
        while retry < 5:
            try:
                analysis, _ = Network.make_request(messages, model=selected_model, attempt=1, temperature=0.0, where="analyze_edge_cases")
                return f"=== Edge Case Analysis ===\n\n{analysis}"
            except Exception:
                retry += 1
                selected_model = Exchange_Model(selected_model)
                time.sleep(1)
        return "Error: Failed to analyze edge cases"

    @ToolManager.tool
    def log_strategy(self, approach: str, reasoning: str) -> str:
        """Record a high-level strategy before attempting it.
        Arguments:
            approach: Brief description of the approach
            reasoning: Why you think this will work
        Output:
            Confirmation with strategy ID for later reference.
        """
        self.strategy_counter += 1
        strategy = {
            "id": self.strategy_counter,
            "approach": approach,
            "reasoning": reasoning,
            "success": None,
            "reason": None,
            "timestamp": time.time(),
        }
        self.strategies.append(strategy)
        return f"Strategy #{self.strategy_counter} logged: {approach}\nReasoning: {reasoning}\nUse mark_strategy_outcome to record results."

    @ToolManager.tool
    def mark_strategy_outcome(self, strategy_id: int, success: bool, reason: str) -> str:
        """Record whether a strategy worked.
        Arguments:
            strategy_id: ID from log_strategy (e.g., 1, 2, 3)
            success: True if approach worked, False otherwise
            reason: Why it succeeded/failed
        Output:
            Updated strategy status.
        """
        for strat in self.strategies:
            if strat["id"] == strategy_id:
                strat["success"] = success
                strat["reason"] = reason
                status = "SUCCEEDED" if success else "FAILED"
                return f"Strategy #{strategy_id} marked as {status}\nReason: {reason}"
        return f"Error: Strategy #{strategy_id} not found"

    @ToolManager.tool
    def list_attempted_strategies(self) -> str:
        """View all strategies tried, with outcomes.
        Arguments:
            None
        Output:
            Formatted list of all strategies with outcomes.
        """
        if not self.strategies:
            return "No strategies recorded yet. Use log_strategy before attempting significant changes."
        output = ["=== STRATEGY HISTORY ===\n"]
        succeeded = [s for s in self.strategies if s["success"] is True]
        failed = [s for s in self.strategies if s["success"] is False]
        pending = [s for s in self.strategies if s["success"] is None]
        output.append(f"Summary: {len(succeeded)} succeeded, {len(failed)} failed, {len(pending)} pending\n")
        for status, strategies in [("SUCCEEDED", succeeded), ("FAILED", failed), ("PENDING", pending)]:
            if strategies:
                output.append(f"\n{status}:")
                for s in strategies:
                    output.append(f"\n  [{s['id']}] {s['approach']}")
                    output.append(f"      Reasoning: {s['reasoning']}")
                    if s["reason"]:
                        output.append(f"      Outcome: {s['reason']}")
        return "\n".join(output)

    @ToolManager.tool
    def create_hypothesis(self, description: str, evidence: str) -> str:
        """Create a hypothesis about the bug's root cause.
        Arguments:
            description: What you think is causing the bug
            evidence: What evidence supports this theory
        Output:
            Confirmation with hypothesis ID for tracking.
        """
        self.hypothesis_counter += 1
        hypothesis = {
            "id": self.hypothesis_counter,
            "description": description,
            "evidence": evidence,
            "status": "untested",
            "findings": None,
            "created_step": self._current_step,
            "tested_step": None,
            "timestamp": time.time(),
        }
        self.hypotheses.append(hypothesis)
        return f"Hypothesis #{self.hypothesis_counter} created: {description}\nEvidence: {evidence}\nStatus: untested"

    @ToolManager.tool
    def test_hypothesis(self, hypothesis_id: int, outcome: str, findings: str) -> str:
        """Record the result of testing a hypothesis.
        Arguments:
            hypothesis_id: ID from create_hypothesis
            outcome: One of 'confirmed', 'rejected', or 'inconclusive'
            findings: What you discovered
        Output:
            Updated hypothesis status.
        """
        if outcome not in ["confirmed", "rejected", "inconclusive"]:
            return f"Error: outcome must be 'confirmed', 'rejected', or 'inconclusive', got '{outcome}'"
        for hyp in self.hypotheses:
            if hyp["id"] == hypothesis_id:
                hyp["status"] = outcome
                hyp["findings"] = findings
                hyp["tested_step"] = self._current_step
                status_emoji = {"confirmed": "✅", "rejected": "❌", "inconclusive": "❓"}.get(outcome, "")
                return f"{status_emoji} Hypothesis #{hypothesis_id} marked as {outcome.upper()}\nFindings: {findings}"
        return f"Error: Hypothesis #{hypothesis_id} not found"

    @ToolManager.tool
    def list_hypotheses(self) -> str:
        """View all hypotheses with their test status.
        Arguments:
            None
        Output:
            Formatted list of all hypotheses with status and findings.
        """
        if not self.hypotheses:
            return "No hypotheses recorded yet."
        output = ["=== HYPOTHESIS TRACKER ===\n"]
        untested = [h for h in self.hypotheses if h["status"] == "untested"]
        confirmed = [h for h in self.hypotheses if h["status"] == "confirmed"]
        rejected = [h for h in self.hypotheses if h["status"] == "rejected"]
        inconclusive = [h for h in self.hypotheses if h["status"] == "inconclusive"]
        output.append(f"Summary: {len(confirmed)} confirmed, {len(rejected)} rejected, {len(inconclusive)} inconclusive, {len(untested)} untested\n")
        for status, hypotheses in [("✅ CONFIRMED", confirmed), ("❌ REJECTED", rejected), ("❓ INCONCLUSIVE", inconclusive), ("🔍 UNTESTED", untested)]:
            if hypotheses:
                output.append(f"\n{status}:")
                for h in hypotheses:
                    output.append(f"\n  [{h['id']}] {h['description']}")
                    output.append(f"      Evidence: {h['evidence']}")
                    if h['findings']:
                        output.append(f"      Findings: {h['findings']}")
        return "\n".join(output)

    @ToolManager.tool
    def run_shell_cmd(self, command: str) -> str:
        '''
        Runs shell commands for the repository. This tool executes shell commands directly.
        Arguments:
            command: A shell command to be run.
        Output:
            The stdout results of the command. Your working directory is the root of the project.
        '''
        if not command:
            return "Error: No command provided."

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                timeout=150
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
            return output
        except subprocess.TimeoutExpired:
            return f"Error: Command '{command}' timed out after 150 seconds"
        except Exception as e:
            return f"Error running command: {str(e)}"

    @ToolManager.tool
    def finish_find_files_to_fix(self, files: List[str]):
        """
        Signals completion of the file finding workflow execution
        Arguments:
            files: The list of files to fix.
        """
        self.files_to_fix = files
        return files

    @ToolManager.tool
    def finish_root_cause_analysis(self, root_cause: str, files_to_edit: list, confidence: str = "high") -> str:
        """
        Signal that root cause analysis is complete. MUST be called before any file edits.
        Arguments:
            root_cause: confirmed root cause description (1-2 sentences)
            files_to_edit: list of file paths that need to be modified
            confidence: confidence level - high/medium/low
        Output:
            Confirmation that RCA is complete and editing is now allowed.
        """
        self._rca_complete = True
        self._rca_data = {"root_cause": root_cause, "files": files_to_edit, "confidence": confidence}
        logger.info(f"✅ [RCA] Complete — files: {files_to_edit} confidence: {confidence}")
        return f"RCA complete. Root cause confirmed. Ready to edit: {files_to_edit}"

    @ToolManager.tool
    def finish(self):
        """
        Signals completion of the current workflow execution. Validates solution before finishing.
        Arguments:
            None
        Output:
            Review patch prompt with validation results, or "finish" if all checks pass
        """
        if self.is_fix_task:
            if self._last_run_tests_step is None:
                return textwrap.dedent(
                    """
                    ⚠️ VERIFICATION REQUIRED - Cannot Finish Yet

                    You have not executed `run_tests`. Run the repository-defined verification workflow, ensure it passes, then call `finish` again.
                    """
                ).strip()
            if self._last_edit_step is not None and self._last_run_tests_step < self._last_edit_step:
                return textwrap.dedent(
                    """
                    ⚠️ VERIFICATION REQUIRED - Cannot Finish Yet

                    You edited code after your last verification run. Run `run_tests` again after the last edit and ensure it passes, then call `finish`.
                    """
                ).strip()
            if self._last_run_tests_passed is False:
                # [v3] Allow finish if failure is a framework config error, not a fix failure
                _framework_errors = (
                    'ImproperlyConfigured',
                    'django.core.exceptions',
                    'DJANGO_SETTINGS_MODULE',
                    "Apps aren't loaded yet",
                )
                _last_out = getattr(self, '_last_test_output', '')
                if not any(e in _last_out for e in _framework_errors):
                    return textwrap.dedent(
                        """
                        ⚠️ VERIFICATION FAILED - Cannot Finish Yet

                        Your latest verification run did not pass. Fix the failures, re-run `run_tests`, then call `finish`.
                        """
                    ).strip()

        # Validate patch
        validation_result = self.validate_patch_application()
        if "Patch validation passed" not in validation_result:
            return f"⚠️ Patch Validation Issue:\n{validation_result}\n\nPlease review and fix before calling finish."

        if self.is_fix_task and self.solution_verifier:
            regression_review = self.solution_verifier.verify_solution()
            if "REGRESSION_AND_BUG_CHECK_PASSED" in regression_review:
                return "finish"
            else:
                return textwrap.dedent(
                    f"""
                    ⚠️ VERIFICATION FAILED - Cannot Finish Yet

                    Your solution is not ready. Please address:

                    {regression_review}

                    REQUIRED: Fix BOTH the original bug AND all regressions, then call `finish` again.
                    """
                ).strip()

        return "finish"


def _safe_call_tool(tool_manager, tool_name: str, tool_args):
    tool_fn = tool_manager.get_tool(tool_name)
    if isinstance(tool_fn, str):
        return tool_fn
    if tool_args is None or tool_args == {}:
        return tool_fn()
    if not isinstance(tool_args, dict):
        return tool_fn()
    try:
        sig = inspect.signature(tool_fn)
        allowed = set(sig.parameters.keys())
        allowed.discard("self")
    except Exception:
        allowed = set(tool_args.keys())
    cleaned = {k: v for k, v in tool_args.items() if k in allowed}
    try:
        for k in list(cleaned.keys()):
            v = cleaned[k]
            p = sig.parameters.get(k)
            ann = str(getattr(p, "annotation", ""))
            if v is not None and isinstance(v, str) and ("List" in ann or "list" in ann):
                cleaned[k] = v.split() if v.strip() else []
    except Exception:
        pass
    return tool_fn(**cleaned) if cleaned else tool_fn()


def execute_agent_workflow(
    cot: COT,
    tool_manager: FixTaskToolManager,
    system_prompt: str,
    instance_prompt: str,
    n_max_steps: int,
    timeout: int,
    models: List[Model],
    finish_tool_name: str = "finish",
    cost_limit: float = 1.0,
    cost_usage_threshold: float = 0.15,
    reject_observation_token_threshold: int = REJECT_OBSERVATION_TOKEN_THRESHOLD,
    save_observation_to_file_token_threshold: int = SAVE_OBSERVATION_TO_FILE_TOKEN_THRESHOLD,
    budget: 'TimeBudget' = None,
    memory: 'InvestigationMemory' = None,
) -> tuple:
    global run_id
    start_time = time.time()
    raw_text = ""
    total_attempts = 0
    error_counter = {}
    next_thought = next_tool_name = next_tool_args = None
    modified_files = set()
    current_model_index = 0

    consecutive_no_tool = 0
    for step in range(n_max_steps):
        if _watchdog_abort.is_set():
            logger.warning(f"⚠️ [WATCHDOG] Abort flag set at step {step} — returning partial diff")
            break
        if budget is not None and budget.is_expired():
            logger.warning(f"⏰ [BUDGET] TimeBudget expired at step {step} — returning partial diff")
            break
        cost_usage = Network.get_cost_usage()
        if cost_usage.get("used_cost_usd", 0) > cost_usage.get("max_cost_usd", float("inf")) * cost_limit - cost_usage_threshold:
            logger.info("Cost limit reached, stopping")
            break
        if time.time() - start_time > timeout:
            cot.add_action(
                COT.Action(
                    next_thought="global timeout reached",
                    next_tool_name="",
                    next_tool_args={},
                    observation="",
                    is_error=True,
                    inference_error_counter={},
                    request_data=[],
                )
            )
            break

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt},
        ]
        messages.extend(cot.to_str())
        messages.append({"role": "system", "content": STOP_INSTRUCTION})
        if memory is not None:
            mem_block = memory.to_prompt_block()
            if mem_block:
                messages.append({"role": "system", "content": mem_block})
        if step >= 4 and not modified_files:
            logger.warning(f"⚠️ [FORCE-EDIT] No file edits after {step + 1} steps — injecting forced-edit warning")
            urgency = "CRITICAL" if step >= 7 else "WARNING"
            messages.append({
                "role": "user",
                "content": (
                    f"⚠️ {urgency}: You have made {step + 1} tool calls with NO file edits. "
                    "You MUST make a file change in this step. "
                    "Use `apply_code_edit` to modify an existing source file, OR "
                    "`create_new_file(file_path, full_content, overwrite=True)` to replace it. "
                    "Do NOT call run_code, read_file, or grep_search — only apply_code_edit or create_new_file is allowed now. "
                    "An empty diff scores ZERO. An imperfect fix scores > 0. Act NOW."
                ),
            })

        selected_model = models[current_model_index % len(models)]
        temperature = 0.0

        if cot.is_thought_repeated():
            last_thought = cot.thoughts[-1]
            messages.append(
                {
                    "role": "user",
                    "content": DO_NOT_REPEAT_TOOL_CALLS.format(
                        previous_response=f"next_thought:{last_thought.next_thought}\n next_tool_name:{last_thought.next_tool_name}\n next_tool_args:{last_thought.next_tool_args}"
                    ),
                }
            )
            temperature = 0.5
            if cot.repeated_thoughts >= 2:
                model_idx = (cot.repeated_thoughts - 2) % len(models)
                selected_model = models[model_idx]

        # Update cot reference in tool_manager
        if hasattr(tool_manager, "is_fix_task") and tool_manager.is_fix_task:
            if not tool_manager.cot:
                tool_manager.cot = cot
                tool_manager.solution_verifier = SolutionVerifier(cot=cot, problem_statement=tool_manager.problem_statement)
            elif tool_manager.cot != cot:
                tool_manager.cot = cot
                if tool_manager.solution_verifier:
                    tool_manager.solution_verifier.cot = cot

        try:
            models_to_try = [selected_model] + [m for m in models if m != selected_model]
            (
                next_thought,
                next_tool_name,
                next_tool_args,
                raw_text,
                total_attempts,
                error_counter,
                messages,
                used_model,
            ) = Network.inference(messages, model=models_to_try, run_id=run_id, temperature=temperature)
            logger.info(f"Step {step}: tool={next_tool_name}")
        except Exception as e:
            logger.error(f"Inference error at step {step}: {e}")
            continue

        if next_tool_name is None:
            consecutive_no_tool += 1
            if consecutive_no_tool >= 3:
                logger.warning(f"[T68] 3 consecutive no-tool-call responses at step {step} — breaking loop to return partial patch")
                break
        else:
            consecutive_no_tool = 0

        tool_names_list = next_tool_name if isinstance(next_tool_name, list) else [next_tool_name]
        tool_args_list = next_tool_args if isinstance(next_tool_args, list) else [next_tool_args]

        tool_manager._current_step = step
        tool_manager._cot_snapshot_cache = [
            {"thought": t.next_thought, "tool": t.next_tool_name, "args": str(t.next_tool_args)[:200], "success": not t.is_error}
            for t in cot.thoughts[-10:]
        ]

        if tool_names_list and hasattr(tool_manager, '_rca_complete'):
            if not tool_manager._rca_complete and step < 5:
                edit_tools = {"apply_code_edit", "create_new_file", "modify_test_case"}
                blocked = [t for t in tool_names_list if t in edit_tools]
                if blocked:
                    logger.warning(f"⛔ [RCA GATE] Blocking {blocked} at step {step} — RCA not complete")
                    _rca_observation = (
                        "⛔ You must call `finish_root_cause_analysis(root_cause, files_to_edit)` first. "
                        "Identify the root cause and which files to edit before making changes."
                    )
                    cot.add_action(COT.Action(
                        next_thought=next_thought,
                        next_tool_name=next_tool_name,
                        next_tool_args=next_tool_args,
                        observation=_rca_observation,
                        is_error=True,
                        inference_error_counter={},
                        request_data=messages,
                    ))
                    continue

        all_observations = []
        all_successful = True
        for idx, (tool_name, tool_args) in enumerate(zip(tool_names_list, tool_args_list)):
            try:
                if tool_name and ('"' in tool_name or "'" in tool_name):
                    tool_name = tool_name.replace('"', "").replace("'", "")
                observation = _safe_call_tool(tool_manager, tool_name, tool_args)
                if tool_name == "apply_code_edit" and tool_args and "file_path" in tool_args:
                    if "ok, code edit applied successfully" in str(observation).lower():
                        modified_files.add(tool_args["file_path"])
                estimated_tokens = Utils.count_tokens(str(observation))
                if estimated_tokens > reject_observation_token_threshold:
                    observation = (
                        f"Error: Tool output from '{tool_name}' exceeded token limit ({estimated_tokens} tokens). "
                        f"Please use more specific queries."
                    )
                elif estimated_tokens > save_observation_to_file_token_threshold:
                    observation_path, line_count = tool_manager._save_large_observation(str(observation), tool_name)
                    observation = (
                        f"Tool output from `{tool_name}` exceeded token limit ({estimated_tokens} tokens). "
                        f"Full output saved to: {observation_path}. The file has {line_count} lines."
                    )
                all_observations.append(observation)
            except ToolManager.Error as e:
                all_observations.append(f"Tool {idx+1} ({tool_name}) error: {e.message}")
                all_successful = False
            except Exception as e:
                all_observations.append(f"Tool {idx+1} ({tool_name}) exception: {str(e)}\n{traceback.format_exc()}")
                all_successful = False

        for _tm_name, _tm_args, _tm_obs in zip(tool_names_list, tool_args_list, all_observations):
            if memory is not None:
                if _tm_name in ("get_file_content", "get_function_body") and isinstance(_tm_args, dict) and _tm_args.get("file_path"):
                    fp = _tm_args["file_path"]
                    if fp not in memory.files_examined:
                        memory.files_examined.append(fp)
                elif _tm_name == "grep_search" and "ok" in str(_tm_obs).lower():
                    memory.call_sites_found.append(str(_tm_args.get("query", "") if isinstance(_tm_args, dict) else "")[:50])

        # Check for finish tool
        if finish_tool_name in tool_names_list:
            for name, obs in zip(tool_names_list, all_observations):
                if name == finish_tool_name and obs == "finish":
                    return tool_manager.get_final_git_patch(), True

        if len(all_observations) == 1:
            combined_observation = all_observations[0]
        else:
            combined_observation = "\n\n--- Tool Call Results ---\n" + "\n\n".join(
                [f"Tool {i+1} ({tool_names_list[i]}):\n{obs}" for i, obs in enumerate(all_observations)]
            )

        cot.add_action(
            COT.Action(
                next_thought=next_thought,
                next_tool_name=next_tool_name,
                next_tool_args=next_tool_args,
                observation=combined_observation,
                is_error=not all_successful,
                raw_response=raw_text,
                total_attempts=total_attempts,
                inference_error_counter=error_counter,
                request_data=messages,
            )
        )

    return tool_manager.get_final_git_patch(), False


def set_env_for_agent():
    logger.debug("Setting up environment for agent")
    work_dir = os.getcwd()
    original_cwd = work_dir
    pythonpath = os.environ.get("PYTHONPATH", "")
    if work_dir not in pythonpath.split(":"):
        os.environ["PYTHONPATH"] = f"{work_dir}:{pythonpath}"
    lib_dir = os.path.join(work_dir, "lib")
    if os.path.exists(lib_dir) and lib_dir not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] += f":{lib_dir}"
    try:
        with open(os.path.join(work_dir, "sitecustomize.py"), "w") as f:
            f.write(VERSION_COMPATIBILITY_FIX)
    except Exception:
        pass
    try:
        os.chdir(work_dir)
        if not os.path.exists(".git"):
            subprocess.run(["git", "init"], check=True, capture_output=True)
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir], capture_output=True)
            subprocess.run(["git", "config", "--global", "user.email", "agent@sandbox.local"], check=True, capture_output=True)
            subprocess.run(["git", "config", "--global", "user.name", "sandbox_agent"], check=True, capture_output=True)
            subprocess.run(["git", "add", "."], check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "Initial commit"], check=False, capture_output=True)
        else:
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir], capture_output=True)
    except Exception as e:
        logger.warning(f"Error setting up environment: {e}")
    finally:
        os.chdir(original_cwd)


def select_best_solution(solutions: List[dict], problem_statement: str) -> dict:
    if not solutions:
        return None
    if len(solutions) == 1:
        return solutions[0]
    SELECT_BEST_PROMPT = textwrap.dedent(
        """
        You are an expert code reviewer. Select the SINGLE best solution.
        Evaluate on: correctness, edge case coverage, logical soundness.
        Return ONLY valid JSON: {"selected_index": <0-based int>, "reasoning": "<why>"}
        """
    )
    solutions_context = ""
    for i, sol in enumerate(solutions):
        code = sol.get("solution_code", "") or sol.get("patch", "")
        if len(code) > 60000:
            code = code[:60000]
        solutions_context += f"\n\n=========== SOLUTION {i} ===========\n```\n{code}\n```\n\nSummary: {sol.get('summary', '')}"
    retry = 0
    selected_model = QWEN_MODEL_NAME
    while retry < 10:
        try:
            messages = [
                {"role": "system", "content": SELECT_BEST_PROMPT},
                {"role": "user", "content": f"Problem:\n{problem_statement}\n\n{solutions_context}\n\nSelect best."},
            ]
            result = Network.make_request(messages, model=selected_model, temperature=0.0, where="select_best_solution")
            response_text = result[0] if isinstance(result, tuple) else result
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                selection = json.loads(json_match.group())
                selected_index = selection.get("selected_index", 0)
                if 0 <= selected_index < len(solutions):
                    return solutions[selected_index]
        except Exception:
            pass
        retry += 1
        if retry > 5:
            selected_model = Exchange_Model(selected_model)
        time.sleep(2)
    return solutions[0]


def fix_task_solve_workflow(
    problem_statement: str,
    *,
    timeout: int,
    run_id_1: str,
    enhancement: str = "",
    n_max_steps: int = MAX_FIX_TASK_STEPS,
    root_cause_analysis: str = None,
):
    repo_path = os.getenv("REPO_PATH", "/sandbox/repo")
    repo_dir = repo_path.split("/")[-1]
    if os.path.exists(repo_dir):
        os.chdir(repo_dir)

    set_env_for_agent()

    global run_id, _current_tool_manager
    run_id = run_id_1
    logger.info(f"🆔 [WORKFLOW] Run ID: {run_id}")

    # Problem decomposition
    decomposition = None
    decomposition_text = ""
    try:
        decomposition = _problem_decomposer.decompose(problem_statement)
        decomposition_text = _problem_decomposer.format_for_prompt(decomposition)
    except Exception as e:
        logger.warning(f"⚠️ Problem decomposition failed: {e}")

    cot = COT(
        latest_observations_to_keep=LATEST_OBSERVATIONS_TO_KEEP,
        summarize_batch_size=SUMMARIZE_BATCH_SIZE,
    )
    tool_manager = FixTaskToolManager(
        available_tools=[
            "analyze_edge_cases",
            "generate_test_cases_from_root_cause",
            "list_directory_structure",
            "get_file_content",
            "get_function_body",
            "find_symbol_references",
            "grep_search",
            "search_in_file",
            "apply_code_edit",
            "modify_test_case",
            "create_new_file",
            "run_code",
            "run_tests",
            "think",
            "log_strategy",
            "mark_strategy_outcome",
            "list_attempted_strategies",
            "create_hypothesis",
            "test_hypothesis",
            "list_hypotheses",
            "finish_root_cause_analysis",
            "finish",
        ],
        problem_statement=problem_statement,
        is_fix_task=True,
        cot=cot,
    )

    _current_tool_manager = tool_manager

    # Pre-populate hypotheses from decomposition
    if decomposition and decomposition.get("initial_hypotheses"):
        for hyp in decomposition["initial_hypotheses"]:
            if isinstance(hyp, dict) and hyp.get("description"):
                tool_manager.hypothesis_counter += 1
                tool_manager.hypotheses.append({
                    "id": tool_manager.hypothesis_counter,
                    "description": hyp.get("description", ""),
                    "evidence": hyp.get("confirming_evidence", "From problem analysis"),
                    "status": "untested",
                    "findings": None,
                    "created_step": 0,
                    "tested_step": None,
                    "timestamp": time.time(),
                })

    tool_manager.problem_decomposition = decomposition

    system_prompt = FIX_TASK_SYSTEM_PROMPT.format(
        tools_docs=tool_manager.get_tool_docs(),
        problem_statement=problem_statement,
        format_prompt=FORMAT_PROMPT_FIX,
    )

    enhanced_problem = problem_statement
    if decomposition_text:
        enhanced_problem = problem_statement + "\n\n---\n\n# Structured Problem Analysis\n\n" + decomposition_text
    if enhancement:
        enhanced_problem = enhanced_problem + "\n\n---\n\n# Additional Context\n\n" + enhancement
    instance_prompt = FIX_TASK_INSTANCE_PROMPT_TEMPLATE.format(problem_statement=enhanced_problem)
    if root_cause_analysis:
        instance_prompt += "\n\n---\n\n# Preliminary analysis\n\n" + root_cause_analysis

    logger.info("🚀 [WORKFLOW] Executing agent workflow...")
    budget = TimeBudget(total_secs=timeout)
    memory = InvestigationMemory()
    patch, is_success = execute_agent_workflow(
        cot,
        tool_manager,
        system_prompt,
        instance_prompt,
        n_max_steps,
        timeout,
        AGENT_MODELS,
        finish_tool_name="finish",
        budget=budget,
        memory=memory,
    )
    return patch, is_success


def process_fix_task(problem_text: str, enhancement: str = "") -> str:
    cwd = os.getcwd()
    global run_id, agent_start_time
    patch_text = ""
    try:
        results = []
        for attempt in range(3):
            cost_usage = Network.get_cost_usage()
            elapsed_time = time.time() - agent_start_time
            if elapsed_time > 850 or cost_usage.get("used_cost_usd", 0) > cost_usage.get("max_cost_usd", float("inf")) - 0.6:
                logger.info(f"Stopping after {attempt} attempts (elapsed={elapsed_time:.0f}s)")
                break
            # Without this, if watchdog fired on attempt 1, attempts 2+3 abort at step 0
            _watchdog_abort.clear()
            os.system("git reset --hard")
            os.system("git clean -fd")
            remaining_time = max(10, INNER_TIMEOUT - elapsed_time)
            patch_text, is_success = fix_task_solve_workflow(
                problem_text,
                timeout=int(remaining_time),
                run_id_1=run_id,
                enhancement=enhancement,
            )
            modified_files = ToolManager.get_modified_files_list()
            modified_files_content = {}
            result = ""
            if modified_files:
                temp_file_ops = FileOperationsUtil(new_files_created=[])
                temp_file_ops.file_system_manager = FileSystemManager()
                temp_file_ops.search_manager = SearchManager()
                for file_path in modified_files:
                    file_content = temp_file_ops.get_file_content(file_path, limit=-1)
                    modified_files_content[file_path] = file_content
                result = "\n\n".join([f"{fp}\n{content}" for fp, content in modified_files_content.items()])
            observation = "Success" if is_success else "Failed"
            if len(results) == 0 or is_success:
                results.append({
                    "solution_code": result,
                    "patch": patch_text,
                    "modified_files": modified_files,
                    "modified_files_content": modified_files_content,
                    "summary": observation,
                })
        best_solution = select_best_solution(results, problem_text) if results else None
        os.system("git reset --hard")
        os.system("git clean -fd")
        if best_solution and best_solution.get("modified_files_content"):
            file_ops = FileOperationsUtil(new_files_created=[])
            for file_path, file_content in best_solution["modified_files_content"].items():
                try:
                    file_ops.save(file_path, file_content)
                except Exception as e:
                    logger.error(f"Error restoring file {file_path}: {e}")
            if best_solution.get("patch"):
                patch_text = best_solution["patch"]
    except Exception as e:
        logger.error(f"Error in process_fix_task: {e}, {traceback.format_exc()}")
    finally:
        os.chdir(cwd)
    return patch_text

def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo"):
    """
    Main entry point called by Ridges runtime.
    Returns git diff string (unified patch format).
    """
    global DEFAULT_PROXY_URL, run_id, agent_start_time, AGENT_MODELS, INNER_TIMEOUT
    agent_start_time = time.time()

    run_id = os.getenv("EVALUATION_RUN_ID", "")
    repo_dir = os.path.abspath(repo_dir)
    sys.path.insert(0, repo_dir)
    if os.path.exists(repo_dir):
        os.chdir(repo_dir)

    set_env_for_agent()

    logger.info("🔍 [T68] Running model health probe...")
    try:
        probed_models = _build_healthy_model_list(run_id)
        if probed_models:
            AGENT_MODELS = probed_models
            logger.info(f"✅ [T68] Healthy model order: {[m.name for m in AGENT_MODELS]}")
    except Exception as e:
        logger.warning(f"⚠️ [T68] Model probe failed: {e}. Using default order.")

    INNER_TIMEOUT = max(300, AGENT_TIMEOUT - 100)
    timeout = INNER_TIMEOUT

    _watchdog_abort.clear()
    install_watchdog(AGENT_TIMEOUT)

    result = None
    task_completed = threading.Event()

    def run_task():
        nonlocal result
        try:
            global _current_tool_manager
            _current_tool_manager = ToolManager()
            problem_statement = input_dict.get("problem_statement", "")
            result = process_fix_task(problem_statement, "")
        except Exception as e:
            logger.error(f"Task error: {e}\n{traceback.format_exc()}")
        finally:
            task_completed.set()

    task_thread = threading.Thread(target=run_task, daemon=True)
    task_thread.start()
    task_thread.join(timeout=timeout)

    if task_thread.is_alive():
        logger.warning(f"Task timed out after {timeout}s")

    global _current_tool_manager
    if _current_tool_manager is not None:
        try:
            final_patch = _current_tool_manager.get_final_git_patch()
            if final_patch and isinstance(final_patch, str) and not final_patch.startswith("Error"):
                result = final_patch
        except Exception as e:
            logger.error(f"Failed to get final patch: {e}")
        finally:
            _current_tool_manager = None

    try:
        subprocess.Popen(["git", "reset", "--hard"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

    return result or ""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SN62 Agent")
    parser.add_argument("--problem", type=str, default="Fix the bug in the code.", help="Problem statement")
    parser.add_argument("--repo-dir", type=str, default="repo", help="Repository directory")
    args = parser.parse_args()
    output = agent_main({"problem_statement": args.problem}, repo_dir=args.repo_dir)
    print(output)
