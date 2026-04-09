from __future__ import annotations

import re
import time


class ExecutionGuards:
    """
    Per-turn execution budget guards.
    Instantiated fresh each turn in agent_loop.py.
    """

    MAX_REASONING_STEPS = 6
    MAX_TOOL_CALLS = 4

    def __init__(self) -> None:
        self.reason_steps = 0
        self.tool_calls = 0

    def allow_reason(self) -> bool:
        self.reason_steps += 1
        return self.reason_steps <= self.MAX_REASONING_STEPS

    def allow_tool(self) -> bool:
        self.tool_calls += 1
        return self.tool_calls <= self.MAX_TOOL_CALLS


class InputGuard:
    """
    Validates and sanitizes raw user input before it enters the pipeline.
    Called once at the top of handle_turn in orchestrator.py.
    """

    MAX_INPUT_LENGTH = 4000

    # Prompt injection patterns — attempts to override system instructions
    _INJECTION_PATTERNS: list[re.Pattern] = [
        re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
        re.compile(r"you\s+are\s+now\s+", re.I),
        re.compile(r"forget\s+(everything|all|your)\s+", re.I),
        re.compile(r"(act|behave|pretend|respond)\s+as\s+if\s+you\s+(are|were|have no)", re.I),
        re.compile(r"your\s+new\s+(instructions?|rules?|persona)", re.I),
        re.compile(r"disregard\s+(your|all|the)\s+(previous|prior|system)", re.I),
        re.compile(r"<\s*system\s*>", re.I),
        re.compile(r"\[INST\]", re.I),
    ]

    # Hard-blocked content — never process these regardless of context
    _BLOCKED_PATTERNS: list[re.Pattern] = [
        re.compile(r"\b(make|build|create|synthesize|produce)\b.{0,40}\b(bomb|explosive|weapon|poison|malware|virus|ransomware)\b", re.I),
        re.compile(r"\b(how\s+to\s+)?(hack|exploit|bypass)\b.{0,40}\b(password|system|auth|database)\b", re.I),
        re.compile(r"\bchild\s+(porn|abuse|sexual)\b", re.I),
        re.compile(r"\b(doxx|doxing|swatt?ing)\b", re.I),
    ]

    def validate(self, text: str) -> tuple[bool, str | None]:
        """
        Returns (is_valid, rejection_reason).
        If is_valid is False, rejection_reason contains a safe message for the user.
        """
        if not text or not text.strip():
            return False, "Message cannot be empty."

        if len(text) > self.MAX_INPUT_LENGTH:
            return False, f"Message is too long. Please keep it under {self.MAX_INPUT_LENGTH} characters."

        for pattern in self._INJECTION_PATTERNS:
            if pattern.search(text):
                return False, "I can't process that request."

        for pattern in self._BLOCKED_PATTERNS:
            if pattern.search(text):
                return False, "I'm not able to help with that."

        return True, None

    def sanitize(self, text: str) -> str:
        """
        Strips null bytes and normalizes whitespace.
        Always safe to call — never raises.
        """
        try:
            text = text.replace("\x00", "")
            text = re.sub(r"[ \t]{10,}", " ", text)
            return text.strip()
        except Exception:
            return text


class RateLimiter:
    """
    In-memory per-user rate limiter.
    Uses a sliding window — tracks timestamps of recent requests.
    Replace backing store with Redis in Phase 16 for multi-instance deployments.
    """

    WINDOW_SECONDS = 60
    MAX_REQUESTS_PER_WINDOW = 20

    def __init__(self) -> None:
        # user_id -> list of unix timestamps
        self._windows: dict[int, list[float]] = {}

    def allow(self, user_id: int) -> bool:
        now = time.time()
        cutoff = now - self.WINDOW_SECONDS

        timestamps = self._windows.get(user_id, [])
        # Evict expired timestamps
        timestamps = [t for t in timestamps if t > cutoff]
        timestamps.append(now)
        self._windows[user_id] = timestamps

        return len(timestamps) <= self.MAX_REQUESTS_PER_WINDOW

    def remaining(self, user_id: int) -> int:
        now = time.time()
        cutoff = now - self.WINDOW_SECONDS
        timestamps = [t for t in self._windows.get(user_id, []) if t > cutoff]
        return max(0, self.MAX_REQUESTS_PER_WINDOW - len(timestamps))


class ToolSandbox:
    """
    Validates tool calls before execution.
    Prevents tools from being called with dangerous or malformed arguments.
    """

    # Tools that are allowed to make external network calls
    _NETWORK_TOOLS: set[str] = {"web_search", "web"}

    # Tools that can write to storage
    _WRITE_TOOLS: set[str] = {"memory", "file", "image"}

    # Arguments that should never contain raw code strings
    _NO_CODE_ARGS: set[str] = {"query", "instruction", "prompt"}

    _CODE_INJECTION_PATTERN = re.compile(
        r"(import\s+os|subprocess|__import__|eval\(|exec\(|open\()", re.I
    )

    def validate_tool_call(
        self,
        tool_name: str,
        args: dict,
    ) -> tuple[bool, str | None]:
        """
        Returns (is_safe, rejection_reason).
        """
        for arg_name, arg_value in args.items():
            if not isinstance(arg_value, str):
                continue
            if arg_name in self._NO_CODE_ARGS:
                if self._CODE_INJECTION_PATTERN.search(arg_value):
                    return False, f"Unsafe content detected in tool argument '{arg_name}'."

        return True, None


# Module-level singletons
input_guard = InputGuard()
rate_limiter = RateLimiter()
tool_sandbox = ToolSandbox()
