import os
from contextlib import contextmanager
import contextvars
from typing import Iterable, Optional

import tiktoken


class TokenBudgetExceeded(RuntimeError):
    """Raised when adding text would exceed the configured token budget."""

    def __init__(self, attempted: int, cap: int, label: Optional[str] = None) -> None:
        label_text = f" while processing {label}" if label else ""
        super().__init__(
            f"Token budget exceeded{label_text}. Attempted {attempted} tokens with a cap of {cap}."
        )
        self.attempted = attempted
        self.cap = cap
        self.label = label


def resolve_encoding(name: Optional[str]):
    candidate = name or os.getenv("TOKEN_GUARD_ENCODING", "cl100k_base")
    try:
        return tiktoken.encoding_for_model(candidate)
    except Exception:
        pass
    try:
        return tiktoken.get_encoding(candidate)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


class TokenBudget:
    """Tracks token consumption and raises when a configurable cap is reached."""

    def __init__(
        self,
        encoding_name: Optional[str] = None,
        max_context_tokens: Optional[int] = None,
        cap_ratio: float = 0.9,
    ) -> None:
        self.encoding = resolve_encoding(encoding_name)
        total_limit = max_context_tokens or int(os.getenv("MODEL_CONTEXT_LIMIT", "128000"))
        self.max_context_tokens = total_limit
        self.cap = max(1, int(total_limit * cap_ratio))
        self.consumed = 0

    @property
    def consumed(self) -> int:
        return self.consumed

    def reset(self) -> None:
        self.consumed = 0

    def count_text(self, text: Optional[str]) -> int:
        if not text:
            return 0
        try:
            return len(self.encoding.encode(text))
        except Exception:
            return len(text.split())

    def initialize(self, base_texts: Iterable[Optional[str]]) -> None:
        self.reset()
        total = 0
        for text in base_texts:
            total += self.count_text(text)
        if total > self.cap:
            raise TokenBudgetExceeded(total, self.cap, label="conversation_history")
        self._consumed = total

    def consume_tokens(self, tokens: int, label: Optional[str] = None) -> None:
        if tokens <= 0:
            return
        attempted = self.consumed + tokens
        if attempted > self.cap:
            raise TokenBudgetExceeded(attempted, self.cap, label=label)
        self.consumed = attempted

    def consume_text(self, text: Optional[str], label: Optional[str] = None) -> None:
        tokens = self.count_text(text)
        self.consume_tokens(tokens, label=label)


guard_var: contextvars.ContextVar[Optional[TokenBudget]] = contextvars.ContextVar(
    "token_guard", default=None
)
global_guard: Optional[TokenBudget] = None


def get_active_guard() -> Optional[TokenBudget]:
    guard = guard_var.get()
    if guard is not None:
        return guard
    return global_guard


@contextmanager
def activate_guard(guard: TokenBudget):
    global global_guard
    token = guard_var.set(guard)
    previous_guard = global_guard
    global_guard = guard
    try:
        yield guard
    finally:
        guard_var.reset(token)
        global_guard = previous_guard
