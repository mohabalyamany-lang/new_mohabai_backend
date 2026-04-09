from __future__ import annotations

# runtime_orchestrator.py is retired.
# Both /chat and /stream-chat now use ConversationOrchestrator from app.services.orchestrator.
# This file is kept only to prevent import errors during transition.
# Safe to delete once you confirm no other file imports runtime_orchestrator.

from app.services.orchestrator import ConversationOrchestrator  # noqa: F401
