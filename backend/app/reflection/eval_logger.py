from __future__ import annotations

import json
from datetime import UTC, datetime

from sqlalchemy.orm import Session


class EvalLogger:
    """
    Persists automated evaluation scores per turn.
    Uses a simple JSON append to the Turn.planner_trace field so no
    new DB migration is needed right now. Can be promoted to its own
    table in Phase 16 when observability is formalized.
    """

    def log(
        self,
        db: Session,
        turn_id: int,
        user_message: str,
        assistant_reply: str,
        scores: dict,
    ) -> None:
        try:
            from app.db.models import Turn  # local import to avoid circular

            turn = db.query(Turn).filter(Turn.id == turn_id).first()
            if turn is None:
                return

            existing_trace = turn.planner_trace or []
            existing_trace.append({
                "stage": "eval",
                "summary": f"overall={scores.get('overall', '?')}",
                "details": {
                    "scores": scores,
                    "flags": scores.get("flags", []),
                    "logged_at": datetime.now(UTC).isoformat(),
                    "user_message_preview": user_message[:120],
                    "reply_preview": assistant_reply[:120],
                },
            })
            turn.planner_trace = existing_trace
            db.flush()
        except Exception:
            pass  # Logging must never break a response


eval_logger = EvalLogger()
