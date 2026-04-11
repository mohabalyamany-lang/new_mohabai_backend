from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.models import Conversation, User
from app.db.session import get_db_session
from app.schemas.chat import ChatRequest
from app.services.orchestrator import ConversationOrchestrator
from app.services.execution_engine import ExecutionEngine
from app.tools.registry import ToolRegistry

router = APIRouter(prefix="/stream-chat", tags=["chat"])


def sse_event(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("")
async def stream_chat_endpoint(
    payload: ChatRequest,
    db: Session = Depends(get_db_session),
    user: User = Depends(get_current_user),
) -> StreamingResponse:
    conversation = db.query(Conversation).filter(
        Conversation.id == payload.conversation_id
    ).first()

    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    tool_registry = ToolRegistry()
    orchestrator = ConversationOrchestrator(
        db=db,
        tool_registry=tool_registry,
    )

    async def event_stream() -> AsyncIterator[str]:
        result = await orchestrator.handle_turn(
            conversation=conversation,
            user_message=payload.message,
            user_id=user.id,
        )

        # ━━━ Meta event with turn_id for frontend deduplication ━━━
        yield sse_event({
            "type": "meta",
            "conversation_id": result.conversation_id,
            "turn_id": result.turn_id,
            "planner_action": result.planner_action,
            "planner_trace": result.planner_trace,
        })

        if not result.ok:
            yield sse_event({
                "type": "error",
                "error": result.error or "Unknown error",
                "turn_id": result.turn_id,
            })
            yield sse_event({"type": "done", "turn_id": result.turn_id})
            return

        # ━━━ Check for multi-intent — route to execution engine ━━━
        is_multi = any(
            t.get("stage") == "multi_intent"
            for t in (result.planner_trace or [])
        )

        if is_multi:
            # Multi-intent: execution engine handles sequential steps
            # NOTE: This requires the orchestrator to expose the raw
            # PlannerResult with steps. If your orchestrator doesn't
            # do this yet, this path will use the normal single-intent
            # flow below. The execution engine is ready — just needs
            # the planner result passed through from the orchestrator.
            #
            # TODO: Add `result.planner_result` to orchestrator output
            # Then uncomment below:
            #
            # engine = ExecutionEngine(tool_registry)
            # exec_result = await engine.execute_plan(
            #     planner_result=result.planner_result,
            #     original_message=payload.message,
            #     conversation=conversation,
            #     turn=turn,
            #     db=db,
            # )
            # if exec_result.final_response:
            #     yield sse_event({
            #         "type": "content",
            #         "content": exec_result.final_response,
            #         "turn_id": result.turn_id,
            #     })
            # for artifact in exec_result.final_artifacts:
            #     if artifact.get("artifact_type") == "image":
            #         yield sse_event({
            #             "type": "image",
            #             "url": artifact.get("storage_url"),
            #             "prompt": artifact.get("effective_prompt"),
            #             "turn_id": result.turn_id,
            #         })
            # if exec_result.error:
            #     yield sse_event({
            #         "type": "error",
            #         "error": exec_result.error,
            #         "turn_id": result.turn_id,
            #     })
            # yield sse_event({"type": "done", "turn_id": result.turn_id})
            # return

            # Fallback: use normal flow until orchestrator exposes planner result
            pass

        # ━━━ Normal single-intent flow ━━━
        text = result.assistant_text or ""
        if text:
            yield sse_event({
                "type": "content",
                "content": text,
                "turn_id": result.turn_id,
            })

        if result.tool_result:
            tool_result = result.tool_result

            # Surface image artifacts to the frontend explicitly
            for artifact in tool_result.get("artifacts", []):
                if artifact.get("artifact_type") == "image":
                    yield sse_event({
                        "type": "image",
                        "url": artifact.get("storage_url"),
                        "prompt": artifact.get("effective_prompt"),
                        "turn_id": result.turn_id,
                    })

            yield sse_event({
                "type": "tool_result",
                "tool_result": tool_result,
                "turn_id": result.turn_id,
            })

        # ━━━ Done event includes turn_id ━━━
        yield sse_event({"type": "done", "turn_id": result.turn_id})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
