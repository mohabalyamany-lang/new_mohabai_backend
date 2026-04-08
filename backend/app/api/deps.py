from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from app.db.models import SessionToken, User
from app.db.session import get_db_session


def _extract_bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    prefix = "bearer "
    if not authorization.lower().startswith(prefix):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header",
        )

    token = authorization[len(prefix):].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
        )
    return token


def get_current_session_token(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db_session),
) -> SessionToken:
    token = _extract_bearer_token(authorization)

    session_token = (
        db.query(SessionToken)
        .filter(
            SessionToken.access_token == token,
            SessionToken.is_revoked.is_(False),
        )
        .first()
    )
    if session_token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session",
        )
    return session_token


def get_current_user(
    session_token: SessionToken = Depends(get_current_session_token),
    db: Session = Depends(get_db_session),
) -> User:
    user = db.query(User).filter(User.id == session_token.user_id, User.is_active.is_(True)).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )
    return user
