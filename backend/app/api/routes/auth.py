from __future__ import annotations

from datetime import timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_current_session_token
from app.core.security import create_access_token, create_refresh_token, hash_password, verify_password
from app.db.models import SessionToken, User
from app.db.session import get_db_session
from app.schemas.auth import AuthResponse, LoginRequest, MeResponse, RegisterRequest

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=AuthResponse)
async def register(
    payload: RegisterRequest,
    db: Session = Depends(get_db_session),
) -> AuthResponse:
    existing = (
        db.query(User)
        .filter(
            or_(
                User.username == payload.username,
                User.email == payload.email if payload.email else False,
            )
        )
        .first()
    )
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User already exists")

    user = User(
        username=payload.username,
        email=payload.email,
        password_hash=hash_password(payload.password),
        is_active=True,
    )
    db.add(user)
    db.flush()

    access_token = create_access_token(str(user.id), extra={"username": user.username})
    refresh_token = create_refresh_token(str(user.id), extra={"username": user.username})

    session_token = SessionToken(
        user_id=user.id,
        access_token=access_token,
        refresh_token=refresh_token,
    )
    db.add(session_token)
    db.commit()
    db.refresh(user)

    return AuthResponse(
        ok=True,
        user_id=user.id,
        username=user.username,
        access_token=access_token,
        refresh_token=refresh_token,
    )


@router.post("/login", response_model=AuthResponse)
async def login(
    payload: LoginRequest,
    db: Session = Depends(get_db_session),
) -> AuthResponse:
    user = (
        db.query(User)
        .filter(
            or_(
                User.username == payload.username_or_email,
                User.email == payload.username_or_email,
            )
        )
        .first()
    )
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    access_token = create_access_token(str(user.id), extra={"username": user.username})
    refresh_token = create_refresh_token(str(user.id), extra={"username": user.username})

    session_token = SessionToken(
        user_id=user.id,
        access_token=access_token,
        refresh_token=refresh_token,
    )
    db.add(session_token)
    db.commit()

    return AuthResponse(
        ok=True,
        user_id=user.id,
        username=user.username,
        access_token=access_token,
        refresh_token=refresh_token,
    )


@router.post("/logout")
async def logout(
    session_token: SessionToken = Depends(get_current_session_token),
    db: Session = Depends(get_db_session),
) -> dict[str, bool]:
    session_token.is_revoked = True
    db.commit()
    return {"ok": True}


@router.get("/me", response_model=MeResponse)
async def me(user: User = Depends(get_current_user)) -> MeResponse:
    return MeResponse(
        ok=True,
        user_id=user.id,
        public_id=user.public_id,
        username=user.username,
        email=user.email,
    )
