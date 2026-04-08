from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=80)
    email: EmailStr | None = None
    password: str = Field(min_length=8, max_length=200)


class LoginRequest(BaseModel):
    username_or_email: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=1, max_length=200)


class AuthResponse(BaseModel):
    ok: bool
    user_id: int
    username: str
    access_token: str
    refresh_token: str


class MeResponse(BaseModel):
    ok: bool
    user_id: int
    public_id: str
    username: str
    email: str | None
