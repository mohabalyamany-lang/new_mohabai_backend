from datetime import datetime

from pydantic import BaseModel, ConfigDict


class APIModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class HealthResponse(APIModel):
    ok: bool
    service: str
    environment: str


class ErrorResponse(APIModel):
    ok: bool = False
    error: str


class Timestamped(APIModel):
    created_at: datetime
