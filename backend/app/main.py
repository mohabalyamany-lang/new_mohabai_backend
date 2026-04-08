from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.chat import router as chat_router
from app.config import get_settings
from app.core.logging import configure_logging
from app.core.telemetry import configure_telemetry


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    configure_telemetry(app)
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict[str, str | bool]:
        return {
            "ok": True,
            "service": settings.app_name,
            "environment": settings.app_env,
        }

    app.include_router(chat_router, prefix=settings.api_v1_prefix)

    return app


app = create_app()