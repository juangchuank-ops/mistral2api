"""mistral2api - Mistral Chat (chat.mistral.ai) to OpenAI-compatible API.

Converts the Mistral Chat web interface internal API into OpenAI-compatible
endpoints. Supports guest (anonymous) chat without login.
"""
from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import ConfigManager
from app.routes import router

config_manager = ConfigManager()


def create_app() -> FastAPI:
    app = FastAPI(
        title="mistral2api",
        description="OpenAI-compatible API wrapper for chat.mistral.ai (guest mode)",
        version="1.0.0",
    )

    # CORS - allow all origins for easy integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config_manager.config.host,
        port=config_manager.config.port,
        reload=True,
    )