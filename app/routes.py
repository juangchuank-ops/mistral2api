"""OpenAI-compatible API routes for mistral2api."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from starlette.responses import JSONResponse

from .config import ConfigManager
from .logger import logger
from .mistral_client import (
    AVAILABLE_MODELS,
    MistralClient,
    parse_curl_cookies,
)
from .models import (
    ChatCompletionChunk,
    ChatCompletionChoice,
    ChatCompletionChoiceNonStream,
    ChatCompletionDelta,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ErrorDetail,
    ErrorResponse,
    ModelInfo,
    ModelList,
    Usage,
)

router = APIRouter()
config_manager = ConfigManager()

# Cache the HTML template
_HTML_PATH = None


def _get_html_template() -> str:
    global _HTML_PATH
    if _HTML_PATH is None:
        from pathlib import Path
        html_file = Path(__file__).parent / "web" / "index.html"
        _HTML_PATH = html_file.read_text(encoding="utf-8")
    return _HTML_PATH


# ── Auth ───────────────────────────────────────────────────────────────

def verify_api_key(request: Request) -> str:
    """Extract and validate API key from Authorization header."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message="Missing or invalid Authorization header. Use: Bearer YOUR_API_KEY",
                    type="invalid_request_error",
                )
            ).model_dump(),
        )
    token = auth[7:].strip()
    if not config_manager.is_valid_api_key(token):
        raise HTTPException(
            status_code=401,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message="Invalid API key",
                    type="authentication_error",
                )
            ).model_dump(),
        )
    return token


# ── Models ─────────────────────────────────────────────────────────────

@router.get("/v1/models")
async def list_models(_api_key: str = Depends(verify_api_key)):
    """List available models in OpenAI format."""
    models = [
        ModelInfo(id=m, owned_by="mistral")
        for m in AVAILABLE_MODELS
    ]
    return ModelList(data=models)


# ── Chat Completions ───────────────────────────────────────────────────

def _build_messages_for_mistral(messages: list[ChatMessage]) -> str:
    """Convert OpenAI messages format to a single prompt string for Mistral.

    Mistral Chat API expects a single 'content' string, not a messages array.
    We format multi-turn conversations as:
        [system]: ...
        [user]: ...
        [assistant]: ...
        [user]: latest question
    """
    parts = []
    for msg in messages:
        role = msg.role
        content = msg.content
        if isinstance(content, list):
            # Handle multimodal content - extract text parts
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        text_parts.append("[Image]")
                else:
                    text_parts.append(str(part))
            content = "\n".join(text_parts) if text_parts else ""

        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        elif role == "tool":
            parts.append(f"Tool: {content}")

    return "\n\n".join(parts)


def _build_sse_chunk(
    chunk_id: str,
    model: str,
    created: int,
    content: Optional[str] = None,
    role: Optional[str] = None,
    finish_reason: Optional[str] = None,
) -> str:
    """Build an SSE-formatted chat completion chunk."""
    delta = ChatCompletionDelta()
    if role:
        delta.role = "assistant"
    if content is not None:
        delta.content = content

    choice = ChatCompletionChoice(
        delta=delta,
        finish_reason=finish_reason,
    )

    chunk = ChatCompletionChunk(
        id=chunk_id,
        model=model,
        created=created,
        choices=[choice],
    )

    data = chunk.model_dump(exclude_none=True)
    return f"data: {json.dumps(data)}\n\n"


def _get_client_cookies() -> Optional[Dict[str, str]]:
    """Get cookies from config for the MistralClient, if configured."""
    if config_manager.is_valid_cookie():
        return dict(config_manager.config.cookies)
    return None


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    _api_key: str = Depends(verify_api_key),
):
    """OpenAI-compatible chat completions endpoint.

    Supports both streaming (stream=True) and non-streaming modes.
    Uses cookie-based auth if cookies are configured, otherwise guest mode.
    """
    model = request.model
    if model not in AVAILABLE_MODELS:
        model = config_manager.config.default_model

    # Build prompt from messages
    prompt = _build_messages_for_mistral(request.messages)

    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if request.stream:
        return StreamingResponse(
            _stream_response(prompt, model, chunk_id, created),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )
    else:
        # Non-streaming: collect all tokens
        return await _non_stream_response(prompt, model, chunk_id, created)


async def _stream_response(
    prompt: str,
    model: str,
    chunk_id: str,
    created: int,
):
    """Generate SSE stream for chat completion."""
    cookies = _get_client_cookies()
    role_sent = False

    async with MistralClient(
        timeout=config_manager.config.request_timeout,
        cookies=cookies,
    ) as client:
        try:
            async for token in client.chat_stream(prompt, model):
                if not role_sent:
                    yield _build_sse_chunk(chunk_id, model, created, role="assistant")
                    role_sent = True
                yield _build_sse_chunk(chunk_id, model, created, content=token)

            # Send final chunk
            if not role_sent:
                yield _build_sse_chunk(chunk_id, model, created, role="assistant")
            yield _build_sse_chunk(chunk_id, model, created, finish_reason="stop")
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Streaming error for model {model}: {e}", exc_info=True)
            error_data = json.dumps({
                "error": {
                    "message": f"Mistral API error: {str(e)}",
                    "type": "upstream_error",
                }
            })
            yield f"data: {error_data}\n\n"
            yield "data: [DONE]\n\n"


async def _non_stream_response(
    prompt: str,
    model: str,
    chunk_id: str,
    created: int,
):
    """Non-streaming chat completion."""
    cookies = _get_client_cookies()

    async with MistralClient(
        timeout=config_manager.config.request_timeout,
        cookies=cookies,
    ) as client:
        try:
            full_text, _ = await client.chat_complete(prompt, model)

            # Approximate token count (rough estimate: ~4 chars per token)
            # Note: This is not accurate - real tokenization requires a proper tokenizer
            prompt_tokens = max(1, len(prompt) // 4)
            completion_tokens = max(1, len(full_text) // 4)

            return ChatCompletionResponse(
                id=chunk_id,
                created=created,
                model=model,
                choices=[
                    ChatCompletionChoiceNonStream(
                        message=ChatCompletionMessage(content=full_text),
                        finish_reason="stop",
                    )
                ],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )
        except Exception as e:
            logger.error(f"Non-streaming error for model {model}: {e}", exc_info=True)
            raise HTTPException(
                status_code=502,
                detail=ErrorResponse(
                    error=ErrorDetail(
                        message=f"Mistral API error: {str(e)}",
                        type="upstream_error",
                    )
                ).model_dump(),
            )


# ── Web UI ─────────────────────────────────────────────────────────────

@router.get("/")
@router.get("/admin")
async def admin_page():
    """Serve the Web UI management panel."""
    return HTMLResponse(_get_html_template())


# ── Health ─────────────────────────────────────────────────────────────

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "mistral2api"}


# ── Admin: Cookie Management ──────────────────────────────────────────

@router.get("/admin/cookie")
async def get_cookie_status():
    """Get current cookie status (whether cookies are set, type)."""
    has_cookies = config_manager.is_valid_cookie()
    cookie_keys = list(config_manager.config.cookies.keys()) if has_cookies else []

    # Determine cookie type
    cookie_type = "none"
    if has_cookies:
        if any(k.startswith("ory_session") for k in cookie_keys):
            cookie_type = "authenticated"
        elif "csrftoken" in cookie_keys:
            cookie_type = "guest"
        else:
            cookie_type = "custom"

    return {
        "has_cookies": has_cookies,
        "cookie_type": cookie_type,
        "cookie_keys": cookie_keys,
        "cookie_count": len(cookie_keys),
    }


@router.post("/admin/cookie/save")
async def save_cookies(request: Request):
    """Save cookies from JSON body or cURL string.

    Accepts:
    - JSON body with `cookies` dict: {"cookies": {"name": "value", ...}}
    - JSON body with `curl` string: {"curl": "curl -b 'key=value' ..."}
    - JSON body with `raw` cookie string: {"raw": "name1=value1; name2=value2"}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message="Invalid JSON body",
                    type="invalid_request_error",
                )
            ).model_dump(),
        )

    cookies: Dict[str, str] = {}

    if "cookies" in body and isinstance(body["cookies"], dict):
        cookies = body["cookies"]
    elif "curl" in body and isinstance(body["curl"], str):
        cookies = parse_curl_cookies(body["curl"])
    elif "raw" in body and isinstance(body["raw"], str):
        # Parse raw cookie string: "name1=value1; name2=value2"
        for pair in body["raw"].split(";"):
            pair = pair.strip()
            if "=" in pair:
                name, _, value = pair.partition("=")
                name = name.strip()
                value = value.strip()
                if name and value:
                    cookies[name] = value
    else:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message="Body must contain 'cookies' (dict), 'curl' (string), or 'raw' (string)",
                    type="invalid_request_error",
                )
            ).model_dump(),
        )

    if not cookies:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message="No valid cookies found in request",
                    type="invalid_request_error",
                )
            ).model_dump(),
        )

    config_manager.set_cookies(cookies)

    return {
        "status": "ok",
        "message": f"Saved {len(cookies)} cookie(s)",
        "cookie_keys": list(cookies.keys()),
    }


@router.post("/admin/cookie/test")
async def test_cookies():
    """Test if current cookies work by making a simple request to Mistral."""
    if not config_manager.is_valid_cookie():
        return {
            "status": "error",
            "message": "No cookies configured. Save cookies first.",
            "valid": False,
        }

    async with MistralClient(
        timeout=30,
        cookies=dict(config_manager.config.cookies),
    ) as client:
        try:
            valid = await client.validate_cookies()
            return {
                "status": "ok" if valid else "error",
                "message": "Cookies are valid" if valid else "Cookies are invalid or expired",
                "valid": valid,
            }
        except Exception as e:
            logger.error(f"Cookie test failed: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Test failed: {str(e)}",
                "valid": False,
            }


@router.delete("/admin/cookie")
async def clear_cookies():
    """Clear stored cookies."""
    config_manager.clear_cookies()
    return {
        "status": "ok",
        "message": "All cookies cleared",
    }


@router.post("/admin/cookie/import_curl")
async def import_curl_cookies(request: Request):
    """Parse a cURL command string to extract cookies.

    Body: {"curl": "curl -H 'Cookie: ...' https://..."}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message="Invalid JSON body",
                    type="invalid_request_error",
                )
            ).model_dump(),
        )

    curl_cmd = body.get("curl", "")
    if not curl_cmd or not isinstance(curl_cmd, str):
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message="Body must contain 'curl' field with a cURL command string",
                    type="invalid_request_error",
                )
            ).model_dump(),
        )

    cookies = parse_curl_cookies(curl_cmd)
    if not cookies:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message="No cookies found in cURL command. Use -b or --cookie flag.",
                    type="invalid_request_error",
                )
            ).model_dump(),
        )

    config_manager.set_cookies(cookies)

    return {
        "status": "ok",
        "message": f"Imported {len(cookies)} cookie(s) from cURL",
        "cookie_keys": list(cookies.keys()),
    }


# ── Admin: System Status ──────────────────────────────────────────────

@router.get("/admin/status")
async def system_status():
    """Return system status (cookies set? guest mode? models available?)."""
    has_cookies = config_manager.is_valid_cookie()
    cookie_type = "none"
    if has_cookies:
        cookie_keys = list(config_manager.config.cookies.keys())
        if any(k.startswith("ory_session") for k in cookie_keys):
            cookie_type = "authenticated"
        elif "csrftoken" in cookie_keys:
            cookie_type = "guest"
        else:
            cookie_type = "custom"

    auth_mode = "cookie" if has_cookies else "guest"

    return {
        "status": "ok",
        "service": "mistral2api",
        "auth_mode": auth_mode,
        "cookies_configured": has_cookies,
        "cookie_type": cookie_type,
        "available_models": AVAILABLE_MODELS,
        "default_model": config_manager.config.default_model,
        "api_keys_count": len(config_manager.config.api_keys),
    }
