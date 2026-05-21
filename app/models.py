"""Pydantic models for OpenAI-compatible API."""
from __future__ import annotations

import time
import uuid
from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ── OpenAI Request Models ─────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, List[dict]]
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "mistral-large-latest"
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    user: Optional[str] = None


# ── OpenAI Response Models ─────────────────────────────────────────────

class ChatCompletionDelta(BaseModel):
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionDelta = Field(default_factory=ChatCompletionDelta)
    finish_reason: Optional[Literal["stop", "length", "content_filter", "tool_calls"]] = None


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None


class ChatCompletionChoiceNonStream(BaseModel):
    index: int = 0
    message: ChatCompletionMessage = Field(default_factory=ChatCompletionMessage)
    finish_reason: Optional[Literal["stop", "length", "content_filter", "tool_calls"]] = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoiceNonStream]
    usage: Optional[Usage] = None


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mistral"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelInfo]


# ── Error Model ────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    message: str
    type: str
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail