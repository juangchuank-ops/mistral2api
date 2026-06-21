"""Mistral Chat internal API client (guest mode + cookie-based auth).

Reverse-engineered from chat.mistral.ai web interface.
Captures the internal tRPC and REST API endpoints used by the web chat.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import AsyncIterator, Dict, Optional

import httpx

from .logger import logger

CHAT_API_URL = "https://chat.mistral.ai/api/chat"
NEW_CHAT_URL = "https://chat.mistral.ai/api/trpc/message.newChat?batch=1"
SITE_URL = "https://chat.mistral.ai/chat"

# Default models available on Mistral Chat (guest mode + authenticated)
AVAILABLE_MODELS = [
    "mistral-large-latest",
    "mistral-small-latest",
    "mistral-medium-latest",
    "mistral-8b-latest",
    "pixtral-large-latest",
    "codestral-latest",
]

MODEL_MAP = {
    "mistral-large-latest": "mistral-large-2507",
    "mistral-small-latest": "mistral-small-2505",
    "mistral-medium-latest": "mistral-medium-latest",
    "mistral-8b-latest": "ministral-8b-latest",
    "pixtral-large-latest": "pixtral-large-2505",
    "codestral-latest": "codestral-latest",
}

# Regex for Mistral's custom streaming line protocol: number:content
_STREAM_LINE_RE = re.compile(r'^(\d+):(.*)$')


class MistralClient:
    """Async HTTP client for Mistral Chat internal API.

    Supports both guest mode (auto session init) and cookie-based auth
    (for logged-in users who provide their own session cookies).

    Supports async context manager protocol for automatic cleanup:
        async with MistralClient(cookies=cookies) as client:
            async for token in client.chat_stream(prompt, model):
                print(token)
    """

    def __init__(
        self,
        timeout: int = 120,
        cookies: Optional[Dict[str, str]] = None,
    ) -> None:
        self._client: Optional[httpx.AsyncClient] = None
        self._timeout = timeout
        self._provided_cookies: Optional[Dict[str, str]] = cookies
        self._guest_initialized = False
        self._retry_count = 0
        self._max_retries = 2

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/148.0.0.0 Safari/537.36"
                    ),
                    "Accept": "*/*",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Origin": "https://chat.mistral.ai",
                    "Referer": "https://chat.mistral.ai/chat",
                },
                follow_redirects=True,
            )
            # If user provided cookies, set them immediately
            if self._provided_cookies:
                for name, value in self._provided_cookies.items():
                    self._client.cookies.set(name, value, domain=".mistral.ai")
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.debug("MistralClient closed")

    async def __aenter__(self) -> "MistralClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - ensures cleanup."""
        await self.close()

    def _resolve_model(self, model: str) -> str:
        """Map user-facing model name to internal model ID."""
        return MODEL_MAP.get(model, model)

    async def init_guest_session(self) -> Dict[str, str]:
        """Visit the Mistral chat page to establish guest session cookies.

        Returns the cookies acquired from the session (ory_session_*, csrftoken, etc.).
        """
        client = await self._get_client()
        logger.info("Initializing guest session...")
        resp = await client.get(SITE_URL)
        resp.raise_for_status()

        # Extract cookies from the jar
        cookies: Dict[str, str] = {}
        for cookie in client.cookies.jar:
            cookies[cookie.name] = cookie.value

        self._guest_initialized = True
        logger.info(f"Guest session initialized with {len(cookies)} cookies")
        return cookies

    def _extract_csrf_token(self) -> Optional[str]:
        """Extract CSRF token from the client's cookie jar."""
        if self._client is None:
            return None
        for cookie in self._client.cookies.jar:
            if cookie.name == "csrftoken":
                return cookie.value
        return None

    def _build_headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Build proper headers including CSRF token, Origin, Referer, etc."""
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Origin": "https://chat.mistral.ai",
            "Referer": "https://chat.mistral.ai/chat",
        }
        csrf = self._extract_csrf_token()
        if csrf:
            headers["X-CSRFTOKEN"] = csrf
        if extra:
            headers.update(extra)
        return headers

    async def validate_cookies(self) -> bool:
        """Test if current cookies are valid by making a simple request."""
        try:
            client = await self._get_client()
            resp = await client.get(SITE_URL, follow_redirects=True)
            valid = resp.status_code == 200
            logger.debug(f"Cookie validation result: {valid}")
            return valid
        except Exception as e:
            logger.warning(f"Cookie validation failed: {e}")
            return False

    async def _ensure_session(self) -> None:
        """Ensure a valid session exists. Initialize guest session if needed."""
        client = await self._get_client()

        # If user provided cookies, just validate them
        if self._provided_cookies:
            valid = await self.validate_cookies()
            if not valid:
                raise RuntimeError(
                    "Provided cookies are invalid or expired. "
                    "Please update your session cookies."
                )
            return

        # Guest mode: initialize session if not already done
        if not self._guest_initialized:
            await self.init_guest_session()

    async def create_chat(self, prompt: str) -> str:
        """Create a new chat session and return the chatId.

        Calls the tRPC endpoint to initialize a conversation.
        """
        await self._ensure_session()
        client = await self._get_client()
        payload = {
            "0": {
                "json": {
                    "chatId": None,
                    "content": prompt,
                    "rag": False,
                },
                "meta": {
                    "values": {
                        "chatId": ["undefined"],
                    },
                },
            },
        }
        headers = self._build_headers({
            "x-trpc-source": "nextjs-react",
        })
        resp = await client.post(
            NEW_CHAT_URL,
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list) and len(data) > 0:
            item = data[0]
            chat_id = (
                item.get("result", {})
                .get("data", {})
                .get("json", {})
                .get("chatId")
            )
            if chat_id:
                return chat_id

        raise RuntimeError(f"Failed to create chat: {resp.text[:500]}")

    async def send_message(
        self,
        chat_id: str,
        prompt: str,
        model: str,
        mode: str = "append",
    ) -> httpx.Response:
        """Send a message to an existing chat and return the streaming response."""
        await self._ensure_session()
        client = await self._get_client()
        internal_model = self._resolve_model(model)

        payload = {
            "chatId": chat_id,
            "model": internal_model,
            "messageInput": prompt,
            "messageId": uuid.uuid4().hex,
            "mode": mode,
        }
        headers = self._build_headers()
        return await client.send(
            client.build_request(
                method="POST",
                url=CHAT_API_URL,
                json=payload,
                headers=headers,
            ),
            stream=True,
        )

    @staticmethod
    def parse_sse_chunk(line: str) -> Optional[str]:
        """Parse a single SSE line from Mistral's streaming response.

        Mistral's custom line protocol:
          - `0:"text content"` — text delta (JSON-quoted string)
          - `1:{json}` — metadata/done signals (ignored for text extraction)

        Only yields text from type-0 chunks. Handles JSON escaping properly.
        """
        match = _STREAM_LINE_RE.match(line)
        if not match:
            return None

        chunk_type = match.group(1)
        content_raw = match.group(2)

        if chunk_type != "0":
            # Non-text chunks (metadata, done signals, etc.) — skip
            return None

        # Content is JSON-quoted string - use json.loads for proper unescaping
        try:
            content = json.loads(content_raw)
            return content if content else None
        except json.JSONDecodeError:
            # Fallback: if it's not valid JSON, treat as plain text
            logger.warning(f"Failed to parse SSE chunk as JSON: {content_raw[:100]}")
            return content_raw if content_raw else None

    async def _chat_stream_once(
        self,
        prompt: str,
        model: str = "mistral-large-latest",
        chat_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Single attempt at streaming (no retry)."""
        if chat_id is None:
            chat_id = await self.create_chat(prompt)
            response = await self.send_message(chat_id, prompt, model, mode="retry")
        else:
            response = await self.send_message(chat_id, prompt, model, mode="append")

        try:
            buffer = ""
            async for chunk in response.aiter_bytes():
                buffer += chunk.decode("utf-8", errors="replace")
                lines = buffer.split("\n")
                buffer = lines.pop()  # Keep incomplete line in buffer

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    content = self.parse_sse_chunk(line)
                    if content:
                        yield content

            # Process remaining buffer
            if buffer.strip():
                content = self.parse_sse_chunk(buffer.strip())
                if content:
                    yield content
        finally:
            await response.aclose()

    async def chat_stream(
        self,
        prompt: str,
        model: str = "mistral-large-latest",
        chat_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens from Mistral.

        Yields text deltas as they arrive from the SSE stream.
        Includes retry logic: on 401/403, re-establishes guest session (max 2 retries).
        """
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                # On retry, re-initialize the session
                if attempt > 0:
                    self._client = None
                    self._guest_initialized = False
                    if self._provided_cookies:
                        # Re-apply provided cookies
                        client = await self._get_client()
                        for name, value in self._provided_cookies.items():
                            client.cookies.set(name, value, domain=".mistral.ai")
                    else:
                        await self.init_guest_session()

                async for token in self._chat_stream_once(prompt, model, chat_id):
                    yield token
                return  # Success — exit

            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code in (401, 403) and attempt < self._max_retries:
                    continue  # Retry
                raise
            except Exception as e:
                last_error = e
                if attempt < self._max_retries:
                    continue  # Retry on any error
                raise

        # Should not reach here, but just in case
        if last_error:
            raise last_error

    async def chat_complete(
        self,
        prompt: str,
        model: str = "mistral-large-latest",
        chat_id: Optional[str] = None,
    ) -> tuple[str, Optional[str]]:
        """Non-streaming chat completion. Returns (full_text, chat_id)."""
        full_text = ""
        new_chat_id = chat_id
        async for token in self.chat_stream(prompt, model, chat_id):
            full_text += token
        return full_text, new_chat_id


# ── cURL Cookie Parsing Utility ────────────────────────────────────────

def parse_curl_cookies(curl_command: str) -> Dict[str, str]:
    """Parse a cURL command string to extract cookies.

    Handles both `-b 'key=value; ...'` and multiple `-b key=value` formats.
    Returns a dict of cookie name → value.
    """
    cookies: Dict[str, str] = {}

    # Match -b or --cookie followed by cookie string
    # Pattern: -b "name1=value1; name2=value2" or --cookie '...'
    cookie_patterns = [
        re.compile(r'(?:-b|--cookie)\s+["\']([^"\']+)["\']'),
        re.compile(r'(?:-b|--cookie)\s+(\S+)'),
    ]

    for pattern in cookie_patterns:
        match = pattern.search(curl_command)
        if match:
            cookie_str = match.group(1)
            # Split by semicolon or ampersand
            for pair in re.split(r'[;&]', cookie_str):
                pair = pair.strip()
                if '=' in pair:
                    name, _, value = pair.partition('=')
                    name = name.strip()
                    value = value.strip()
                    if name and value:
                        cookies[name] = value
            break

    return cookies
