"""
llm_client.py - the one place that knows which LLM backend is on the other end.

Talks to the Prediction Guard API's OpenAI-style `/chat/completions` endpoint.
Callers hand it ollama-shaped messages - `{"role", "content", "images": [path]}`
- and get the reply text back, so the harness never has to care how a
screenshot travels over the wire.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

import requests

DEFAULT_URL = "http://localhost:8080"


class LLMError(Exception):
    """The API answered, but not with a reply (bad request, auth, outage)."""


class ChatClient:
    def __init__(self, url: str | None = None, apiKey: str | None = None,
                 timeout: float = 120.0):
        self.url = (url or DEFAULT_URL).rstrip("/")
        self.apiKey = apiKey
        if not self.apiKey:
            raise LLMError("no Prediction Guard API key (pass --api-key)")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.apiKey}",
            "Content-Type": "application/json",
        })

    def chat(self, model: str, messages: list, temperature: float,
             maxTokens: int, reasoningEffort: str | None = None) -> str:
        """One completion. Returns the reply text, falling back to the
        reasoning when a thinking model spent its budget before answering."""
        body = {
            "model": model,
            "messages": [_toWire(m) for m in messages],
            "temperature": temperature,
            "max_tokens": maxTokens,
        }
        if reasoningEffort:
            body["reasoning_effort"] = reasoningEffort

        try:
            resp = self.session.post(f"{self.url}/chat/completions",
                                     json=body, timeout=self.timeout)
        except requests.RequestException as exc:
            raise LLMError(f"could not reach {self.url}: {exc}") from exc
        if resp.status_code != 200:
            raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")

        try:
            message = resp.json()["choices"][0]["message"]
        except (ValueError, KeyError, IndexError) as exc:
            raise LLMError(f"unexpected response: {resp.text[:300]}") from exc
        content = (message.get("content") or "").strip()
        return content or (message.get("reasoning")
                           or message.get("reasoning_content") or "")


def _toWire(message: dict) -> dict:
    """Ollama's `images: [path]` becomes OpenAI-style content parts."""
    images = message.get("images")
    if not images:
        return {"role": message["role"], "content": message["content"]}
    parts = [{"type": "text", "text": message["content"]}]
    for image in images:
        parts.append({"type": "image_url",
                      "image_url": {"url": _dataUrl(Path(image))}})
    return {"role": message["role"], "content": parts}


def _dataUrl(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"
