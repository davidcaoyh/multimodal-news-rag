"""Small, dependency-free Ollama client for local text and vision generation."""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_HOST = "http://127.0.0.1:11434"


def encode_image(path: str | Path) -> str:
    """Return an image as base64 for Ollama's REST API."""
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def chat(
    prompt: str,
    *,
    model: str,
    image_paths: list[str | Path] | None = None,
    host: str = DEFAULT_HOST,
    temperature: float = 0.0,
    seed: int = 42,
    max_tokens: int = 512,
    response_format: str | dict[str, Any] | None = None,
    timeout: int = 300,
) -> dict[str, Any]:
    """Run one non-streaming local inference and return text plus timing metadata."""
    message: dict[str, Any] = {"role": "user", "content": prompt}
    if image_paths:
        message["images"] = [encode_image(path) for path in image_paths]
    payload: dict[str, Any] = {
        "model": model,
        "messages": [message],
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "seed": seed,
            "num_predict": max_tokens,
        },
    }
    if response_format is not None:
        payload["format"] = response_format

    request = urllib.request.Request(
        host.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = json.load(response)
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {host}. Start it with `ollama serve`."
        ) from exc

    message_out = raw.get("message", {})
    return {
        "text": (message_out.get("content") or "").strip(),
        "thinking": (message_out.get("thinking") or "").strip(),
        "wall_seconds": time.perf_counter() - started,
        "total_seconds": raw.get("total_duration", 0) / 1e9,
        "load_seconds": raw.get("load_duration", 0) / 1e9,
        "prompt_tokens": raw.get("prompt_eval_count"),
        "output_tokens": raw.get("eval_count"),
        "done_reason": raw.get("done_reason"),
    }

