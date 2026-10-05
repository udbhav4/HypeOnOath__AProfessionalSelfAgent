"""
Step 7 -- generation via a local Ollama model (Phase 1 only; ADR-4 moves to
a hosted LLM in Phase 2).

Uses Ollama's HTTP /api/chat endpoint through the standard library, so no
extra dependency is added for one POST request. temperature=0 keeps eval
runs as repeatable as the model allows; num_ctx is raised because Ollama's
default context window is too small for k=8 chunks plus rules.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from hypeonoath.core import config
from hypeonoath.serving.prompt import Prompt


class LLMUnavailableError(RuntimeError):
    """Ollama is not reachable / the model is missing -- actionable message."""


def generate(
    prompt: Prompt,
    model: str = config.OLLAMA_MODEL,
    base_url: str = config.OLLAMA_URL,
    timeout: float = config.OLLAMA_TIMEOUT_SECONDS,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_ctx": config.OLLAMA_NUM_CTX},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404:
            raise LLMUnavailableError(
                f"Ollama has no model {model!r}. Run: ollama pull {model}"
            ) from exc
        raise LLMUnavailableError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise LLMUnavailableError(
            f"Cannot reach Ollama at {base_url} ({exc}). Install Ollama, start it, "
            f"and run: ollama pull {model}"
        ) from exc

    try:
        return body["message"]["content"].strip()
    except (KeyError, TypeError) as exc:
        raise LLMUnavailableError(f"Unexpected Ollama response: {body!r}") from exc
