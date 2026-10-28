"""One place to talk to the model. Works with any OpenAI-compatible server.
Default: a free local model via Ollama (https://ollama.com)."""
import os
import threading
from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1/"),
    api_key=os.getenv("LLM_API_KEY", "ollama"),  # Ollama ignores the key, but the client needs one
)
MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")

STATS = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
_LOCK = threading.Lock()  # team specialists call the model from several threads at once


def reset_stats():
    for key in STATS:
        STATS[key] = 0


def chat(messages, **kwargs):
    """Send a list of messages, get back the model's reply message."""
    response = client.chat.completions.create(model=MODEL, messages=messages, temperature=0, **kwargs)
    with _LOCK:
        STATS["calls"] += 1
        if getattr(response, "usage", None):
            STATS["prompt_tokens"] += response.usage.prompt_tokens or 0
            STATS["completion_tokens"] += response.usage.completion_tokens or 0
    return response.choices[0].message
