"""Answer generation backends: Anthropic Claude (provider) or Ollama (local)."""

from typing import Protocol

import anthropic
import requests

SYSTEM_PROMPT = (
    "You are a question-answering assistant. Answer the user's question using "
    "only the provided context passages. If the context does not contain the "
    "answer, say so explicitly instead of guessing. Be concise and cite the "
    "source passage numbers you used, e.g. [2]."
)


def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i}] (source: {c['source']})\n{c['text']}" for i, c in enumerate(chunks, 1)
    )
    return (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer based only on the context above."
    )


class Generator(Protocol):
    def generate(self, prompt: str) -> str: ...


class AnthropicGenerator:
    """Generation via the Claude API."""

    def __init__(self, model: str = "claude-opus-5", max_tokens: int = 16000):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            detail = (
                response.stop_details.explanation
                if response.stop_details
                else "no detail provided"
            )
            return f"[The model declined to answer: {detail}]"
        return "".join(b.text for b in response.content if b.type == "text").strip()


class OllamaGenerator:
    """Generation via a locally running Ollama server."""

    def __init__(self, model: str = "llama3.1:8b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")

    def generate(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
