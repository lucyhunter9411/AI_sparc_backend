"""
Prompt-builder helpers for LLM calls.
"""
from __future__ import annotations
from langchain.prompts import PromptTemplate
from functools import lru_cache
from typing import Any
from app.services import llm as _raw    # existing module with llm_models etc.
import requests
import os 

OPENAI_API_KEY = os.getenv("OPENAI_KEY", "")

# We keep the signature minimal – later we can evolve it.
def build_chat_prompt(
    custom_template: str,
    history: list[str],
    query: str,
    context: str,
    *,
    language: str = "",
    username: str = "",
    total_count: str = "25",
    hands_up_count: str = "5",
    overview: str = "",
    greeting_msg: str = "",
) -> str:
    """
    Render the final prompt string used for the LLM completion.
    """
    print("---------------------build_chat_prompt function is running correctly.")
    tmpl = PromptTemplate(
        input_variables=[
            "history",
            "query",
            "context",
            "language",
            "username",
            "totalCount",
            "handsUpCount",
            "overview",
            "greeting_msg",
        ],
        template=custom_template,
    )
    print("---------------------tmpl:", tmpl)
    return tmpl.format(
        history="\n".join(history),
        query=query,
        context=context,
        language=language,
        username=username,
        totalCount=total_count,
        handsUpCount=hands_up_count,
        overview=overview,
        greeting_msg=greeting_msg,
    )
# --------------------------------------------------------------------------- #
#                              L L M   F A Ç A D E                            #
# --------------------------------------------------------------------------- #
"""
Public helpers to fetch / switch the active LLM and run a completion.
Keeps `main.py` blissfully unaware of LangChain client internals.
"""


# ── active-model state (kept module-local) ───────────────────────────────────
_active_name: str = _raw.get_selected_llm_name()


def set_default_llm(model_name: str) -> None:
    """
    Switch the global default LLM used by `predict`.
    Raises KeyError if the name is unknown.
    """
    global _active_name
    if model_name not in _raw.llm_models:
        raise KeyError(f"Unknown model '{model_name}'")
    _active_name = model_name


def get_default_llm_name() -> str:
    """Return current default model name."""
    return _active_name


def get_llm(model_name: str | None = None):
    """
    Return a concrete LangChain chat-model client.
    """
    name = model_name or _active_name
    try:
        return _raw.llm_models[name]
    except KeyError as exc:
        raise KeyError(f"LLM '{name}' not configured") from exc


def predict(prompt: str, *, model_name: str | None = None, **kwargs: Any) -> str:
    """
    Convenience wrapper around `<ChatModel>.predict`.
    Extra kwargs pass straight through.
    """
    return get_llm(model_name).predict(prompt, **kwargs)


def generate_openai_response(prompt: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    try:
        print("⏳ Sending request to OpenAI...")
        response = requests.post(url, headers=headers, json=data, timeout=10)  # Force timeout after 10s
        response.raise_for_status()
        print("✅ Response received")
        print("🔍 Full Response JSON:", response.json())
        print("🔍 Full Response JSON:", response.json()['choices'][0]['message']['content'].strip())
        return response.json()['choices'][0]['message']['content'].strip()
    except requests.exceptions.Timeout:
        print("❌ Request timed out!")
        return "Request timed out."
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        print("Response text:", getattr(e.response, 'text', 'No response body'))
        return "Request failed."
