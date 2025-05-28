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
import httpx, asyncio

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

async def predict(
    prompt: str,
    *,
    model_name: str | None = None,
    **kwargs: Any,
) -> str:
    """
    Async wrapper around LangChain chat-models.

    • If the selected LLM exposes `.apredict`, use it.
    • Otherwise fall back to `.predict` in a worker thread so the
      FastAPI event-loop is never blocked.
    """
    llm = get_llm(model_name)

    # native async path (works for most LangChain chat models)
    if hasattr(llm, "apredict"):
        return await llm.apredict(prompt, **kwargs)

    # sync model – run in a thread pool
    return await asyncio.to_thread(llm.predict, prompt, **kwargs)

