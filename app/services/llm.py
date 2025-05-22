"""
app/services/llm.py   – lazy-loaded Azure OpenAI chat models
"""
from functools import lru_cache
from typing import Dict, Iterator, ItemsView, KeysView, ValuesView

from langchain_community.chat_models import AzureChatOpenAI  # type: ignore
from app.core.config import get_settings

# ───────────────────────── configuration ──────────────────────────
settings = get_settings()

_DEPLOYMENTS: Dict[str, str] = {
    k: v
    for k, v in {
        "GPT-41-mini":  getattr(settings, "azure_deploy_41mini", None),
        "GPT-4":        getattr(settings, "azure_deploy_4", None),
        "GPT-35-Turbo": getattr(settings, "azure_deploy_35", None),
    }.items()
    if v
}
if not _DEPLOYMENTS:
    raise RuntimeError("No Azure OpenAI deployments configured.")

# ───────────────────────── factory & cache ────────────────────────
@lru_cache(maxsize=None)
def _make_llm(model_name: str) -> AzureChatOpenAI:  # noqa: N802
    if model_name not in _DEPLOYMENTS:
        raise KeyError(f"Unknown model '{model_name}'")
    return AzureChatOpenAI(
        openai_api_base=settings.azure_openai_base,
        openai_api_version=settings.azure_openai_version,
        openai_api_key=settings.azure_openai_key,
        openai_api_type="azure",
        deployment_name=_DEPLOYMENTS[model_name],
    )

# ───────────────────────── lazy mapping ───────────────────────────
class _LazyLLMMapping:
    """Dict-like object that builds LLM clients only when accessed."""

    __slots__ = ()

    # ---- required for “obj[key]” --------------------------------
    def __getitem__(self, key: str) -> AzureChatOpenAI:
        return _make_llm(key)          # ← *** this makes it subscriptable ***

    # ---- optional helpers so in, len(), iteration work ---------
    def __contains__(self, key: object) -> bool:  # type: ignore[override]
        return key in _DEPLOYMENTS

    def __iter_(_self) -> Iterator[str]:
        return iter(_DEPLOYMENTS)

    def __len__(self) -> int:
        return len(_DEPLOYMENTS)

    def keys(self) -> KeysView[str]:        # noqa: D401
        return _DEPLOYMENTS.keys()          # type: ignore[return-value]

    def items(self) -> ItemsView[str, AzureChatOpenAI]:
        return {k: _make_llm(k) for k in _DEPLOYMENTS}.items()  # type: ignore[return-value]

    def values(self) -> ValuesView[AzureChatOpenAI]:
        return {k: _make_llm(k) for k in _DEPLOYMENTS}.values()  # type: ignore[return-value]

    def __repr__(self) -> str:  # noqa: D401
        return f"<LazyLLMMapping models={list(self.keys())}>"

# exported object --------------------------------------------------
llm_models = _LazyLLMMapping()  # type: ignore[var-annotated]

# convenience wrappers (unchanged API) -----------------------------
_DEFAULT_NAME = (
    "GPT-41-mini" if "GPT-41-mini" in _DEPLOYMENTS else next(iter(_DEPLOYMENTS))
)

def get_llm_models() -> Dict[str, AzureChatOpenAI]:  # noqa: N802
    return {name: _make_llm(name) for name in _DEPLOYMENTS}

def get_selected_llm() -> AzureChatOpenAI:  # noqa: N802
    return _make_llm(_DEFAULT_NAME)

def get_selected_llm_name() -> str:  # noqa: N802
    return _DEFAULT_NAME

_all_ = [
    "llm_models",
    "get_llm_models",
    "get_selected_llm",
    "get_selected_llm_name",
]