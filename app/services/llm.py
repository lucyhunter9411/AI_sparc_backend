# app/services/llm.py
from langchain_community.chat_models import AzureChatOpenAI
from app.core.config import get_settings

settings = get_settings()

# Public dict of all available LLM clients
llm_models = {
    name: AzureChatOpenAI(
        openai_api_base=settings.azure_openai_base,
        openai_api_version=settings.azure_openai_version,
        openai_api_key=settings.azure_openai_key,
        openai_api_type="azure",
        deployment_name=deployment,
    )
    for name, deployment in {
        "GPT-41-mini":  settings.azure_deploy_41mini,
        "GPT-4":        settings.azure_deploy_4,
        "GPT-35-Turbo": settings.azure_deploy_35,
    }.items() if deployment
}

# Default model name (must match one of the keys above)
_DEFAULT_LLM_NAME = "GPT-41-mini"

def get_llm_models() -> dict[str, AzureChatOpenAI]:
    """FastAPI dependency or helper: returns the full dict of LLM clients."""
    return llm_models


def get_selected_llm() -> AzureChatOpenAI:
    """Returns the configured default LLM instance."""
    try:
        return llm_models[_DEFAULT_LLM_NAME]
    except KeyError:
        raise RuntimeError(f"Unknown default model `{_DEFAULT_LLM_NAME}`")
    
def get_selected_llm_name() -> str:
    """Helper: returns the name of the default LLM."""
    return _DEFAULT_LLM_NAME

# Explicitly declare the public API of this module
__all__ = ["llm_models", "get_llm_models", "get_selected_llm", "get_selected_llm_name"]
