import re

_BAD = {"hack", "kill", "violence", "explosive", "attack", "threat"}

def sanitize_text(text: str) -> str:
    """Mask a short list of forbidden words (same as in main.py)."""
    for word in _BAD:
        text = re.sub(rf"\b{word}\b", "****", text, flags=re.IGNORECASE)
    return text