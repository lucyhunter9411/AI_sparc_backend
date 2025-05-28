from __future__ import annotations
from typing import Any, Dict
from pydantic import BaseModel
import time

class WSMessage(BaseModel):
    """
    Canonical wrapper for every WebSocket payload.

    type : str   – routing key, e.g. "audio", "stt_result", "ping"
    data : dict  – payload specific to the type
    ts   : float – Unix time on sender (seconds)
    """
    type: str
    data: Dict[str, Any]
    ts:  float = time.time()   # auto-stamp when you build a message
