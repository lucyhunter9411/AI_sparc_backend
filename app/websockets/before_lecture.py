"""
/ws/{robot_id}/before/lecture
=============================

• A client connects, sends **one** "register" frame to declare its role
  (``{"type":"register","data":{"client":"speech"}}`` or
   ``{"type":"register","data":{"client":"audio"}}``).

• "speech" frames contain a base-64 WAV clip; the backend

      1. runs STT (Whisper / Azure etc.),
      2. adds the user turn to the running conversation history,
      3. builds a prompt with the trimmed history,
      4. gets an LLM reply,
      5. appends the assistant turn to history,
      6. calls TTS,
      7. broadcasts the TTS WAV **only** to sockets that registered
         as *audio*.

• "audio" frames would be handled symmetrically if you ever send them
  the other way (left as a TODO).

• "ping" → "pong" is unchanged.

Everything heavy (STT, LLM, TTS) runs in an `asyncio.create_task()` so
the socket never blocks long enough to miss keep-alive pings.
"""

from __future__ import annotations

import asyncio, logging, time
from typing import Any

from fastapi import APIRouter, Depends, WebSocket
from starlette.websockets import WebSocketDisconnect

from app.api.deps import get_conn_mgr                  # → ConnectionManager singleton
from app.websockets.connection_manager import ConnectionManager
from app.schemas.ws import WSMessage


# --- services ---------------------------------------------------------------                   # conversation history
from app.services.vision_service import  handle_vision_data
from app.services.shared_data import set_connected_audio_clients, set_audio_source, get_audio_source
from app.services.audio_chat_pipeline import pipeline
import main
# ---------------------------------------------------------------------------

router = APIRouter()
log = logging.getLogger(__name__)


# ───────────────────────── websocket entrypoint ─────────────────────────────
@router.websocket("/ws/{robot_id}/before/lecture")
async def before_lecture(
    ws: WebSocket,
    robot_id: str,
    mgr: ConnectionManager = Depends(get_conn_mgr),
) -> None:
    """
    Main handler for the before-lecture channel.
    """
    await mgr.connect(robot_id, ws)
    vision_task = asyncio.create_task(handle_vision_data("st_waiting", robot_id, ws))

    try:
        while True:
            raw = await ws.receive_json()
            msg = WSMessage.model_validate(raw)
            log.debug("[%s] 📥 %s", robot_id, msg)

            # ––– handshake --------------------------------------------------
            if msg.type == "register":
                role = (msg.data or {}).get("client")
                mgr.tag(ws, role or "unknown")
                if role == "audio":
                    set_connected_audio_clients(robot_id, ws)

                if role == "speech":
                    set_audio_source(robot_id, "speech")
                if role == "frontend":
                    set_audio_source(robot_id, "frontend")
                
                log.info("[%s] ➕ %s client registered", robot_id, role)
                continue

            # ––– keep-alive -------------------------------------------------
            if msg.type == "ping":
                await ws.send_json(
                    WSMessage(type="pong", data={}, ts=time.time()).model_dump()
                )
                continue

            # ––– main speech pipeline ---------------------------------------
            if msg.type == "speech":
                audio_source = get_audio_source(robot_id)
                result = await pipeline(robot_id, msg, audio_source)
                
                # Send user's transcribed text
                if audio_source == "frontend":
                    await mgr.send_role(robot_id, "frontend", result["in_msg"])
                elif audio_source == "speech":
                    await mgr.send_role(robot_id, "speech", result["in_msg"])

                # Send assistant's response
                out_msg = {
                    "robot_id": robot_id,
                    "type": "model",
                    "text": result["assistant_text"],
                    "audio": list(result["wav_bytes"]),
                    "ts": time.time(),
                }
                
                if audio_source == "frontend":
                    await mgr.send_role(robot_id, "frontend", out_msg)
                elif audio_source == "speech":
                    await mgr.send_role(robot_id, "audio", out_msg)
                    
                continue

            # ––– (future) audio->speech routing -----------------------------
            if msg.type == "audio":
                # TODO – if you ever send clips the other way.
                continue

            log.warning("[%s] ⚠️ unknown message type %s", robot_id, msg.type)

    except WebSocketDisconnect:
        log.info("[%s] client disconnected", robot_id)

    finally:
        mgr.disconnect(robot_id, ws)
        vision_task.cancel()
        try:
            await vision_task
        except asyncio.CancelledError:
            log.info("Vision task cancelled cleanly.")
