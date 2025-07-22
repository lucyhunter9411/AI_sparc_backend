from __future__ import annotations

import asyncio, logging, time
from typing import Any

from fastapi import APIRouter, Depends, WebSocket
from starlette.websockets import WebSocketDisconnect
from app.vector_db.image_retrieve_based_answer import  retrieve_image_safe

from app.api.deps import get_conn_mgr                  # → ConnectionManager singleton
from app.websockets.connection_manager import ConnectionManager
from app.schemas.ws import WSMessage
from app.services.tv_interface import send_image_to_devices
from app.api.deps import get_db

# --- services ---------------------------------------------------------------                   # conversation history
from app.services.vision_service import  handle_vision_data
from app.services.shared_data import set_connected_audio_clients, set_audio_source, get_audio_source
from app.services.audio_chat_pipeline import pipeline
import os
# ---------------------------------------------------------------------------

from dotenv import load_dotenv
from pathlib import Path
# explicitly point at your .env
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

router = APIRouter()
log = logging.getLogger(__name__)

# ───────────────────────── websocket entrypoint ─────────────────────────────
@router.websocket("/ws/{robot_id}/before/lecture")
async def before_lecture(
    ws: WebSocket,
    robot_id: str,
    mgr: ConnectionManager = Depends(get_conn_mgr),
    db=Depends(get_db)  # Use dependency injection to get the database session
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
                    # Create a task to run image_retrieve_based_answer independently
                    log.info(f"[{robot_id}] Starting retrieve_image call for frontend source")
                    log.info(f"[{robot_id}] User text: {result['user_text']}")
                    log.info(f"[{robot_id}] Assistant text: {result['assistant_text']}")
                    try:
                        closest_image_path_ini = await retrieve_image_safe(result["user_text"], result["assistant_text"], robot_id, top_k=1)
                        closest_image_path = closest_image_path_ini
                    except Exception as e:
                        log.error(f"[{robot_id}] retrieve_image failed: {e}", exc_info=True)
                        closest_image_path = None
                    log.info(f"[{robot_id}] retrieve_image completed successfully: {closest_image_path}")

                    # Make send_image_to_devices non-blocking so it doesn't prevent audio response
                    try:
                        await send_image_to_devices(robot_id, db, closest_image_path, log)
                    except Exception as e:
                        log.error(f"[{robot_id}] send_image_to_devices failed: {e}", exc_info=True)
                        # Continue execution even if image sending fails

                    # Send assistant's response
                    out_msg = {
                        "robot_id": robot_id,
                        "type": "model",
                        "text": result["assistant_text"],
                        "audio": list(result["wav_bytes"]),
                        "ts": time.time(),
                        "image_path": closest_image_path
                    }
                    await mgr.send_role(robot_id, "frontend", out_msg)
                    
                elif audio_source == "speech":
                    await mgr.send_role(robot_id, "speech", result["in_msg"])
                    # Create a task to run image_retrieve_based_answer independently
                    log.info(f"[{robot_id}] Starting retrieve_image call for speech source")
                    log.info(f"[{robot_id}] User text: {result['user_text']}")
                    log.info(f"[{robot_id}] Assistant text: {result['assistant_text']}")
                    try:
                        closest_image_path_ini = await retrieve_image_safe(result["user_text"], result["assistant_text"], robot_id, top_k=1)
                        closest_image_path = closest_image_path_ini
                    except Exception as e:
                        log.error(f"[{robot_id}] retrieve_image failed: {e}", exc_info=True)
                        closest_image_path = None
                    log.info(f"[{robot_id}] retrieve_image completed successfully: {closest_image_path}")
                    
                    # Make send_image_to_devices non-blocking so it doesn't prevent audio response
                    try:
                        await send_image_to_devices(robot_id, db, closest_image_path, log)
                    except Exception as e:
                        log.error(f"[{robot_id}] send_image_to_devices failed: {e}", exc_info=True)
                        # Continue execution even if image sending fails

                    # Send assistant's response
                    out_msg = {
                        "robot_id": robot_id,
                        "type": "model",
                        "text": "",
                        "ts": time.time(),
                        "image_path": closest_image_path
                    }
                    await mgr.send_role(robot_id, "frontend", out_msg)
                    
                    # Send assistant's response
                    out_msg = {
                        "robot_id": robot_id,
                        "type": "model",
                        "text": result["assistant_text"],
                        "audio": list(result["wav_bytes"]),
                        "ts": time.time(),
                    }
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
