from __future__ import annotations

import asyncio, logging, time
import zlib  # For compression
from typing import Any

from fastapi import APIRouter, Depends, WebSocket
from starlette.websockets import WebSocketDisconnect
from app.vector_db.image_retrieve_based_answer import retrieve_image_safe

from app.api.deps import get_conn_mgr
from app.websockets.connection_manager import ConnectionManager
from app.schemas.ws import WSMessage
from app.services.tv_interface import send_image_to_devices
from app.api.deps import get_db

from app.services.vision_service import handle_vision_data
from app.services.shared_data import set_connected_audio_clients, set_audio_source, get_audio_source
from app.services.audio_chat_pipeline import pipeline
import os

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

router = APIRouter()
log = logging.getLogger(__name__)

async def send_audio_in_chunks(robot_id, mgr, audio_data, role, chunk_size=1024):
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i + chunk_size]
        await mgr.send_role(robot_id, role, {"type": "audio_chunk", "data": chunk})
        await asyncio.sleep(0.01)

async def process_and_send_audio(robot_id, mgr, result, audio_source, closest_image_path=None):
    compressed_audio = zlib.compress(result["wav_bytes"])

    if audio_source == "frontend":
        await send_audio_in_chunks(robot_id, mgr, compressed_audio, "frontend")

        out_msg = {
            "robot_id": robot_id,
            "type": "model",
            "text": result["assistant_text"],
            "ts": time.time(),
            "image_path": closest_image_path
        }
        await mgr.send_role(robot_id, "frontend", out_msg)

    elif audio_source == "speech":
        await send_audio_in_chunks(robot_id, mgr, compressed_audio, "audio")

        image_msg = {
            "robot_id": robot_id,
            "type": "model",
            "text": "",
            "ts": time.time(),
            "image_path": closest_image_path
        }
        await mgr.send_role(robot_id, "frontend", image_msg)

        audio_msg = {
            "robot_id": robot_id,
            "type": "model",
            "text": result["assistant_text"],
            "ts": time.time()
        }
        await mgr.send_role(robot_id, "audio", audio_msg)

@router.websocket("/ws/{robot_id}/before/lecture")
async def before_lecture(
    ws: WebSocket,
    robot_id: str,
    mgr: ConnectionManager = Depends(get_conn_mgr),
    db=Depends(get_db)
) -> None:
    await mgr.connect(robot_id, ws)
    vision_task = asyncio.create_task(handle_vision_data("st_waiting", robot_id, ws))

    try:
        while True:
            raw = await ws.receive_json()
            msg = WSMessage.model_validate(raw)
            log.debug("[%s] 📥 %s", robot_id, msg)

            if msg.type == "register":
                role = (msg.data or {}).get("client")
                mgr.tag(ws, role or "unknown")
                if role == "audio":
                    set_connected_audio_clients(robot_id, ws)
                if role in ["speech", "frontend"]:
                    set_audio_source(robot_id, role)
                log.info("[%s] ➕ %s client registered", robot_id, role)
                continue

            if msg.type == "ping":
                await ws.send_json(
                    WSMessage(type="pong", data={}, ts=time.time()).model_dump()
                )
                continue

            if msg.type == "speech":
                audio_source = get_audio_source(robot_id)
                result = await pipeline(robot_id, msg, audio_source)

                await mgr.send_role(robot_id, audio_source, result["in_msg"])

                log.info(f"[{robot_id}] Starting retrieve_image call for {audio_source} source")
                log.info(f"[{robot_id}] User text: {result['user_text']}")
                log.info(f"[{robot_id}] Assistant text: {result['assistant_text']}")
                try:
                    closest_image_path_ini = await retrieve_image_safe(result["user_text"], result["assistant_text"], robot_id, top_k=1)
                    closest_image_path = closest_image_path_ini
                except Exception as e:
                    log.error(f"[{robot_id}] retrieve_image failed: {e}", exc_info=True)
                    closest_image_path = None
                log.info(f"[{robot_id}] retrieve_image completed successfully: {closest_image_path}")

                await send_image_to_devices(robot_id, db, closest_image_path, log)

                await process_and_send_audio(robot_id, mgr, result, audio_source, closest_image_path)
                continue

            if msg.type == "audio":
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
