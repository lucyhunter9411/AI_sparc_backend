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

import asyncio, logging, time, json
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
from app.services.shared_data import set_connected_audio_clients, set_audio_source, get_audio_source, get_saveConv
from app.services.audio_chat_pipeline import pipeline
import os
import base64
# ---------------------------------------------------------------------------

from dotenv import load_dotenv
from pathlib import Path
# explicitly point at your .env
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

router = APIRouter()
log = logging.getLogger(__name__)

CHUNK_SIZE = os.getenv("CHUNK_SIZE")

# Convert CHUNK_SIZE to an integer, with a default value if not set or invalid
try:
    CHUNK_SIZE = int(CHUNK_SIZE)
except (TypeError, ValueError):
    # Set a default value if CHUNK_SIZE is not set or is not a valid integer
    # WebSocket limit is 4MB (4194304), so use 2MB to account for JSON metadata overhead
    CHUNK_SIZE = 2097152  # 2MB - safe size accounting for JSON overhead


def chunk_audio(audio_data, chunk_size):
    """Split audio data into chunks with sequence numbers and total count."""
    total_chunks = (len(audio_data) + chunk_size - 1) // chunk_size
    chunks = []
    for i in range(total_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = audio_data[start:end]
        chunks.append({
            "sequence_number": i,
            "total_chunks": total_chunks,
            "data": list(chunk)
        })
    return chunks
    
def chunk_audio_base64(audio_data, chunk_size):
    """Split audio data into chunks with sequence numbers and total count."""
    total_chunks = (len(audio_data) + chunk_size - 1) // chunk_size
    chunks = []
    for i in range(total_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = audio_data[start:end]
        chunks.append({
            "sequence_number": i,
            "total_chunks": total_chunks,
            "data": base64.b64encode(chunk).decode('utf-8')  # Use base64 instead of list
        })
    return chunks

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
    from main import save_conv_into_db
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
                save_conv_flag = get_saveConv(robot_id)

                if save_conv_flag != "unsave":
                    await save_conv_into_db(result["user_text"], result["assistant_text"], db)
                    log.info(f"[{robot_id}]'s conversation is saved in the database successfully!")

                # Send user's transcribed text
                if audio_source == "frontend":
                    await mgr.send_role(robot_id, "frontend", result["in_msg"])
                    
                    audio_chunks = chunk_audio(result["wav_bytes"], CHUNK_SIZE)
                    log.info(f"[{robot_id}] Created {len(audio_chunks)} audio chunks for frontend")
                    
                    # Send each chunk with its metadata
                    for i, chunk in enumerate(audio_chunks):
                        out_msg = {
                            "robot_id": robot_id,
                            "type": "model",
                            "text": result["assistant_text"] if i == 0 else "",  # Only send text with first chunk
                            "audio_chunk": chunk,  # Send each chunk with its metadata
                            "ts": time.time(),
                            "image_path": None,  # Initially send None
                            "is_image_update": False # Not an image update yet
                        }
                        
                        # Log message size for monitoring
                        msg_size = len(json.dumps(out_msg).encode('utf-8'))
                        log.info(f"[{robot_id}] 🔊 Sending audio chunk {i+1}/{len(audio_chunks)} to frontend, message size: {msg_size:,} bytes")
                        
                        await mgr.send_role(robot_id, "frontend", out_msg)
                        log.info(f"[{robot_id}] ✅ Audio chunk {i+1}/{len(audio_chunks)} sent to frontend successfully")
                    
                    # Now handle image retrieval asynchronously and send image update
                    async def handle_image_retrieval():
                        try:
                            log.info(f"[{robot_id}] Starting retrieve_image call for frontend source")
                            log.info(f"[{robot_id}] User text: {result['user_text']}")
                            log.info(f"[{robot_id}] Assistant text: {result['assistant_text']}")
                            
                            closest_image_path = await retrieve_image_safe(result["user_text"], result["assistant_text"], robot_id, top_k=1)
                            log.info(f"[{robot_id}] retrieve_image completed successfully: {closest_image_path}")
                            
                            # Only send image update if we actually found an image
                            if closest_image_path:
                                # Send image update message to frontend (maintains compatibility)
                                image_update_msg = {
                                    "robot_id": robot_id,
                                    "type": "model",  # Use same type for compatibility
                                    "text": "",  # No additional text
                                    "audio_chunk": None,  # No additional audio
                                    "ts": time.time(),
                                    "image_path": closest_image_path,  # Include image path
                                    "is_image_update": True  # Flag to indicate this is an image update
                                }
                                await mgr.send_role(robot_id, "frontend", image_update_msg)
                                
                                # Also send to TV devices (maintains TV interface functionality)
                                try:
                                    await send_image_to_devices(robot_id, db, closest_image_path, log)
                                except Exception as e:
                                    log.error(f"[{robot_id}] send_image_to_devices failed: {e}", exc_info=True)
                            else:
                                log.info(f"[{robot_id}] No relevant image found - no WebSocket message sent (this is normal)")
                                
                        except Exception as e:
                            log.error(f"[{robot_id}] retrieve_image failed: {e}", exc_info=True)
                            log.info(f"[{robot_id}] Continuing without image due to retrieval failure or no relevant content")
                    
                    # Create task for image retrieval (non-blocking)
                    asyncio.create_task(handle_image_retrieval())
                    
                elif audio_source == "speech":
                    await mgr.send_role(robot_id, "speech", result["in_msg"])
                    
                    # Send audio immediately without waiting for image retrieval
                    log.info(f"[{robot_id}] 🔊 Starting audio sending to audio clients")
                    audio_chunks = chunk_audio_base64(result["wav_bytes"], CHUNK_SIZE)
                    log.info(f"[{robot_id}] 🔊 Sending {len(audio_chunks)} audio chunks to audio clients")
                    
                    # Send each chunk with its metadata
                    for i, chunk in enumerate(audio_chunks):
                        out_msg = {
                            "robot_id": robot_id,
                            "type": "model",
                            "text": result["assistant_text"] if i == 0 else "",  # Only send text with first chunk
                            "audio_chunk": chunk,  # Send each chunk with its metadata
                            "ts": time.time(),
                        }
                        
                        # Log message size for monitoring
                        msg_size = len(json.dumps(out_msg).encode('utf-8'))
                        log.info(f"[{robot_id}] 🔊 Sending audio chunk {i+1}/{len(audio_chunks)} to audio clients, message size: {msg_size:,} bytes")
                        
                        await mgr.send_role(robot_id, "audio", out_msg)
                    
                    log.info(f"[{robot_id}] ✅ Audio chunks sent to audio clients successfully")
                    
                    # Send assistant's response to frontend (without image initially)
                    out_msg = {
                        "robot_id": robot_id,
                        "type": "model",
                        "text": "",
                        "ts": time.time(),
                        "image_path": None  # Initially send None
                    }
                    await mgr.send_role(robot_id, "frontend", out_msg)
                    
                    # Now handle image retrieval asynchronously and send image update
                    async def handle_image_retrieval():
                        try:
                            log.info(f"[{robot_id}] Starting retrieve_image call for speech source")
                            log.info(f"[{robot_id}] User text: {result['user_text']}")
                            log.info(f"[{robot_id}] Assistant text: {result['assistant_text']}")
                            
                            closest_image_path = await retrieve_image_safe(result["user_text"], result["assistant_text"], robot_id, top_k=1)
                            log.info(f"[{robot_id}] retrieve_image completed successfully: {closest_image_path}")
                            
                            # Only send image update if we actually found an image
                            if closest_image_path:
                                # Send image update message to frontend
                                image_update_msg = {
                                    "robot_id": robot_id,
                                    "type": "model",
                                    "text": "",
                                    "ts": time.time(),
                                    "image_path": closest_image_path,
                                    "is_image_update": True
                                }
                                await mgr.send_role(robot_id, "frontend", image_update_msg)
                                
                                # Also send to TV devices
                                try:
                                    await send_image_to_devices(robot_id, db, closest_image_path, log)
                                except Exception as e:
                                    log.error(f"[{robot_id}] send_image_to_devices failed: {e}", exc_info=True)
                            else:
                                log.info(f"[{robot_id}] No relevant image found - no WebSocket message sent (this is normal)")
                                
                        except Exception as e:
                            log.error(f"[{robot_id}] retrieve_image failed: {e}", exc_info=True)
                            log.info(f"[{robot_id}] Continuing without image due to retrieval failure")
                    
                    # Create task for image retrieval (non-blocking)
                    asyncio.create_task(handle_image_retrieval())
                    
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
