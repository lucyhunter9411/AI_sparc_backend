"""
/ws/{robot_id}/before/lecture
=============================

• A client connects, sends **one** “register” frame to declare its role
  (``{"type":"register","data":{"client":"speech"}}`` or
   ``{"type":"register","data":{"client":"audio"}}``).

• “speech” frames contain a base-64 WAV clip; the backend

      1. runs STT (Whisper / Azure etc.),
      2. adds the user turn to the running conversation history,
      3. builds a prompt with the trimmed history,
      4. gets an LLM reply,
      5. appends the assistant turn to history,
      6. calls TTS,
      7. broadcasts the TTS WAV **only** to sockets that registered
         as *audio*.

• “audio” frames would be handled symmetrically if you ever send them
  the other way (left as a TODO).

• “ping” → “pong” is unchanged.

Everything heavy (STT, LLM, TTS) runs in an `asyncio.create_task()` so
the socket never blocks long enough to miss keep-alive pings.
"""

from __future__ import annotations

import asyncio, base64, json, logging, tempfile, time
from typing import Any

from fastapi import APIRouter, Depends, WebSocket
from starlette.websockets import WebSocketDisconnect

from app.api.deps import get_conn_mgr                  # → ConnectionManager singleton
from app.websockets.connection_manager import ConnectionManager
from app.schemas.ws import WSMessage

from app.utils.censor import sanitize_text
# --- services ---------------------------------------------------------------
from app.services.stt_service import transcribe_audio
from app.services.llm_service import build_chat_prompt as build_prompt
from app.services.llm_service import predict as llm_predict
from app.utils.audio import generate_audio_stream
from app.services import shared_data                           # conversation history
import main
# ---------------------------------------------------------------------------

router = APIRouter()
log = logging.getLogger(__name__)


# ───────────────────────── helper coroutines ────────────────────────────────
async def _tts_bytes(text: str, lang_hint: str = "") -> bytes:
    """
    Run the blocking TTS service in a worker thread; return WAV bytes.
    """
    wav_io = await asyncio.to_thread(generate_audio_stream, text, lang_hint)
    wav_io.seek(0)
    return wav_io.read()


async def _pipeline(
    robot_id: str,
    mgr: ConnectionManager,
    msg: WSMessage,
) -> None:
    """
    Full STT → LLM → TTS pipeline for one 'speech' clip.
    """

    # 1. decode wav clip to a temp file ------------------------------------------------
    b64_clip: str = msg.data["audio"]
    pcm_bytes = base64.b64decode(b64_clip)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(pcm_bytes)
        wav_path = tmp.name

    # 2. speech-to-text ---------------------------------------------------------------
    stt_backend = msg.data.get("backend", "whisper-1")
    user_text = await transcribe_audio(wav_path, stt_backend)
    log.info("[%s] 📝 transcript: %s", robot_id, user_text)

    # update conversation history --------------------------------------------------
   
    history = shared_data.conversations[robot_id]
    history.append(user_text)          # ← store plain string, not dict

    # ───── 2-b. Retrieve supporting context from FAISS  ───────────────
    retrieved_docs = main.faiss_text_db.similarity_search(user_text, k=5)
    if retrieved_docs: log.info("FIASS match found")
    retrieved_texts = (
       "\n".join(sanitize_text(doc.page_content) for doc in retrieved_docs)
        if retrieved_docs else "No relevant context found."             
    )

    # 3. build prompt --------------------------------------------------------------
    prompt = build_prompt(
        query=user_text,
        context=retrieved_texts,                           
        history=list(history),                      # deque → list[str]
        custom_template=getattr(main, "custom_prompt_template", ""),
    )
   
    # 4. LLM --------------------------------------------------------------------------
    assistant_text = await llm_predict(prompt)
    history.append(assistant_text)     # ← store plain string
    log.info("🤖 LLM response received")
    #log.info("[%s] 🤖 reply: %s", robot_id, assistant_text)

    # 5. text-to-speech ---------------------------------------------------------------
    wav_bytes = await _tts_bytes(assistant_text)
    secs = len(wav_bytes) / (16_000 * 2)      # 16-kHz mono 16-bit
    log.info("[%s] 🔊 TTS %.2fs (%.1f kB)", robot_id, secs, len(wav_bytes)/1024)

    # 6. broadcast to *audio* clients only -------------------------------------------
    out_msg: dict[str, Any] = {
        "robot_id": robot_id,
        "type":     "model",
        "text":     assistant_text,
        "audio":    base64.b64encode(wav_bytes).decode(),
        "ts":       time.time(),
    }
    await mgr.send_role(robot_id, "audio", out_msg)


# ───────────────────────── websocket entrypoint ─────────────────────────────
@router.websocket("/ws/{robot_id}/before/lecture")
async def lecture_ws(
    ws: WebSocket,
    robot_id: str,
    mgr: ConnectionManager = Depends(get_conn_mgr),
) -> None:
    """
    Main handler for the before-lecture channel.
    """
    await mgr.connect(robot_id, ws)

    try:
        while True:
            raw = await ws.receive_json()
            msg = WSMessage.model_validate(raw)
            log.debug("[%s] 📥 %s", robot_id, msg)

            # ––– handshake --------------------------------------------------
            if msg.type == "register":
                role = (msg.data or {}).get("client")
                mgr.tag(ws, role or "unknown")
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
                asyncio.create_task(_pipeline(robot_id, mgr, msg))
                await ws.send_json(
                    WSMessage(type="speech_ok", data={}, ts=time.time()).model_dump()
                )
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
