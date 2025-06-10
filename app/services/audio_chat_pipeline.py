import base64
import tempfile
import time
import logging
from typing import Any

import asyncio

from app.services.stt_service import transcribe_audio
from app.services.llm_service import build_chat_prompt as build_prompt
from app.services.llm_service import predict as llm_predict
from app.utils.censor import sanitize_text
from app.services.shared_data import get_saveConv
from app.services import shared_data
import main
import wave


log = logging.getLogger(__name__)


async def _tts_bytes(text: str, lang_hint: str = "") -> bytes:
    from app.utils.audio import generate_audio_stream
    wav_io = await asyncio.to_thread(generate_audio_stream, text, lang_hint)
    wav_io.seek(0)
    return wav_io.read()

async def save_audio_to_file(audio_bytes, robot_id):
    audio_path = f"{robot_id}received_audio.wav"
    with wave.open(audio_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_bytes)
    log.info(f"✅ Audio saved to {audio_path}")

async def pipeline(robot_id: str, msg, audio_source) -> dict:
    """
    Full STT → LLM → TTS pipeline for one 'speech' clip.
    Returns a dictionary containing the processed data.
    """
    # 1. decode wav clip to a temp file
    audio_data = msg.data["audio"]  # Get raw audio (no base64 decoding)
    pcm_bytes = bytes(audio_data)  # Convert list of integers back to bytes
    save_state = get_saveConv(robot_id)
    if save_state == "save":
        await save_audio_to_file(pcm_bytes, robot_id)
    else:
        pass
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(pcm_bytes)
        wav_path = tmp.name

    # 2. speech-to-text
    stt_backend = msg.data.get("backend", "whisper-1")
    user_text = await transcribe_audio(wav_path, stt_backend)

    in_msg: dict[str, Any] = {
        "type": "user",
        "text": user_text,
        "ts": time.time(),
    }

    log.info("[%s] 📝 transcript: %s", robot_id, user_text)

    # update conversation history
    history = shared_data.conversations[robot_id]
    history.append(user_text)

    # Retrieve supporting context from FAISS
    retrieved_docs = main.faiss_text_db.similarity_search(user_text, k=5)
    if retrieved_docs:
        log.info("FAISS match found:", retrieved_docs)
    retrieved_texts = (
        "\n".join(sanitize_text(doc.page_content) for doc in retrieved_docs)
        if retrieved_docs else "No relevant context found."
    )
    log.info("retrieved_texts:", retrieved_texts)

    # build prompt
    prompt = build_prompt(
        query=user_text,
        context=retrieved_texts,
        history=list(history),
        custom_template=getattr(main, "custom_prompt_template", ""),
    )

    # LLM prediction
    assistant_text = await llm_predict(prompt)
    history.append(assistant_text)
    log.info("🤖 LLM response received")

    # text-to-speech
    wav_bytes = await _tts_bytes(assistant_text)
    secs = len(wav_bytes) / (16_000 * 2)  # 16-kHz mono 16-bit
    log.info("[%s] 🔊 TTS %.2fs (%.1f kB)", robot_id, secs, len(wav_bytes) / 1024)
    
    # Return all processed data
    return {
        "user_text": user_text,
        "assistant_text": assistant_text,
        "wav_bytes": wav_bytes,
        "in_msg": in_msg,
        "audio_source": audio_source
    }
    
