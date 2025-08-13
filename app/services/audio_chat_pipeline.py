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
from app.services.shared_data import get_saveConv, set_qna_flag
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

def analyze_performance(timing_data: dict, robot_id: str):
    """Analyze pipeline performance and log insights."""
    total = timing_data["total"]
    components = {
        "STT": timing_data["stt"],
        "FAISS": timing_data["faiss"], 
        "LLM": timing_data["llm"],
        "TTS": timing_data["tts"]
    }
    
    # Find the slowest component
    slowest = max(components.items(), key=lambda x: x[1])
    
    # Calculate percentages
    percentages = {name: (duration/total)*100 for name, duration in components.items()}
    
    log.info("[%s] 📊 Performance Analysis - Slowest: %s (%.1f%%, %.3fs)", 
             robot_id, slowest[0], percentages[slowest[0]], slowest[1])
    
    # Alert if any component is taking too long
    if slowest[1] > 2.0:  # More than 2 seconds
        log.warning("[%s] ⚠️  %s is taking longer than expected: %.3fs", 
                   robot_id, slowest[0], slowest[1])

async def pipeline(robot_id: str, msg, audio_source) -> dict:
    """
    Full STT → LLM → TTS pipeline for one 'speech' clip.
    """
    pipeline_start = time.perf_counter()
    
    # 1. decode wav clip to a temp file
    # b64_clip: str = msg.data["audio"]
    # pcm_bytes = base64.b64decode(b64_clip)
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
    stt_start = time.perf_counter()
    stt_backend = msg.data.get("backend", "whisper-1")
    user_text = await transcribe_audio(wav_path, stt_backend)
    stt_duration = time.perf_counter() - stt_start

    in_msg: dict[str, Any] = {
        "type": "user",
        "text": user_text,
        "ts": time.time(),
    }

    log.info("[%s] 📝 STT completed in %.3fs - transcript: %s", robot_id, stt_duration, user_text)

    # update conversation history
    history = shared_data.conversations[robot_id]
    history.append(user_text)

    # Retrieve supporting context from FAISS
    faiss_start = time.perf_counter()
    retrieved_docs = main.faiss_text_db.similarity_search(user_text, k=5)
    faiss_duration = time.perf_counter() - faiss_start
    
    if retrieved_docs:
        log.info("[%s] 🔍 FAISS retrieval completed in %.3fs - %d docs found", robot_id, faiss_duration, len(retrieved_docs))
    else:
        log.info("[%s] 🔍 FAISS retrieval completed in %.3fs - no docs found", robot_id, faiss_duration)
        
    retrieved_texts = (
        "\n".join(sanitize_text(doc.page_content) for doc in retrieved_docs)
        if retrieved_docs else "No relevant context found."
    )

    # build prompt
    prompt = build_prompt(
        query=user_text,
        context=retrieved_texts,
        history=list(history),
        custom_template=getattr(main, "custom_prompt_template", ""),
    )

    # LLM prediction
    llm_start = time.perf_counter()
    assistant_text = await llm_predict(prompt)
    llm_duration = time.perf_counter() - llm_start

    history.append(assistant_text)
    log.info("[%s] 🤖 LLM prediction completed in %.3fs - response received", robot_id, llm_duration)

    # text-to-speech
    tts_start = time.perf_counter()
    wav_bytes = await _tts_bytes(assistant_text)
    tts_duration = time.perf_counter() - tts_start
    
    secs = len(wav_bytes) / (16_000 * 2)  # 16-kHz mono 16-bit
    log.info("[%s] 🔊 TTS completed in %.3fs - audio: %.2fs duration (%.1f kB)", robot_id, tts_duration, secs, len(wav_bytes) / 1024)
    
    set_qna_flag(robot_id, False)
    
    # Calculate total pipeline time
    total_duration = time.perf_counter() - pipeline_start
    
    # Log comprehensive timing summary
    log.info("[%s] ⏱️  Pipeline completed in %.3fs - Breakdown: STT=%.3fs, FAISS=%.3fs, LLM=%.3fs, TTS=%.3fs", 
             robot_id, total_duration, stt_duration, faiss_duration, llm_duration, tts_duration)
    
    # Performance analysis
    timing_data = {
        "total": total_duration,
        "stt": stt_duration,
        "faiss": faiss_duration,
        "llm": llm_duration,
        "tts": tts_duration
    }
    analyze_performance(timing_data, robot_id)
    
    # Return all processed data
    return {
        "user_text": user_text,
        "assistant_text": assistant_text,
        "wav_bytes": wav_bytes,
        "in_msg": in_msg,
        "audio_source": audio_source,
        "timing": timing_data
    }
    
