"""
Speech-to-text (STT) adapter.

The logic is lifted unchanged from main.py so that later we can
mock / swap it without touching the WebSocket handler.
"""
from __future__ import annotations
import os, logging, base64, requests, wave
import speech_recognition as sr

logger          = logging.getLogger(__name__)
OPENAI_API_KEY  = os.getenv("OPENAI_KEY", "")

async def transcribe_audio(file_path: str, model: str) -> str:
    """
    Transcribe an audio file using either OpenAI Whisper endpoints
    *or* local Google Speech Recognition as a fallback.
    """
    if model in {"whisper-1", "gpt-4o-transcribe", "GPT-41-mini-transcribe"}:
        logger.info("Processing STT via OpenAI: %s", model)
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        with open(file_path, "rb") as f:
            files = {"file": (file_path, f, "audio/wav")}
            resp  = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers=headers,
                data={"model": model},
                files=files
            )
        if resp.status_code == 200:
            return resp.json().get("text", "")
        logger.error("❌ STT error %s – %s", resp.status_code, resp.text)
        return "Transcription failed."

    # ───── local recogniser fallback ────────────
    recogniser = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as src:
            audio = recogniser.record(src)
        return recogniser.recognize_google(audio, language=model)
    except Exception as exc:                                # noqa: BLE001
        logger.error("Local STT failed: %s", exc, exc_info=True)
        return "Transcription failed."
