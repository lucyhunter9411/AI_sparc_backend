# app/utils/audio.py
"""
Audio helpers: text-to-speech synthesis & duration util.

Pulled verbatim from main.py so existing behaviour is unchanged.
A later step will thin this out further (e.g. inject cfg instead of os.getenv).
"""
from __future__ import annotations
import base64, io, os, wave, logging
from langdetect import detect
import azure.cognitiveservices.speech as speechsdk

logger = logging.getLogger(__name__)

def generate_audio_stream(text: str, language: str):
    subscription_key = os.getenv("SUBSCRIPTION_KEY", "")
    region           = os.getenv("REGION", "")
    speech_config    = speechsdk.SpeechConfig(subscription=subscription_key, region=region)

    lang_code = detect(text)
    lang_map  = {"en": "English", "hi": "Hindi", "te": "Telugu"}
    language  = lang_map.get(lang_code, "English")
    logger.info("language detected: %s", language)

    if language == "English":
        speech_config.speech_synthesis_voice_name = "en-IN-AashiNeural"
    elif language == "Hindi":
        speech_config.speech_synthesis_voice_name = "hi-IN-AnanyaNeural"
    elif language == "Telugu":
        speech_config.speech_synthesis_voice_name = "te-IN-ShrutiNeural"
    else:
        raise ValueError("Unsupported language")

    buffer = io.BytesIO()
    synth  = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=speechsdk.audio.AudioOutputConfig(stream=speechsdk.audio.PushAudioOutputStream(buffer))
    )
    result = synth.speak_text_async(text).get()
    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        raise RuntimeError(f"TTS failed – {result.reason}")

    buffer.seek(0)
    pcm = buffer.read()

    wav_stream = io.BytesIO()
    with wave.open(wav_stream, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit PCM
        wf.setframerate(16000)
        wf.writeframes(pcm)

    wav_stream.seek(0)
    return wav_stream


def get_audio_length(audio_stream) -> float | None:
    """
    Return duration in seconds for a 16 kHz mono 16-bit WAV/PCM stream.
    """
    try:
        pos = audio_stream.tell()
        audio_stream.seek(0)
        data = audio_stream.read()
        audio_stream.seek(pos)

        if not data:
            return None
        sample_rate, bit_depth, channels = 16000, 2, 1
        return len(data) / (sample_rate * channels * bit_depth)
    except Exception as exc:        # noqa: BLE001
        logger.error("Error getting audio length: %s", exc, exc_info=True)
        return None
