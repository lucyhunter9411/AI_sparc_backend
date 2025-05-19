"""
Unit-tests for app.services.stt_service – no real HTTP or SR calls.
"""
from types import SimpleNamespace
import io, asyncio, pytest
import app.services.stt_service as stt


@pytest.fixture
def small_wav(tmp_path):
    """Create a 0.3-s silent WAV and return its path."""
    import wave
    p = tmp_path / "tiny.wav"
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\0\0" * int(0.3 * 16000))
    return p


def test_transcribe_openai(monkeypatch, small_wav):
    """OpenAI path returns JSON['text']."""
    def fake_post(_url, *, headers, data, files):
        class _Resp:
            status_code = 200
            def json(self): return {"text": "ok"}
        return _Resp()

    monkeypatch.setattr(stt, "requests", SimpleNamespace(post=fake_post))
    txt = asyncio.run(stt.transcribe_audio(str(small_wav), "whisper-1"))
    assert txt == "ok"


def test_transcribe_google(monkeypatch, small_wav):
    """Google SR fallback returns recognize_google value."""
    class _Rec:
        def record(self, src): return "dummy"
        def recognize_google(self, audio_data, language): return "hola"

    monkeypatch.setattr(
        stt,
        "sr",
        SimpleNamespace(
            Recognizer=lambda: _Rec(),
            AudioFile=lambda fp: io.BytesIO(),
        ),
    )
    txt = asyncio.run(stt.transcribe_audio(str(small_wav), "es-ES"))
    assert txt == "hola"
