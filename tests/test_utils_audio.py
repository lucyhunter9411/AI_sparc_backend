"""
Unit-tests for app.utils.audio – fully offline.

We monkey-patch azure.cognitiveservices.speech so that:
* generate_audio_stream() writes 1 second of silent PCM to the buffer
* get_audio_length() returns ≈1 second
"""
from types import SimpleNamespace
import io, pytest
import app.utils.audio as audio


@pytest.fixture(autouse=True)
def fake_speech_sdk(monkeypatch):
    """Replace the parts of speechsdk that utils.audio touches."""
    SILENT_PCM = b"\0\0" * 16000        # 1 s of silent PCM

    # --- minimal fake SpeechConfig object ----------------------------------
    class _SpeechConfig:
        def __init__(self, *_, **__):
            self.speech_synthesis_voice_name = ""

    # --- fake synthesizer ---------------------------------------------------
    class _Synthesizer:
        def __init__(self, *, audio_config=None, **_):
            self._buffer = audio_config            # the BytesIO we pass in
        def speak_text_async(self, _txt):
            class _Future:
                def __init__(self, buf): self._buf = buf
                def get(self_self):
                    self_self._buf.write(SILENT_PCM)
                    return SimpleNamespace(reason="OK")
            return _Future(self._buffer)

    # -- FAKE speechsdk namespace -------------------------------------------
    fake_sdk = SimpleNamespace(
        SpeechConfig=_SpeechConfig,
        audio=SimpleNamespace(
            PushAudioOutputStream=lambda buf: buf,
            AudioOutputConfig=lambda *, stream=None, **__: stream,
        ),
        SpeechSynthesizer=_Synthesizer,
        ResultReason=SimpleNamespace(SynthesizingAudioCompleted="OK"),
    )
    monkeypatch.setitem(audio.__dict__, "speechsdk", fake_sdk)

def test_generate_audio_stream_length(monkeypatch):
    # Force language detection to a known code so branching passes
    monkeypatch.setattr(audio, "detect", lambda _txt: "en")

    stream = audio.generate_audio_stream("hello", "English")
    assert isinstance(stream, io.BytesIO)

    duration = audio.get_audio_length(stream)
    assert 0.9 < duration < 1.1          # ≈1 s for the silent PCM we wrote
