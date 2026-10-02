"""
Scout's speaking voice: Google Cloud Text-to-Speech.

The browser does the listening (Web Speech API); this turns Scout's replies
into natural-sounding audio. The voice is a setting (SCOUT_TTS_VOICE).
"""

import logging

from google.api_core.exceptions import GoogleAPICallError
from google.cloud import texttospeech

logger = logging.getLogger(__name__)

FALLBACK_VOICE = "en-US-Neural2-F"


class SpeechSynthesizer:
    def __init__(self, voice_name: str):
        self._client = texttospeech.TextToSpeechClient()
        self._voice_name = voice_name

    def synthesize(self, text: str, speaking_rate: float = 1.0) -> bytes:
        """MP3 audio for the text. Falls back to a standard voice if the chosen one fails."""
        try:
            return self._synthesize(text, self._voice_name, speaking_rate)
        except GoogleAPICallError as error:
            logger.warning("Voice %s failed (%s); using %s.", self._voice_name, error.message, FALLBACK_VOICE)
            return self._synthesize(text, FALLBACK_VOICE, speaking_rate)

    def _synthesize(self, text: str, voice_name: str, speaking_rate: float) -> bytes:
        language_code = "-".join(voice_name.split("-")[:2])      # "en-US-Chirp3-HD-..." -> "en-US"
        response = self._client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=text),
            voice=texttospeech.VoiceSelectionParams(language_code=language_code, name=voice_name),
            audio_config=texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=max(0.5, min(speaking_rate, 2.0)),
            ),
        )
        return response.audio_content
