from pathlib import Path

from mlx_audio.tts.utils import load_model, get_model_path
from pydantic import BaseModel, Field
from typing_extensions import override
import mlx.core as mx
import soundfile as sf

from ..chat.gpu_lock import mlx_generation_lock
from .schema import TTSRequest


# Cache for TTS models to avoid reloading on every request
class _TTSCache:
    def __init__(self):
        self.model = None
        self.model_path = None

    def get_model(self, model_path: str):
        if self.model is None or self.model_path != model_path:
            print(f"[TTS] Loading model: {model_path}")
            resolved = get_model_path(model_path)
            self.model = load_model(resolved)
            self.model_path = model_path
        return self.model

_tts_cache = _TTSCache()


class TTSModelAdapter(BaseModel):
    """Base class to adapt different TTS models to support the audio endpoint."""

    path_or_hf_repo: str | Path = Field(
        None, title="The path or the huggingface repository to load the model from."
    )

    def generate_audio(self, request: TTSRequest, output_path: str | Path) -> bool:
        """
        Generate audio from input text.
        ¨
        Args:
            request (TTSRequest): The request object containing the input text and other parameters.
            output_path (str | Path): The path to save the generated audio file.

        Returns:
            bool: True if the audio was generated successfully, False otherwise.
        """
        pass

    @classmethod
    def from_path_or_hf_repo(cls, path_or_hf_repo: str = None) -> "TTSModelAdapter":
        # Default to Kokoro if no model specified
        return MlxAudioModel(path_or_hf_repo=path_or_hf_repo or "mlx-community/Kokoro-82M-4bit")


class MlxAudioModel(TTSModelAdapter):
    path_or_hf_repo: str = Field("mlx-community/Kokoro-82M-4bit")

    @override
    def generate_audio(self, request: TTSRequest, output_path: str | Path) -> bool:
        self.path_or_hf_repo = request.model
        voice = request.voice if hasattr(request, "voice") else "af_sky"

        # Determine lang_code based on model type
        is_marvis = "marvis" in self.path_or_hf_repo.lower() or "sesame" in self.path_or_hf_repo.lower()

        if is_marvis:
            lang_code = None
        else:
            # Kokoro uses voice prefix as lang_code
            lang_code = voice[:1] if voice else "a"

        extra_params = request.get_extra_params() or {}

        # Use cached model - avoids reloading and HF HTTP requests
        model = _tts_cache.get_model(self.path_or_hf_repo)

        # Build generate kwargs
        gen_kwargs = dict(
            text=request.input,
            voice=voice,
            speed=request.speed,
            verbose=False,
            **extra_params,
        )
        if lang_code:
            gen_kwargs["lang_code"] = lang_code

        # Generate audio
        results = model.generate(**gen_kwargs)

        # Collect audio
        audio_list = []
        for result in results:
            audio_list.append(result.audio)

        if audio_list:
            audio = mx.concatenate(audio_list, axis=0) if len(audio_list) > 1 else audio_list[0]
            sf.write(str(output_path), audio, model.sample_rate)

        return Path(output_path).exists()


class TTSService:
    model: TTSModelAdapter

    def __init__(self, path_or_hf_repo: str | Path | None = None):
        self.model = TTSModelAdapter.from_path_or_hf_repo(path_or_hf_repo)
        self.sample_audio_path = Path("sample.wav")

    async def generate_speech(
        self,
        request: TTSRequest,
    ) -> bytes:
        try:
            # Use GPU lock to prevent concurrent Metal access (causes crashes)
            with mlx_generation_lock:
                self.model.generate_audio(
                    request=request, output_path=self.sample_audio_path
                )
            with open(self.sample_audio_path, "rb") as audio_file:
                audio_content = audio_file.read()
            self.sample_audio_path.unlink(missing_ok=True)
            return audio_content
        except Exception as e:
            raise Exception(f"Error reading audio file: {str(e)}")
