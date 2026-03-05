import uuid
import os
import tempfile
import logging
import requests
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from TTS.api import TTS
from .config import (
    S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY,
    S3_BUCKET, S3_PUBLIC_URL, S3_INTERNAL_URL, S3_USE_PATH_STYLE,
)
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class TTSServiceError(Exception):
    """Base exception for all TTS service errors."""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


class ModelLoadError(TTSServiceError):
    """Raised when a TTS model cannot be loaded."""
    def __init__(self, model_name: str, cause: Exception):
        super().__init__(
            f"Failed to load model '{model_name}': {cause}",
            status_code=503,
        )


class TTSSynthesisError(TTSServiceError):
    """Raised when TTS/VC synthesis fails."""
    def __init__(self, cause: Exception):
        super().__init__(f"TTS synthesis failed: {cause}", status_code=500)


class S3ConnectionError(TTSServiceError):
    """Raised when the S3 endpoint is unreachable or misconfigured."""
    def __init__(self, cause: Exception):
        super().__init__(f"S3 connection error: {cause}", status_code=503)


class S3UploadError(TTSServiceError):
    """Raised when an S3 upload fails."""
    def __init__(self, cause: Exception):
        super().__init__(f"S3 upload failed: {cause}", status_code=502)


class ReferenceAudioDownloadError(TTSServiceError):
    """Raised when a reference audio URL cannot be fetched."""
    def __init__(self, url: str, cause: Exception):
        super().__init__(
            f"Failed to download reference audio from '{url}': {cause}",
            status_code=502,
        )


# ---------------------------------------------------------------------------
# Lazy model cache
# ---------------------------------------------------------------------------

models = {}

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using device: %s", device)


def get_model(model_name: str):
    if model_name not in models:
        logger.info("Loading TTS model: %s to use device %s", model_name, device)
        try:
            models[model_name] = TTS(model_name).to(device)
        except Exception as e:
            logger.error("Failed to load model '%s': %s", model_name, e, exc_info=True)
            raise ModelLoadError(model_name, e) from e
        logger.info("Model loaded: %s", model_name)
    return models[model_name]


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _s3_client():
    logger.debug(
        "Creating S3 client — endpoint=%s bucket=%s path_style=%s",
        S3_ENDPOINT, S3_BUCKET, S3_USE_PATH_STYLE,
    )
    kwargs = dict(
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )
    if S3_USE_PATH_STYLE:
        kwargs["config"] = Config(s3={"addressing_style": "path"})
    return boto3.client("s3", **kwargs)


def _ensure_bucket(s3) -> None:
    try:
        s3.head_bucket(Bucket=S3_BUCKET)
        logger.debug("Bucket '%s' already exists", S3_BUCKET)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("404", "NoSuchBucket"):
            logger.warning("Bucket '%s' not found — creating it", S3_BUCKET)
            s3.create_bucket(Bucket=S3_BUCKET)
            logger.info("Bucket '%s' created", S3_BUCKET)
        else:
            logger.error("Unexpected S3 error checking bucket: %s", e)
            raise


def upload_to_s3(local_path: str, key: str) -> str:
    logger.info("Uploading %s → s3://%s/%s", local_path, S3_BUCKET, key)
    try:
        s3 = _s3_client()
        _ensure_bucket(s3)
    except ClientError as e:
        logger.error("S3 bucket access error: %s", e, exc_info=True)
        raise S3ConnectionError(e) from e
    except Exception as e:
        logger.error("S3 client/config error: %s", e, exc_info=True)
        raise S3ConnectionError(e) from e

    try:
        s3.upload_file(local_path, S3_BUCKET, key)
    except ClientError as e:
        logger.error("S3 upload failed: %s", e, exc_info=True)
        raise S3UploadError(e) from e

    url = f"{S3_PUBLIC_URL}/{key}"
    logger.info("Upload complete: %s", url)
    return url


# ---------------------------------------------------------------------------
# Public service functions
# ---------------------------------------------------------------------------

def generate_tts(model_name: str, text: str) -> str:
    logger.info("generate_tts called — model=%s text=%r", model_name, text)
    model = get_model(model_name)  # raises ModelLoadError on failure

    os.makedirs("tmp", exist_ok=True)
    output_file = f"tmp/{uuid.uuid4()}.wav"
    try:
        logger.debug("Rendering TTS to %s", output_file)
        try:
            model.tts_to_file(text=text, file_path=output_file)
        except Exception as e:
            logger.error("TTS synthesis failed: %s", e, exc_info=True)
            raise TTSSynthesisError(e) from e
        logger.info("WAV generated: %s", output_file)

        key = f"tts/{uuid.uuid4()}.wav"
        return upload_to_s3(output_file, key)  # raises S3ConnectionError / S3UploadError
    finally:
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
            except OSError as e:
                logger.warning("Failed to remove temp file %s: %s", output_file, e)


def generate_vc(model_name: str, reference_audio_urls: list[str], text: str, language: str) -> str:
    logger.info("generate_vc called — model=%s text=%r language=%s refs=%d",
                model_name, text, language, len(reference_audio_urls))
    model = get_model(model_name)  # raises ModelLoadError on failure

    os.makedirs("tmp", exist_ok=True)
    output_file = f"tmp/{uuid.uuid4()}.wav"
    try:
        with tempfile.TemporaryDirectory() as ref_dir:
            local_reference_paths = []
            for i, url in enumerate(reference_audio_urls):
                if S3_INTERNAL_URL and S3_PUBLIC_URL and url.startswith(S3_PUBLIC_URL):
                    url = S3_INTERNAL_URL + url[len(S3_PUBLIC_URL):]

                local_path = os.path.join(ref_dir, f"ref_{i}.wav")
                logger.debug("Downloading reference audio %d: %s", i, url)
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                except requests.exceptions.Timeout as e:
                    logger.error("Timeout downloading reference audio %s: %s", url, e)
                    raise ReferenceAudioDownloadError(url, e) from e
                except requests.exceptions.ConnectionError as e:
                    logger.error("Connection error downloading reference audio %s: %s", url, e)
                    raise ReferenceAudioDownloadError(url, e) from e
                except requests.exceptions.HTTPError as e:
                    logger.error("HTTP error downloading reference audio %s: %s", url, e)
                    raise ReferenceAudioDownloadError(url, e) from e
                except requests.exceptions.RequestException as e:
                    logger.error("Error downloading reference audio %s: %s", url, e)
                    raise ReferenceAudioDownloadError(url, e) from e

                with open(local_path, "wb") as f:
                    f.write(response.content)
                local_reference_paths.append(local_path)

            logger.debug("Rendering voice-clone TTS to %s", output_file)
            try:
                model.tts_to_file(
                    text=text,
                    speaker_wav=local_reference_paths,
                    language=language,
                    file_path=output_file,
                )
            except Exception as e:
                logger.error("Voice clone synthesis failed: %s", e, exc_info=True)
                raise TTSSynthesisError(e) from e

        logger.info("Voice-clone WAV generated: %s", output_file)

        key = f"vc/{uuid.uuid4()}.wav"
        return upload_to_s3(output_file, key)  # raises S3ConnectionError / S3UploadError
    finally:
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
            except OSError as e:
                logger.warning("Failed to remove temp file %s: %s", output_file, e)
