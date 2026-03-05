from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from .tts_service import (
    generate_tts, generate_vc,
    TTSServiceError, ModelLoadError, TTSSynthesisError,
    S3ConnectionError, S3UploadError, ReferenceAudioDownloadError,
)
from .config import API_KEY

app = FastAPI(title="AI TTS Worker")


@app.get("/health")
async def health():
    return {"status": "ok"}


class TTSRequest(BaseModel):
    modelName: str
    text: str
    outputFormat: str = "wav"


class VCRequest(BaseModel):
    modelName: str
    text: str
    language: str
    referenceAudioUrls: list[str]


def verify_api_key(x_api_key: str | None) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _ok(url: str) -> dict:
    return {"status": 200, "url": url, "message": "Audio generated successfully"}


def _error(status: int, message: str, exc_detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "exception": {"status_code": status, "message": exc_detail},
            "status": status,
            "url": None,
            "message": message,
        },
    )


@app.post("/api/generate-tts")
async def generate_tts_endpoint(req: TTSRequest, x_api_key: str = Header(None)):
    try:
        verify_api_key(x_api_key)
    except HTTPException as e:
        return _error(e.status_code, "Authentication failed", e.detail)

    try:
        url = generate_tts(req.modelName, req.text)
        return _ok(url)
    except ModelLoadError as e:
        return _error(e.status_code, "TTS model could not be loaded", str(e))
    except TTSSynthesisError as e:
        return _error(e.status_code, "Audio synthesis failed", str(e))
    except S3ConnectionError as e:
        return _error(e.status_code, "Storage service is unavailable", str(e))
    except S3UploadError as e:
        return _error(e.status_code, "Failed to upload generated audio", str(e))
    except TTSServiceError as e:
        return _error(e.status_code, "TTS service error", str(e))
    except Exception as e:
        return _error(500, "An unexpected error occurred", str(e))


@app.post("/api/generate-vc")
async def generate_vc_endpoint(req: VCRequest, x_api_key: str = Header(None)):
    try:
        verify_api_key(x_api_key)
    except HTTPException as e:
        return _error(e.status_code, "Authentication failed", e.detail)

    try:
        url = generate_vc(req.modelName, req.referenceAudioUrls, req.text, req.language)
        return _ok(url)
    except ModelLoadError as e:
        return _error(e.status_code, "TTS model could not be loaded", str(e))
    except ReferenceAudioDownloadError as e:
        return _error(e.status_code, "Failed to fetch reference audio", str(e))
    except TTSSynthesisError as e:
        return _error(e.status_code, "Voice cloning synthesis failed", str(e))
    except S3ConnectionError as e:
        return _error(e.status_code, "Storage service is unavailable", str(e))
    except S3UploadError as e:
        return _error(e.status_code, "Failed to upload generated audio", str(e))
    except TTSServiceError as e:
        return _error(e.status_code, "TTS service error", str(e))
    except Exception as e:
        return _error(500, "An unexpected error occurred", str(e))
