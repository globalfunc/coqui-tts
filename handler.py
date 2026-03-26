import runpod
from app.tts_service import (
    generate_tts, generate_vc,
    TTSServiceError, ModelLoadError, TTSSynthesisError,
    S3ConnectionError, S3UploadError, ReferenceAudioDownloadError,
)


def handler(job):
    job_input = job["input"]
    job_type = job_input.get("type")

    try:
        if job_type == "tts":
            model_name = job_input["modelName"]
            text = job_input["text"]
            url = generate_tts(model_name, text)
            return {"status": 200, "url": url, "message": "Audio generated successfully"}

        elif job_type == "vc":
            model_name = job_input["modelName"]
            text = job_input["text"]
            language = job_input["language"]
            reference_audio_urls = job_input["referenceAudioUrls"]
            url = generate_vc(model_name, reference_audio_urls, text, language)
            return {"status": 200, "url": url, "message": "Audio generated successfully"}

        else:
            return {"error": f"Unknown job type: {job_type!r}. Expected 'tts' or 'vc'."}

    except ModelLoadError as e:
        return {"error": str(e), "status": e.status_code, "message": "TTS model could not be loaded"}
    except ReferenceAudioDownloadError as e:
        return {"error": str(e), "status": e.status_code, "message": "Failed to fetch reference audio"}
    except TTSSynthesisError as e:
        return {"error": str(e), "status": e.status_code, "message": "Audio synthesis failed"}
    except S3ConnectionError as e:
        return {"error": str(e), "status": e.status_code, "message": "Storage service is unavailable"}
    except S3UploadError as e:
        return {"error": str(e), "status": e.status_code, "message": "Failed to upload generated audio"}
    except TTSServiceError as e:
        return {"error": str(e), "status": e.status_code, "message": "TTS service error"}
    except Exception as e:
        return {"error": str(e), "status": 500, "message": "An unexpected error occurred"}


runpod.serverless.start({"handler": handler})
