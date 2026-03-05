import os

API_KEY = os.getenv("WORKER_API_KEY", "da2sEDa3DegfZXx@#$_")

S3_ENDPOINT = os.getenv("S3_ENDPOINT")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_PUBLIC_URL = os.getenv("S3_PUBLIC_URL")
S3_INTERNAL_URL = os.getenv("S3_INTERNAL_URL")
S3_USE_PATH_STYLE = os.getenv("S3_USE_PATH_STYLE", "false").lower() == "true"
