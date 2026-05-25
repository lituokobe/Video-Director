import os
from pathlib import Path

# Get project folder dir
current_file = Path(__file__).resolve()
project_dir = current_file.parent.parent

# Define the paths
ENV_PATH = project_dir / ".env"
LOG_PATH = project_dir / "logs"
# TTS_DATA_PATH = project_dir / "data/tts_data.json"
TTS_DATA_PATH = r"http://videovueapi.km360.cn/static/materia/tts_data.json"

HOST = os.getenv("HOST_DOCKER_INTERNAL", "127.0.0.1") #HOST_DOCKER_INTERNAL exists in Docker deployment
EMBED_SERVICE_URL = f"http://{HOST}:8083"
MILVUS_URL = f"http://{HOST}:19530"