import json
from pathlib import Path
import requests
from requests import HTTPError, RequestException
from config.constant_config import DIMENSION, COLLECTION_NAME_PREFIX
from config.path_config import EMBED_SERVICE_URL
from functionals.logger import video_director_logger

def user_id_to_collection_name(user_id:str|int|None)->str:
    return COLLECTION_NAME_PREFIX + str(user_id)

async def embed_query(query:str) -> list:
    try:
        response = requests.post(
            f"{EMBED_SERVICE_URL}/embed",
            json={"input": query}
        )
        response.raise_for_status()  # Explicitly catches 4xx/5xx HTTP errors
        vector = response.json()["embeddings"][0]

        if not isinstance(vector, list):
            e_m = f"❌ 嵌入'{query[:5]}...'向量格式无效, 预期为 list, 实际为 {type(vector).__name__}"
            video_director_logger.error(e_m)
            raise ValueError(e_m)

        if len(vector) != DIMENSION:
            e_m = f"❌ '{query[:5]}...'嵌入向量维度错误, 预期为 {DIMENSION}, 实际为 {len(vector)}"
            video_director_logger.error(e_m)
            raise ValueError(e_m)

    except HTTPError as e:
        e_m = f"❌ 嵌入'{query[:5]}...'HTTP 错误 {e.response.status_code}: {e}"
        video_director_logger.error(e_m)
        raise RuntimeError(e_m) from e

    except RequestException as e:
        # Catches ConnectionError, Timeout, TooManyRedirects, etc.
        e_m = f"❌ 嵌入'{query[:5]}...'网络请求异常: {type(e).__name__} - {e}"
        video_director_logger.error(e_m)
        raise ConnectionError(e_m) from e

    except Exception as e:
        e_m = f"❌ 嵌入'{query[:5]}...'未知异常: {type(e).__name__} - {e}"
        video_director_logger.error(e_m)
        raise RuntimeError(e_m) from e

    return vector

def calculate_total_duration(script:str)->float:
    return 0.3*len(script) + 2

def load_tts_data(path_or_url):
    # Case 1: HTTPS or HTTP URL
    if isinstance(path_or_url, str) and path_or_url.startswith(("http://", "https://")):
        try:
            response = requests.get(path_or_url, timeout=10)
            response.raise_for_status()  # Raises HTTPError for 4xx/5xx responses
            return response.json()
        except RequestException as e:
            raise FileNotFoundError(f"❌ TTS data 在以下地址未找到: {path_or_url} — {e}")

    # Case 2: Local file path
    else:
        local_path = Path(path_or_url)
        if not local_path.exists():
            raise FileNotFoundError(f"❌ TTS data 在以下地址未找到: {path_or_url}")
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)

if __name__ == "__main__":
    from config.path_config import TTS_DATA_PATH
    data = load_tts_data(TTS_DATA_PATH)
    print(type(data))
    print(data)
    print(type(data[1]))
    print(data[1])