DIMENSION = 1024
COLLECTION_NAME_PREFIX = "video_generator_"
TOP_K = 7 # For image, bgm, tts
TOP_K_FOOTAGE_OPENING = 3
TOP_K_FOOTAGE_REGULAR = 4
TOP_K_RECOMMENDATION = 10
TOP_K_MULT = 4
N_SCRIPT = 5
DEFAULT_TOTAL_DURATION: int = 20
MATERIAL_TYPE = ("footage_regular", "footage_opening", "image", "bgm")
MATERIAL_CN = {
    "footage_regular": "普通视频素材",
    "footage_opening": "开场白视频素材",
    "image": "图片",
    "bgm": "背景音乐"
}

TASK_TYPE_CN = {
    1: "普通成片",
    2: "普通成片重新生成",
    3: "爆款裂变",
    4: "爆款裂变重新生成"
}

DEFAULT_QUERY = "适合商业会展"
DEFAULT_FOOTAGE_DESC = "适合商业会展的视频素材"
DEFAULT_IMAGE_DESC = "适合商业会展的图片素材"
DEFAULT_BGM_DESC = "适合商业会展的背景音乐"
DEFAULT_TTS_DESC = "适合商业会展的解说音色"
DEFAULT_DURATION = 10.0


LOG_KEEPING_HOURS = 24
LOG_PREFIX = "video_director"