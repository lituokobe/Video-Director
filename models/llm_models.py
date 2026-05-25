import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from config.path_config import ENV_PATH

# Load environment variables
if ENV_PATH and os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)

# Get APIs
ALI_API_KEY = os.getenv("ALI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Get URLs
ALI_BASE_URL= "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# Set up LLM
llm_reasoning = ChatOpenAI(
    temperature=0,
    model='qwen3.6-max-preview', #'qwen3-max-2026-01-23', 'qwen3.6-max-preview'
    api_key=ALI_API_KEY,
    base_url=ALI_BASE_URL)

llm_regular = ChatOpenAI(
    temperature=0,
    model='qwen3.6-flash', #'qwen3.6-max-preview'
    api_key=ALI_API_KEY,
    base_url=ALI_BASE_URL)

if __name__ == "__main__":
    response = llm_regular.invoke("Hi, how are you?")
    print(response.content)