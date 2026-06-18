from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    huggingface_api_key: str = ""
    default_model: str = "meta-llama/Llama-3.2-8B-Instruct"
    hf_inference_url: str = "https://router.huggingface.co/hf-inference/models"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    request_timeout: int = 60

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()