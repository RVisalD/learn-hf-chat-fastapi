from fastapi import APIRouter
from app.schemas import ModelInfo
from app.config import settings

router = APIRouter()

RECOMMENDED_MODELS = [
    ModelInfo(
        model_id="Qwen/Qwen2.5-72B-Instruct",
        description="Qwen 72B instruct-tuned — fast, capable, great for chat",
        task="text-generation",
    ),
    ModelInfo(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        description="Meta Llama 3.2 3B instruct — compact and efficient",
        task="text-generation",
    ),
    ModelInfo(
        model_id="microsoft/Phi-3-mini-4k-instruct",
        description="Microsoft Phi-3 mini — small but surprisingly capable",
        task="text-generation",
    ),
    ModelInfo(
        model_id="google/flan-t5-large",
        description="Google FLAN-T5 large — good for summarization & Q&A",
        task="text2text-generation",
    ),
    ModelInfo(
        model_id="HuggingFaceH4/zephyr-7b-beta",
        description="Zephyr 7B — fine-tuned for helpful chat responses",
        task="text-generation",
    ),
]


@router.get("/models", response_model=list[ModelInfo], summary="List recommended models")
async def list_models():
    """Returns a curated list of models that work well with this API."""
    return RECOMMENDED_MODELS


@router.get("/models/default", summary="Get the currently configured default model")
async def get_default_model():
    return {"default_model": settings.default_model}