from fastapi import APIRouter, HTTPException, Depends
from app.schemas import ChatRequest, ChatResponse
from app.services.hf_client import HuggingFaceClient, get_hf_client
from app.config import settings

router = APIRouter()


@router.post("/chat", response_model=ChatResponse, summary="Chat with Meta Llama")
async def chat(
    request: ChatRequest,
    client: HuggingFaceClient = Depends(get_hf_client),
):
    """
    Send a list of messages to Meta Llama and receive a response.

    This API only supports the configured Meta Llama model.
    """
    if not settings.huggingface_api_key:
        raise HTTPException(status_code=500, detail="HUGGINGFACE_API_KEY is not configured.")

    try:
        generated_text = await client.chat(
            messages=request.messages,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Hugging Face API error: {str(e)}")

    return ChatResponse(model=settings.default_model, generated_text=generated_text)
