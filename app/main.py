from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import chat, models
from app.config import settings

app = FastAPI(
    title="Hugging Face LLM API",
    description="FastAPI wrapper for Hugging Face Inference API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
# app.include_router(models.router, prefix="/api/v1", tags=["models"])


# @app.get("/", tags=["health"])
# async def root():
#     return {"status": "ok", "message": "Hugging Face LLM API is running"}


# @app.get("/health", tags=["health"])
# async def health_check():
#     return {"status": "healthy", "model": settings.default_model}