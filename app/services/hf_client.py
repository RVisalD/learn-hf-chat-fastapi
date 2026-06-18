import asyncio
from typing import AsyncGenerator, List, Optional
from app.config import settings
from app.schemas import Message

from huggingface_hub import InferenceClient


class HuggingFaceClient:
    """Wrapper around `huggingface_hub.InferenceClient` exposing async methods
    compatible with the rest of the codebase.
    """

    def __init__(self):
        # Create the InferenceClient using the configured API key (if present)
        api_key = settings.huggingface_api_key or None
        self.client = InferenceClient(api_key=api_key)

    @staticmethod
    def _model_with_policy(model_id: str) -> str:
        if not model_id:
            return model_id
        return model_id if ":" in model_id else f"{model_id}:fastest"

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        return_full_text: bool = False,
    ) -> str:
        model_id = model or settings.default_model
        model_id = self._model_with_policy(model_id)

        params = {
            "max_new_tokens": max_new_tokens or settings.max_new_tokens,
            "temperature": temperature or settings.temperature,
            "top_p": top_p or settings.top_p,
            "return_full_text": return_full_text,
        }

        def call():
            try:
                return self.client.text_generation(model=model_id, inputs=prompt, parameters=params)
            except TypeError:
                # Some versions of the SDK expect positional args instead of keywords
                return self.client.text_generation(model_id, prompt, params)

        result = await asyncio.to_thread(call)

        # Normalize possible return formats
        if isinstance(result, dict):
            # Some clients return {'generated_text': '...'} or {'generated_texts': [...]}
            if "generated_text" in result:
                return result["generated_text"]
            if "generated_texts" in result and result["generated_texts"]:
                return result["generated_texts"][0]
            # fallback to string conversion
            return str(result)

        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict) and "generated_text" in first:
                return first["generated_text"]
            return str(first)

        return str(result)

    async def chat(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        model_id = model or settings.default_model
        model_id = self._model_with_policy(model_id)

        # Convert messages to the shape expected by the InferenceClient
        payload_messages = [{"role": m.role, "content": m.content} for m in messages]

        def call():
            # Use chat completions API
            return self.client.chat.completions.create(model=model_id, messages=payload_messages)

        result = await asyncio.to_thread(call)

        # Attempt to extract text from common response shapes
        try:
            # Some responses include choices -> message
            choices = getattr(result, "choices", None) or (result.get("choices") if isinstance(result, dict) else None)
            if choices:
                first = choices[0]
                msg = first.get("message") if isinstance(first, dict) else getattr(first, "message", None)
                if msg:
                    return msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", str(msg))
        except Exception:
            pass

        # Fallback to stringifying the result
        return str(result)

    async def stream_generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """Best-effort streaming implementation. If the underlying client supports
        streaming, it will be used; otherwise yield the full completion once.
        """
        model_id = model or settings.default_model
        model_id = self._model_with_policy(model_id)

        params = {
            "max_new_tokens": max_new_tokens or settings.max_new_tokens,
            "temperature": temperature or settings.temperature,
            "top_p": top_p or settings.top_p,
        }

        # Try streaming via the client if available
        try:
            def call_stream():
                try:
                    return self.client.text_generation(model=model_id, inputs=prompt, parameters=params, stream=True)
                except TypeError:
                    return self.client.text_generation(model_id, prompt, params, True)

            stream_iter = await asyncio.to_thread(call_stream)
            # If stream_iter is an iterator/generator, iterate and yield items
            if hasattr(stream_iter, "__iter__"):
                for chunk in stream_iter:
                    # each chunk may be bytes/str/dict
                    if isinstance(chunk, dict) and "generated_text" in chunk:
                        yield chunk["generated_text"]
                    else:
                        yield str(chunk)
                return
        except Exception:
            # Fall through to non-streaming behavior
            pass

        # Fallback: produce the full text as a single chunk
        full = await self.generate(prompt=prompt, model=model_id, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        yield full


def get_hf_client() -> HuggingFaceClient:
    return HuggingFaceClient()