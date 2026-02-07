import os
import sys
import logging
import torch
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from config import settings
from schemas import ChatRequest, ChatCompletionResponse, ModelList, ModelCard, Choice, ChatMessage

# Logging
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AirLLM-Gateway")

# Configuration AMD GPU
os.environ["HSA_OVERRIDE_GFX_VERSION"] = settings.hsa_override_gfx_version


try:
    from airllm import AutoModel
except ImportError:
    logger.critical("❌ The 'airllm' library is missing.")
    sys.exit(1)

# Server state management
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application lifecycle (Startup -> Serving -> Shutdown).
    """
    logger.info(f"🚀 Initializing Gateway on {settings.host}:{settings.port}")
    logger.info(f"⚙️  AMD GPU Override: {settings.hsa_override_gfx_version}")
    logger.info(f"🧠 Loading model: {settings.model_id} ({settings.compression})")
    
    try:
        start_load = time.time()
        ml_models["llm"] = AutoModel.from_pretrained(
            settings.model_id,
            compression=settings.compression,
            profiling_mode=False
        )
        duration = time.time() - start_load
        logger.info(f"✅ Model loaded successfully in {duration:.2f}s")
        
    except Exception as e:
        logger.critical(f"🔥 Critical failure loading model: {e}")
        raise e

    # Service ready
    yield

    # Cleanup on shutdown
    logger.info("🛑 Shutting down Gateway. Cleaning up VRAM...")
    ml_models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- APPLICATION FASTAPI ---
app = FastAPI(
    title="AirLLM AMD Gateway",
    version="0.0.1",
    description="OpenAI-compatible API for Layer-wise Inference on AMD GPUs",
    lifespan=lifespan
)

# --- UTILS ---

def format_prompt_llama3(messages: list[ChatMessage]) -> str:
    """
    Applies the Llama-3 chat template manually.
    """
    prompt = "<|begin_of_text|>"
    for msg in messages:
        prompt += f"<|start_header_id|>{msg.role}<|end_header_id|>\n\n{msg.content}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

import time

# --- ENDPOINTS ---

@app.get("/health")
async def health_check():
    """
    Docstring for health_check
    """
    if "llm" not in ml_models:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ready", "model": settings.model_id}

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """
    Discovery Endpoint for LibreChat.
    """
    return ModelList(data=[
        ModelCard(id=settings.model_display_name, root=settings.model_id)
    ])

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatRequest):
    """Main inference endpoint."""
    
    # 1. Check if model is available
    if "llm" not in ml_models:
        raise HTTPException(status_code=503, detail="Model is loading or failed.")
    
    model = ml_models["llm"]
    logger.info(f"📩 Inference requested for: {request.model}")

    # 2. Prepare prompt
    formatted_prompt = format_prompt_llama3(request.messages)
    
    # 3. Tokenization
    try:
        input_tokens = model.tokenizer(
            [formatted_prompt],
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=settings.max_length,
            padding=False
        )
    except Exception as e:
        logger.error(f"Tokenization error: {e}")
        raise HTTPException(status_code=400, detail="Tokenization failed")

    # Inference (CPU RAM -> GPU VRAM -> CPU RAM)
    start_time = time.time()
    try:
        with torch.no_grad():
            generation_output = model.generate(
                input_tokens['input_ids'].cuda(),
                max_new_tokens=request.max_tokens,
                use_cache=True,
                return_dict_in_generate=True
            )
    except Exception as e:
        logger.error(f"AirLLM Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    
    duration = time.time() - start_time
    logger.info(f"⚡ GGeneration completed in {duration:.2f}s")

    raw_output = model.tokenizer.decode(generation_output.sequences[0])
    
    response_text = raw_output.replace(formatted_prompt, "")
    for special in ["<|eot_id|>", "<|begin_of_text|>", "<|end_of_text|>"]:
        response_text = response_text.replace(special, "")
    
    last_user_msg = request.messages[-1].content
    if last_user_msg in response_text:
         response_text = response_text.split(last_user_msg)[-1]

    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        model=request.model,
        choices=[
            Choice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text.strip()),
                finish_reason="stop"
            )
        ],
        usage={
            "prompt_tokens": len(input_tokens['input_ids'][0]),
            "completion_tokens": len(generation_output.sequences[0]),
            "total_tokens": 0
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=False)