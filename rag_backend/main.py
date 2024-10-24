try:
    from rag_engine import RAGEngine as engine
    from logger import logger as logger
except ImportError:
    from .rag_engine import RAGEngine as engine
    from .logger import logger as logger

from contextlib import asynccontextmanager
import os
from fastapi import Depends, FastAPI, HTTPException, Header, Request, Response
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel

import redis.asyncio as redis

# Request validation
class PromptRequest(BaseModel):
    user_input: str
    few_shot: bool = True
    format_response: bool = True
    history: str = "",

# Lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        redis_client = redis.from_url(redis_url,
                                      encoding="utf-8",
                                      decode_responses=True,
                                      ssl_cert_reqs=None) # In production, we should use a proper certificate
        try:
            await FastAPILimiter.init(redis_client)
            yield
        except Exception as e:
            logger.error(f"Rate limiting will not be appled - Error initializing rate limiter: {e}")
            yield
        finally:
            if redis_client:
                await redis_client.close()
                yield
    else:
        logger.warning("No Redis URL found, rate limiting will not be applied")
        yield

# Rate limiting with Redis
async def get_rate_limiter():
    if FastAPILimiter.redis:
        return RateLimiter(times=10, seconds=60)
    else:
        return None

app = FastAPI(lifespan=lifespan)

@app.post("/prompt")
async def prompt(
    request: Request,
    response: Response,
    prompt_request: PromptRequest,
    openai_api_key: str = Header(..., alias="X-OpenAI-API-Key"),
    anthropic_api_key: str = Header(..., alias="X-Anthropic-API-Key"),
    rate_limiter: RateLimiter | None = Depends(get_rate_limiter)
):
    if rate_limiter:
        await rate_limiter(request, response)
    
    try:
        logger.info(f"Received request: {prompt_request}")
        
        eng = engine(openai_api_key=openai_api_key, anthropic_api_key=anthropic_api_key)
        
        result = await eng.get_answer(
            prompt_request.user_input, 
            prompt_request.few_shot, 
            prompt_request.format_response, 
            prompt_request.history
        )
        
        logger.info(f"Prompt response: {result}")
        
        return result
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e
