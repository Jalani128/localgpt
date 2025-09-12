from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import asyncio
import json
import os

import logging
from inference import process_query

# configure module logger
logger = logging.getLogger("localgpt2.server")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


class QueryRequest(BaseModel):
    query: str


app = FastAPI(title="localgpt2-server", version="0.2")

# Allow CORS from localhost origins (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint for Railway
@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "localgpt2-server", "version": "0.2"}

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    try:
        body_bytes = await request.body()
        body_text = body_bytes.decode("utf-8") if body_bytes else ""
    except Exception:
        body_text = "<could not read body>"

    # restore the request stream for downstream consumers
    if 'body_bytes' in locals():
        async def receive() -> dict:
            return {"type": "http.request", "body": body_bytes}
        request._receive = receive  # type: ignore[attr-defined]

    logger.info(f"Incoming {request.method} {request.url} body={body_text}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code} for {request.method} {request.url}")
    return response


@app.post("/api/query")
async def post_query(req: QueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    try:
        # process_query is blocking; run it in a thread to avoid blocking the event loop
        logger.info(f"Processing query (endpoint): {req.query}")
        try:
            # run with timeout to prevent long blocking
            result = await asyncio.wait_for(asyncio.to_thread(process_query, req.query), timeout=30.0)
        except asyncio.TimeoutError:
            logger.error("process_query timed out")
            raise HTTPException(status_code=504, detail="processing timeout")
            
        # Log complete analytics
        logger.info("=== ANALYTICS ===")
        logger.info(f"Query: {req.query}")
        logger.info(f"Result keys: {list(result.keys())}")
        logger.info(f"State: {result.get('state')}")
        logger.info(f"Valid: {result.get('valid')}")
        logger.info(f"Provider count: {len(result.get('providers', []))}")
        logger.info(f"Usage report: {result.get('usage_report', {})}")
        logger.info(f"Complete JSON response: {json.dumps(result, indent=2, ensure_ascii=False)}")
        logger.info("=== END ANALYTICS ===")
        
        logger.info(f"Processed result for query: {req.query} -> keys={list(result.keys())}")
    except Exception as e:
        logger.exception("Error while processing query")
        raise HTTPException(status_code=500, detail=f"processing error: {e}")

    return result


if __name__ == "__main__":
    # Get port from environment variable (Railway sets this) or default to 8000
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    
    # Run with: python server.py
    uvicorn.run("server:app", host=host, port=port, log_level="info")
