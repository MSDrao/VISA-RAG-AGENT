"""FastAPI application setup."""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from api.middleware.rate_limit import limiter
from api.routes import query as query_router
from api.routes import feedback as feedback_router
from api.routes.feedback import init_feedback_db

from retrieval.query_processor import QueryProcessor
from retrieval.retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker
from generation.llm_client import LLMClient
from guardrails.classifier import GuardrailClassifier
from guardrails.safety import SafetyLayer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api")

APP_TITLE = os.getenv("APP_TITLE", "U.S. Immigration Assistant")
API_VERSION = os.getenv("APP_VERSION", "1.0.0")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1")
ALLOW_DOCS = os.getenv("ENABLE_API_DOCS", "true").lower() == "true"
DEFAULT_ORIGINS = "http://localhost:8501,http://127.0.0.1:8501"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", DEFAULT_ORIGINS).split(",")
    if origin.strip()
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize heavy components once at startup."""
    logger.info("Starting up visa-rag-agent API...")

    init_feedback_db()

    logger.info("Loading QueryProcessor...")
    app.state.query_processor = QueryProcessor()

    logger.info("Loading HybridRetriever (+ BM25 index)...")
    app.state.retriever = HybridRetriever()

    logger.info("Loading CrossEncoderReranker...")
    app.state.reranker = CrossEncoderReranker()

    logger.info("Loading LLMClient...")
    app.state.llm_client = LLMClient()

    app.state.classifier = GuardrailClassifier()
    app.state.safety = SafetyLayer()

    logger.info("All components ready. API is live.")
    yield

    logger.info("Shutting down...")


app = FastAPI(
    title=APP_TITLE,
    description=(
        "Answers U.S. immigration and visa questions grounded in "
        "official government sources. Not legal advice."
    ),
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if ALLOW_DOCS else None,
    redoc_url="/redoc" if ALLOW_DOCS else None,
    openapi_url="/openapi.json" if ALLOW_DOCS else None,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

app.include_router(query_router.router,    prefix="/api/v1", tags=["Query"])
app.include_router(feedback_router.router, prefix="/api/v1", tags=["Feedback"])


@app.get("/health")
async def health():
    chunk_count = app.state.retriever.collection.count()
    return {
        "status": "ok",
        "chunks_in_index": chunk_count,
        "model": CHAT_MODEL,
        "ready": chunk_count > 0,
    }


@app.get("/")
async def root():
    return {
        "name": APP_TITLE,
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
    }
