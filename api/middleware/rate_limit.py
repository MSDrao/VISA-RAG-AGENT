"""
Rate Limiting — api/middleware/rate_limit.py
Three-layer rate limiting strategy:
  Layer 1: Request rate (slowapi) — 10/min, 50/hr per IP
  Layer 2: Session token budget — tracked in app.state
  Layer 3: Upstream retries — handled in llm_client.py via tenacity
"""

import os
from slowapi import Limiter
from slowapi.util import get_remote_address

# Limiter instance — imported by main.py and all routes
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],          # Per-route limits only — no global default
)

QUERY_LIMIT       = os.getenv("RATE_LIMIT_PER_MINUTE", "10")
QUERY_LIMIT_HOUR  = os.getenv("RATE_LIMIT_PER_HOUR", "50")

MAX_TOKENS_PER_SESSION = int(os.getenv("MAX_TOKENS_PER_SESSION", 50000))
MAX_TURNS_PER_SESSION  = 20
