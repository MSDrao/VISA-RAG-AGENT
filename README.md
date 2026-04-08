# U.S. Immigration Assistant

A retrieval-augmented application for U.S. immigration and visa questions, grounded in official government sources and selected institutional guidance.

## Ownership

Copyright (c) 2026 Sandesh Rao and Sowmya.
All rights reserved.

See [LICENSE](LICENSE).

## Repository notes

- This project is intended to answer general educational questions, not provide legal advice.
- Do not commit `.env`, local vector-store data, or feedback data.
- Review [SECURITY.md](SECURITY.md) before publishing or deploying.

## Local environment variables

Copy `.env.example` to `.env` and fill in the required values.

## Free deployment

The most practical free deployment path for the current architecture is a Docker-based Hugging Face Space.

Why this option:
- it fits the existing split architecture better than forcing a full refactor
- it supports secret management
- it can run the FastAPI backend and Streamlit frontend in one container

Files included for this:
- `Dockerfile`
- `.dockerignore`
- `scripts/start_hf_space.sh`

### Important deployment constraint

This app needs a populated Chroma index to answer normal RAG questions well.

That means a free deployment is only fully useful if one of these is true:

1. you include a prebuilt `data/chroma/` snapshot in the deployment source, or
2. you use external/persistent storage for the vector data, or
3. you accept that only live operational questions will work reliably at first

Free Hugging Face Spaces do not give you reliable persistent local storage by default, so this is best treated as:
- a demo deployment
- a prototype deployment
- not a production deployment

### Hugging Face Space setup

1. Create a new Hugging Face Space
2. Choose `Docker` as the SDK
3. Push this repository to the Space
4. Add secrets in the Space settings:
   - `OPENAI_API_KEY`
   - `OPENAI_CHAT_MODEL`
   - `LANGFUSE_PUBLIC_KEY` if you use it
   - `LANGFUSE_SECRET_KEY` if you use it
   - `LANGFUSE_HOST` if you use it

Recommended values:

```text
OPENAI_CHAT_MODEL=gpt-4.1
ENABLE_API_DOCS=false
```

### If you want the deployed demo to answer more than live-data questions

You should prepare a deployment-safe Chroma snapshot and make it available to the container at `data/chroma/`.

Without that, questions that depend on the main RAG index will have little or no useful context.

## Main components

- API: `api/`
- Retrieval: `retrieval/`
- Ingestion: `ingestion/` and `scripts/ingest.py`
- Generation: `generation/`
- Live current-data handling: `services/live_official_data.py`
- Frontend: `frontend/app.py`
- Architecture walkthrough: `docs/project_architecture_walkthrough.md`
