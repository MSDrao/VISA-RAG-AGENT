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

## Main components

- API: `api/`
- Retrieval: `retrieval/`
- Ingestion: `ingestion/` and `scripts/ingest.py`
- Generation: `generation/`
- Live current-data handling: `services/live_official_data.py`
- Frontend: `frontend/app.py`
- Architecture walkthrough: `docs/project_architecture_walkthrough.md`
