# Security and Privacy Notes

## Before publishing this repository

1. Do not commit `.env` or any file containing API keys.
2. Do not commit `data/chroma/`, `data/raw/`, or `data/feedback.db`.
3. Review `config/sources.yaml` and docs for any internal-only URLs before publishing updates.
4. Rotate any API keys that were ever stored outside your local machine.

## Runtime recommendations

- Set `ENABLE_API_DOCS=false` if you deploy the API publicly.
- Set `ALLOWED_ORIGINS` to the exact frontend origins you control.
- Keep feedback storage local unless you have a retention policy and user consent.
- Avoid storing raw production user questions in logs.
- Treat `data/feedback.db` as user data.

## Current repository protections

- `.env` and local data files are ignored by `.gitignore`
- session IDs written to feedback storage are hashed before persistence
- API docs can be disabled by environment variable
- CORS origins are configurable through environment variables

## Recommended next steps for a public deployment

- add authentication before exposing the API on the internet
- terminate TLS at a trusted reverse proxy
- move session and feedback storage to managed infrastructure with backups and access control
- define a retention period for feedback data
