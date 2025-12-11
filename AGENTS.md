# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Build & Package Management
- **Python version**: `>=3.11.12, <3.13` (strict requirement)
- **Dependency management**: Use `uv sync` (not pip/poetry). Lock file: `uv.lock`
- **Install**: `uv sync --all-extras` for full dev environment

## Code Style
- **Ruff**: `line-length=120`, ignore E501 (handled automatically)
- **Linting**: `ruff check .` or `uv run ruff check .`

## Docker Build
- **Multi-platform**: Builds target `linux/amd64` and `linux/arm64`
- **Entrypoint**: `scripts/entrypoint.sh` auto-creates pgvector extension

## Testing
- **E2E tests**: `pytest tests/e2e_test/` (requires running databases)
- **Unit tests**: `pytest tests/unit_test/`
- **Graph storage tests**: See `tests/e2e_test/graphstorage/README.md` for setup

## Frontend
- **Framework**: Next.js 15.4.8 with Turbopack
- **API client**: Auto-generated from OpenAPI spec via `web/src/api/api.ts`
- **Dev server**: `npm run dev` (uses Turbopack)

## Key Configuration
- **Chunking**: `CHUNK_SIZE=400`, `CHUNK_OVERLAP=20` (non-standard defaults)
- **Celery**: `--pool=threads --concurrency=16`, uses DatabaseScheduler