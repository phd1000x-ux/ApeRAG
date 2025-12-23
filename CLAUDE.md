# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

### Package Management
- **Python version**: `>=3.11.12, <3.13` (strict requirement)
- **Package manager**: Use `uv sync` (not pip/poetry). Lock file: `uv.lock`
- **Full install**: `uv sync --all-groups --all-extras`
- **Setup dev environment**: `make dev` (creates venv, installs tools, hooks)
- **Activate venv**: `source .venv/bin/activate`

### Running Services
```bash
# Start infrastructure (PostgreSQL, Redis, Qdrant, Elasticsearch)
make compose-infra

# Apply database migrations
make migrate

# Run backend (FastAPI on port 8000)
make run-backend

# Run Celery worker (background tasks)
make run-celery

# Run frontend (Next.js on port 3000)
make run-frontend
```

### Testing
```bash
make test                # All tests
make unit-test           # Unit tests only (fast, no external deps)
make e2e-test            # E2E tests (requires running services)

# Single test file
uv run pytest tests/unit_test/test_model_service.py -v

# Single test function
uv run pytest tests/unit_test/test_model_service.py::TestModelService::test_get_models -v
```

### Code Quality
```bash
make format              # Auto-fix with ruff
make lint                # Check with ruff
make static-check        # MyPy type checking

# Or directly:
uvx ruff check --fix ./aperag ./tests
uvx ruff format ./aperag ./tests
```

### Code Generation
```bash
make generate-models           # OpenAPI → Pydantic models (aperag/schema/view_models.py)
make generate-frontend-sdk     # OpenAPI → TypeScript client (web/src/api/)
make makemigration             # Generate Alembic migration
```

## Architecture Overview

ApeRAG is a production-ready RAG platform with these core components:

### Backend (FastAPI + Celery)
- **Entry point**: `aperag/app.py` - FastAPI application
- **API specs**: `aperag/api/` - OpenAPI YAML specs defining all endpoints
- **Views**: `aperag/views/` - Request handlers
- **Services**: `aperag/service/` - Business logic layer
- **DB models**: `aperag/db/models.py` - SQLAlchemy/SQLModel definitions
- **Migrations**: `aperag/migration/` - Alembic migrations
- **Tasks**: `aperag/tasks/` - Celery async tasks
- **Config**: `config/celery.py` - Celery configuration with beat schedules

### Index Types
Five index types in `aperag/index/`:
- **Vector**: Qdrant-based semantic search
- **Full-text**: Elasticsearch
- **Graph**: Modified LightRAG implementation in `aperag/graph/`
- **Summary**: Document summaries for retrieval
- **Vision**: Image/chart analysis

### LLM Integration
- **LiteLLM proxy**: `aperag/llm/` - Unified LLM interface
- **Custom embeddings**: `aperag/llm/embed/embedding_service.py` - Direct HTTP bypass for specific models
- **Embedding cache**: SHA256-based Redis caching (`embedding:{hash}` key pattern)

### Frontend (Next.js 15)
- **Location**: `web/`
- **API client**: Auto-generated from OpenAPI at `web/src/api/api.ts`
- **Build**: Uses Turbopack

### Data Flow
1. Documents uploaded → `aperag/source/` handles ingestion
2. Parsing → `aperag/docparser/` extracts content (optional DocRay for complex docs)
3. Indexing → Celery tasks create vector/fulltext/graph indexes
4. Query → `aperag/query/` orchestrates hybrid retrieval
5. Chat → `aperag/chat/` handles conversation with retrieved context

## Non-Standard Defaults
- **Line length**: 120 (ruff configured in pyproject.toml)
- **Chunking**: `CHUNK_SIZE=400`, `CHUNK_OVERLAP=20` (smaller than typical RAG)
- **Celery**: `--pool=threads --concurrency=16` (thread pool, not gevent/prefork)
- **Embedding batch**: `EMBEDDING_MAX_CHUNKS_IN_BATCH=3` for large documents

## Key Configuration Notes

### LiteLLM
- `LITELLM_DROP_PARAMS` doesn't work client-side with proxy - requires server-side `drop_params: true`

### Graph Index (LightRAG)
- Config: `aperag/graph/lightrag_manager.py` → `LightRAGConfig`
- Chunk tokens: 1024, overlap: 128
- Entity extraction language: Defaults to match input text language
- Prompts: `aperag/graph/lightrag/prompt.py`

### Content Sanitization
`_sanitize_content()` in embedding service removes: `\xa0`, `\u200b`, `\u200c`, `\u200d`, `\ufeff`

## Docker Compose Profiles
```bash
make compose-up                           # Full application
make compose-up WITH_NEO4J=1              # Add Neo4j for graph storage
make compose-up WITH_DOCRAY=1             # Add DocRay parsing
make compose-up WITH_DOCRAY=1 WITH_GPU=1  # DocRay with GPU
make compose-down                         # Stop services
make compose-down REMOVE_VOLUMES=1        # Stop and delete data
```

## Troubleshooting

### Graph Index Failures
- **400 Bad Request**: Reduce `EMBEDDING_MAX_CHUNKS_IN_BATCH` (try 3)
- **Disk full**: `docker builder prune -f && docker image prune -f`
- **Redis MISCONF**: `docker restart aperag-redis`

### Wrong Entity Language
Check Collection's `knowledge_graph_config.language` setting - should be "The same language like input text"

## Change Log

### 2024-12-23: Knowledge Graph 언어 기본값 수정
- **문제**: 한글 문서를 graph index 시 엔티티가 영어로 추출됨
- **원인**: `collection.yaml`의 `knowledge_graph_config.language` 기본값이 "English"로 설정되어 있었음
- **해결**: 기본값을 "The same language like input text"로 변경
- **수정 파일**:
  - `aperag/api/components/schemas/collection.yaml` (라인 119, 120, 160)
  - `aperag/schema/view_models.py` (make generate-models로 자동 생성)
- **참고**: 기존에 생성된 Collection은 DB에 저장된 값("English")이 유지됨. 새로 생성되는 Collection에만 적용됨.
