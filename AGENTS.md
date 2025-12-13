# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Build & Package Management
- **Python version**: `>=3.11.12, <3.13` (strict requirement)
- **Package manager**: Use `uv sync` (not pip/poetry). Lock file: `uv.lock`
- **Install**: `uv sync --all-extras` for full dev environment

## Code Style
- **Line length**: 120 (not standard 88/79)
- **Linting**: `ruff check .` or `uv run ruff check .`

## Docker
- **Multi-platform**: Builds target `linux/amd64` and `linux/arm64`
- **pgvector**: Auto-created by `scripts/entrypoint.sh` - don't create manually

## Frontend
- Next.js 15.4.8 with Turbopack
- API client auto-generated from OpenAPI spec via `web/src/api/api.ts`

## Non-Standard Defaults
- **Chunking**: `CHUNK_SIZE=400`, `CHUNK_OVERLAP=20` (most RAG systems use 1000/200)
- **Celery**: `--pool=threads --concurrency=16` (not gevent/prefork), uses DatabaseScheduler

## LLM & Embedding Configuration

### LiteLLM Proxy
- **Client-side limitations**: `LITELLM_DROP_PARAMS` doesn't work with proxy - requires server-side `drop_params: true`

### Custom Embedding Implementation
- **Direct HTTP bypass**: Implemented in `aperag/llm/embed/embedding_service.py` to bypass LiteLLM for specific model combinations
- **Model selection**: Uses `custom_embedding_models` dictionary to define which model combinations use direct HTTP
  ```python
  custom_embedding_models = {
      "openai/cohere.embed-multilingual-v3": True,
  }
  ```
- **Decision logic**: `_should_use_direct_http()` determines when to bypass LiteLLM

### Embedding Cache System
- **SHA256-based caching**: Redis-backed with 1000x performance improvement (2000ms → 2ms)
- **Cache key**: SHA256 hash of provider, model, and texts (see `_generate_cache_key()`)
- **Cache format**: `embedding:{hash}` key pattern with JSON-encoded data

### Cohere Models
- **Bedrock integration**: Requires `embedding_types: ["float"]` parameter for JSONArray format
- **Parameter handling**: `input_type` parameter removed to prevent base64 encoding errors

### Redis Configuration
- **Fallback handling**: System continues if Redis unavailable, with warnings logged
- Connection tested during initialization with `ping()` command

### Error Handling
- **Base64 encoding**: Special handling for dimension probe requests to avoid encoding issues
- Custom error types: `EmbeddingError`, `BatchProcessingError`, `EmptyTextError`

### Content Sanitization
- **Purpose**: Prevents `400 Bad Request` errors from embedding APIs when content contains special characters
- **Implementation**: `_sanitize_content()` method in `aperag/llm/embed/embedding_service.py`
- **Characters handled**:
  - `\xa0` (non-breaking space) → regular space
  - `\u200b` (zero-width space) → removed
  - `\u200c` (zero-width non-joiner) → removed
  - `\u200d` (zero-width joiner) → removed
  - `\ufeff` (BOM) → removed
  - Multiple consecutive spaces → single space
- **Applied to**: Direct HTTP embedding requests before sending to API

## Graph Index (LightRAG)

### Configuration
- **Location**: `aperag/graph/lightrag_manager.py` → `LightRAGConfig` class
- **Chunk token size**: `CHUNK_TOKEN_SIZE = 1024` (tokens per chunk)
- **Chunk overlap**: `CHUNK_OVERLAP_TOKEN_SIZE = 128`
- **Max batch size**: `MAX_BATCH_SIZE = 32`

### Embedding Batch Size for Graph Index
- **Environment variable**: `EMBEDDING_MAX_CHUNKS_IN_BATCH` (default: 10)
- **Recommended for large documents**: Set to `3` to avoid `400 Bad Request` errors
- **Location**: Set in `docker-compose.yml` under celeryworker service:
  ```yaml
  environment:
    - EMBEDDING_MAX_CHUNKS_IN_BATCH=3
  ```
- **Why needed**: Large markdown files with complex tables and special characters can cause embedding API failures when batch size is too large

### Troubleshooting Graph Index Failures
1. **400 Bad Request from embedding API**:
   - Reduce `EMBEDDING_MAX_CHUNKS_IN_BATCH` (try 3)
   - Ensure content sanitization is applied
   - Check for special characters (`\xa0`, emoji, etc.) in source documents

2. **Disk full errors** (`psycopg2.errors.DiskFull`):
   - Clean Docker resources: `docker builder prune -f` and `docker image prune -f`
   - Check disk usage: `docker exec aperag-postgres df -h /`

3. **Redis MISCONF errors**:
   - Restart Redis: `docker restart aperag-redis`
   - Usually occurs after disk was full and Redis couldn't save RDB snapshots

### Reconciliation
- **Task**: `reconcile_indexes_task` runs every hour (3600s) by default
- **Configuration**: `config/celery.py` → `beat_schedule`
- **Manual trigger**: Rebuild indexes via frontend UI or API

### Language Configuration
- **Default**: `"The same language like input text"` - extracts entities in the same language as the source document
- **Schema location**: `aperag/schema/view_models.py` → `KnowledgeGraphConfig.language`
- **Runtime location**: `aperag/graph/lightrag_manager.py` → uses `kg_config.language` from Collection settings
- **Override**: Can be set per-collection via frontend UI (Collection Settings → Knowledge Graph Config → Language)
- **Common values**: `"English"`, `"Korean"`, `"Chinese"`, `"The same language like input text"`
- **Note**: If entities are extracted in wrong language (e.g., English entities from Korean document), check Collection's `knowledge_graph_config.language` setting

### Entity Extraction Prompt Language Instructions
- **Location**: `aperag/graph/lightrag/prompt.py`
- **Prompts affected**: `entity_extraction`, `entity_continue_extraction`
- **Fields with language instruction** (`must use **same language** as input text`):
  - `entity_name`: Entity names match input text language
  - `entity_description`: Entity descriptions match input text language
  - `relationship_description`: Relationship descriptions match input text language
  - `relationship_keywords`: Relationship keywords match input text language
- **Behavior**: Korean text → Korean output, English text → English output, mixed → follows each section's language
- **Note**: Without explicit language instructions on each field, LLM tends to generate English descriptions/keywords even for non-English documents (due to English examples in prompt)