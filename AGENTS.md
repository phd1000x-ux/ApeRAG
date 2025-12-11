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

## LLM & Embedding Configuration

### LiteLLM Proxy Configuration
- **Client-side limitations**: `LITELLM_DROP_PARAMS` doesn't work with proxy - requires server-side `drop_params: true`
- **Parameter handling**: All extra parameters are dropped by default to avoid encoding issues

### Custom Embedding Implementation
- **Direct HTTP requests**: Implemented in `aperag/llm/embed/embedding_service.py` to bypass LiteLLM for specific model combinations
- **Model selection**: Uses `custom_embedding_models` dictionary to define which model combinations should use direct HTTP
  ```python
  custom_embedding_models = {
      "openai/cohere.embed-multilingual-v3": True,  # Example model
  }
  ```
- **Decision logic**: `_should_use_direct_http()` method determines when to use direct HTTP requests based on provider/model
- **HTTP client**: Uses httpx client in `_embed_batch_direct_http()` method for direct API calls
- **Request format**: Standard OpenAI-compatible `/embeddings` endpoint with proper authentication

### Embedding Cache System
- **SHA256-based caching**: Redis-backed caching system with 1000x performance improvement (2000ms → 2ms)
- **Cache key generation**: `_generate_cache_key()` creates deterministic keys using provider, model, and texts
  ```python
  cache_data = {
      "provider": self.embedding_provider,
      "model": self.model,
      "texts": list(texts)
  }
  cache_string = json.dumps(cache_data, sort_keys=True)
  return hashlib.sha256(cache_string.encode()).hexdigest()
  ```
- **Cache operations**:
  - `_get_from_cache()` retrieves cached embeddings using the SHA256 key
  - `_cache_embeddings()` stores new embeddings with TTL from settings
- **Cache format**: Uses `embedding:{hash}` key pattern with JSON-encoded data including embeddings, timestamp, and metadata

### Cohere Models Configuration
- **Bedrock integration**: Requires `embedding_types: ["float"]` parameter for JSONArray format
- **Base64 encoding issues**: Cohere-specific parameters are currently commented out to avoid encoding problems
- **Provider support**: Handles both `bedrock` and `openai` providers when using Cohere models
- **Parameter handling**: `input_type` parameter is removed to prevent base64 encoding errors

### Redis Configuration
- **Authentication**: Uses `REDIS_PASSWORD` environment variable for secure connections
- **Connection settings**: Configured via `settings.redis_host`, `settings.redis_port`, and `settings.redis_password`
- **Fallback handling**: System continues to function if Redis is unavailable, with warnings logged
- **Connection testing**: Redis connection is tested during initialization with `ping()` command

### Error Handling
- **Custom error types**: Implements `EmbeddingError`, `BatchProcessingError`, and `EmptyTextError` for better error management
- **Base64 encoding**: Special handling for dimension probe requests to avoid base64 encoding issues
- **Empty content**: Detects and handles empty or whitespace-only content with appropriate error messages
- **Dimension validation**: Validates embedding dimensions and logs warnings for inconsistent results
- **HTTP error handling**: Proper exception handling for both LiteLLM and direct HTTP requests