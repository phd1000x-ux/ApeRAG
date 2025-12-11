# AGENTS.md - Ask Mode

## Package Structure (Non-Obvious)

### `aperag/concurrent_control/`
- **Not** a standard asyncio wrapper - custom lock abstraction
- `LockManager`: Factory pattern for Redis vs Threading locks
- `RedisLock`: Distributed lock using Lua scripts (not SETNX pattern)
- `ThreadingLock`: Async wrapper for `threading.Lock` (for single-process)

### `aperag/llm/`
- LiteLLM integration with SHA256-based Redis caching
- Cache key generation in `litellm_cache_key.py` - non-standard hashing
- Multimodal support: `vision_llm.py`, `multimodal_embedding.py`

### `aperag/graphindex/`
- **LightRAG implementation**: Entity extraction, merging, graph storage
- `lightrag_ops/`: 3 storage backends (NetworkX, Postgres, Neo4j)
- Case normalization and entity merging in `lightrag_entity_extraction_and_merging.py`

### `aperag/websearch/`
- Custom provider system (`provider_manager.py`) - not a wrapper library
- LLM.txt parsing support (`llm_txt_provider.py`)
- Parallel search across multiple providers

### `aperag/docparser/`
- Chunking: `CHUNK_SIZE=400`, `CHUNK_OVERLAP=20` (non-standard)
- Multi-format support: PDF, DOCX, MD, etc.
- See `test_chunking.py` for expected behavior

## Confusing Names

### `aperag/mcp/`
- **MCP = Model Context Protocol**, not "Model Configuration Protocol"
- Servers in `aperag/mcp/servers/` - registered via `server_manager.py`
- Used for context management in LLM workflows

### `scripts/entrypoint.sh`
- **Critical**: Auto-creates pgvector extension - don't create manually
- Runs migrations and initialization
- Multi-platform support (amd64/arm64)

### `tests/e2e_test/graphstorage/`
- Contains oracle data (`graph_storage_oracle.py`) for correctness testing
- Requires all 3 backends (NetworkX, Postgres, Neo4j) to be available
- See `README.md` in that directory for setup

## Model Configuration

### `models/` directory
- Custom JSON format for Alibaba Bailian, OpenRouter
- `generate_model_configs.py`: Script to generate configs
- Not compatible with standard LiteLLM model lists

## Database Architecture

### Dual Engine System
- `get_db_engine()`: Sync engine (for Celery tasks)
- `get_async_db_engine()`: Async engine (for FastAPI)
- **Important**: Connection pools are separately configured

### Graph Storage Strategies
- `NetworkXStorage`: In-memory (default, for testing)
- `PostgresGraphStorage`: Uses native graph queries, not ORM
- `Neo4jStorage`: Direct Cypher queries
- Switch via `GRAPH_STORAGE_TYPE` env var