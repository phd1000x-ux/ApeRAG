# AGENTS.md - Code Mode

## Custom Utilities & Patterns

### Concurrent Control (`aperag/concurrent_control/`)
- **ThreadingLock**: For single-process async coordination
- **RedisLock**: For distributed locking across workers
- Use `LockManager.get_lock(key, lock_type)` - auto-selects based on Redis availability
- **Critical**: Never use standard `asyncio.Lock` for cross-worker tasks

### LLM Caching (`aperag/llm/`)
- LiteLLM + Redis caching enabled by default (1000x performance on repeated queries)
- Cache keys use SHA256 hash of (model, messages, temperature, etc.)
- **Important**: Modifying prompt structure invalidates cache - design prompts carefully
- Set `LITELLM_CACHE=False` to disable during development

### Database Patterns
- **Dual engines**: Sync (`get_db_engine()`) and Async (`get_async_db_engine()`)
- Connection pool: Custom settings in `aperag/db/database.py`
- **pgvector**: Extension auto-created by `scripts/entrypoint.sh` - don't create manually

### LightRAG Graph Storage
- 3 strategies: `NetworkXStorage`, `PostgresGraphStorage`, `Neo4jStorage`
- **Default**: `NetworkXStorage` (in-memory, for testing)
- **Production**: Use `PostgresGraphStorage` (set `GRAPH_STORAGE_TYPE=postgres`)
- Graph ops in `aperag/graphindex/lightrag_ops/`

## Hidden Dependencies
- **Celery tasks**: Import from `aperag.tasks.*` - auto-registered via `@shared_task`
- **MCP servers**: Located in `aperag/mcp/servers/` - registered in `aperag/mcp/server_manager.py`
- **Web search**: Custom provider system in `aperag/websearch/provider_manager.py`

## Code Style Enforcements
- Use `uv run ruff format .` before commits
- Line length: 120 (not 88/79)
- Import sorting: Ruff handles automatically