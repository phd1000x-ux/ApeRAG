# AGENTS.md - Architect Mode

## Architecture Constraints (Non-Standard)

### LightRAG Graph Storage Design
- **3 backend strategies**: NetworkXStorage, PostgresGraphStorage, Neo4jStorage
- **Default (NetworkXStorage)**: In-memory, no persistence - testing only
- **Production (PostgresGraphStorage)**: Native graph queries, not traditional ORM
- **Trade-offs**:
  - NetworkX: Fast, no I/O, but no persistence
  - Postgres: Persistent, SQL-based, moderate performance
  - Neo4j: Best for graph traversals, requires separate service
- Switch via `GRAPH_STORAGE_TYPE` env var - no runtime switching

### Database Connection Pool Architecture
- **Dual engine system**: Sync + Async engines (not shared pool)
- Custom pool sizes in `aperag/db/database.py` - different from SQLAlchemy defaults
- **Critical**: Celery tasks use sync engine, FastAPI uses async engine
- pgvector extension auto-created by `scripts/entrypoint.sh` - don't add to migrations

### Concurrent Control Strategy
- **LockManager**: Factory pattern for lock selection (Redis vs Threading)
- **Redis presence detection**: Auto-selects distributed lock if Redis available
- **Constraint**: Standard `asyncio.Lock` cannot be used for cross-worker coordination
- RedisLock uses Lua scripts for atomicity - not standard SETNX pattern

### LLM Caching Architecture
- **Cache layer**: LiteLLM + Redis (not application-level cache)
- **Cache key**: SHA256 hash of (model, messages, temperature, etc.)
- **Performance**: 1000x speedup on cache hits
- **Design constraint**: Prompt structure changes invalidate entire cache - design prompts carefully
- Disable with `LITELLM_CACHE=False` (not `CACHE_ENABLED`)

### Chunking Strategy
- **Non-standard defaults**: `CHUNK_SIZE=400`, `CHUNK_OVERLAP=20`
- Most RAG systems use 1000/200 - intentionally smaller for graph extraction
- Affects LightRAG entity extraction quality
- Trade-off: More chunks = better granularity, but slower indexing

### MCP Server Architecture
- **MCP = Model Context Protocol** (context management for LLMs)
- Servers in `aperag/mcp/servers/` - auto-registered via `server_manager.py`
- **Not** a microservice - runs in same process
- Used for context injection and prompt routing

### Web Search Provider System
- Custom provider abstraction in `aperag/websearch/provider_manager.py`
- Not a wrapper for existing library - implements parallel search
- LLM.txt parsing support for site-specific search optimization

### Celery Task Design
- **Pool**: `--pool=threads --concurrency=16` (not default gevent/prefork)
- **Scheduler**: DatabaseScheduler (stores tasks in PostgreSQL, not memory)
- Tasks in `aperag/tasks/*` - auto-registered via `@shared_task`
- **Important**: Use sync DB engine in tasks, not async

### Multi-Platform Build
- Docker builds for `linux/amd64` and `linux/arm64`
- `scripts/entrypoint.sh`: Platform-agnostic initialization
- pgvector extension creation in entrypoint - not in Dockerfile

### Frontend Architecture
- Next.js 15.4.8 with Turbopack (not Webpack)
- API client auto-generated from OpenAPI spec - don't hand-write
- Source: `web/src/api/api.ts` - regenerate when backend schema changes