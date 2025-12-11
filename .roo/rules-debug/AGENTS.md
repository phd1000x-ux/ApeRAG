# AGENTS.md - Debug Mode

## OpenTelemetry & Tracing
- **Jaeger UI**: Default at `http://localhost:16686` when enabled
- **Selective instrumentation**: Control via `OTEL_INSTRUMENT_*` env vars
  - `OTEL_INSTRUMENT_FASTAPI=true/false`
  - `OTEL_INSTRUMENT_SQLALCHEMY=true/false`
  - `OTEL_INSTRUMENT_MCP=true/false`
- **Span export**: HTTP endpoint at `http://localhost:4318/v1/traces`
- See `docs/jaeger-tracing.md` for detailed setup

## Celery Debugging
- **Worker logs**: Use `--loglevel=debug` in `scripts/start-celery-worker.sh`
- **Flower UI**: Monitor at `http://localhost:5555` (when enabled)
- **Task inspection**: `celery -A aperag.tasks inspect active`
- **Scheduler**: DatabaseScheduler stores tasks in PostgreSQL, not memory

## Database Debugging
- **Connection pool stats**: Check `aperag/db/database.py` - custom pool sizes differ from defaults
- **pgvector issues**: Verify extension created by `scripts/entrypoint.sh`, not manually
- **Graph storage**: Set `LOG_LEVEL=DEBUG` to see NetworkX/Postgres/Neo4j query logs

## LLM Cache Debugging
- **Cache hits**: Look for "LiteLLM cache hit" in logs
- **Cache key**: SHA256 of model+messages+params - check `tests/unit_test/llm/test_litellm_cache_key.py`
- **Disable caching**: Set `LITELLM_CACHE=False` (not `CACHE_ENABLED`)
- **Redis connection**: Verify `REDIS_URL` - cache silently fails without Redis

## Concurrent Control Issues
- **Lock type mismatch**: `LockManager` auto-selects Redis vs Threading based on availability
- **Deadlocks**: Check `aperag/concurrent_control/` - RedisLock uses Lua scripts for atomicity
- **Cross-worker sync**: Standard `asyncio.Lock` won't work - must use RedisLock

## Environment Variables
- **Non-standard defaults**: `CHUNK_SIZE=400`, `CHUNK_OVERLAP=20` (most projects use 1000/200)
- **Graph storage**: `GRAPH_STORAGE_TYPE=postgres|neo4j|networkx` (default: networkx)
- **Model configs**: See `models/generate_model_configs.py` for custom provider formats