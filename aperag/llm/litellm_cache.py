# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LiteLLM Cache Configuration

This module configures LiteLLM's built-in caching functionality with:
- Redis cache storage
- Custom cache key generation
- Cache hit/miss tracking
- Cache statistics
"""

import logging
import os
from typing import Any, Dict

import litellm
from litellm.types.caching import LiteLLMCacheType

logger = logging.getLogger(__name__)

# Configure litellm drop_params from environment variables
# This must be set before any litellm API calls
if os.getenv("LITELLM_DROP_PARAMS", "").lower() == "true":
    litellm.drop_params = True
    logger.info("LiteLLM drop_params enabled from LITELLM_DROP_PARAMS environment variable")

# Also try parsing LITELLM_SETTINGS JSON
litellm_settings_str = os.getenv("LITELLM_SETTINGS")
if litellm_settings_str:
    try:
        import json
        litellm_settings = json.loads(litellm_settings_str)
        if litellm_settings.get("drop_params"):
            litellm.drop_params = True
            logger.info("LiteLLM drop_params enabled from LITELLM_SETTINGS environment variable")
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Failed to parse LITELLM_SETTINGS: {e}")

# Local in-memory statistics
# Note: These are simple integer operations that may not be thread-safe
# in multi-threaded environments, but are acceptable for monitoring purposes.
# In multi-process environments (e.g., Celery prefork), each process maintains its own stats.
_cache_stats = {
    "hits": 0,
    "misses": 0,
    "added": 0,
    "total_requests": 0,
}


# doc: https://docs.litellm.ai/docs/caching/all_caches#enabling-cache
# All parameters for cache: https://docs.litellm.ai/docs/caching/all_caches#cache-initialization-parameters
def setup_litellm_cache(default_type=LiteLLMCacheType.DISK):
    from litellm.caching.caching import CacheMode

    from aperag.config import settings

    if not settings.cache_enabled:
        return

    litellm.enable_cache(
        type=default_type,
        mode=CacheMode.default_on,
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_password,
        ttl=settings.cache_ttl,
        disk_cache_dir="/tmp/litellm_cache",
    )
    # Setup custom cache handlers with local stats tracking
    # Note: Only setup if cache was successfully initialized
    if litellm.cache is not None:
        setup_custom_get_cache()
        setup_custom_add_cache()
        logger.info("LiteLLM cache with local statistics initialized")


def disable_litellm_cache():
    litellm.disable_cache()


def setup_custom_get_cache_key():
    def custom_get_cache_key(*args, **kwargs):
        # Generate SHA256-based cache key for better performance and collision avoidance
        import hashlib
        import json
        
        # Create a deterministic string from all relevant parameters
        cache_data = {
            "model": kwargs.get("model", ""),
            "messages": kwargs.get("messages", ""),
            "temperature": kwargs.get("temperature", ""),
            "logit_bias": kwargs.get("logit_bias", ""),
            "max_tokens": kwargs.get("max_tokens", ""),
            "top_p": kwargs.get("top_p", ""),
            "frequency_penalty": kwargs.get("frequency_penalty", ""),
            "presence_penalty": kwargs.get("presence_penalty", ""),
        }
        
        # For embeddings, include the input texts
        if "input" in kwargs:
            cache_data["input"] = kwargs.get("input", [])
        
        # Create deterministic string and hash it
        cache_string = json.dumps(cache_data, sort_keys=True)
        key = hashlib.sha256(cache_string.encode()).hexdigest()
        
        logger.debug(f"Generated SHA256 cache key: {key[:16]}...")
        return key

    if litellm.cache is not None:
        litellm.cache.get_cache_key = custom_get_cache_key


def setup_custom_add_cache():
    """
    Wraps litellm.cache.add_cache to include local statistics for cache additions.
    """
    if litellm.cache is None:
        return

    # Store the original method
    original_add_cache = litellm.cache.add_cache

    def custom_add_cache(result, *args, **kwargs):
        # Update local stats - simple increment, may not be atomic in multi-threaded env
        global _cache_stats
        _cache_stats["added"] += 1
        logger.debug("LiteLLM Cache ADD")

        # Call the original caching function
        return original_add_cache(result, *args, **kwargs)

    # Replace the method
    litellm.cache.add_cache = custom_add_cache


def setup_custom_get_cache():
    """
    Wraps litellm.cache.get_cache to include local hit/miss statistics.
    """
    if litellm.cache is None:
        return

    # Store the original method
    original_get_cache = litellm.cache.get_cache

    def custom_get_cache(*args, **kwargs):
        # Call the original function to get the result from cache
        result = original_get_cache(*args, **kwargs)

        # Update local stats - simple increment, may not be atomic in multi-threaded env
        global _cache_stats
        _cache_stats["total_requests"] += 1
        if result is not None:
            _cache_stats["hits"] += 1
            logger.debug("LiteLLM Cache HIT")
            if _cache_stats["hits"] % 100 == 0:
                logger.info(
                    f"Cache HIT count: {_cache_stats['hits']}, total requests: {_cache_stats['total_requests']}"
                )
                logger.info(f"Cache HIT rate: {_cache_stats['hits'] / _cache_stats['total_requests']:.2%}")
        else:
            _cache_stats["misses"] += 1
            logger.debug("LiteLLM Cache MISS")

        return result

    # Replace the method
    litellm.cache.get_cache = custom_get_cache


def get_cache_stats() -> Dict[str, Any]:
    """
    Get local in-memory cache statistics for the current process.

    Returns:
        Dict containing cache statistics including hit rate calculation.
    """
    # Create a copy to avoid modification during read
    stats = _cache_stats.copy()

    # Calculate hit rate
    if stats["total_requests"] > 0:
        stats["hit_rate"] = round(stats["hits"] / stats["total_requests"], 4)
    else:
        stats["hit_rate"] = 0.0

    # Add metadata
    stats["cache_type"] = "local_memory"
    stats["note"] = "Process-specific stats, not thread-safe"

    return stats


def clear_cache_stats() -> None:
    """Reset local in-memory cache statistics for the current process."""
    global _cache_stats
    _cache_stats = {
        "hits": 0,
        "misses": 0,
        "added": 0,
        "total_requests": 0,
    }
    logger.info("Local cache statistics cleared")
