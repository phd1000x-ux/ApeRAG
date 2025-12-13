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

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Sequence, Tuple

import httpx
import litellm
from redis import Redis

from aperag.config import settings
from aperag.llm.llm_error_types import (
    BatchProcessingError,
    EmbeddingError,
    EmptyTextError,
    wrap_litellm_error,
)

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(
        self,
        embedding_provider: str,
        embedding_model: str,
        embedding_service_url: str,
        embedding_service_api_key: str,
        embedding_max_chunks_in_batch: int,
        multimodal: bool = False,
        caching: bool = True,
    ):
        self.embedding_provider = embedding_provider
        self.model = embedding_model
        self.api_base = embedding_service_url
        self.api_key = embedding_service_api_key
        self.max_chunks = embedding_max_chunks_in_batch
        self.max_workers = 8
        self.multimodal = multimodal
        self.caching = caching
        
        # Initialize Redis client for SHA256-based caching
        self.redis_client = None
        if self.caching and settings.cache_enabled:
            try:
                self.redis_client = Redis(
                    host=settings.redis_host,
                    port=settings.redis_port,
                    password=settings.redis_password,
                    decode_responses=True
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Redis client initialized for embedding cache")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis client for embedding cache: {e}")
                self.redis_client = None
        
        # Custom embedding models that bypass LiteLLM
        self.custom_embedding_models = {
            # Add specific model combinations that should use direct HTTP requests
            # Format: "provider/model": True
            # Examples will be added based on specific requirements
            "openai/cohere.embed-multilingual-v3": True,  # Use direct HTTP for this model to avoid embedding_types issue
        }

    def embed_documents(self, contents: List[str]) -> List[List[float]]:
        """
        Embed multiple documents in parallel batches.

        Args:
            contents: List of documents (texts or base64-encoded images) to embed

        Returns:
            List of embedding vectors in the same order as input contents
        """
        # Validate inputs
        if not contents:
            raise EmptyTextError(0)
        
        # Check for empty contents
        empty_indices = [i for i, text in enumerate(contents) if not text or not text.strip()]
        if empty_indices:
            logger.warning(f"Found {len(empty_indices)} empty content at indices: {empty_indices}")
            if len(empty_indices) == len(contents):
                raise EmptyTextError(len(empty_indices))
        
        # Special handling for dimension probe to avoid base64 encoding issues
        # Check if this is a dimension probe request
        is_dimension_probe = all(text == "dimension_probe" for text in contents if text)
        
        if is_dimension_probe:
            # Use a simpler, ASCII-only text for dimension probing
            # This avoids base64 encoding issues with certain providers
            logger.info("Using ASCII-only text for dimension probe to avoid encoding issues")
            contents = ["dimension_probe"]  # Override with ASCII-only version

        try:
            # Clean contents by replacing newlines with spaces
            clean_contents = [t.replace("\n", " ") if t and t.strip() else " " for t in contents]
            # Determine batch size (use max_chunks or process all at once if not set)
            batch_size = self.max_chunks or len(clean_contents)

            # Store results with original indices to ensure correct ordering
            results_dict: Dict[int, List[float]] = {}

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = []

                # Submit batches for processing with their starting indices
                for start in range(0, len(clean_contents), batch_size):
                    batch = clean_contents[start : start + batch_size]
                    # Pass both the batch and starting index to track position
                    future = pool.submit(self._embed_batch_with_indices, batch, start)
                    futures.append(future)

                # Process completed futures and store results by index
                failed_batches = []
                for future in as_completed(futures):
                    try:
                        # Get results with their original indices
                        batch_results = future.result()
                        for idx, embedding in batch_results:
                            results_dict[idx] = embedding
                    except Exception as e:
                        failed_batches.append(str(e))
                        logger.error(f"Batch processing failed: {e}")

                if failed_batches:
                    raise BatchProcessingError(
                        batch_size=batch_size,
                        reason=f"Failed to process {len(failed_batches)} batches: {failed_batches[:3]} "
                        f"contents: {contents}",
                    )

            # Reconstruct the result list in the original order
            results = [results_dict[i] for i in range(len(clean_contents))]
            return results
        except (EmptyTextError, BatchProcessingError, EmbeddingError):
            # Re-raise our custom embedding errors
            raise
        except Exception as e:
            logger.error(f"Document embedding failed: {str(e)}")
            raise wrap_litellm_error(e, "embedding", self.embedding_provider, self.model) from e

    async def aembed_documents(self, contents: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.embed_documents, contents)

    def embed_query(self, content: str) -> List[float]:
        """
        Embed a single query content.

        Args:
            content: content to embed

        Returns:
            List of floats representing the embedding vector
        """
        if not content or not content.strip():
            raise EmptyTextError(1)

        try:
            return self.embed_documents([content])[0]
        except (EmptyTextError, EmbeddingError):
            # Re-raise our custom embedding errors
            raise
        except Exception as e:
            logger.error(f"Query embedding failed: {str(e)}")
            raise wrap_litellm_error(e, "embedding", self.embedding_provider, self.model) from e

    async def aembed_query(self, content: str) -> List[float]:
        return await asyncio.to_thread(self.embed_query, content)

    def is_multimodal(self) -> bool:
        return self.multimodal

    def _embed_batch_with_indices(self, batch: Sequence[str], start_idx: int) -> List[Tuple[int, List[float]]]:
        """Process a batch of texts and return embeddings with their original indices."""
        try:
            embeddings = self._embed_batch(batch)
            # Return each embedding with its corresponding index in the original list
            return [(start_idx + i, embedding) for i, embedding in enumerate(embeddings)]
        except Exception as e:
            logger.error(f"Batch embedding with indices failed: {str(e)}")
            # Convert litellm errors for batch processing
            raise wrap_litellm_error(e, "embedding", self.embedding_provider, self.model) from e

    def _embed_batch(self, batch: Sequence[str]) -> List[List[float]]:
        """
        Embed a batch of contents using either direct HTTP requests or litellm.

        Args:
            batch: Sequence of contents to embed

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding fails
        """
        model_key = f"{self.embedding_provider}/{self.model}"
        
        # Check if we should use direct HTTP requests for this model
        if model_key in self.custom_embedding_models or self._should_use_direct_http():
            return self._embed_batch_direct_http(batch)
        
        # Check SHA256-based cache first
        if self.redis_client:
            cached_embeddings = self._get_from_cache(batch)
            if cached_embeddings:
                logger.debug(f"Cache hit for batch of {len(batch)} items")
                return cached_embeddings
        
        # Use LiteLLM for embedding
        try:
            # Prepare request parameters
            request_params = {
                "custom_llm_provider": self.embedding_provider,
                "model": self.model,
                "api_base": self.api_base,
                "api_key": self.api_key,
                "input": list(batch),
                "caching": self.caching,
                # Drop all extra parameters to avoid encoding issues
                "drop_params": True,
            }
            
            # Remove all Cohere-specific parameters to avoid base64 encoding issues
            # These parameters are causing the base64 encoding error
            # if self.embedding_provider == "bedrock" and "cohere" in self.model.lower():
            #     request_params["embedding_types"] = ["float"]
            #     # Remove input_type parameter to avoid base64 encoding issues
            #     # request_params["input_type"] = "search_document"
            # # Also add for openai provider when using cohere models through bedrock
            # elif self.embedding_provider == "openai" and "cohere" in self.model.lower():
            #     request_params["embedding_types"] = ["float"]
            #     # Remove input_type parameter to avoid base64 encoding issues
            #     # request_params["input_type"] = "search_document"
            
            response = litellm.embedding(**request_params)

            if not response or "data" not in response:
                raise EmbeddingError(
                    "Invalid response format from embedding API",
                    {"provider": self.embedding_provider, "model": self.model, "batch_size": len(batch)},
                )

            embeddings = [item["embedding"] for item in response["data"]]

            # Validate embedding dimensions
            if embeddings and len(set(len(emb) for emb in embeddings)) > 1:
                dimensions = [len(emb) for emb in embeddings]
                logger.warning(f"Inconsistent embedding dimensions: {set(dimensions)}")

            # Cache the results using SHA256
            if self.redis_client and embeddings:
                self._cache_embeddings(batch, embeddings)

            return embeddings
        except Exception as e:
            logger.error(f"Batch embedding API call failed: {str(e)}")
            # Convert litellm errors to our custom types
            raise wrap_litellm_error(e, "embedding", self.embedding_provider, self.model) from e
    
    def _should_use_direct_http(self) -> bool:
        """
        Determine if direct HTTP requests should be used instead of LiteLLM.
        
        Returns:
            bool: True if direct HTTP should be used
        """
        # Add logic to determine when to use direct HTTP requests
        # This can be based on provider, model, or other conditions
        direct_http_providers = ["custom_provider1", "custom_provider2"]  # Add as needed
        return self.embedding_provider in direct_http_providers
    
    def _embed_batch_direct_http(self, batch: Sequence[str]) -> List[List[float]]:
        """
        Embed a batch using direct HTTP requests, bypassing LiteLLM.
        
        Args:
            batch: Sequence of contents to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingError: If embedding fails
        """
        # Check SHA256-based cache first
        if self.redis_client:
            cached_embeddings = self._get_from_cache(batch)
            if cached_embeddings:
                logger.debug(f"Cache hit for direct HTTP batch of {len(batch)} items")
                return cached_embeddings
        
        try:
            # Sanitize content before sending to API
            sanitized_batch = [self._sanitize_content(text) for text in batch]

            # Prepare request based on provider
            headers = {"Authorization": f"Bearer {self.api_key}"}
            data = {
                "model": self.model,
                "input": sanitized_batch
            }
            
            # Remove all Cohere-specific parameters to avoid base64 encoding issues
            # These parameters are causing the base64 encoding error
            # if self.embedding_provider == "bedrock" and "cohere" in self.model.lower():
            #     data["embedding_types"] = ["float"]
            #     # Remove input_type parameter to avoid base64 encoding issues
            #     # data["input_type"] = "search_document"
            # # Also add for openai provider when using cohere models through bedrock
            # elif self.embedding_provider == "openai" and "cohere" in self.model.lower():
            #     data["embedding_types"] = ["float"]
            #     # Remove input_type parameter to avoid base64 encoding issues
            #     # data["input_type"] = "search_document"
            
            # Make direct HTTP request
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{self.api_base}/embeddings",
                    headers=headers,
                    json=data
                )
                response.raise_for_status()
                
                result = response.json()
                
                if "data" not in result:
                    raise EmbeddingError(
                        "Invalid response format from direct HTTP embedding API",
                        {"provider": self.embedding_provider, "model": self.model, "response": result},
                    )
                
                embeddings = [item["embedding"] for item in result["data"]]
                
                # Validate embedding dimensions
                if embeddings and len(set(len(emb) for emb in embeddings)) > 1:
                    dimensions = [len(emb) for emb in embeddings]
                    logger.warning(f"Inconsistent embedding dimensions: {set(dimensions)}")
                
                # Cache the results using SHA256
                if self.redis_client and embeddings:
                    self._cache_embeddings(batch, embeddings)
                
                return embeddings
                
        except httpx.HTTPError as e:
            logger.error(f"Direct HTTP embedding request failed: {str(e)}")
            raise EmbeddingError(
                f"Direct HTTP embedding request failed: {str(e)}",
                {"provider": self.embedding_provider, "model": self.model},
            ) from e
        except Exception as e:
            logger.error(f"Direct HTTP embedding failed: {str(e)}")
            raise EmbeddingError(
                f"Direct HTTP embedding failed: {str(e)}",
                {"provider": self.embedding_provider, "model": self.model},
            ) from e

    def _sanitize_content(self, text: str) -> str:
        """
        Sanitize content before sending to embedding API.

        Args:
            text: Text to sanitize

        Returns:
            str: Sanitized text
        """
        if not text:
            return text

        # Replace non-breaking spaces with regular spaces
        sanitized = text.replace('\xa0', ' ')

        # Replace other problematic Unicode characters
        sanitized = sanitized.replace('\u200b', '')  # Zero-width space
        sanitized = sanitized.replace('\u200c', '')  # Zero-width non-joiner
        sanitized = sanitized.replace('\u200d', '')  # Zero-width joiner
        sanitized = sanitized.replace('\ufeff', '')  # BOM

        # Normalize multiple spaces to single space
        sanitized = re.sub(r' +', ' ', sanitized)

        return sanitized.strip()

    def _generate_cache_key(self, texts: Sequence[str]) -> str:
        """
        Generate SHA256 hash-based cache key for the given texts.
        
        Args:
            texts: Sequence of texts to embed
            
        Returns:
            str: SHA256 hash-based cache key
        """
        # Create a deterministic string from all inputs
        cache_data = {
            "provider": self.embedding_provider,
            "model": self.model,
            "texts": list(texts)
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_string.encode()).hexdigest()
    
    def _get_from_cache(self, batch: Sequence[str]) -> List[List[float]] | None:
        """
        Retrieve embeddings from SHA256-based cache.
        
        Args:
            batch: Sequence of texts to embed
            
        Returns:
            List of embedding vectors or None if not found
        """
        if not self.redis_client:
            return None
        
        try:
            cache_key = f"embedding:{self._generate_cache_key(batch)}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                cache_entry = json.loads(cached_data)
                logger.debug(f"Cache hit for key: {cache_key}")
                return cache_entry["embeddings"]
            
            return None
        except Exception as e:
            logger.warning(f"Failed to retrieve from cache: {e}")
            return None
    
    def _cache_embeddings(self, batch: Sequence[str], embeddings: List[List[float]]) -> None:
        """
        Cache embeddings using SHA256-based key.
        
        Args:
            batch: Sequence of texts that were embedded
            embeddings: List of embedding vectors to cache
        """
        if not self.redis_client:
            return
        
        try:
            cache_key = f"embedding:{self._generate_cache_key(batch)}"
            cache_data = {
                "embeddings": embeddings,
                "timestamp": time.time(),
                "provider": self.embedding_provider,
                "model": self.model
            }
            
            # Cache with TTL from settings
            ttl = getattr(settings, 'cache_ttl', 3600)  # Default 1 hour
            self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(cache_data)
            )
            logger.debug(f"Cached embeddings with key: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to cache embeddings: {e}")
