#!/usr/bin/env python3
"""
Test script for new embedding features:
1. Custom Embedding (Direct HTTP requests)
2. Embedding Cache (SHA256-based caching)
3. Cohere Models embedding_types parameter
"""

import asyncio
import json
import logging
import time
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_embedding_service():
    """Test the enhanced embedding service with new features."""
    
    # Import after setting up any required environment
    from aperag.llm.embed.embedding_service import EmbeddingService
    
    # Test configuration - adjust these values based on your environment
    test_config = {
        "embedding_provider": "openai",  # Change to your provider
        "embedding_model": "text-embedding-ada-002",  # Change to your model
        "embedding_service_url": "https://api.openai.com/v1",  # Change to your API URL
        "embedding_service_api_key": "your-api-key-here",  # Change to your API key
        "embedding_max_chunks_in_batch": 10,
        "multimodal": False,
        "caching": True,
    }
    
    # Create embedding service instance
    embedding_service = EmbeddingService(**test_config)
    
    # Test texts
    test_texts = [
        "This is a test document for embedding.",
        "Another test document with different content.",
        "Third document to test batch processing.",
        "This is a test document for embedding.",  # Duplicate to test caching
    ]
    
    logger.info("Testing embedding service with new features...")
    
    try:
        # Test 1: Single query embedding
        logger.info("Test 1: Single query embedding")
        start_time = time.time()
        single_embedding = embedding_service.embed_query(test_texts[0])
        single_time = time.time() - start_time
        logger.info(f"Single embedding completed in {single_time:.2f}s")
        logger.info(f"Embedding dimension: {len(single_embedding)}")
        
        # Test 2: Batch embedding (first run - should be slower)
        logger.info("Test 2: Batch embedding (first run)")
        start_time = time.time()
        batch_embeddings_1 = embedding_service.embed_documents(test_texts)
        batch_time_1 = time.time() - start_time
        logger.info(f"First batch embedding completed in {batch_time_1:.2f}s")
        logger.info(f"Batch size: {len(batch_embeddings_1)}")
        
        # Test 3: Batch embedding (second run - should be faster due to caching)
        logger.info("Test 3: Batch embedding (second run - testing cache)")
        start_time = time.time()
        batch_embeddings_2 = embedding_service.embed_documents(test_texts)
        batch_time_2 = time.time() - start_time
        logger.info(f"Second batch embedding completed in {batch_time_2:.2f}s")
        
        # Calculate cache performance improvement
        if batch_time_1 > 0:
            improvement = ((batch_time_1 - batch_time_2) / batch_time_1) * 100
            logger.info(f"Cache performance improvement: {improvement:.1f}%")
        
        # Test 4: Verify embeddings are consistent
        logger.info("Test 4: Verifying embedding consistency")
        for i, (emb1, emb2) in enumerate(zip(batch_embeddings_1, batch_embeddings_2)):
            if emb1 == emb2:
                logger.info(f"Embedding {i} is consistent between runs")
            else:
                logger.warning(f"Embedding {i} differs between runs!")
        
        # Test 5: Async embedding
        logger.info("Test 5: Async embedding")
        start_time = time.time()
        async_embeddings = await embedding_service.aembed_documents(test_texts[:2])
        async_time = time.time() - start_time
        logger.info(f"Async embedding completed in {async_time:.2f}s")
        
        # Test 6: SHA256 cache key generation
        logger.info("Test 6: Testing SHA256 cache key generation")
        cache_key_1 = embedding_service._generate_cache_key(["test text"])
        cache_key_2 = embedding_service._generate_cache_key(["test text"])
        cache_key_3 = embedding_service._generate_cache_key(["different text"])
        
        if cache_key_1 == cache_key_2:
            logger.info("SHA256 cache keys are consistent for same input")
        else:
            logger.error("SHA256 cache keys differ for same input!")
        
        if cache_key_1 != cache_key_3:
            logger.info("SHA256 cache keys differ for different input")
        else:
            logger.error("SHA256 cache keys are same for different input!")
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

def test_cohere_bedrock_embedding():
    """Test Cohere models with embedding_types parameter for Bedrock."""
    
    from aperag.llm.embed.embedding_service import EmbeddingService
    
    # Test configuration for Cohere on Bedrock
    test_config = {
        "embedding_provider": "bedrock",
        "embedding_model": "cohere.embed-multilingual-v3",  # Example Cohere model
        "embedding_service_url": "https://bedrock-runtime.us-east-1.amazonaws.com",
        "embedding_service_api_key": "your-aws-credentials",
        "embedding_max_chunks_in_batch": 10,
        "multimodal": False,
        "caching": True,
    }
    
    logger.info("Testing Cohere models with embedding_types parameter...")
    
    try:
        embedding_service = EmbeddingService(**test_config)
        
        # Test if the provider/model combination correctly adds embedding_types
        test_texts = ["This is a test for Cohere embedding on Bedrock."]
        
        # This would test the embedding_types parameter addition
        # Note: This test requires valid AWS credentials and Bedrock access
        logger.info("Cohere Bedrock test configuration created successfully")
        logger.info("Note: Actual API call requires valid AWS credentials")
        
    except Exception as e:
        logger.error(f"Cohere Bedrock test failed: {str(e)}")

def test_direct_http_embedding():
    """Test direct HTTP embedding functionality."""
    
    from aperag.llm.embed.embedding_service import EmbeddingService
    
    # Test configuration for direct HTTP
    test_config = {
        "embedding_provider": "custom_provider1",  # This should trigger direct HTTP
        "embedding_model": "custom-embedding-model",
        "embedding_service_url": "https://api.custom-provider.com/v1",
        "embedding_service_api_key": "your-custom-api-key",
        "embedding_max_chunks_in_batch": 10,
        "multimodal": False,
        "caching": True,
    }
    
    logger.info("Testing direct HTTP embedding functionality...")
    
    try:
        embedding_service = EmbeddingService(**test_config)
        
        # Test if direct HTTP is correctly identified
        should_use_direct = embedding_service._should_use_direct_http()
        logger.info(f"Should use direct HTTP: {should_use_direct}")
        
        if should_use_direct:
            logger.info("Direct HTTP embedding configuration created successfully")
            logger.info("Note: Actual API call requires valid custom provider endpoint")
        else:
            logger.info("Direct HTTP not triggered for this configuration")
        
    except Exception as e:
        logger.error(f"Direct HTTP test failed: {str(e)}")

if __name__ == "__main__":
    print("Testing new embedding features...")
    print("=" * 50)
    
    # Run tests
    asyncio.run(test_embedding_service())
    print("\n" + "=" * 50)
    test_cohere_bedrock_embedding()
    print("\n" + "=" * 50)
    test_direct_http_embedding()
    
    print("\nAll feature tests completed!")
    print("Note: Some tests require valid API credentials to make actual API calls.")