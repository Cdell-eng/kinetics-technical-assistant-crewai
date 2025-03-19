"""
Caching system for CrewAI responses
"""

import json
import os
from datetime import datetime, timedelta
import hashlib
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ResponseCache:
    def __init__(self, cache_dir="G:/APPS/Technical Assistant/cache", ttl_hours=24):
        """
        Initialize the response cache
        
        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time-to-live in hours for cache entries
        """
        self.cache_dir = cache_dir
        self.ttl_hours = ttl_hours
        
        # Create cache directory if it doesn't exist
        try:
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Cache directory initialized: {cache_dir}")
        except Exception as e:
            logger.error(f"Error creating cache directory: {str(e)}")
    
    def _get_cache_key(self, query: str) -> str:
        """
        Generate a unique cache key for a query
        
        Args:
            query: The user's query
            
        Returns:
            String hash to use as filename
        """
        # Use hash to create a filename-safe key
        return hashlib.md5(query.encode()).hexdigest()
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached response if it exists and is not expired
        
        Args:
            query: The user's query
            
        Returns:
            Cached response or None if not found/expired
        """
        cache_key = self._get_cache_key(query)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            # Check if cache is expired
            timestamp = datetime.fromisoformat(cached_data.get('timestamp', '2000-01-01T00:00:00'))
            if datetime.now() - timestamp > timedelta(hours=self.ttl_hours):
                # Cache expired
                logger.info(f"Cache expired for key: {cache_key}")
                return None
            
            logger.info(f"Cache hit for query: {query[:50]}...")
            return cached_data.get('response')
            
        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            return None
    
    def set(self, query: str, response: Dict[str, Any]) -> bool:
        """
        Store a response in the cache
        
        Args:
            query: The user's query
            response: The response data to cache
            
        Returns:
            True if successful, False otherwise
        """
        cache_key = self._get_cache_key(query)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            cached_data = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'response': response
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cached_data, f, indent=2)
            
            logger.info(f"Cached response for query: {query[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error writing to cache: {e}")
            return False
    
    def invalidate(self, query: str) -> bool:
        """
        Invalidate a cached response
        
        Args:
            query: The query to invalidate
            
        Returns:
            True if successful, False otherwise
        """
        cache_key = self._get_cache_key(query)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                logger.info(f"Invalidated cache for query: {query[:50]}...")
                return True
            except Exception as e:
                logger.error(f"Error invalidating cache: {e}")
                return False
        
        return False
    
    def clear_all(self) -> bool:
        """
        Clear all cached responses
        
        Returns:
            True if successful, False otherwise
        """
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, filename))
            
            logger.info("Cleared all cached responses")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def clear_expired(self) -> int:
        """
        Clear expired cache entries
        
        Returns:
            Number of entries cleared
        """
        cleared_count = 0
        
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    cache_file = os.path.join(self.cache_dir, filename)
                    
                    try:
                        with open(cache_file, 'r') as f:
                            cached_data = json.load(f)
                        
                        timestamp = datetime.fromisoformat(cached_data.get('timestamp', '2000-01-01T00:00:00'))
                        if datetime.now() - timestamp > timedelta(hours=self.ttl_hours):
                            os.remove(cache_file)
                            cleared_count += 1
                            
                    except Exception as e:
                        logger.error(f"Error processing cache file {filename}: {e}")
                        continue
            
            logger.info(f"Cleared {cleared_count} expired cache entries")
            return cleared_count
            
        except Exception as e:
            logger.error(f"Error clearing expired cache: {e}")
            return 0