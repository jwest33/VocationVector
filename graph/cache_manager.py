"""
Cache Manager for tracking processed records and avoiding redundant processing

This module provides caching functionality to:
- Track processed jobs, resumes, and matches
- Skip reprocessing unchanged data
- Invalidate cache when source files change
- Persist cache across pipeline runs
"""

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached record"""
    key: str  # Unique identifier (e.g., file path, job query)
    content_hash: str  # Hash of the content
    processed_at: str  # ISO timestamp
    metadata: Dict[str, Any]  # Additional metadata
    result: Optional[Any] = None  # Cached processing result


class CacheManager:
    """Manages caching for pipeline processing"""
    
    def __init__(self, cache_dir: str = "data/.cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate cache files for different types
        self.job_cache_file = self.cache_dir / "jobs_cache.json"
        self.resume_cache_file = self.cache_dir / "resumes_cache.json"
        self.match_cache_file = self.cache_dir / "matches_cache.json"
        
        # Load existing caches
        self.job_cache = self._load_cache(self.job_cache_file)
        self.resume_cache = self._load_cache(self.resume_cache_file)
        self.match_cache = self._load_cache(self.match_cache_file)
        
        logger.info(f"CacheManager initialized with cache dir: {self.cache_dir}")
    
    def _load_cache(self, cache_file: Path) -> Dict[str, CacheEntry]:
        """Load cache from file"""
        if not cache_file.exists():
            return {}
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return {
                    k: CacheEntry(**v) for k, v in data.items()
                }
        except Exception as e:
            logger.warning(f"Failed to load cache from {cache_file}: {e}")
            return {}
    
    def _save_cache(self, cache: Dict[str, CacheEntry], cache_file: Path):
        """Save cache to file"""
        try:
            data = {
                k: asdict(v) for k, v in cache.items()
            }
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save cache to {cache_file}: {e}")
    
    def _compute_content_hash(self, content: Any) -> str:
        """Compute hash of content"""
        if isinstance(content, (str, bytes)):
            if isinstance(content, str):
                content = content.encode('utf-8')
        elif isinstance(content, dict):
            content = json.dumps(content, sort_keys=True).encode('utf-8')
        elif isinstance(content, Path):
            # For files, hash the file content
            if content.exists():
                with open(content, 'rb') as f:
                    content = f.read()
            else:
                return ""
        else:
            content = str(content).encode('utf-8')
        
        return hashlib.sha256(content).hexdigest()
    
    def _get_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get file metadata for cache tracking"""
        if not file_path.exists():
            return {}
        
        stat = file_path.stat()
        return {
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "path": str(file_path)
        }
    
    # Job caching methods
    def is_job_query_cached(self, query: str, location: str, max_jobs: int) -> Tuple[bool, Optional[List[Dict]]]:
        """Check if a job query has been cached recently"""
        cache_key = f"{query}|{location}|{max_jobs}"
        
        if cache_key in self.job_cache:
            entry = self.job_cache[cache_key]
            
            # Check if cache is still valid (24 hours for job queries)
            processed_at = datetime.fromisoformat(entry.processed_at)
            if datetime.now() - processed_at < timedelta(hours=24):
                logger.info(f"Job query cached and valid: {cache_key}")
                return True, entry.result
            else:
                logger.info(f"Job query cache expired: {cache_key}")
        
        return False, None
    
    def cache_job_query(self, query: str, location: str, max_jobs: int, results: List[Dict]):
        """Cache job query results"""
        cache_key = f"{query}|{location}|{max_jobs}"
        
        entry = CacheEntry(
            key=cache_key,
            content_hash=self._compute_content_hash(results),
            processed_at=datetime.now().isoformat(),
            metadata={
                "query": query,
                "location": location,
                "max_jobs": max_jobs,
                "job_count": len(results)
            },
            result=results
        )
        
        self.job_cache[cache_key] = entry
        self._save_cache(self.job_cache, self.job_cache_file)
        logger.info(f"Cached job query: {cache_key} ({len(results)} jobs)")
    
    # Resume caching methods
    def is_resume_processed(self, file_path: Path) -> Tuple[bool, Optional[Dict]]:
        """Check if a resume has already been processed with same content"""
        file_path = Path(file_path)
        cache_key = str(file_path.absolute())
        
        if not file_path.exists():
            return False, None
        
        current_hash = self._compute_content_hash(file_path)
        
        if cache_key in self.resume_cache:
            entry = self.resume_cache[cache_key]
            
            # Check if file content hasn't changed
            if entry.content_hash == current_hash:
                logger.info(f"Resume already processed: {file_path.name}")
                return True, entry.result
            else:
                logger.info(f"Resume changed, needs reprocessing: {file_path.name}")
        
        return False, None
    
    def cache_resume(self, file_path: Path, processed_data: Dict):
        """Cache processed resume data"""
        file_path = Path(file_path)
        cache_key = str(file_path.absolute())
        
        entry = CacheEntry(
            key=cache_key,
            content_hash=self._compute_content_hash(file_path),
            processed_at=datetime.now().isoformat(),
            metadata=self._get_file_metadata(file_path),
            result=processed_data
        )
        
        self.resume_cache[cache_key] = entry
        self._save_cache(self.resume_cache, self.resume_cache_file)
        logger.info(f"Cached resume: {file_path.name}")
    
    # Match caching methods
    def get_match_cache_key(self, resume_ids: List[str], job_ids: List[str]) -> str:
        """Generate cache key for matching operation"""
        # Sort IDs to ensure consistent keys
        resume_key = "|".join(sorted(resume_ids))
        job_key = "|".join(sorted(job_ids))
        return f"{resume_key}::{job_key}"
    
    def is_matching_cached(self, resume_ids: List[str], job_ids: List[str]) -> Tuple[bool, Optional[List[Dict]]]:
        """Check if matching results are cached for given resumes and jobs"""
        cache_key = self.get_match_cache_key(resume_ids, job_ids)
        
        if cache_key in self.match_cache:
            entry = self.match_cache[cache_key]
            
            # Matching cache is valid for 7 days
            processed_at = datetime.fromisoformat(entry.processed_at)
            if datetime.now() - processed_at < timedelta(days=7):
                logger.info(f"Matching results cached: {len(resume_ids)} resumes x {len(job_ids)} jobs")
                return True, entry.result
        
        return False, None
    
    def cache_matching(self, resume_ids: List[str], job_ids: List[str], results: List[Dict]):
        """Cache matching results"""
        cache_key = self.get_match_cache_key(resume_ids, job_ids)
        
        entry = CacheEntry(
            key=cache_key,
            content_hash=self._compute_content_hash(results),
            processed_at=datetime.now().isoformat(),
            metadata={
                "resume_count": len(resume_ids),
                "job_count": len(job_ids),
                "match_count": len(results)
            },
            result=results
        )
        
        self.match_cache[cache_key] = entry
        self._save_cache(self.match_cache, self.match_cache_file)
        logger.info(f"Cached matching results: {len(results)} matches")
    
    # Cache management methods
    def clear_cache(self, cache_type: Optional[str] = None):
        """Clear cache (optionally specific type)"""
        if cache_type == "jobs" or cache_type is None:
            self.job_cache = {}
            self._save_cache(self.job_cache, self.job_cache_file)
            logger.info("Cleared job cache")
        
        if cache_type == "resumes" or cache_type is None:
            self.resume_cache = {}
            self._save_cache(self.resume_cache, self.resume_cache_file)
            logger.info("Cleared resume cache")
        
        if cache_type == "matches" or cache_type is None:
            self.match_cache = {}
            self._save_cache(self.match_cache, self.match_cache_file)
            logger.info("Cleared match cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "jobs": {
                "count": len(self.job_cache),
                "queries": list(self.job_cache.keys())
            },
            "resumes": {
                "count": len(self.resume_cache),
                "files": [Path(k).name for k in self.resume_cache.keys()]
            },
            "matches": {
                "count": len(self.match_cache)
            }
        }
    
    def invalidate_expired(self):
        """Remove expired cache entries"""
        now = datetime.now()
        
        # Clean job cache (24 hour expiry)
        expired_jobs = []
        for key, entry in self.job_cache.items():
            if now - datetime.fromisoformat(entry.processed_at) > timedelta(hours=24):
                expired_jobs.append(key)
        
        for key in expired_jobs:
            del self.job_cache[key]
        
        if expired_jobs:
            self._save_cache(self.job_cache, self.job_cache_file)
            logger.info(f"Removed {len(expired_jobs)} expired job cache entries")
        
        # Clean match cache (7 day expiry)
        expired_matches = []
        for key, entry in self.match_cache.items():
            if now - datetime.fromisoformat(entry.processed_at) > timedelta(days=7):
                expired_matches.append(key)
        
        for key in expired_matches:
            del self.match_cache[key]
        
        if expired_matches:
            self._save_cache(self.match_cache, self.match_cache_file)
            logger.info(f"Removed {len(expired_matches)} expired match cache entries")
