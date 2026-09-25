"""
Persistent query cache for Medical RAG Pipeline.
Uses SQLite to store and retrieve past queries and their responses.
"""

import copy
import json
import sqlite3
import hashlib
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Optional, Dict

from .config import (
    QUERY_CACHE_ENABLED,
    QUERY_CACHE_DIR,
    QUERY_CACHE_EXPIRY_DAYS,
    QUERY_CACHE_NAMESPACE,
    QUERY_CACHE_KEY_VERSION,
)

logger = logging.getLogger(__name__)

class QueryCache:
    """
    Persistent SQLite-based cache for RAG pipeline responses.
    """
    
    def __init__(
        self,
        cache_dir: str = QUERY_CACHE_DIR,
        expiry_days: int = QUERY_CACHE_EXPIRY_DAYS,
        namespace: str = QUERY_CACHE_NAMESPACE,
        key_version: int = QUERY_CACHE_KEY_VERSION,
        enabled: Optional[bool] = None,
    ):
        self.enabled = QUERY_CACHE_ENABLED if enabled is None else enabled
        self.cache_dir = Path(cache_dir)
        self.db_path = self.cache_dir / "query_cache.db"
        self.expiry_days = expiry_days
        self.namespace = namespace.strip() or "default"
        self.key_version = key_version
        self._schema_initialized = False
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._memory_lock = threading.Lock()
        
        if self.enabled:
            self._init_db()
            
    def _init_db(self):
        """Initialize the SQLite database and create tables if they don't exist."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                self._ensure_schema(conn)
            logger.info(f"✅ Query cache initialized at {self.db_path}")
            self.prune_expired()
        except Exception as e:
            logger.error(f"❌ Failed to initialize query cache: {e}")
            self.enabled = False

    def _connect(self) -> sqlite3.Connection:
        """Create a SQLite connection configured for concurrent API access."""
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def _ensure_schema(self, conn: sqlite3.Connection):
        """Create the cache schema if the database file is empty or incomplete."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                query TEXT,
                response TEXT,
                timestamp DATETIME,
                hit_count INTEGER DEFAULT 0
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON cache(timestamp)")
        self._schema_initialized = True

    def _ensure_schema_if_needed(self, conn: sqlite3.Connection):
        db_missing = not self.db_path.exists()
        db_empty = False
        try:
            db_empty = self.db_path.exists() and self.db_path.stat().st_size == 0
        except OSError:
            db_empty = True

        if self._schema_initialized and not db_missing and not db_empty:
            return
        self._ensure_schema(conn)

    def _generate_key(self, query: str, *, namespace: Optional[str] = None, **kwargs) -> str:
        """Generate a unique SHA-256 hash for the query and its parameters."""
        # Normalize query: strip whitespace and lowercase
        normalized_query = query.strip().lower()

        key_payload = {
            "namespace": (namespace or self.namespace).strip() or "default",
            "key_version": self.key_version,
            "query": normalized_query,
            "params": kwargs,
        }
        key_json = json.dumps(key_payload, sort_keys=True, default=str)
        return hashlib.sha256(key_json.encode("utf-8")).hexdigest()

    def _is_expired(self, timestamp_str: str) -> bool:
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
        except (TypeError, ValueError):
            return True
        return datetime.now() - timestamp > timedelta(days=self.expiry_days)

    def _get_memory_entry(self, key: str) -> Optional[Any]:
        with self._memory_lock:
            entry = self._memory_cache.get(key)
            if not entry:
                return None
            if self._is_expired(entry.get("timestamp", "")):
                self._memory_cache.pop(key, None)
                return None
            return copy.deepcopy(entry.get("value"))

    def _set_memory_entry(self, key: str, value: Any, timestamp: Optional[str] = None) -> None:
        entry_timestamp = timestamp or datetime.now().isoformat()
        with self._memory_lock:
            self._memory_cache[key] = {
                "timestamp": entry_timestamp,
                "value": copy.deepcopy(value),
            }

    def get_entry(self, query: str, *, namespace: Optional[str] = None, **kwargs) -> Optional[Any]:
        """Retrieve a cached value from an optional logical namespace."""
        if not self.enabled:
            return None

        key = self._generate_key(query, namespace=namespace, **kwargs)
        memory_value = self._get_memory_entry(key)
        if memory_value is not None:
            return memory_value

        try:
            with self._connect() as conn:
                self._ensure_schema_if_needed(conn)
                cursor = conn.execute(
                    "SELECT response, timestamp FROM cache WHERE key = ?",
                    (key,),
                )
                row = cursor.fetchone()
                if not row:
                    return None

                response_json, timestamp_str = row
                if self._is_expired(timestamp_str):
                    conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                    return None

                conn.execute(
                    "UPDATE cache SET hit_count = hit_count + 1 WHERE key = ?",
                    (key,),
                )
                parsed = json.loads(response_json)
                self._set_memory_entry(key, parsed, timestamp=timestamp_str)
                return copy.deepcopy(parsed)
        except Exception as e:
            logger.error(f"Error reading from cache: {e}")

        return None

    def set_entry(self, query: str, response: Any, *, namespace: Optional[str] = None, **kwargs):
        """Store a cached value in an optional logical namespace."""
        if not self.enabled:
            return

        key = self._generate_key(query, namespace=namespace, **kwargs)
        timestamp = datetime.now().isoformat()
        try:
            serialized_response = self._serialize(response)
            with self._connect() as conn:
                self._ensure_schema_if_needed(conn)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache (key, query, response, timestamp, hit_count)
                    VALUES (?, ?, ?, ?, COALESCE((SELECT hit_count FROM cache WHERE key = ?), 0))
                    """,
                    (key, query, serialized_response, timestamp, key)
                )
            self._set_memory_entry(key, response, timestamp=timestamp)
            logger.debug(f"💾 Cached response for query: '{query[:50]}...'")
        except Exception as e:
            logger.error(f"Error writing to cache: {e}")

    def get(self, query: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Retrieve a cached response if it exists and is not expired."""
        cached = self.get_entry(query, **kwargs)
        if cached is not None:
            logger.info(f"🎯 Cache hit for query: '{query[:50]}...'")
        return cached

    def set(self, query: str, response: Dict[str, Any], **kwargs):
        """Store a response in the cache."""
        self.set_entry(query, response, **kwargs)

    def prune_expired(self):
        """Remove all expired entries from the cache."""
        if not self.enabled:
            return
        try:
            expiry_date = (datetime.now() - timedelta(days=self.expiry_days)).isoformat()
            with self._connect() as conn:
                self._ensure_schema_if_needed(conn)
                cursor = conn.execute("DELETE FROM cache WHERE timestamp < ?", (expiry_date,))
                if cursor.rowcount > 0:
                    logger.info(f"🧹 Pruned {cursor.rowcount} expired cache entries")
        except Exception as e:
            logger.error(f"Error pruning cache: {e}")

    def _serialize(self, obj: Any) -> str:
        """Helper to serialize object to JSON string, handling special types."""
        def default(item):
            import math
            try:
                import numpy as np
            except Exception:
                np = None

            if isinstance(item, datetime):
                return item.isoformat()
            if isinstance(item, Path):
                return str(item)
            if isinstance(item, (set, tuple)):
                return list(item)
            if hasattr(item, 'to_dict'):
                return item.to_dict()
            if isinstance(item, float):
                if math.isnan(item) or math.isinf(item):
                    return None
                return float(item)
            if np is not None and isinstance(item, np.floating):
                if math.isnan(item) or math.isinf(item):
                    return None
                return float(item)
            if np is not None and isinstance(item, np.integer):
                return int(item)
            if np is not None and isinstance(item, np.ndarray):
                return item.tolist()
            return str(item)
            
        return json.dumps(obj, default=default)
