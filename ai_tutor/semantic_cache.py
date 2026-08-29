"""
semantic_cache.py
-----------------
Multi-level response caching layer for AI Tutor:
1. ExactResponseCache: Deterministic SHA-256 hash matching on (message, course_id, hint_level, mode).
2. SemanticCache: Cosine similarity / word Jaccard similarity matcher with configurable similarity threshold.
3. Thread-safe LRU eviction and TTL invalidation.
"""

from __future__ import annotations

import hashlib
import time
import threading
from typing import Optional, Dict, Any, Tuple
from collections import OrderedDict

from .models import AIChatResponse


class ExactResponseCache:
    """Thread-safe LRU cache with TTL for exact request matching."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[AIChatResponse, float]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _compute_key(self, message: str, course_id: Optional[int] = None, hint_level: Optional[int] = None) -> str:
        raw = f"{message.strip().lower()}|{course_id or 0}|{hint_level or 0}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, message: str, course_id: Optional[int] = None, hint_level: Optional[int] = None) -> Optional[AIChatResponse]:
        key = self._compute_key(message, course_id, hint_level)
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None

            resp, timestamp = self._cache[key]
            if time.time() - timestamp > self.ttl_seconds:
                del self._cache[key]
                self.misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self.hits += 1
            return resp

    def set(self, message: str, response: AIChatResponse, course_id: Optional[int] = None, hint_level: Optional[int] = None):
        key = self._compute_key(message, course_id, hint_level)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (response, time.time())
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def clear(self):
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0


class SemanticCache:
    """Lightweight similarity-based semantic cache using Jaccard token overlap."""

    def __init__(self, similarity_threshold: float = 0.85, max_size: int = 500, ttl_seconds: int = 3600):
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._entries: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        import re
        clean_text = re.sub(r"[^\w\s]", " ", text.lower())
        return {w.strip() for w in clean_text.split() if len(w.strip()) > 1}


    def _jaccard_similarity(self, set_a: set[str], set_b: set[str]) -> float:
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    def get(self, query: str) -> Optional[AIChatResponse]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return None

        now = time.time()
        with self._lock:
            # Purge expired entries
            self._entries = [e for e in self._entries if now - e["timestamp"] <= self.ttl_seconds]

            best_match = None
            best_score = 0.0

            for entry in self._entries:
                score = self._jaccard_similarity(query_tokens, entry["tokens"])
                if score > best_score and score >= self.similarity_threshold:
                    best_score = score
                    best_match = entry["response"]

            return best_match

    def set(self, query: str, response: AIChatResponse):
        tokens = self._tokenize(query)
        if not tokens:
            return

        with self._lock:
            if len(self._entries) >= self.max_size:
                self._entries.pop(0)
            self._entries.append({
                "tokens": tokens,
                "response": response,
                "timestamp": time.time()
            })

    def clear(self):
        with self._lock:
            self._entries.clear()
