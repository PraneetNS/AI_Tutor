"""
learner_store.py
----------------
Persistence layer for LearnerState.

Architecture
~~~~~~~~~~~~
- ``BaseLearnerStateStore``: Abstract interface for loading and saving LearnerState.
- ``InMemoryLearnerStateStore``: Thread-safe, zero-dependency store for development and testing.
- ``PostgresRedisLearnerStateStore``: Dual-tier storage for production:
    * Postgres: Source of truth (persists JSONB representation in ``learner_states`` table).
    * Redis: Fast read-through cache for low-latency retrieval on incoming requests.
    * Auto-migrates Postgres schema on startup.
"""

from __future__ import annotations

import json
import logging
import threading
from abc import ABC, abstractmethod
from typing import Dict, Optional

from .models import LearnerState

logger = logging.getLogger("ai_tutor.learner_store")


class BaseLearnerStateStore(ABC):
    """Abstract interface for LearnerState storage."""

    @abstractmethod
    def load(self, student_id: str) -> Optional[LearnerState]:
        """
        Load current LearnerState for the given student_id.
        Returns None if no state exists yet.
        """
        pass

    @abstractmethod
    def save(self, state: LearnerState) -> None:
        """
        Persist the updated LearnerState.
        """
        pass


class InMemoryLearnerStateStore(BaseLearnerStateStore):
    """Thread-safe in-memory store for unit tests and local development."""

    def __init__(self) -> None:
        self._store: Dict[str, LearnerState] = {}
        self._lock = threading.Lock()

    def load(self, student_id: str) -> Optional[LearnerState]:
        with self._lock:
            state = self._store.get(str(student_id))
            return state.model_copy(deep=True) if state else None

    def save(self, state: LearnerState) -> None:
        with self._lock:
            self._store[str(state.student_id)] = state.model_copy(deep=True)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


class PostgresRedisLearnerStateStore(BaseLearnerStateStore):
    """
    Production-ready dual-tier LearnerState store:
    - Postgres: Durable source of truth in table ``learner_states``.
    - Redis: Cache layer with configurable TTL (default 3600s).

    Libraries ``psycopg2`` and ``redis`` are imported lazily.
    """

    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS learner_states (
        student_id      TEXT PRIMARY KEY,
        state_data      JSONB NOT NULL,
        schema_version  INT NOT NULL DEFAULT 1,
        updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """

    UPSERT_SQL = """
    INSERT INTO learner_states (student_id, state_data, schema_version, updated_at)
    VALUES (%(student_id)s, %(state_data)s, %(schema_version)s, %(updated_at)s)
    ON CONFLICT (student_id) DO UPDATE SET
        state_data = EXCLUDED.state_data,
        schema_version = EXCLUDED.schema_version,
        updated_at = EXCLUDED.updated_at;
    """

    SELECT_SQL = """
    SELECT state_data FROM learner_states WHERE student_id = %s;
    """

    def __init__(
        self,
        postgres_dsn: str,
        redis_url: Optional[str] = None,
        cache_ttl_seconds: int = 3600,
        auto_migrate: bool = True
    ) -> None:
        self.postgres_dsn = postgres_dsn
        self.redis_url = redis_url
        self.cache_ttl_seconds = cache_ttl_seconds
        self._redis_client = None

        if auto_migrate:
            self._migrate()

    def _get_redis(self):
        if self._redis_client is None and self.redis_url:
            try:
                import redis
                self._redis_client = redis.Redis.from_url(self.redis_url, decode_responses=True)
            except Exception as e:
                logger.warning(f"Failed to initialize Redis client: {e}")
        return self._redis_client

    def _migrate(self) -> None:
        try:
            import psycopg2
            with psycopg2.connect(self.postgres_dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(self.CREATE_TABLE_SQL)
                conn.commit()
            logger.info("Postgres schema for learner_states verified.")
        except Exception as e:
            logger.error(f"Failed to migrate learner_states table: {e}")
            raise

    def load(self, student_id: str) -> Optional[LearnerState]:
        key = f"learner_state:{student_id}"

        # 1. Try Redis read-through
        r = self._get_redis()
        if r:
            try:
                cached = r.get(key)
                if cached:
                    return LearnerState.model_validate_json(cached)
            except Exception as e:
                logger.warning(f"Redis cache read failed for student {student_id}: {e}")

        # 2. Fall back to Postgres (Source of Truth)
        try:
            import psycopg2
            with psycopg2.connect(self.postgres_dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(self.SELECT_SQL, (str(student_id),))
                    row = cur.fetchone()
            if row:
                raw_json = row[0]
                state = LearnerState.model_validate(raw_json if isinstance(raw_json, dict) else json.loads(raw_json))
                # Populate cache
                if r:
                    try:
                        r.setex(key, self.cache_ttl_seconds, state.model_dump_json())
                    except Exception as e:
                        logger.warning(f"Redis write-through failed: {e}")
                return state
        except Exception as e:
            logger.error(f"Postgres read failed for student {student_id}: {e}")

        return None

    def save(self, state: LearnerState) -> None:
        # 1. Write to Postgres source of truth
        try:
            import psycopg2
            import psycopg2.extras
            params = {
                "student_id": str(state.student_id),
                "state_data": psycopg2.extras.Json(state.model_dump()),
                "schema_version": state.schema_version,
                "updated_at": state.updated_at,
            }
            with psycopg2.connect(self.postgres_dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(self.UPSERT_SQL, params)
                conn.commit()
        except Exception as e:
            logger.error(f"Postgres save failed for student {state.student_id}: {e}")
            raise

        # 2. Mirror to Redis cache
        r = self._get_redis()
        if r:
            try:
                key = f"learner_state:{state.student_id}"
                r.setex(key, self.cache_ttl_seconds, state.model_dump_json())
            except Exception as e:
                logger.warning(f"Redis cache write failed for student {state.student_id}: {e}")
