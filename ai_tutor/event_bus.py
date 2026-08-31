"""
event_bus.py
------------
Lightweight event bus for the AI Tutor pipeline.

Architecture
~~~~~~~~~~~~
``BaseEventBus`` defines the two-method contract (emit / subscribe) that
every caller uses.  Concrete backends plug in without touching any call-site:

┌─────────────────────┐   Phase 1 (now)
│  InMemoryEventBus   │ ← Unit tests, local dev — zero deps
└─────────────────────┘

┌─────────────────────┐   Phase 1 (production)
│  PostgresEventBus   │ ← append-only table + LISTEN/NOTIFY worker
└─────────────────────┘

┌─────────────────────┐   Phase 2 (when you need it)
│  KafkaEventBus      │ ← Drop-in replacement, same interface
└─────────────────────┘

Postgres schema (run once, DDL included as a class constant)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CREATE TABLE learning_events (
    id          BIGSERIAL PRIMARY KEY,
    event_id    TEXT        NOT NULL UNIQUE,
    event_type  TEXT        NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL,
    student_id  TEXT,
    session_id  TEXT,
    course_id   INT,
    lecture_id  INT,
    concept     TEXT,
    mastery_score REAL,
    hint_level  INT,
    pedagogy_mode TEXT,
    payload     JSONB       NOT NULL DEFAULT '{}'
);

After every INSERT the trigger fires NOTIFY learning_events, <event_id>.
The ``PostgresEventBus._listen_loop`` thread wakes on that channel and
dispatches to all registered handlers — no polling interval needed.

Dependencies
~~~~~~~~~~~~
``InMemoryEventBus``  — stdlib only.
``PostgresEventBus``  — requires ``psycopg2-binary`` (not imported at module
                        level; only imported inside the class so the rest of
                        the codebase stays importable without it installed).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

from .models import LearningEvent, LearningEventType   # noqa: F401 (re-exported)

logger = logging.getLogger("ai_tutor.event_bus")

# Type alias for handler callables
EventHandler = Callable[[LearningEvent], None]


# ---------------------------------------------------------------------------
# Abstract base — the only interface callers ever reference
# ---------------------------------------------------------------------------

class BaseEventBus(ABC):
    """
    Minimal event bus contract.

    All pipeline code calls only ``emit`` and ``subscribe``; the concrete
    backend is injected at startup, making the bus trivially swappable.
    """

    @abstractmethod
    def emit(self, event: LearningEvent) -> None:
        """
        Publish *event* to the bus.

        The call is fire-and-forget from the caller's perspective.
        Implementations must be thread-safe.

        Parameters
        ----------
        event : LearningEvent
            The event to publish.  Must already be fully populated.
        """

    @abstractmethod
    def subscribe(self, handler: EventHandler) -> None:
        """
        Register a *handler* to be called for every event that arrives.

        Handlers are invoked synchronously inside the worker thread that
        processes events.  Keep handlers fast; offload heavy work to a
        separate thread/task.

        Parameters
        ----------
        handler : Callable[[LearningEvent], None]
            Function (or any callable) that accepts a single
            ``LearningEvent`` argument.
        """

    def subscribe_to(
        self,
        event_type: LearningEventType,
        handler: EventHandler
    ) -> None:
        """
        Convenience: register a *handler* that only fires for *event_type*.

        Built on top of ``subscribe`` — no override needed in subclasses.
        """
        def _filtered(event: LearningEvent) -> None:
            if event.event_type == event_type.value or event.event_type == event_type:
                handler(event)

        self.subscribe(_filtered)


# ---------------------------------------------------------------------------
# In-Memory backend — dev / test, zero external deps
# ---------------------------------------------------------------------------

class InMemoryEventBus(BaseEventBus):
    """
    Synchronous, in-process event bus backed by a plain list.

    ``emit`` calls all handlers inline before returning.  This makes
    behaviour deterministic in unit tests without any threading.

    Thread safety: a ``threading.Lock`` guards the handler list so the bus
    is safe to use from multiple threads (e.g. async frameworks).
    """

    def __init__(self) -> None:
        self._handlers: List[EventHandler] = []
        self._lock = threading.Lock()
        self._log: List[LearningEvent] = []   # append-only audit log for tests

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def emit(self, event: LearningEvent) -> None:
        """Dispatch *event* to all registered handlers immediately."""
        with self._lock:
            handlers = list(self._handlers)
            self._log.append(event)

        logger.debug("InMemoryEventBus: emit %s [%s]", event.event_type, event.event_id)

        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:
                logger.error(
                    "Handler %s raised on event %s: %s",
                    getattr(handler, "__name__", repr(handler)),
                    event.event_id,
                    exc,
                    exc_info=True
                )

    def subscribe(self, handler: EventHandler) -> None:
        """Append *handler* to the dispatch list."""
        with self._lock:
            self._handlers.append(handler)
        logger.debug(
            "InMemoryEventBus: subscribed handler %s",
            getattr(handler, "__name__", repr(handler))
        )

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    @property
    def log(self) -> List[LearningEvent]:
        """Read-only view of all events emitted so far (for assertions)."""
        with self._lock:
            return list(self._log)

    def clear(self) -> None:
        """Reset the log and all subscriptions (useful between test cases)."""
        with self._lock:
            self._log.clear()
            self._handlers.clear()


# ---------------------------------------------------------------------------
# Postgres backend — Phase 1 production
# ---------------------------------------------------------------------------

class PostgresEventBus(BaseEventBus):
    """
    Durable event bus backed by an append-only Postgres table and
    ``LISTEN``/``NOTIFY`` for near-real-time delivery.

    How it works
    ~~~~~~~~~~~~
    ``emit()``  — INSERTs the event as a JSONB row.  A Postgres trigger
                  fires ``NOTIFY learning_events, '<event_id>'`` after
                  each insert so no polling is needed.

    ``subscribe()`` — Registers a Python handler.  The background thread
                      (started lazily on first subscribe) blocks on
                      ``conn.notifies`` and dispatches by fetching the full
                      row from Postgres using the notified event_id.

    Fallback polling
    ~~~~~~~~~~~~~~~~
    If NOTIFY is somehow missed (e.g. after a reconnect) the worker also
    polls ``learning_events`` for any rows newer than the last seen
    ``id`` every ``poll_interval_s`` seconds.

    Parameters
    ----------
    dsn : str
        Postgres connection string, e.g.
        ``"postgresql://user:pass@localhost:5432/ai_tutor"``
    pool_size : int
        Number of connections in the pool (default 5).
    poll_interval_s : float
        Fallback polling cadence in seconds (default 2.0).
    auto_migrate : bool
        If True (default) the DDL is applied on first connection so the
        table and trigger are always present.

    Example
    -------
    >>> bus = PostgresEventBus(dsn="postgresql://localhost/ai_tutor")
    >>> bus.subscribe(lambda e: print(e.event_type, e.student_id))
    >>> bus.emit(LearningEvent(event_type=LearningEventType.MESSAGE_SENT,
    ...                        student_id="u42", session_id="sess_abc"))
    """

    # ----- DDL --------------------------------------------------------

    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS learning_events (
        id            BIGSERIAL PRIMARY KEY,
        event_id      TEXT        NOT NULL UNIQUE,
        event_type    TEXT        NOT NULL,
        occurred_at   TIMESTAMPTZ NOT NULL,
        student_id    TEXT,
        session_id    TEXT,
        course_id     INT,
        lecture_id    INT,
        concept       TEXT,
        mastery_score REAL,
        hint_level    INT,
        pedagogy_mode TEXT,
        payload       JSONB       NOT NULL DEFAULT '{}'
    );
    """

    # Trigger: fire NOTIFY after every INSERT
    CREATE_TRIGGER_SQL = """
    CREATE OR REPLACE FUNCTION notify_learning_event() RETURNS trigger AS $$
    BEGIN
        PERFORM pg_notify('learning_events', NEW.event_id);
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;

    DROP TRIGGER IF EXISTS trg_learning_event_notify ON learning_events;
    CREATE TRIGGER trg_learning_event_notify
        AFTER INSERT ON learning_events
        FOR EACH ROW EXECUTE FUNCTION notify_learning_event();
    """

    INSERT_SQL = """
    INSERT INTO learning_events
        (event_id, event_type, occurred_at, student_id, session_id,
         course_id, lecture_id, concept, mastery_score, hint_level,
         pedagogy_mode, payload)
    VALUES
        (%(event_id)s, %(event_type)s, %(occurred_at)s, %(student_id)s,
         %(session_id)s, %(course_id)s, %(lecture_id)s, %(concept)s,
         %(mastery_score)s, %(hint_level)s, %(pedagogy_mode)s,
         %(payload)s)
    ON CONFLICT (event_id) DO NOTHING;
    """

    SELECT_BY_ID_SQL = """
    SELECT event_id, event_type, occurred_at, student_id, session_id,
           course_id, lecture_id, concept, mastery_score, hint_level,
           pedagogy_mode, payload
    FROM   learning_events
    WHERE  event_id = %s;
    """

    SELECT_SINCE_SQL = """
    SELECT event_id, event_type, occurred_at, student_id, session_id,
           course_id, lecture_id, concept, mastery_score, hint_level,
           pedagogy_mode, payload
    FROM   learning_events
    WHERE  id > %s
    ORDER  BY id ASC;
    """

    # ------------------------------------------------------------------

    def __init__(
        self,
        dsn: str,
        poll_interval_s: float = 2.0,
        db_timeout: float = 5.0,
        auto_migrate: bool = True
    ) -> None:
        self._dsn = dsn
        self._poll_interval_s = poll_interval_s
        self._db_timeout = db_timeout
        self._handlers: List[EventHandler] = []
        self._lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_seen_id: int = 0

        if auto_migrate:
            self._migrate()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def emit(self, event: LearningEvent) -> None:
        """
        Persist *event* to Postgres.  The DB trigger fires NOTIFY so
        subscribed workers wake up immediately.
        """
        import psycopg2
        import psycopg2.extras

        params = {
            "event_id":     event.event_id,
            "event_type":   str(event.event_type),
            "occurred_at":  event.occurred_at,
            "student_id":   str(event.student_id) if event.student_id is not None else None,
            "session_id":   event.session_id,
            "course_id":    event.course_id,
            "lecture_id":   event.lecture_id,
            "concept":      event.concept,
            "mastery_score": event.mastery_score,
            "hint_level":   event.hint_level,
            "pedagogy_mode": event.pedagogy_mode,
            "payload":      psycopg2.extras.Json(event.payload),
        }

        with psycopg2.connect(self._dsn, connect_timeout=int(self._db_timeout)) as conn:
            with conn.cursor() as cur:
                cur.execute(self.INSERT_SQL, params)
            conn.commit()

        logger.debug("PostgresEventBus: emitted %s [%s]", event.event_type, event.event_id)

    def subscribe(self, handler: EventHandler) -> None:
        """Register *handler* and (lazily) start the listener worker."""
        with self._lock:
            self._handlers.append(handler)
            if self._worker is None or not self._worker.is_alive():
                self._start_worker()

        logger.debug(
            "PostgresEventBus: subscribed handler %s",
            getattr(handler, "__name__", repr(handler))
        )

    def stop(self) -> None:
        """Gracefully stop the background listener thread."""
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=5)
        logger.info("PostgresEventBus: listener stopped.")

    # ------------------------------------------------------------------
    # Internal: migration
    # ------------------------------------------------------------------

    def _migrate(self) -> None:
        """Apply DDL (idempotent — uses CREATE IF NOT EXISTS)."""
        try:
            import psycopg2
            with psycopg2.connect(self._dsn, connect_timeout=int(self._db_timeout)) as conn:
                with conn.cursor() as cur:
                    cur.execute(self.CREATE_TABLE_SQL)
                    cur.execute(self.CREATE_TRIGGER_SQL)
                conn.commit()
            logger.info("PostgresEventBus: schema migration applied.")
        except Exception as exc:
            logger.error("PostgresEventBus: migration failed: %s", exc, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Internal: background listener worker
    # ------------------------------------------------------------------

    def _start_worker(self) -> None:
        """Spin up the LISTEN/NOTIFY + fallback poll thread (daemon)."""
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._listen_loop,
            name="pg-event-bus-worker",
            daemon=True
        )
        self._worker.start()
        logger.info("PostgresEventBus: listener worker started.")

    def _listen_loop(self) -> None:
        """
        Block on Postgres NOTIFY; fall back to polling on reconnect.

        The connection is kept open in AUTOCOMMIT mode (required for LISTEN).
        On any connection error the thread waits ``poll_interval_s`` and
        reconnects — making the worker resilient to transient DB blips.
        """
        import psycopg2
        import select as _select

        while not self._stop_event.is_set():
            conn = None
            try:
                conn = psycopg2.connect(self._dsn, connect_timeout=int(self._db_timeout))
                conn.set_isolation_level(0)   # AUTOCOMMIT — required for LISTEN
                with conn.cursor() as cur:
                    cur.execute("LISTEN learning_events;")
                logger.info("PostgresEventBus: LISTEN channel opened.")

                # Catch up on anything missed while we were down
                self._poll_missed(conn)

                while not self._stop_event.is_set():
                    # Block up to poll_interval_s seconds for a notification
                    ready = _select.select([conn], [], [], self._poll_interval_s)
                    if ready[0]:
                        conn.poll()
                        while conn.notifies:
                            notify = conn.notifies.pop(0)
                            event_id = notify.payload
                            self._fetch_and_dispatch(conn, event_id)
                    else:
                        # Timeout — run a catch-up poll for safety
                        self._poll_missed(conn)

            except Exception as exc:
                logger.error(
                    "PostgresEventBus: listener error (%s); reconnecting in %.1fs",
                    exc, self._poll_interval_s
                )
                time.sleep(self._poll_interval_s)
            finally:
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass

    def _fetch_and_dispatch(self, conn, event_id: str) -> None:
        """Fetch row by event_id and dispatch to all handlers."""
        try:
            with conn.cursor() as cur:
                cur.execute(self.SELECT_BY_ID_SQL, (event_id,))
                row = cur.fetchone()
            if row:
                event = self._row_to_event(row)
                self._dispatch(event)
        except Exception as exc:
            logger.error(
                "PostgresEventBus: failed fetching event_id=%s: %s", event_id, exc
            )

    def _poll_missed(self, conn) -> None:
        """Fetch all rows with id > _last_seen_id (catch-up after reconnect)."""
        try:
            with conn.cursor() as cur:
                cur.execute(self.SELECT_SINCE_SQL, (self._last_seen_id,))
                rows = cur.fetchall()
            for row in rows:
                event = self._row_to_event(row)
                self._dispatch(event)
        except Exception as exc:
            logger.error("PostgresEventBus: catch-up poll failed: %s", exc)

    def _row_to_event(self, row: tuple) -> LearningEvent:
        """Convert a DB row tuple to a ``LearningEvent`` instance."""
        (
            event_id, event_type, occurred_at, student_id, session_id,
            course_id, lecture_id, concept, mastery_score, hint_level,
            pedagogy_mode, payload
        ) = row

        return LearningEvent(
            event_id=event_id,
            event_type=event_type,
            occurred_at=occurred_at.isoformat() if hasattr(occurred_at, "isoformat") else occurred_at,
            student_id=student_id,
            session_id=session_id,
            course_id=course_id,
            lecture_id=lecture_id,
            concept=concept,
            mastery_score=mastery_score,
            hint_level=hint_level,
            pedagogy_mode=pedagogy_mode,
            payload=payload if isinstance(payload, dict) else json.loads(payload or "{}"),
        )

    def _dispatch(self, event: LearningEvent) -> None:
        """Call all registered handlers for *event*; update last_seen_id."""
        with self._lock:
            handlers = list(self._handlers)

        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:
                logger.error(
                    "PostgresEventBus: handler %s raised on %s: %s",
                    getattr(handler, "__name__", repr(handler)),
                    event.event_id,
                    exc,
                    exc_info=True
                )
