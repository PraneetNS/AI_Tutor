from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from threading import Lock
from pydantic import BaseModel, Field
from .models import ChatMessage, PedagogyState, Role


class SessionData(BaseModel):
    session_id: str
    messages: List[ChatMessage] = Field(default_factory=list)
    pedagogy_state: PedagogyState = Field(default_factory=PedagogyState)


class BaseSessionStore(ABC):
    """Abstract Base Class for Session Storage (In-Memory, Redis, DB, etc.)"""

    @abstractmethod
    def get_session(self, session_id: str) -> SessionData:
        """Retrieve existing session or initialize a fresh one."""
        pass

    @abstractmethod
    def save_session(self, session_data: SessionData) -> None:
        """Persist session state and history."""
        pass

    @abstractmethod
    def append_message(self, session_id: str, role: Role, content: str) -> ChatMessage:
        """Append a message turn to the session history."""
        pass

    @abstractmethod
    def update_pedagogy_state(self, session_id: str, state: PedagogyState) -> None:
        """Update pedagogical tracking state for the session."""
        pass

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Remove a session from storage."""
        pass


class InMemorySessionStore(BaseSessionStore):
    """
    Thread-safe In-Memory session store.
    Easily swappable with RedisSessionStore in production.
    """

    def __init__(self):
        self._store: Dict[str, SessionData] = {}
        self._lock = Lock()

    def get_session(self, session_id: str) -> SessionData:
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = SessionData(session_id=session_id)
            return self._store[session_id].model_copy(deep=True)

    def save_session(self, session_data: SessionData) -> None:
        with self._lock:
            self._store[session_data.session_id] = session_data.model_copy(deep=True)

    def append_message(self, session_id: str, role: Role, content: str) -> ChatMessage:
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = SessionData(session_id=session_id)
            msg = ChatMessage(role=role, content=content)
            self._store[session_id].messages.append(msg)
            return msg

    def update_pedagogy_state(self, session_id: str, state: PedagogyState) -> None:
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = SessionData(session_id=session_id)
            self._store[session_id].pedagogy_state = state.model_copy(deep=True)

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            return self._store.pop(session_id, None) is not None
