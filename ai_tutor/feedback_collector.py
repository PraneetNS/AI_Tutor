"""
feedback_collector.py
---------------------
Collects and aggregates explicit learner feedback (ratings, helpfulness tags,
comments) on AI Tutor interactions to support reinforcement learning from human feedback (RLHF)
and continuous pedagogy refinement.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class FeedbackItem(BaseModel):
    """Student feedback on a specific tutor turn."""
    feedback_id: Optional[str] = None
    session_id: str = Field(..., description="Conversation session ID")
    student_id: Optional[str] = Field(default=None, description="Student ID")
    rating: int = Field(..., ge=1, le=5, description="1 to 5 star rating")
    helpful: bool = Field(default=True, description="Thumbs up / down binary rating")
    tags: List[str] = Field(default_factory=list, description="Categorical tags ('clear', 'too_fast', 'great_hint')")
    comment: Optional[str] = Field(default=None, description="Optional free-text feedback")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp"
    )


class FeedbackCollector:
    """Thread-safe feedback collector and statistical aggregator."""

    def __init__(self):
        self._lock = threading.Lock()
        self._feedbacks: List[FeedbackItem] = []

    def record(self, item: FeedbackItem) -> FeedbackItem:
        with self._lock:
            self._feedbacks.append(item)
            return item

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._feedbacks)
            if total == 0:
                return {
                    "total_count": 0,
                    "avg_rating": 0.0,
                    "helpful_percentage": 100.0,
                    "tag_counts": {}
                }

            avg_rating = sum(f.rating for f in self._feedbacks) / total
            helpful_count = sum(1 for f in self._feedbacks if f.helpful)
            tag_counts: Dict[str, int] = {}
            for f in self._feedbacks:
                for tag in f.tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

            return {
                "total_count": total,
                "avg_rating": round(avg_rating, 2),
                "helpful_percentage": round((helpful_count / total) * 100, 1),
                "tag_counts": tag_counts
            }

    def clear(self):
        with self._lock:
            self._feedbacks.clear()
