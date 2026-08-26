import os
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from .models import ChatMessage, ClassificationResult


class ReviewLogger:
    """
    Structured logger for classifier auditing.
    Persists all OFF_TOPIC and low-confidence classification turns to a JSONL file
    and outputs warning logs for monitoring.
    """

    def __init__(self, log_filepath: Optional[str] = None):
        if log_filepath:
            self.log_path = Path(log_filepath)
        else:
            default_dir = Path(os.getenv("AI_TUTOR_LOG_DIR", "logs"))
            default_dir.mkdir(parents=True, exist_ok=True)
            self.log_path = default_dir / "classifier_review.jsonl"

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("ai_tutor.classifier.review")

    def log_review_event(
        self,
        student_message: str,
        classification: ClassificationResult,
        conversation_history: Optional[List[ChatMessage]] = None,
        session_id: Optional[str] = None,
        reason: str = "OFF_TOPIC_OR_LOW_CONFIDENCE"
    ) -> Dict[str, Any]:
        """Record the flagged interaction."""
        recent_context = []
        if conversation_history:
            recent_context = [
                {"role": msg.role.value, "content": msg.content}
                for msg in conversation_history[-3:]
            ]

        event = {
            "timestamp": classification.timestamp,
            "session_id": session_id,
            "flag_reason": reason,
            "label": classification.label.value,
            "confidence": round(classification.confidence, 4),
            "rationale": classification.rationale,
            "student_message": student_message,
            "recent_context": recent_context
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write review log to {self.log_path}: {e}")

        self.logger.info(
            f"[CLASSIFIER REVIEW FLAGGED] Reason={reason} | Label={classification.label.value} "
            f"| Confidence={classification.confidence:.2f} | Msg='{student_message[:60]}...'"
        )

        return event
