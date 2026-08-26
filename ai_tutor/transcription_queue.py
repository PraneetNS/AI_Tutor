from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import uuid
import logging

logger = logging.getLogger("ai_tutor.transcription_queue")


class TranscriptionJob:
    def __init__(
        self,
        job_id: str,
        lecture_id: int,
        lecture_title: str,
        lecture_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.job_id = job_id
        self.lecture_id = lecture_id
        self.lecture_title = lecture_title
        self.lecture_type = lecture_type
        self.metadata = metadata or {}
        self.status = "QUEUED"
        self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "lecture_id": self.lecture_id,
            "lecture_title": self.lecture_title,
            "lecture_type": self.lecture_type,
            "status": self.status,
            "metadata": self.metadata,
            "created_at": self.created_at
        }


class TranscriptionQueue(ABC):
    """
    Abstract interface for async video/audio transcription pipeline.
    Lectures without transcripts are queued non-blockingly for background processing.
    """

    @abstractmethod
    def enqueue(
        self,
        lecture_id: int,
        lecture_title: str,
        lecture_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Enqueue a lecture for speech-to-text / transcript extraction.
        Returns job_id. Must be non-blocking.
        """
        pass

    @abstractmethod
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Check status of a queued transcription job."""
        pass


class InMemoryTranscriptionQueue(TranscriptionQueue):
    """
    In-memory stub implementation of TranscriptionQueue.
    Logs queued jobs and maintains job records without blocking execution.
    """

    def __init__(self):
        self._jobs: Dict[str, TranscriptionJob] = {}
        self._queued_lecture_ids = set()

    def enqueue(
        self,
        lecture_id: int,
        lecture_title: str,
        lecture_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        # Avoid duplicate queuing for same lecture
        if lecture_id in self._queued_lecture_ids:
            for job in self._jobs.values():
                if job.lecture_id == lecture_id:
                    return job.job_id

        job_id = f"job_transcribe_{uuid.uuid4().hex[:8]}"
        job = TranscriptionJob(
            job_id=job_id,
            lecture_id=lecture_id,
            lecture_title=lecture_title,
            lecture_type=lecture_type,
            metadata=metadata
        )
        self._jobs[job_id] = job
        self._queued_lecture_ids.add(lecture_id)

        logger.info(
            f"[TRANSCRIPTION QUEUED] JobID={job_id} | LectureID={lecture_id} "
            f"| Title='{lecture_title}' | Type={lecture_type}"
        )
        return job_id

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self._jobs.get(job_id)
        return job.to_dict() if job else None

    def list_jobs(self) -> List[Dict[str, Any]]:
        return [j.to_dict() for j in self._jobs.values()]
