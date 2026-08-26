import os
import re
import math
import logging
from typing import List, Dict, Any, Optional, Callable
import requests

from .models import Chunk
from .knowledge_source import KnowledgeSource
from .transcription_queue import TranscriptionQueue, InMemoryTranscriptionQueue

logger = logging.getLogger("ai_tutor.lms_knowledge_source")

VIDEO_TYPES = {"video", "youtube", "vimeo", "audio", "mp4"}


class LMSKnowledgeSource(KnowledgeSource):
    """
    KnowledgeSource implementation backed by the LMS API contract:
    GET /api/lms/courses/{course_id}/content

    - Fetches course hierarchy (Course > Lessons > Lectures)
    - Converts lessons/lectures into Chunk objects
    - Stubs TranscriptionQueue for video/audio lectures without transcripts (non-blocking)
    - Caches course payloads to minimize redundant network roundtrips
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        auth_token: Optional[str] = None,
        transcription_queue: Optional[TranscriptionQueue] = None,
        custom_fetcher: Optional[Callable[[int], Dict[str, Any]]] = None
    ):
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.transcription_queue = transcription_queue or InMemoryTranscriptionQueue()
        self.custom_fetcher = custom_fetcher
        self._course_cache: Dict[int, Dict[str, Any]] = {}
        self._parsed_chunks_cache: Dict[int, List[Chunk]] = {}

    def fetch_course_content(self, course_id: int) -> Dict[str, Any]:
        """
        Fetch course content from LMS API or custom fetcher with caching.
        Endpoint: GET /api/lms/courses/{course_id}/content
        """
        if course_id in self._course_cache:
            return self._course_cache[course_id]

        if self.custom_fetcher:
            data = self.custom_fetcher(course_id)
            self._course_cache[course_id] = data
            return data

        url = f"{self.base_url}/api/lms/courses/{course_id}/content"
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        try:
            resp = requests.get(url, headers=headers, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            self._course_cache[course_id] = data
            return data
        except Exception as e:
            logger.error(f"Failed to fetch LMS course content from {url}: {e}")
            return {"course": {"id": course_id, "title": f"Course {course_id}"}, "lessons": []}

    def _convert_lms_to_chunks(self, course_data: Dict[str, Any]) -> List[Chunk]:
        """
        Parse Course > Lesson > Lecture payload into standardized Chunk objects.
        Dispatches missing transcripts to TranscriptionQueue non-blockingly.
        """
        course = course_data.get("course", {})
        course_id = course.get("id")
        course_title = course.get("title", f"Course {course_id}")
        lessons = course_data.get("lessons", [])

        chunks: List[Chunk] = []

        for lesson in lessons:
            lesson_id = lesson.get("id")
            lesson_name = lesson.get("name", f"Lesson {lesson_id}")
            lectures = lesson.get("lectures", [])

            for lecture in lectures:
                lecture_id = lecture.get("id")
                lecture_title = lecture.get("title", f"Lecture {lecture_id}")
                lecture_type = lecture.get("type", "text").lower()
                transcript = lecture.get("transcript") or lecture.get("content") or lecture.get("text")
                url = lecture.get("url") or lecture.get("video_url")

                # Handle video/youtube/vimeo lectures without transcript
                if lecture_type in VIDEO_TYPES and not transcript:
                    # Non-blocking stub enqueue
                    job_id = self.transcription_queue.enqueue(
                        lecture_id=lecture_id,
                        lecture_title=lecture_title,
                        lecture_type=lecture_type,
                        metadata={"course_id": course_id, "lesson_id": lesson_id, "url": url}
                    )

                    # Create a lightweight metadata placeholder chunk so the lecture is indexable
                    fallback_content = (
                        f"Lecture '{lecture_title}' (Type: {lecture_type}) in lesson '{lesson_name}'. "
                        f"Transcript is pending background audio processing (JobID: {job_id}). "
                        f"{lecture.get('description', '')}"
                    )

                    chunks.append(
                        Chunk(
                            content=fallback_content.strip(),
                            source_title=lecture_title,
                            source_id=lecture_id,
                            metadata={
                                "course_id": course_id,
                                "course_title": course_title,
                                "lesson_id": lesson_id,
                                "lesson_name": lesson_name,
                                "lecture_id": lecture_id,
                                "lecture_type": lecture_type,
                                "chunk_id": f"chunk_lec_{lecture_id}_meta",
                                "transcription_pending": True,
                                "transcription_job_id": job_id
                            }
                        )
                    )
                elif transcript:
                    # If lecture has full transcript or content, split into passages
                    passages = self._split_text(transcript, max_chars=400)
                    for idx, passage in enumerate(passages, 1):
                        chunks.append(
                            Chunk(
                                content=passage,
                                source_title=lecture_title,
                                source_id=lecture_id,
                                metadata={
                                    "course_id": course_id,
                                    "course_title": course_title,
                                    "lesson_id": lesson_id,
                                    "lesson_name": lesson_name,
                                    "lecture_id": lecture_id,
                                    "lecture_type": lecture_type,
                                    "chunk_id": f"chunk_lec_{lecture_id}_{idx}",
                                    "transcription_pending": False
                                }
                            )
                        )
                else:
                    # Text/article lecture with description
                    desc = lecture.get("description", f"Material for {lecture_title}")
                    chunks.append(
                        Chunk(
                            content=desc,
                            source_title=lecture_title,
                            source_id=lecture_id,
                            metadata={
                                "course_id": course_id,
                                "course_title": course_title,
                                "lesson_id": lesson_id,
                                "lesson_name": lesson_name,
                                "lecture_id": lecture_id,
                                "lecture_type": lecture_type,
                                "chunk_id": f"chunk_lec_{lecture_id}_desc",
                                "transcription_pending": False
                            }
                        )
                    )

        return chunks

    def _split_text(self, text: str, max_chars: int = 400) -> List[str]:
        """Split long text/transcripts into readable paragraph chunks."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()]

        chunks = []
        current = []
        current_len = 0

        for p in paragraphs:
            if current_len + len(p) > max_chars and current:
                chunks.append("\n\n".join(current))
                current = [p]
                current_len = len(p)
            else:
                current.append(p)
                current_len += len(p)

        if current:
            chunks.append("\n\n".join(current))

        return chunks

    def retrieve(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Retrieve chunks from LMS course hierarchy matching filters and query.
        """
        filters = filters or {}
        course_id = filters.get("course_id", 101)  # Default or extracted from filter
        lecture_id_filter = filters.get("lecture_id")
        lesson_id_filter = filters.get("lesson_id")
        top_k = filters.get("top_k", 3)

        # Get or parse course chunks
        if course_id not in self._parsed_chunks_cache:
            course_data = self.fetch_course_content(course_id)
            self._parsed_chunks_cache[course_id] = self._convert_lms_to_chunks(course_data)

        all_chunks = self._parsed_chunks_cache[course_id]

        # Filter by lecture/lesson
        filtered = []
        for c in all_chunks:
            meta = c.metadata or {}
            if lecture_id_filter is not None and meta.get("lecture_id") != lecture_id_filter:
                continue
            if lesson_id_filter is not None and meta.get("lesson_id") != lesson_id_filter:
                continue
            filtered.append(c)

        if not filtered:
            return []

        # Lexical score ranking against query
        query_tokens = set(re.findall(r"\b\w+\b", query.lower()))
        scored = []
        for c in filtered:
            content_tokens = set(re.findall(r"\b\w+\b", c.content.lower()))
            title_tokens = set(re.findall(r"\b\w+\b", c.source_title.lower()))
            overlap = len(query_tokens.intersection(content_tokens)) + (2 * len(query_tokens.intersection(title_tokens)))
            score = round(overlap / (len(query_tokens) + 1e-5), 3) if query_tokens else 0.5
            
            c_copy = c.model_copy(deep=True)
            c_copy.metadata["relevance_score"] = score
            scored.append((score, c_copy))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]
