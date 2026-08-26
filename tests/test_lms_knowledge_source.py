import pytest
from ai_tutor import (
    KnowledgeSource,
    LMSKnowledgeSource,
    InMemoryTranscriptionQueue,
    Chunk,
    TutorCore,
    MockLLMClient,
    TutorInput
)


# Sample LMS payload matching API contract
SAMPLE_LMS_PAYLOAD = {
    "course": {
        "id": 101,
        "title": "Machine Learning"
    },
    "lessons": [
        {
            "id": 10,
            "name": "ML Basics",
            "lectures": [
                {
                    "id": 50,
                    "title": "Supervised Learning",
                    "type": "video",
                    "transcript": "Supervised learning is trained on labeled pairs (x,y). It learns mapping f(x) -> y."
                },
                {
                    "id": 52,
                    "title": "Deep Learning Intro Video",
                    "type": "youtube",
                    # No transcript provided -> Should stub/enqueue to TranscriptionQueue
                    "url": "https://youtube.com/watch?v=sample123"
                }
            ]
        }
    ]
}


def test_lms_knowledge_source_interface_conformance():
    lms_ks = LMSKnowledgeSource()
    assert isinstance(lms_ks, KnowledgeSource)


def test_lms_course_content_conversion_and_transcription_queue():
    t_queue = InMemoryTranscriptionQueue()
    lms_ks = LMSKnowledgeSource(
        custom_fetcher=lambda course_id: SAMPLE_LMS_PAYLOAD,
        transcription_queue=t_queue
    )

    # 1. Retrieve lecture 50 (has transcript)
    chunks_50 = lms_ks.retrieve(
        query="What is supervised learning?",
        filters={"course_id": 101, "lecture_id": 50}
    )
    assert len(chunks_50) > 0
    c50 = chunks_50[0]
    assert c50.source_id == 50
    assert c50.source_title == "Supervised Learning"
    assert c50.metadata["transcription_pending"] is False
    assert "supervised learning" in c50.content.lower()

    # 2. Retrieve lecture 52 (video/youtube with NO transcript)
    chunks_52 = lms_ks.retrieve(
        query="Tell me about deep learning video",
        filters={"course_id": 101, "lecture_id": 52}
    )
    assert len(chunks_52) > 0
    c52 = chunks_52[0]
    assert c52.source_id == 52
    assert c52.metadata["transcription_pending"] is True
    assert "transcription_job_id" in c52.metadata

    # Check TranscriptionQueue stub received the job
    jobs = t_queue.list_jobs()
    assert len(jobs) == 1
    assert jobs[0]["lecture_id"] == 52
    assert jobs[0]["lecture_type"] == "youtube"
    assert jobs[0]["status"] == "QUEUED"


def test_lms_knowledge_source_with_tutor_core():
    lms_ks = LMSKnowledgeSource(
        custom_fetcher=lambda course_id: SAMPLE_LMS_PAYLOAD
    )
    llm = MockLLMClient()
    tutor = TutorCore(llm_client=llm, knowledge_source=lms_ks)

    input_payload = TutorInput(
        message="Explain supervised learning",
        course_id=101,
        lecture_id=50
    )

    output = tutor.process_turn(input_payload)
    assert output.knowledge_source_used == "LMSKnowledgeSource"
    assert len(output.sources) > 0
    assert output.sources[0].lecture_id == 50
