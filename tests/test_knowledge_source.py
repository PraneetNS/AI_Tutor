import pytest
from ai_tutor import (
    KnowledgeSource,
    MockKnowledgeSource,
    Chunk,
    TutorCore,
    MockLLMClient,
    TutorInput,
    ChatMessage,
    Role
)


def test_mock_knowledge_source_retrieval():
    ks = MockKnowledgeSource()

    # Query matching supervised learning
    chunks = ks.retrieve(query="What is supervised learning and labeled data?")
    assert len(chunks) > 0
    top_chunk = chunks[0]
    assert isinstance(top_chunk, Chunk)
    assert top_chunk.source_id == 50
    assert top_chunk.source_title == "Supervised Learning"
    assert "supervised learning" in top_chunk.content.lower()


def test_knowledge_source_filtering():
    ks = MockKnowledgeSource()

    # Filter by lecture_id 60 (Gradient descent)
    chunks = ks.retrieve(
        query="Tell me about algorithms",
        filters={"course_id": 101, "lecture_id": 60}
    )
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.metadata["lecture_id"] == 60
        assert chunk.source_title == "Gradient Descent & Cost Functions"


def test_tutor_core_with_knowledge_source():
    ks = MockKnowledgeSource()
    llm = MockLLMClient()
    tutor = TutorCore(llm_client=llm, knowledge_source=ks)

    input_data = TutorInput(
        message="Explain supervised learning",
        course_id=101,
        lecture_id=50,
        conversation_history=[]
    )

    output = tutor.process_turn(input_data)

    assert output.knowledge_source_used == "MockKnowledgeSource"
    assert len(output.sources) > 0
    citation = output.sources[0]
    assert citation.lecture_id == 50
    assert citation.title == "Supervised Learning"
    assert citation.chunk_id is not None
