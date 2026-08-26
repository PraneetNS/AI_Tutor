import pytest
from ai_tutor import (
    KnowledgeSource,
    HybridVectorKnowledgeSource,
    Chunk,
    TutorCore,
    MockLLMClient,
    TutorInput
)


def test_hybrid_knowledge_source_interface_conformance():
    ks = HybridVectorKnowledgeSource()
    assert isinstance(ks, KnowledgeSource)


def test_hybrid_retrieval_dense_and_bm25():
    ks = HybridVectorKnowledgeSource()

    # Query targeting gradient descent optimization
    chunks = ks.retrieve("What is gradient descent learning rate and oscillation?")
    assert len(chunks) > 0
    top = chunks[0]
    assert isinstance(top, Chunk)
    assert top.source_id == 60
    assert "gradient descent" in top.content.lower()
    assert "hybrid_score" in top.metadata
    assert "bm25_score" in top.metadata
    assert "dense_similarity" in top.metadata
    assert top.metadata["grounded"] is True


def test_grounding_check_pass_drops_irrelevant_noise():
    ks = HybridVectorKnowledgeSource(grounding_threshold=0.35)

    # Completely irrelevant query having zero semantic connection or keywords with ML course
    irrelevant_query = "how to bake sourdough bread with yeast and flour in oven"
    chunks = ks.retrieve(irrelevant_query)

    # Grounding check pass should drop all irrelevant chunks
    assert len(chunks) == 0


def test_hybrid_retriever_with_tutor_core():
    ks = HybridVectorKnowledgeSource()
    llm = MockLLMClient()
    tutor = TutorCore(llm_client=llm, knowledge_source=ks)

    input_data = TutorInput(
        message="What is the difference between batch and stochastic gradient descent?",
        course_id=101,
        lecture_id=60
    )

    output = tutor.process_turn(input_data)
    assert output.knowledge_source_used == "HybridVectorKnowledgeSource"
    assert len(output.sources) > 0
    assert output.sources[0].lecture_id == 60
