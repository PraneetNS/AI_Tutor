from ai_tutor import (
    HybridVectorKnowledgeSource,
    TutorCore,
    MockLLMClient,
    TutorInput
)

def run_hybrid_rag_demo():
    print("=" * 70)
    print(" HYBRID VECTOR RETRIEVER DEMO (Dense + BM25 + Rerank + Grounding Gate)")
    print("=" * 70)

    # 1. Instantiate HybridVectorKnowledgeSource
    hybrid_source = HybridVectorKnowledgeSource(
        dense_weight=0.5,
        bm25_weight=0.5,
        grounding_threshold=0.35
    )

    tutor = TutorCore(
        llm_client=MockLLMClient(),
        knowledge_source=hybrid_source
    )

    test_scenarios = [
        (
            "Relevant Query (Grounded)",
            "What is the update rule for gradient descent and what happens if learning rate is too high?",
            {"course_id": 101, "lecture_id": 60}
        ),
        (
            "Relevant Keyword + Semantic Query (Grounded)",
            "Explain labeled training data and Mean Squared Error in supervised learning",
            {"course_id": 101, "lecture_id": 50}
        ),
        (
            "Irrelevant / Non-supporting Noise Query (Dropped by Grounding Gate)",
            "How do I bake chocolate chip cookies in an oven with baking powder?",
            {}
        )
    ]

    for title, query, filters in test_scenarios:
        print(f"\n--- Scenario: {title} ---")
        print(f"Query:   '{query}'")
        print(f"Filters: {filters}")

        chunks = hybrid_source.retrieve(query=query, filters=filters)
        print(f"Grounded Chunks Passed to Model: {len(chunks)}")

        if chunks:
            for i, c in enumerate(chunks, 1):
                meta = c.metadata
                print(f"  [{i}] '{c.source_title}' (Lecture {c.source_id})")
                print(f"      Hybrid Score: {meta.get('hybrid_score')} (Dense: {meta.get('dense_similarity')}, BM25: {meta.get('bm25_score')})")
                safe_snippet = c.content[:120].encode('ascii', 'replace').decode('ascii')
                print(f"      Snippet: {safe_snippet}...")
        else:
            print("  [X] All non-supporting chunks dropped by Grounding-Check Pass!")

if __name__ == "__main__":
    run_hybrid_rag_demo()
