from ai_tutor import (
    TutorCore,
    MockKnowledgeSource,
    MockLLMClient,
    TutorInput,
    ChatMessage,
    Role
)

def run_rag_demo():
    print("=" * 65)
    print(" TUTOR CORE + DECOUPLED KNOWLEDGE SOURCE DEMO")
    print("=" * 65)

    # 1. Instantiate the mock knowledge source loaded with Course > Lesson > Lecture fixtures
    knowledge_source = MockKnowledgeSource()

    # 2. Instantiate TutorCore with the abstract knowledge source injected
    tutor = TutorCore(
        llm_client=MockLLMClient(),
        knowledge_source=knowledge_source
    )

    # Scenario: Student asks question scoped to Course 101 and Lecture 50
    input_payload = TutorInput(
        message="Explain supervised learning and how loss functions work",
        course_id=101,
        lecture_id=50,
        conversation_history=[]
    )

    print("\n[Input Request]")
    print(f"Message:    '{input_payload.message}'")
    print(f"Filters:    Course ID={input_payload.course_id}, Lecture ID={input_payload.lecture_id}")

    # Process turn
    output = tutor.process_turn(input_payload)

    print("\n[Tutor Output]")
    print(f"Answer:     {output.answer}")
    print(f"Retriever:  {output.knowledge_source_used}")
    print(f"State:      hint_level={output.pedagogy_state.hint_level} | topic='{output.pedagogy_state.topic}' | stuck={output.pedagogy_state.stuck}")

    print("\n[Retrieved Source Citations]")
    for i, src in enumerate(output.sources, 1):
        print(f"  {i}. Lecture {src.lecture_id}: '{src.title}' (Chunk: {src.chunk_id})")
        print(f"     Score: {src.relevance_score} | Snippet: {src.snippet}")

if __name__ == "__main__":
    run_rag_demo()
