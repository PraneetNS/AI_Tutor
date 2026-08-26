from starlette.testclient import TestClient
from ai_tutor import (
    create_app,
    TutorPipeline,
    HybridVectorKnowledgeSource,
    MockLLMClient,
    DefaultModelAdapter
)

def run_api_demo():
    print("=" * 70)
    print(" END-TO-END TUTOR API & 6-STAGE PIPELINE DEMO")
    print(" Router -> Pedagogy -> (Knowledge) -> Prompt -> Model -> Guardrails")
    print("=" * 70)

    # Wire complete pipeline
    knowledge_source = HybridVectorKnowledgeSource()
    pipeline = TutorPipeline(
        knowledge_source=knowledge_source,
        model_adapter=DefaultModelAdapter(llm_client=MockLLMClient())
    )

    app = create_app(pipeline=pipeline)
    client = TestClient(app)

    scenarios = [
        (
            "CONCEPT Query + course_id (Triggers Socratic + Knowledge Retrieval)",
            {
                "message": "Why does gradient descent oscillate when the learning rate is too large?",
                "course_id": 101,
                "lecture_id": 60,
                "session_id": "sess_student_demo_1"
            }
        ),
        (
            "FACTUAL Query (Direct mode, skips heavy retrieval)",
            {
                "message": "What is the formula for Mean Squared Error?",
                "course_id": 101,
                "session_id": "sess_student_demo_1"
            }
        ),
        (
            "OFF_TOPIC Query (Redirects politely, zero retrieval)",
            {
                "message": "What is the capital of France?",
                "session_id": "sess_student_demo_1"
            }
        )
    ]

    for title, payload in scenarios:
        print(f"\n--- [Test: {title}] ---")
        print(f"POST /api/ai/chat: {payload['message']}")

        resp = client.post("/api/ai/chat", json=payload)
        data = resp.json()

        print(f"Status Code:    {resp.status_code}")
        print(f"Pedagogy Mode:  {data.get('pedagogy_mode')}")
        print(f"Knowledge Used: {data.get('knowledge_source_used')}")
        print(f"Sources Count:  {len(data.get('sources', []))}")
        print(f"Answer:         {data.get('answer')}")

if __name__ == "__main__":
    run_api_demo()
