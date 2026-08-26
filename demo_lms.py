from ai_tutor import (
    LMSKnowledgeSource,
    InMemoryTranscriptionQueue,
    TutorCore,
    MockLLMClient,
    TutorInput
)

# Realistic LMS payload conforming to contract
LMS_MOCK_RESPONSE = {
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
                    "transcript": "Supervised learning models learn mapping f(x) -> y from labeled datasets using cost functions like MSE."
                },
                {
                    "id": 55,
                    "title": "Neural Networks Live Walkthrough",
                    "type": "youtube",
                    "url": "https://youtube.com/watch?v=neural_demo_101",
                    "description": "Deep dive into perceptrons and backpropagation."
                    # No transcript -> Non-blocking enqueue
                }
            ]
        }
    ]
}

def run_lms_demo():
    print("=" * 70)
    print(" LMS KNOWLEDGE SOURCE & ASYNC TRANSCRIPTION QUEUE DEMO")
    print("=" * 70)

    queue = InMemoryTranscriptionQueue()
    lms_source = LMSKnowledgeSource(
        custom_fetcher=lambda course_id: LMS_MOCK_RESPONSE,
        transcription_queue=queue
    )

    tutor = TutorCore(
        llm_client=MockLLMClient(),
        knowledge_source=lms_source
    )

    # 1. Querying lecture with existing transcript
    print("\n--- [Query 1: Lecture with Existing Transcript] ---")
    out1 = tutor.process_turn(TutorInput(
        message="What are supervised learning cost functions?",
        course_id=101,
        lecture_id=50
    ))
    print(f"Tutor Answer: {out1.answer}")
    print(f"Sources:      {[s.title for s in out1.sources]}")

    # 2. Querying video lecture WITHOUT transcript (should NOT block)
    print("\n--- [Query 2: YouTube Lecture Without Transcript (Non-Blocking)] ---")
    out2 = tutor.process_turn(TutorInput(
        message="Tell me about the neural networks walkthrough video",
        course_id=101,
        lecture_id=55
    ))
    print(f"Tutor Answer: {out2.answer}")
    print(f"Sources:      {[s.title for s in out2.sources]}")

    # 3. Inspect TranscriptionQueue status
    print("\n--- [TranscriptionQueue Stub Records] ---")
    jobs = queue.list_jobs()
    for j in jobs:
        print(f"  * Job ID: {j['job_id']} | Lecture: {j['lecture_id']} ({j['lecture_title']}) | Type: {j['lecture_type']} | Status: {j['status']}")

if __name__ == "__main__":
    run_lms_demo()
