import os
from ai_tutor import (
    TutorCore,
    InMemorySessionStore,
    MockLLMClient,
    OpenAILLMClient,
    Role,
    ChatMessage
)

def run_demo():
    print("=" * 60)
    print(" AI TUTOR CORE MODULE DEMO (Stateless + In-Memory Store)")
    print("=" * 60)

    # 1. Initialize store and LLM client
    store = InMemorySessionStore()

    # Use OpenAILLMClient if API key exists, otherwise MockLLMClient
    if os.getenv("OPENAI_API_KEY"):
        print(" Using live OpenAILLMClient (API key detected).")
        llm_client = OpenAILLMClient(model="gpt-4o-mini")
    else:
        print("[INFO] Using MockLLMClient (mock heuristics). Set OPENAI_API_KEY for live API.")
        llm_client = MockLLMClient()

    tutor = TutorCore(llm_client=llm_client)

    session_id = "student_session_001"

    # Multi-turn interaction
    conversation_steps = [
        "Hi! What is supervised learning?",
        "I'm a bit confused on cost function gradients. Could I get a hint?",
        "Can you give me a more specific hint with the math?"
    ]

    for step, student_msg in enumerate(conversation_steps, 1):
        print(f"\n--- Turn {step} ---")
        print(f"Student: {student_msg}")

        # Fetch session state & history
        session = store.get_session(session_id)

        # Call stateless TutorCore
        output = tutor.generate(
            student_message=student_msg,
            conversation_history=session.messages,
            current_state=session.pedagogy_state
        )

        # Update session store
        store.append_message(session_id, Role.USER, student_msg)
        store.append_message(session_id, Role.ASSISTANT, output.answer)
        store.update_pedagogy_state(session_id, output.pedagogy_state)

        # Display result
        print(f"Tutor:   {output.answer}")
        print(f"State:   hint_level={output.pedagogy_state.hint_level} | topic='{output.pedagogy_state.topic}' | stuck={output.pedagogy_state.stuck} | mode={output.pedagogy_state.pedagogy_mode.value}")

    print("\n Demo completed successfully.")

if __name__ == "__main__":
    run_demo()
