import pytest
from ai_tutor import (
    TutorCore,
    MockLLMClient,
    InMemorySessionStore,
    ChatMessage,
    Role,
    PedagogyState,
    PedagogyMode,
    TutorInput
)


def test_stateless_tutor_core_turn():
    client = MockLLMClient()
    tutor = TutorCore(llm_client=client)

    # First turn
    turn1 = tutor.generate(
        student_message="Can you explain supervised learning?",
        conversation_history=[],
        current_state=PedagogyState(hint_level=0, stuck=False)
    )

    assert turn1.pedagogy_state.topic == "Supervised Learning"
    assert turn1.pedagogy_state.stuck is False
    assert turn1.pedagogy_state.hint_level == 0
    assert "supervised learning" in turn1.answer.lower()

    # Second turn with history and student indicating they are stuck
    history = [
        ChatMessage(role=Role.USER, content="Can you explain supervised learning?"),
        ChatMessage(role=Role.ASSISTANT, content=turn1.answer)
    ]
    turn2 = tutor.generate(
        student_message="I'm stuck, can you give me a hint?",
        conversation_history=history,
        current_state=turn1.pedagogy_state
    )

    assert turn2.pedagogy_state.stuck is True
    assert turn2.pedagogy_state.hint_level == 1


def test_session_store_integration():
    store = InMemorySessionStore()
    client = MockLLMClient()
    tutor = TutorCore(llm_client=client)

    session_id = "session_xyz_123"

    # Turn 1
    session = store.get_session(session_id)
    assert len(session.messages) == 0

    student_msg_1 = "What is gradient descent?"
    output1 = tutor.generate(
        student_message=student_msg_1,
        conversation_history=session.messages,
        current_state=session.pedagogy_state
    )

    store.append_message(session_id, Role.USER, student_msg_1)
    store.append_message(session_id, Role.ASSISTANT, output1.answer)
    store.update_pedagogy_state(session_id, output1.pedagogy_state)

    # Verify session persisted
    session_after = store.get_session(session_id)
    assert len(session_after.messages) == 2
    assert session_after.pedagogy_state.topic == "Gradient Descent"

    # Turn 2
    student_msg_2 = "Can I get another hint?"
    output2 = tutor.generate(
        student_message=student_msg_2,
        conversation_history=session_after.messages,
        current_state=session_after.pedagogy_state
    )

    store.append_message(session_id, Role.USER, student_msg_2)
    store.append_message(session_id, Role.ASSISTANT, output2.answer)
    store.update_pedagogy_state(session_id, output2.pedagogy_state)

    final_session = store.get_session(session_id)
    assert len(final_session.messages) == 4
    assert final_session.pedagogy_state.hint_level >= 1
    assert final_session.pedagogy_state.stuck is True
