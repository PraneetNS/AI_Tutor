"""
End-to-end integration tests simulating complete multi-turn student learning journeys:
1. Progressive Socratic scaffolding (Hint 0 -> Hint 1 -> Hint 2 -> Hint 3).
2. Prerequisite mastery unlock path through DAG.
3. Off-topic redirect and student recovery.
"""

from ai_tutor.models import AIChatRequest, PedagogyMode, LearningEvent, LearningEventType
from ai_tutor.pipeline import TutorPipeline
from ai_tutor.knowledge_source import MockKnowledgeSource
from ai_tutor.concept_graph import create_ml_concept_graph
from ai_tutor.learner_model import LearnerModelEngine, BKTUpdater
from ai_tutor.learner_store import InMemoryLearnerStateStore


def test_e2e_multi_turn_socratic_scaffolding_progression():
    pipeline = TutorPipeline(
        knowledge_source=MockKnowledgeSource(),
        model_adapter=None
    )
    session_id = "e2e_student_socratic_journey"

    # Turn 1: Initial conceptual question (Level 0)
    req1 = AIChatRequest(
        session_id=session_id,
        message="I don't understand how gradient descent calculates the step direction.",
        course_id=1,
        lecture_id=1
    )
    res1 = pipeline.process(req1)
    assert res1.pedagogy_mode in (PedagogyMode.SOCRATIC, "socratic")
    assert res1.hint_level >= 1

    # Turn 2: Student asks for another hint (Level 2)
    req2 = AIChatRequest(
        session_id=session_id,
        message="Can you give me another hint about partial derivatives?",
        course_id=1,
        lecture_id=1
    )
    res2 = pipeline.process(req2)
    assert res2.hint_level >= 2

    # Turn 3: Student asks for another hint (Level 3)
    req3 = AIChatRequest(
        session_id=session_id,
        message="Give me one more hint please, still a bit confused.",
        course_id=1,
        lecture_id=1
    )
    res3 = pipeline.process(req3)
    assert res3.hint_level >= 3


def test_e2e_bkt_mastery_unlock_journey():
    store = InMemoryLearnerStateStore()
    engine = LearnerModelEngine(store=store)
    graph = create_ml_concept_graph()

    student_id = "learner_alex_42"

    # Initial state
    state = engine.get_learner_state(student_id)
    assert state is None

    # Step 1: 3 consecutive correct answers on 'loss_functions'
    for _ in range(3):
        engine.process_event(LearningEvent(
            student_id=student_id,
            event_type=LearningEventType.ANSWER_SUBMITTED,
            concept="loss_functions",
            payload={"concept": "loss_functions", "correct": True, "hints_used": 0}
        ))

    state_after = engine.get_learner_state(student_id)
    assert state_after is not None
    mastery = state_after.concept_mastery["loss_functions"].mastery
    assert mastery > 0.60

    # Step 2: Diagnostic check on curriculum position
    pos = graph.compute_curriculum_position(
        learner_state=state_after,
        current_concept="loss_functions"
    )
    assert ("loss_functions" in pos.mastered) or ("loss_functions" in pos.in_progress)
    assert len(pos.next_ready) > 0



def test_e2e_off_topic_redirect_and_recovery():
    pipeline = TutorPipeline(
        knowledge_source=MockKnowledgeSource(),
        model_adapter=None
    )
    session_id = "e2e_redirect_session"

    # Turn 1: Off-topic query
    req1 = AIChatRequest(
        session_id=session_id,
        message="What is the recipe for chocolate chip cookies?",
        course_id=1,
        lecture_id=1
    )
    res1 = pipeline.process(req1)
    assert res1.pedagogy_mode in (PedagogyMode.OFF_TOPIC, "off_topic")

    # Turn 2: Recovery back to course concepts
    req2 = AIChatRequest(
        session_id=session_id,
        message="Back to machine learning, what is a loss function?",
        course_id=1,
        lecture_id=1
    )
    res2 = pipeline.process(req2)
    assert res2.pedagogy_mode in (PedagogyMode.SOCRATIC, PedagogyMode.DIRECT, "socratic", "direct")
    assert len(res2.answer) > 10
