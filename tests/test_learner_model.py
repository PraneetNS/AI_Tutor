"""
tests/test_learner_model.py
---------------------------
Unit tests for LearnerModelEngine, KnowledgeTracer, MisconceptionEngine,
BehavioralModel, and LearnerState persistence.
"""

import pytest
from ai_tutor.models import (
    LearnerState,
    ConceptMastery,
    LearningEvent,
    LearningEventType,
)
from ai_tutor.event_bus import InMemoryEventBus
from ai_tutor.learner_store import InMemoryLearnerStateStore
from ai_tutor.learner_model import (
    KnowledgeTracer,
    MisconceptionEngine,
    BehavioralModel,
    LearnerModelEngine,
)


# ---------------------------------------------------------------------------
# 1. KnowledgeTracer (BKT) Tests
# ---------------------------------------------------------------------------

class TestKnowledgeTracer:
    def test_correct_answer_increases_mastery(self):
        kt = KnowledgeTracer(p_l0=0.10, p_t=0.30, p_s=0.10, p_g=0.20)
        initial = 0.10
        updated = kt.update(initial, is_correct=True)
        assert updated > initial
        # With p_l=0.1, slip=0.1, guess=0.2:
        # P(L|correct) = 0.09 / (0.09 + 0.18) = 0.3333
        # P(L_next) = 0.3333 + (0.6667 * 0.3) = 0.5333
        assert updated == pytest.approx(0.5333, abs=0.01)

    def test_incorrect_answer_lowers_or_damps_mastery(self):
        kt = KnowledgeTracer(p_l0=0.10, p_t=0.30, p_s=0.10, p_g=0.20)
        initial = 0.50
        updated = kt.update(initial, is_correct=False)
        # Incorrect observation should result in lower probability than correct
        correct_updated = kt.update(initial, is_correct=True)
        assert updated < correct_updated

    def test_mastery_converges_to_near_one_on_consecutive_correct(self):
        kt = KnowledgeTracer()
        m = 0.10
        for _ in range(8):
            m = kt.update(m, is_correct=True)
        assert m > 0.95

    def test_process_event_updates_learner_state(self):
        kt = KnowledgeTracer()
        state = LearnerState(student_id="student_1")
        event = LearningEvent(
            event_type=LearningEventType.ANSWER_SUBMITTED,
            student_id="student_1",
            concept="Gradient Descent",
            payload={"correct": True, "concept": "Gradient Descent"}
        )
        mutated = kt.process_event(state, event)
        assert mutated is True
        assert "Gradient Descent" in state.concept_mastery
        assert state.concept_mastery["Gradient Descent"].attempts == 1
        assert state.concept_mastery["Gradient Descent"].correct == 1
        assert state.concept_mastery["Gradient Descent"].mastery > 0.10

    def test_answer_revealed_treated_as_failed_attempt(self):
        kt = KnowledgeTracer()
        state = LearnerState(student_id="student_1")
        event = LearningEvent(
            event_type=LearningEventType.ANSWER_REVEALED,
            student_id="student_1",
            concept="Backpropagation",
            payload={"concept": "Backpropagation"}
        )
        mutated = kt.process_event(state, event)
        assert mutated is True
        assert state.concept_mastery["Backpropagation"].attempts == 1
        assert state.concept_mastery["Backpropagation"].correct == 0


# ---------------------------------------------------------------------------
# 2. MisconceptionEngine Tests
# ---------------------------------------------------------------------------

class TestMisconceptionEngine:
    def test_detects_known_misconception(self):
        me = MisconceptionEngine()
        state = LearnerState(student_id="student_1")
        event = LearningEvent(
            event_type=LearningEventType.ANSWER_SUBMITTED,
            student_id="student_1",
            concept="Gradient Descent",
            payload={
                "correct": False,
                "response": "Gradient descent is always stuck in local minima and cannot escape."
            }
        )
        mutated = me.process_event(state, event)
        assert mutated is True
        assert len(state.misconceptions) == 1
        m = state.misconceptions[0]
        assert m.key == "gd_local_minimum_paralysis"
        assert m.concept == "Gradient Descent"
        assert m.confidence == 0.5
        assert m.hit_count == 1

    def test_increments_confidence_on_repeated_hit(self):
        me = MisconceptionEngine(confidence_step=0.2)
        state = LearnerState(student_id="student_1")
        event1 = LearningEvent(
            event_type=LearningEventType.ANSWER_SUBMITTED,
            student_id="student_1",
            concept="Gradient Descent",
            payload={"correct": False, "response": "It gets always stuck in local minima"}
        )
        event2 = LearningEvent(
            event_type=LearningEventType.ANSWER_SUBMITTED,
            student_id="student_1",
            concept="Gradient Descent",
            payload={"correct": False, "response": "A saddle point trap is impossible to leave"}
        )
        me.process_event(state, event1)
        me.process_event(state, event2)

        assert len(state.misconceptions) == 1
        m = state.misconceptions[0]
        assert m.hit_count == 2
        assert m.confidence == pytest.approx(0.7, abs=0.01)

    def test_ignores_correct_answers(self):
        me = MisconceptionEngine()
        state = LearnerState(student_id="student_1")
        event = LearningEvent(
            event_type=LearningEventType.ANSWER_SUBMITTED,
            student_id="student_1",
            concept="Gradient Descent",
            payload={"correct": True, "response": "Local minima can sometimes be escaped."}
        )
        mutated = me.process_event(state, event)
        assert mutated is False
        assert len(state.misconceptions) == 0

    def test_ignores_unrelated_text(self):
        me = MisconceptionEngine()
        state = LearnerState(student_id="student_1")
        event = LearningEvent(
            event_type=LearningEventType.ANSWER_SUBMITTED,
            student_id="student_1",
            concept="Gradient Descent",
            payload={"correct": False, "response": "I think the answer is 42."}
        )
        mutated = me.process_event(state, event)
        assert mutated is False
        assert len(state.misconceptions) == 0


# ---------------------------------------------------------------------------
# 3. BehavioralModel Tests
# ---------------------------------------------------------------------------

class TestBehavioralModel:
    def test_tracks_hints_and_sessions(self):
        bm = BehavioralModel()
        state = LearnerState(student_id="student_1")

        # Session 1
        bm.process_event(state, LearningEvent(event_type=LearningEventType.SESSION_STARTED, student_id="student_1"))
        bm.process_event(state, LearningEvent(event_type=LearningEventType.HINT_REQUESTED, student_id="student_1"))
        bm.process_event(state, LearningEvent(event_type=LearningEventType.HINT_REQUESTED, student_id="student_1"))

        assert state.behavior.sessions_total == 1
        assert state.behavior.total_hints_used == 2
        assert state.behavior.hints_per_session == 2.0

        # Session 2
        bm.process_event(state, LearningEvent(event_type=LearningEventType.SESSION_STARTED, student_id="student_1"))
        bm.process_event(state, LearningEvent(event_type=LearningEventType.HINT_REQUESTED, student_id="student_1"))

        assert state.behavior.sessions_total == 2
        assert state.behavior.total_hints_used == 3
        assert state.behavior.hints_per_session == 1.5

    def test_tracks_persistence_turns(self):
        bm = BehavioralModel()
        state = LearnerState(student_id="student_1")

        bm.process_event(state, LearningEvent(event_type=LearningEventType.SESSION_STARTED, student_id="student_1"))
        bm.process_event(state, LearningEvent(event_type=LearningEventType.MESSAGE_SENT, student_id="student_1"))
        bm.process_event(state, LearningEvent(event_type=LearningEventType.MESSAGE_SENT, student_id="student_1"))
        bm.process_event(state, LearningEvent(event_type=LearningEventType.MESSAGE_SENT, student_id="student_1"))

        assert state.behavior.total_turns == 3
        assert state.behavior.avg_persistence == 3.0


# ---------------------------------------------------------------------------
# 4. InMemoryLearnerStateStore Tests
# ---------------------------------------------------------------------------

class TestInMemoryLearnerStateStore:
    def test_save_and_load_round_trip(self):
        store = InMemoryLearnerStateStore()
        state = LearnerState(student_id="student_99")
        state.concept_mastery["Loss Functions"] = ConceptMastery(concept="Loss Functions", mastery=0.8)
        store.save(state)

        loaded = store.load("student_99")
        assert loaded is not None
        assert loaded.student_id == "student_99"
        assert loaded.concept_mastery["Loss Functions"].mastery == 0.8

    def test_returns_deep_copy_isolation(self):
        store = InMemoryLearnerStateStore()
        state = LearnerState(student_id="student_99")
        store.save(state)

        loaded1 = store.load("student_99")
        loaded1.behavior.sessions_total = 100

        loaded2 = store.load("student_99")
        assert loaded2.behavior.sessions_total == 0


# ---------------------------------------------------------------------------
# 5. Full LearnerModelEngine Integration Tests
# ---------------------------------------------------------------------------

class TestLearnerModelEngine:
    def test_engine_subscribes_to_bus_and_persists_state(self):
        bus = InMemoryEventBus()
        store = InMemoryLearnerStateStore()
        engine = LearnerModelEngine(store=store, bus=bus)

        # Emit session started
        bus.emit(LearningEvent(
            event_type=LearningEventType.SESSION_STARTED,
            student_id="student_alpha"
        ))

        # Emit quiz answer with misconception
        bus.emit(LearningEvent(
            event_type=LearningEventType.ANSWER_SUBMITTED,
            student_id="student_alpha",
            concept="Gradient Descent",
            payload={
                "correct": False,
                "response": "The model is always stuck in local minima forever."
            }
        ))

        # Retrieve updated state
        state = engine.get_learner_state("student_alpha")
        assert state is not None
        assert state.student_id == "student_alpha"
        assert state.behavior.sessions_total == 1
        assert "Gradient Descent" in state.concept_mastery
        assert state.concept_mastery["Gradient Descent"].attempts == 1
        assert len(state.misconceptions) == 1
        assert state.misconceptions[0].key == "gd_local_minimum_paralysis"

    def test_engine_isolates_multiple_students(self):
        bus = InMemoryEventBus()
        store = InMemoryLearnerStateStore()
        engine = LearnerModelEngine(store=store, bus=bus)

        bus.emit(LearningEvent(
            event_type=LearningEventType.ANSWER_SUBMITTED,
            student_id="student_A",
            concept="Overfitting",
            payload={"correct": True}
        ))
        bus.emit(LearningEvent(
            event_type=LearningEventType.ANSWER_SUBMITTED,
            student_id="student_B",
            concept="Backpropagation",
            payload={"correct": False}
        ))

        state_a = engine.get_learner_state("student_A")
        state_b = engine.get_learner_state("student_B")

        assert "Overfitting" in state_a.concept_mastery
        assert "Backpropagation" not in state_a.concept_mastery

        assert "Backpropagation" in state_b.concept_mastery
        assert "Overfitting" not in state_b.concept_mastery
