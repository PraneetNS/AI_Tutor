"""
tests/test_event_bus.py
-----------------------
Unit tests for LearningEvent schema and InMemoryEventBus.
PostgresEventBus is integration-tested separately (requires a live DB).
"""

import pytest
from ai_tutor.models import LearningEvent, LearningEventType
from ai_tutor.event_bus import InMemoryEventBus


# ---------------------------------------------------------------------------
# LearningEvent schema tests
# ---------------------------------------------------------------------------

class TestLearningEventSchema:
    def test_auto_generates_event_id(self):
        e = LearningEvent(event_type=LearningEventType.MESSAGE_SENT)
        assert e.event_id, "event_id should be auto-populated"
        assert len(e.event_id) == 36, "event_id should be a UUID4 string"

    def test_auto_generates_occurred_at(self):
        e = LearningEvent(event_type=LearningEventType.SESSION_STARTED)
        assert "T" in e.occurred_at, "occurred_at should be an ISO 8601 timestamp"

    def test_each_event_has_unique_id(self):
        e1 = LearningEvent(event_type=LearningEventType.MESSAGE_SENT)
        e2 = LearningEvent(event_type=LearningEventType.MESSAGE_SENT)
        assert e1.event_id != e2.event_id

    def test_optional_fields_default_to_none(self):
        e = LearningEvent(event_type=LearningEventType.KNOWLEDGE_RETRIEVED)
        assert e.student_id is None
        assert e.session_id is None
        assert e.course_id is None
        assert e.concept is None
        assert e.mastery_score is None
        assert e.hint_level is None

    def test_payload_defaults_to_empty_dict(self):
        e = LearningEvent(event_type=LearningEventType.QUIZ_SUBMITTED)
        assert e.payload == {}

    def test_full_construction(self):
        e = LearningEvent(
            event_type=LearningEventType.CONCEPT_MASTERED,
            student_id="u99",
            session_id="sess_abc",
            course_id=1,
            lecture_id=7,
            concept="Gradient Descent",
            mastery_score=0.92,
            hint_level=2,
            pedagogy_mode="socratic",
            payload={"previous_score": 0.4}
        )
        assert e.concept == "Gradient Descent"
        assert e.mastery_score == pytest.approx(0.92)
        assert e.payload["previous_score"] == pytest.approx(0.4)

    def test_mastery_score_bounds(self):
        with pytest.raises(Exception):
            LearningEvent(event_type=LearningEventType.CONCEPT_MASTERED, mastery_score=1.5)
        with pytest.raises(Exception):
            LearningEvent(event_type=LearningEventType.CONCEPT_MASTERED, mastery_score=-0.1)

    def test_hint_level_bounds(self):
        with pytest.raises(Exception):
            LearningEvent(event_type=LearningEventType.HINT_REQUESTED, hint_level=6)

    def test_all_event_types_are_valid(self):
        """Every LearningEventType value should construct without error."""
        for et in LearningEventType:
            e = LearningEvent(event_type=et)
            assert e.event_type == et.value


# ---------------------------------------------------------------------------
# InMemoryEventBus tests
# ---------------------------------------------------------------------------

class TestInMemoryEventBus:

    def _event(self, event_type=LearningEventType.MESSAGE_SENT, **kwargs) -> LearningEvent:
        return LearningEvent(event_type=event_type, **kwargs)

    def test_emit_calls_subscriber(self):
        bus = InMemoryEventBus()
        received = []
        bus.subscribe(received.append)
        bus.emit(self._event())
        assert len(received) == 1

    def test_multiple_subscribers_all_called(self):
        bus = InMemoryEventBus()
        log_a, log_b = [], []
        bus.subscribe(log_a.append)
        bus.subscribe(log_b.append)
        bus.emit(self._event())
        assert len(log_a) == 1
        assert len(log_b) == 1

    def test_handler_receives_correct_event(self):
        bus = InMemoryEventBus()
        received = []
        bus.subscribe(received.append)
        e = self._event(student_id="u42", concept="Loss Function")
        bus.emit(e)
        assert received[0].event_id == e.event_id
        assert received[0].student_id == "u42"
        assert received[0].concept == "Loss Function"

    def test_emit_without_subscribers_is_safe(self):
        bus = InMemoryEventBus()
        bus.emit(self._event())   # should not raise

    def test_faulty_handler_does_not_block_others(self):
        """A handler that raises should not prevent downstream handlers from running."""
        bus = InMemoryEventBus()
        log = []

        def bad_handler(e):
            raise RuntimeError("boom")

        bus.subscribe(bad_handler)
        bus.subscribe(log.append)
        bus.emit(self._event())
        assert len(log) == 1, "good handler must still be called after bad handler raises"

    def test_log_appends_all_emitted_events(self):
        bus = InMemoryEventBus()
        for _ in range(5):
            bus.emit(self._event())
        assert len(bus.log) == 5

    def test_clear_resets_log_and_handlers(self):
        bus = InMemoryEventBus()
        log = []
        bus.subscribe(log.append)
        bus.emit(self._event())      # pre-clear: 1 event, 1 handler call
        bus.clear()                  # wipes both log and handlers
        bus.emit(self._event())      # post-clear: appends to log, but NO handlers registered
        assert len(bus.log) == 1    # only the post-clear event is in log (pre-clear was wiped)
        assert len(log) == 1        # handler only received the pre-clear event

    def test_subscribe_to_filters_by_type(self):
        bus = InMemoryEventBus()
        concept_log = []
        bus.subscribe_to(LearningEventType.CONCEPT_MASTERED, concept_log.append)

        bus.emit(self._event(LearningEventType.MESSAGE_SENT))
        bus.emit(self._event(LearningEventType.CONCEPT_MASTERED, concept="Backprop"))
        bus.emit(self._event(LearningEventType.HINT_REQUESTED))

        assert len(concept_log) == 1
        assert concept_log[0].concept == "Backprop"

    def test_multiple_subscribe_to_different_types(self):
        bus = InMemoryEventBus()
        mastery_log, session_log = [], []
        bus.subscribe_to(LearningEventType.CONCEPT_MASTERED, mastery_log.append)
        bus.subscribe_to(LearningEventType.SESSION_STARTED, session_log.append)

        bus.emit(self._event(LearningEventType.CONCEPT_MASTERED))
        bus.emit(self._event(LearningEventType.SESSION_STARTED))
        bus.emit(self._event(LearningEventType.OFF_TOPIC_REDIRECT))

        assert len(mastery_log) == 1
        assert len(session_log) == 1

    def test_log_is_immutable_snapshot(self):
        """Mutating the returned log list must not affect internal state."""
        bus = InMemoryEventBus()
        bus.emit(self._event())
        snapshot = bus.log
        snapshot.clear()
        assert len(bus.log) == 1
