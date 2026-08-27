"""
tests/test_concept_graph.py
-----------------------------
Unit tests for the ConceptGraph service:
  - get_prerequisites() BFS backward traversal
  - get_unlocked_by() forward traversal
  - compute_curriculum_position() (mastered / in_progress / next_ready / locked)
  - diagnose_root_cause() backward DAG walk with confidence scoring
  - ContextResolver integration: curriculum_position and root_cause_diagnosis
    injected into TeachingStrategy as pre-computed fields

All tests run fully offline — no DB, no API keys required.
"""

from __future__ import annotations

import pytest
from ai_tutor.concept_graph import (
    ConceptGraph,
    InMemoryConceptGraph,
    create_ml_concept_graph,
)
from ai_tutor.models import (
    BehaviorProfile,
    ConceptMastery,
    ConceptNode,
    ConceptPrerequisiteEdge,
    CurriculumPosition,
    LearnerState,
    RootCauseDiagnosis,
)
from ai_tutor.context_resolver import ContextResolver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_graph() -> ConceptGraph:
    """
    Minimal 4-node chain:  A -> B -> C -> D
    (D requires C, C requires B, B requires A)
    """
    concepts = [
        ConceptNode(concept_id="A", domain="math", name="Concept A"),
        ConceptNode(concept_id="B", domain="math", name="Concept B"),
        ConceptNode(concept_id="C", domain="math", name="Concept C"),
        ConceptNode(concept_id="D", domain="math", name="Concept D"),
    ]
    edges = [
        ConceptPrerequisiteEdge(concept_id="B", prerequisite_id="A", weight=1.0),
        ConceptPrerequisiteEdge(concept_id="C", prerequisite_id="B", weight=1.0),
        ConceptPrerequisiteEdge(concept_id="D", prerequisite_id="C", weight=1.0),
    ]
    store = InMemoryConceptGraph(concepts=concepts, edges=edges)
    return ConceptGraph(store=store)


@pytest.fixture
def diamond_graph() -> ConceptGraph:
    """
    Diamond DAG:
          E
         / \
        C   D
         \ /
          B
    (E requires C and D; both require B)
    """
    concepts = [
        ConceptNode(concept_id="B", domain="math", name="Base"),
        ConceptNode(concept_id="C", domain="math", name="Left"),
        ConceptNode(concept_id="D", domain="math", name="Right"),
        ConceptNode(concept_id="E", domain="math", name="Top"),
    ]
    edges = [
        ConceptPrerequisiteEdge(concept_id="C", prerequisite_id="B", weight=1.0),
        ConceptPrerequisiteEdge(concept_id="D", prerequisite_id="B", weight=1.0),
        ConceptPrerequisiteEdge(concept_id="E", prerequisite_id="C", weight=1.0),
        ConceptPrerequisiteEdge(concept_id="E", prerequisite_id="D", weight=1.0),
    ]
    store = InMemoryConceptGraph(concepts=concepts, edges=edges)
    return ConceptGraph(store=store)


@pytest.fixture
def ml_graph() -> ConceptGraph:
    return create_ml_concept_graph()


def make_learner(student_id: str, mastery_map: dict) -> LearnerState:
    """Helper: build a LearnerState with given concept mastery scores."""
    concept_mastery = {
        cid: ConceptMastery(concept=cid, mastery=m)
        for cid, m in mastery_map.items()
    }
    return LearnerState(
        student_id=student_id,
        concept_mastery=concept_mastery,
        behavior=BehaviorProfile(),
    )


# ---------------------------------------------------------------------------
# 1. get_direct_prerequisites
# ---------------------------------------------------------------------------

class TestGetDirectPrerequisites:
    def test_root_concept_has_no_prerequisites(self, simple_graph):
        prereqs = simple_graph.get_direct_prerequisites("A")
        assert prereqs == []

    def test_single_direct_prerequisite(self, simple_graph):
        prereqs = simple_graph.get_direct_prerequisites("B")
        assert len(prereqs) == 1
        assert prereqs[0].concept_id == "A"

    def test_two_direct_prerequisites(self, diamond_graph):
        prereqs = diamond_graph.get_direct_prerequisites("E")
        ids = {p.concept_id for p in prereqs}
        assert ids == {"C", "D"}

    def test_unknown_concept_returns_empty(self, simple_graph):
        assert simple_graph.get_direct_prerequisites("nonexistent") == []


# ---------------------------------------------------------------------------
# 2. get_prerequisites (full ancestry BFS)
# ---------------------------------------------------------------------------

class TestGetPrerequisites:
    def test_full_chain_returns_all_ancestors(self, simple_graph):
        ancestors = simple_graph.get_prerequisites("D")
        ids = {n.concept_id for n in ancestors}
        assert ids == {"A", "B", "C"}

    def test_root_has_no_ancestors(self, simple_graph):
        assert simple_graph.get_prerequisites("A") == []

    def test_diamond_deduplicates_shared_prerequisite(self, diamond_graph):
        """B appears as ancestor of both C and D — must appear only once."""
        ancestors = diamond_graph.get_prerequisites("E")
        ids = [n.concept_id for n in ancestors]
        assert ids.count("B") == 1

    def test_diamond_full_ancestry(self, diamond_graph):
        ancestors = diamond_graph.get_prerequisites("E")
        ids = {n.concept_id for n in ancestors}
        assert ids == {"B", "C", "D"}


# ---------------------------------------------------------------------------
# 3. get_unlocked_by
# ---------------------------------------------------------------------------

class TestGetUnlockedBy:
    def test_mastering_A_unlocks_B(self, simple_graph):
        unlocked = simple_graph.get_unlocked_by("A")
        assert len(unlocked) == 1
        assert unlocked[0].concept_id == "B"

    def test_leaf_concept_unlocks_nothing(self, simple_graph):
        assert simple_graph.get_unlocked_by("D") == []

    def test_shared_base_unlocks_two_concepts(self, diamond_graph):
        unlocked = diamond_graph.get_unlocked_by("B")
        ids = {n.concept_id for n in unlocked}
        assert ids == {"C", "D"}


# ---------------------------------------------------------------------------
# 4. compute_curriculum_position
# ---------------------------------------------------------------------------

class TestComputeCurriculumPosition:
    def test_all_locked_when_no_mastery(self, simple_graph):
        learner = make_learner("s1", {})
        pos = simple_graph.compute_curriculum_position(learner, current_concept="D")
        # A is a root with no prereqs -> next_ready (not locked)
        assert "A" in pos.next_ready
        # B, C, D locked (A not mastered)
        assert "B" in pos.locked or "B" in pos.next_ready  # B depends only on A

    def test_mastered_concept_appears_in_mastered_list(self, simple_graph):
        learner = make_learner("s1", {"A": 0.9, "B": 0.8})
        pos = simple_graph.compute_curriculum_position(learner, current_concept="C")
        assert "A" in pos.mastered
        assert "B" in pos.mastered

    def test_concept_in_progress_appears_in_in_progress(self, simple_graph):
        # A is mastered, B is in_progress (prereq met, mastery >= 0.3)
        learner = make_learner("s1", {"A": 0.9, "B": 0.5})
        pos = simple_graph.compute_curriculum_position(learner, current_concept="B")
        assert "B" in pos.in_progress

    def test_next_ready_concept_after_prerequisite_mastered(self, simple_graph):
        # A mastered, so B is next_ready (mastery = 0)
        learner = make_learner("s1", {"A": 0.85})
        pos = simple_graph.compute_curriculum_position(learner)
        assert "B" in pos.next_ready

    def test_locked_concept_has_unmastered_prerequisite(self, simple_graph):
        learner = make_learner("s1", {})  # nothing mastered
        pos = simple_graph.compute_curriculum_position(learner)
        assert "C" in pos.locked
        assert "D" in pos.locked

    def test_current_concept_is_preserved(self, simple_graph):
        learner = make_learner("s1", {"A": 0.9})
        pos = simple_graph.compute_curriculum_position(learner, current_concept="B")
        assert pos.current_concept == "B"

    def test_full_mastery_all_in_mastered(self, simple_graph):
        learner = make_learner("s1", {"A": 0.9, "B": 0.85, "C": 0.8, "D": 0.92})
        pos = simple_graph.compute_curriculum_position(learner)
        for cid in ["A", "B", "C", "D"]:
            assert cid in pos.mastered


# ---------------------------------------------------------------------------
# 5. diagnose_root_cause
# ---------------------------------------------------------------------------

class TestDiagnoseRootCause:
    def test_no_prerequisites_returns_no_gap(self, simple_graph):
        learner = make_learner("s1", {})
        diagnosis = simple_graph.diagnose_root_cause("A", learner)
        assert diagnosis.likely_root_gap is None
        assert diagnosis.chain_analyzed == []

    def test_chain_of_length_one_returns_immediate_prereq(self, simple_graph):
        # Student struggles with B, only prereq is A (mastery 0.1)
        learner = make_learner("s1", {"A": 0.1, "B": 0.35})
        diagnosis = simple_graph.diagnose_root_cause("B", learner)
        assert diagnosis.likely_root_gap == "A"
        assert diagnosis.struggling_concept == "B"
        assert "A" in diagnosis.chain_analyzed

    def test_identifies_lowest_mastery_in_chain(self, simple_graph):
        # B has decent mastery, but A is very weak — root gap should be A
        learner = make_learner("s1", {"A": 0.05, "B": 0.6, "C": 0.7})
        diagnosis = simple_graph.diagnose_root_cause("D", learner)
        assert diagnosis.likely_root_gap == "A"

    def test_confidence_is_between_zero_and_one(self, simple_graph):
        learner = make_learner("s1", {"A": 0.1, "B": 0.7, "C": 0.8})
        diagnosis = simple_graph.diagnose_root_cause("D", learner)
        assert 0.0 <= diagnosis.confidence <= 1.0

    def test_high_confidence_when_gap_far_below_siblings(self, diamond_graph):
        # E requires C and D. C has low mastery, D has high mastery.
        # C is the root gap; confidence should be HIGH because D is much higher.
        learner = make_learner("s1", {"B": 0.8, "C": 0.1, "D": 0.85})
        diagnosis = diamond_graph.diagnose_root_cause("E", learner)
        assert diagnosis.likely_root_gap == "C"
        assert diagnosis.confidence > 0.5

    def test_chain_analyzed_contains_visited_ancestors(self, simple_graph):
        learner = make_learner("s1", {"A": 0.2, "B": 0.5, "C": 0.6})
        diagnosis = simple_graph.diagnose_root_cause("D", learner)
        assert "A" in diagnosis.chain_analyzed
        assert "B" in diagnosis.chain_analyzed
        assert "C" in diagnosis.chain_analyzed

    def test_max_depth_limits_traversal(self, simple_graph):
        learner = make_learner("s1", {"A": 0.1, "B": 0.5, "C": 0.6})
        # max_depth=1 should only see immediate parents
        diagnosis = simple_graph.diagnose_root_cause("D", learner, max_depth=1)
        assert "C" in diagnosis.chain_analyzed
        assert "A" not in diagnosis.chain_analyzed


# ---------------------------------------------------------------------------
# 6. ML Graph Integration
# ---------------------------------------------------------------------------

class TestMLGraph:
    def test_backpropagation_has_expected_ancestors(self, ml_graph):
        ancestors = ml_graph.get_prerequisites("backpropagation")
        ids = {n.concept_id for n in ancestors}
        # Backprop depends on gradient_descent -> partial_derivatives -> calculus_basics
        assert "gradient_descent" in ids
        assert "partial_derivatives" in ids
        assert "calculus_basics" in ids
        assert "chain_rule" in ids

    def test_transformers_are_locked_without_foundations(self, ml_graph):
        learner = make_learner("s1", {})  # no mastery at all
        pos = ml_graph.compute_curriculum_position(learner)
        assert "transformers" in pos.locked

    def test_backprop_root_cause_when_chain_rule_missing(self, ml_graph):
        # Student knows gradient descent but not chain_rule or partial_derivatives
        learner = make_learner("s1", {
            "supervised_learning": 0.8,
            "loss_functions": 0.85,
            "gradient_descent": 0.78,
            "linear_algebra": 0.65,      # set explicitly so chain_rule is the lowest
            "probability": 0.6,
            "calculus_basics": 0.2,
            "chain_rule": 0.1,           # <-- this is the root gap
            "partial_derivatives": 0.3,
        })
        diagnosis = ml_graph.diagnose_root_cause("backpropagation", learner)
        assert diagnosis.likely_root_gap == "chain_rule"
        assert diagnosis.confidence > 0.3

    def test_mastering_backprop_unlocks_neural_networks(self, ml_graph):
        unlocked = ml_graph.get_unlocked_by("backpropagation")
        ids = {n.concept_id for n in unlocked}
        assert "neural_networks" in ids


# ---------------------------------------------------------------------------
# 7. ContextResolver Integration
# ---------------------------------------------------------------------------

class TestContextResolverIntegration:
    def test_curriculum_position_is_populated_in_strategy(self, ml_graph):
        learner = make_learner("s1", {
            "supervised_learning": 0.85,
            "loss_functions": 0.80,
        })
        resolver = ContextResolver(concept_graph=ml_graph)
        strategy = resolver.resolve(
            learner_state=learner,
            target_concept="gradient_descent",
            consecutive_failures=0,
        )
        assert strategy.curriculum_position is not None
        assert isinstance(strategy.curriculum_position.mastered, list)
        assert isinstance(strategy.curriculum_position.next_ready, list)
        assert isinstance(strategy.curriculum_position.locked, list)

    def test_root_cause_diagnosis_computed_on_consecutive_failures(self, ml_graph):
        learner = make_learner("s1", {
            "calculus_basics": 0.15,
            "chain_rule": 0.1,
            "partial_derivatives": 0.2,
            "gradient_descent": 0.45,
        })
        resolver = ContextResolver(concept_graph=ml_graph)
        strategy = resolver.resolve(
            learner_state=learner,
            target_concept="backpropagation",
            consecutive_failures=2,
        )
        assert strategy.root_cause_diagnosis is not None
        assert strategy.root_cause_diagnosis.struggling_concept == "backpropagation"
        assert strategy.root_cause_diagnosis.likely_root_gap is not None

    def test_root_cause_diagnosis_is_none_when_no_failures(self, ml_graph):
        learner = make_learner("s1", {"gradient_descent": 0.5})
        resolver = ContextResolver(concept_graph=ml_graph)
        strategy = resolver.resolve(
            learner_state=learner,
            target_concept="gradient_descent",
            consecutive_failures=0,
        )
        assert strategy.root_cause_diagnosis is None

    def test_resolver_works_without_concept_graph(self):
        """Existing behavior is preserved when no ConceptGraph is injected."""
        learner = make_learner("s1", {"gradient_descent": 0.6})
        resolver = ContextResolver()   # no concept_graph
        strategy = resolver.resolve(
            learner_state=learner,
            target_concept="gradient_descent",
            consecutive_failures=1,
        )
        assert strategy.curriculum_position is None
        assert strategy.root_cause_diagnosis is None

    def test_to_prompt_sections_includes_curriculum_position(self, ml_graph):
        """OrchestratedContext.to_prompt_sections() embeds the pre-computed data."""
        import json
        from ai_tutor.models import (
            KnowledgeContext, LearningContext, OrchestratedContext,
            PedagogyState, SessionContext,
        )

        learner = make_learner("s1", {
            "supervised_learning": 0.85,
            "loss_functions": 0.8,
            "calculus_basics": 0.15,
        })
        resolver = ContextResolver(concept_graph=ml_graph)
        strategy = resolver.resolve(
            learner_state=learner,
            target_concept="gradient_descent",
            consecutive_failures=2,
        )

        learning_ctx = LearningContext(
            student_id="s1",
            learner_state=learner,
            target_concept="gradient_descent",
            teaching_strategy=strategy,
        )
        ctx = OrchestratedContext(
            student_message="I don't get gradient descent",
            session_context=SessionContext(session_id="sess_1", pedagogy_state=PedagogyState()),
            learning_context=learning_ctx,
            knowledge_context=KnowledgeContext(),
        )

        sections = ctx.to_prompt_sections()
        learner_state_data = json.loads(sections["learner_state"])
        assert "curriculum_position" in learner_state_data
        assert "mastered" in learner_state_data["curriculum_position"]
        assert "next_ready" in learner_state_data["curriculum_position"]

        # Strategy directive should contain root cause text
        if strategy.root_cause_diagnosis and strategy.root_cause_diagnosis.likely_root_gap:
            assert "ROOT CAUSE GAP" in sections["teaching_strategy"]
