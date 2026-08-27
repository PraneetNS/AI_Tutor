"""
concept_graph.py
-----------------
ConceptGraph: A service over the 'concepts' and 'concept_prerequisites' tables.

Provides:
  1. get_prerequisites(concept_id)    -> List[ConceptNode] direct prerequisites
  2. get_unlocked_by(concept_id)      -> List[ConceptNode] concepts unlocked if this is mastered
  3. diagnose_root_cause(user_id, struggling_concept, learner_state)
       -> RootCauseDiagnosis: backward DAG walk returning the lowest-mastery
          prerequisite in the chain as the likely root gap, with confidence
          scored by how much lower it is than sibling prerequisites.

Storage backends:
  - InMemoryConceptGraph: built from plain Python dicts — zero dependencies,
    perfect for tests and seeded course graphs.
  - (Postgres backend can be wired in later via BaseConceptGraphStore.)

Design principle: all curriculum position and root-cause reasoning is computed
deterministically HERE — the LLM never sees raw history and never infers
curriculum state from scratch.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from .models import (
    ConceptNode,
    ConceptPrerequisiteEdge,
    CurriculumPosition,
    LearnerState,
    RootCauseDiagnosis,
)

logger = logging.getLogger("ai_tutor.concept_graph")

# Mastery thresholds
_MASTERY_THRESHOLD = 0.75   # >= this = mastered
_INPROGRESS_THRESHOLD = 0.3  # >= this but < mastered = in_progress


# ---------------------------------------------------------------------------
# Abstract Store Interface
# ---------------------------------------------------------------------------

class BaseConceptGraphStore(ABC):
    """
    Abstract persistence interface for the concept graph.
    Concrete backends: InMemoryConceptGraph, PostgresConceptGraph.
    """

    @abstractmethod
    def get_concept(self, concept_id: str) -> Optional[ConceptNode]:
        """Return the concept node or None if not found."""

    @abstractmethod
    def all_concepts(self) -> List[ConceptNode]:
        """Return every concept registered in the graph."""

    @abstractmethod
    def get_direct_prerequisites(self, concept_id: str) -> List[Tuple[ConceptNode, float]]:
        """
        Return list of (ConceptNode, weight) for direct prerequisites of concept_id.
        """

    @abstractmethod
    def get_direct_dependents(self, concept_id: str) -> List[Tuple[ConceptNode, float]]:
        """
        Return list of (ConceptNode, weight) for concepts that directly depend on concept_id.
        (i.e., concepts that list concept_id as a prerequisite.)
        """


# ---------------------------------------------------------------------------
# In-Memory Store (seeded from plain dicts — no DB needed)
# ---------------------------------------------------------------------------

class InMemoryConceptGraph(BaseConceptGraphStore):
    """
    Fully in-memory concept graph, built from ConceptNode and
    ConceptPrerequisiteEdge objects. Thread-safe for reads.
    """

    def __init__(
        self,
        concepts: Optional[List[ConceptNode]] = None,
        edges: Optional[List[ConceptPrerequisiteEdge]] = None,
    ) -> None:
        self._concepts: Dict[str, ConceptNode] = {}
        # prerequisite edges: concept_id -> list of (prereq_id, weight)
        self._prereq_edges: Dict[str, List[Tuple[str, float]]] = {}
        # dependent edges (reverse): prereq_id -> list of (dependent_id, weight)
        self._dependent_edges: Dict[str, List[Tuple[str, float]]] = {}

        for c in (concepts or []):
            self.add_concept(c)
        for e in (edges or []):
            self.add_edge(e)

    def add_concept(self, concept: ConceptNode) -> None:
        self._concepts[concept.concept_id] = concept
        self._prereq_edges.setdefault(concept.concept_id, [])
        self._dependent_edges.setdefault(concept.concept_id, [])

    def add_edge(self, edge: ConceptPrerequisiteEdge) -> None:
        self._prereq_edges.setdefault(edge.concept_id, []).append(
            (edge.prerequisite_id, edge.weight)
        )
        self._dependent_edges.setdefault(edge.prerequisite_id, []).append(
            (edge.concept_id, edge.weight)
        )

    def get_concept(self, concept_id: str) -> Optional[ConceptNode]:
        return self._concepts.get(concept_id)

    def all_concepts(self) -> List[ConceptNode]:
        return list(self._concepts.values())

    def get_direct_prerequisites(self, concept_id: str) -> List[Tuple[ConceptNode, float]]:
        result = []
        for prereq_id, weight in self._prereq_edges.get(concept_id, []):
            node = self._concepts.get(prereq_id)
            if node:
                result.append((node, weight))
        return result

    def get_direct_dependents(self, concept_id: str) -> List[Tuple[ConceptNode, float]]:
        result = []
        for dep_id, weight in self._dependent_edges.get(concept_id, []):
            node = self._concepts.get(dep_id)
            if node:
                result.append((node, weight))
        return result


# ---------------------------------------------------------------------------
# ConceptGraph Service
# ---------------------------------------------------------------------------

class ConceptGraph:
    """
    High-level service that wraps a BaseConceptGraphStore and provides:

    - get_prerequisites(concept_id)        : all ancestors (BFS, full chain)
    - get_direct_prerequisites(concept_id) : immediate parents only
    - get_unlocked_by(concept_id)          : concepts unlocked when this is mastered
    - compute_curriculum_position(learner_state, current_concept)
        -> CurriculumPosition
    - diagnose_root_cause(struggling_concept, learner_state)
        -> RootCauseDiagnosis
    """

    def __init__(self, store: Optional[BaseConceptGraphStore] = None) -> None:
        self._store = store or InMemoryConceptGraph()

    @property
    def store(self) -> BaseConceptGraphStore:
        return self._store

    # ------------------------------------------------------------------
    # 1. Prerequisite Queries
    # ------------------------------------------------------------------

    def get_direct_prerequisites(self, concept_id: str) -> List[ConceptNode]:
        """Return direct (level-1) prerequisite concepts."""
        return [node for node, _w in self._store.get_direct_prerequisites(concept_id)]

    def get_prerequisites(self, concept_id: str) -> List[ConceptNode]:
        """
        Return ALL ancestor prerequisites via BFS backward traversal.
        The concept itself is excluded from the result.
        """
        visited: Set[str] = set()
        queue: deque[str] = deque([concept_id])
        result: List[ConceptNode] = []

        while queue:
            current = queue.popleft()
            for node, _w in self._store.get_direct_prerequisites(current):
                if node.concept_id not in visited:
                    visited.add(node.concept_id)
                    result.append(node)
                    queue.append(node.concept_id)

        return result

    # ------------------------------------------------------------------
    # 2. Unlock Queries (forward direction)
    # ------------------------------------------------------------------

    def get_unlocked_by(self, concept_id: str) -> List[ConceptNode]:
        """
        Return concepts that become directly unlocked (i.e., have concept_id
        as a direct prerequisite) when this concept is mastered.
        """
        return [node for node, _w in self._store.get_direct_dependents(concept_id)]

    # ------------------------------------------------------------------
    # 3. Curriculum Position (deterministic, pre-LLM computation)
    # ------------------------------------------------------------------

    def compute_curriculum_position(
        self,
        learner_state: LearnerState,
        current_concept: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> CurriculumPosition:
        """
        Walk every concept in the graph and categorise it into:
          - mastered    : P(L) >= _MASTERY_THRESHOLD
          - in_progress : _INPROGRESS_THRESHOLD <= P(L) < _MASTERY_THRESHOLD
          - next_ready  : all direct prerequisites are mastered
          - locked      : one or more prerequisites are NOT yet mastered

        All logic is pure Python arithmetic — the LLM never infers this.
        """
        mastered_set: Set[str] = set()
        in_progress: List[str] = []
        locked: List[str] = []
        next_ready: List[str] = []

        mastery_map = {
            cid: cm.mastery
            for cid, cm in learner_state.concept_mastery.items()
        }

        all_concepts = self._store.all_concepts()
        if domain:
            all_concepts = [c for c in all_concepts if c.domain == domain]

        # First pass: identify mastered concepts
        for concept in all_concepts:
            m = mastery_map.get(concept.concept_id, 0.0)
            if m >= _MASTERY_THRESHOLD:
                mastered_set.add(concept.concept_id)

        # Second pass: categorize non-mastered concepts
        for concept in all_concepts:
            cid = concept.concept_id
            if cid in mastered_set:
                continue

            m = mastery_map.get(cid, 0.0)
            prereqs = self._store.get_direct_prerequisites(cid)
            all_prereqs_mastered = all(
                p.concept_id in mastered_set for p, _w in prereqs
            )

            if all_prereqs_mastered and prereqs:
                # All prerequisites met -> unlocked, but not yet mastered
                if m >= _INPROGRESS_THRESHOLD:
                    in_progress.append(cid)
                else:
                    next_ready.append(cid)
            elif not prereqs:
                # Root concept (no prerequisites) -> always accessible
                if m >= _INPROGRESS_THRESHOLD:
                    in_progress.append(cid)
                else:
                    next_ready.append(cid)
            else:
                locked.append(cid)

        logger.debug(
            "curriculum_position: mastered=%d in_progress=%d next_ready=%d locked=%d",
            len(mastered_set), len(in_progress), len(next_ready), len(locked),
        )

        return CurriculumPosition(
            current_concept=current_concept,
            mastered=sorted(mastered_set),
            in_progress=in_progress,
            locked=locked,
            next_ready=next_ready,
        )

    # ------------------------------------------------------------------
    # 4. Root-Cause Diagnosis (backward DAG walk, deterministic)
    # ------------------------------------------------------------------

    def diagnose_root_cause(
        self,
        struggling_concept: str,
        learner_state: LearnerState,
        max_depth: int = 6,
    ) -> RootCauseDiagnosis:
        """
        Walk backward through the prerequisite DAG from struggling_concept.
        Returns the lowest-mastery prerequisite in the ancestry chain as the
        likely root gap, with a confidence score based on how much lower its
        mastery is compared to siblings in the same level of the chain.

        Algorithm:
          1. BFS backward up to max_depth levels from struggling_concept.
          2. For each visited prerequisite, look up the student's mastery.
          3. Track all prerequisite masteries alongside their siblings (same parent).
          4. The node with the globally lowest mastery in the chain is the root gap.
          5. Confidence = (mean_sibling_mastery - gap_mastery) / (mean_sibling_mastery + ε)
             clamped to [0, 1]. High confidence when the gap is far below its siblings.
        """
        mastery_map = {
            cid: cm.mastery
            for cid, cm in learner_state.concept_mastery.items()
        }

        visited: Set[str] = set()
        chain: List[str] = []   # BFS traversal order (excluding struggling_concept)

        # BFS backward
        queue: deque[Tuple[str, int]] = deque([(struggling_concept, 0)])
        # sibling groups: parent_id -> list of (child_prereq_id, weight)
        sibling_groups: Dict[str, List[Tuple[str, float]]] = {}

        while queue:
            current_id, depth = queue.popleft()
            if depth >= max_depth:
                continue

            direct_prereqs = self._store.get_direct_prerequisites(current_id)
            if direct_prereqs:
                sibling_groups[current_id] = [(p.concept_id, w) for p, w in direct_prereqs]

            for prereq_node, weight in direct_prereqs:
                pid = prereq_node.concept_id
                if pid not in visited:
                    visited.add(pid)
                    chain.append(pid)
                    queue.append((pid, depth + 1))

        if not chain:
            # No prerequisites to analyze — the concept itself IS the root
            return RootCauseDiagnosis(
                struggling_concept=struggling_concept,
                likely_root_gap=None,
                confidence=0.0,
                chain_analyzed=[],
            )

        # Find lowest-mastery node in chain
        candidate_id: Optional[str] = None
        candidate_mastery = float("inf")
        for prereq_id in chain:
            m = mastery_map.get(prereq_id, 0.0)
            if m < candidate_mastery:
                candidate_mastery = m
                candidate_id = prereq_id

        if candidate_id is None:
            return RootCauseDiagnosis(
                struggling_concept=struggling_concept,
                likely_root_gap=None,
                confidence=0.0,
                chain_analyzed=chain,
            )

        # Compute confidence: compare candidate to its sibling prerequisites
        # Find which parent lists candidate_id as a prerequisite
        sibling_masteries: List[float] = []
        for parent_id, siblings in sibling_groups.items():
            sibling_ids = [sid for sid, _w in siblings]
            if candidate_id in sibling_ids:
                for sid in sibling_ids:
                    if sid != candidate_id:
                        sibling_masteries.append(mastery_map.get(sid, 0.0))

        if sibling_masteries:
            mean_sibling = sum(sibling_masteries) / len(sibling_masteries)
            eps = 1e-6
            # How much lower is the candidate compared to siblings?
            gap = mean_sibling - candidate_mastery
            confidence = min(1.0, max(0.0, gap / (mean_sibling + eps)))
        else:
            # No siblings to compare against — medium confidence based on absolute mastery
            # A very low mastery score on a foundational concept is still meaningful
            confidence = min(1.0, max(0.0, 1.0 - candidate_mastery))

        logger.info(
            "diagnose_root_cause: struggling=%r root_gap=%r mastery=%.2f confidence=%.2f chain_len=%d",
            struggling_concept, candidate_id, candidate_mastery, confidence, len(chain),
        )

        return RootCauseDiagnosis(
            struggling_concept=struggling_concept,
            likely_root_gap=candidate_id,
            confidence=round(confidence, 3),
            chain_analyzed=chain,
        )


# ---------------------------------------------------------------------------
# Seeded Machine Learning Concept Graph (sparse starter graph)
# ---------------------------------------------------------------------------

ML_CONCEPTS: List[ConceptNode] = [
    ConceptNode(concept_id="variables",            domain="programming",       name="Variables & Data Types"),
    ConceptNode(concept_id="expressions",          domain="programming",       name="Expressions & Operators"),
    ConceptNode(concept_id="functions",            domain="programming",       name="Functions & Scope"),
    ConceptNode(concept_id="linear_algebra",       domain="mathematics",       name="Linear Algebra Basics"),
    ConceptNode(concept_id="calculus_basics",      domain="mathematics",       name="Calculus Basics (Limits, Derivatives)"),
    ConceptNode(concept_id="chain_rule",           domain="mathematics",       name="Chain Rule (Calculus)"),
    ConceptNode(concept_id="partial_derivatives",  domain="mathematics",       name="Partial Derivatives"),
    ConceptNode(concept_id="probability",          domain="mathematics",       name="Probability & Statistics"),
    ConceptNode(concept_id="supervised_learning",  domain="machine_learning",  name="Supervised Learning"),
    ConceptNode(concept_id="loss_functions",       domain="machine_learning",  name="Loss Functions (MSE, Cross-Entropy)"),
    ConceptNode(concept_id="gradient_descent",     domain="machine_learning",  name="Gradient Descent"),
    ConceptNode(concept_id="backpropagation",      domain="machine_learning",  name="Backpropagation"),
    ConceptNode(concept_id="regularization",       domain="machine_learning",  name="Regularization (L1/L2, Dropout)"),
    ConceptNode(concept_id="neural_networks",      domain="machine_learning",  name="Neural Networks"),
    ConceptNode(concept_id="attention_mechanisms", domain="machine_learning",  name="Attention Mechanisms"),
    ConceptNode(concept_id="transformers",         domain="machine_learning",  name="Transformer Architecture"),
    ConceptNode(concept_id="gradient_descent_variants", domain="machine_learning", name="Gradient Descent Variants (Adam, RMSProp)"),
]

ML_EDGES: List[ConceptPrerequisiteEdge] = [
    # Math foundations
    ConceptPrerequisiteEdge(concept_id="chain_rule",          prerequisite_id="calculus_basics",     weight=1.0),
    ConceptPrerequisiteEdge(concept_id="partial_derivatives", prerequisite_id="calculus_basics",     weight=1.0),
    ConceptPrerequisiteEdge(concept_id="partial_derivatives", prerequisite_id="linear_algebra",      weight=0.8),

    # ML fundamentals
    ConceptPrerequisiteEdge(concept_id="supervised_learning",  prerequisite_id="probability",        weight=0.7),
    ConceptPrerequisiteEdge(concept_id="supervised_learning",  prerequisite_id="linear_algebra",     weight=0.9),
    ConceptPrerequisiteEdge(concept_id="loss_functions",       prerequisite_id="supervised_learning", weight=1.0),
    ConceptPrerequisiteEdge(concept_id="loss_functions",       prerequisite_id="probability",        weight=0.6),

    # Gradient descent
    ConceptPrerequisiteEdge(concept_id="gradient_descent",    prerequisite_id="loss_functions",      weight=1.0),
    ConceptPrerequisiteEdge(concept_id="gradient_descent",    prerequisite_id="partial_derivatives", weight=1.0),

    # Backpropagation (the key chain)
    ConceptPrerequisiteEdge(concept_id="backpropagation",     prerequisite_id="gradient_descent",    weight=1.0),
    ConceptPrerequisiteEdge(concept_id="backpropagation",     prerequisite_id="chain_rule",          weight=1.0),
    ConceptPrerequisiteEdge(concept_id="backpropagation",     prerequisite_id="partial_derivatives", weight=0.9),

    # Neural networks
    ConceptPrerequisiteEdge(concept_id="neural_networks",     prerequisite_id="backpropagation",     weight=1.0),
    ConceptPrerequisiteEdge(concept_id="neural_networks",     prerequisite_id="loss_functions",      weight=0.8),
    ConceptPrerequisiteEdge(concept_id="regularization",      prerequisite_id="neural_networks",     weight=1.0),
    ConceptPrerequisiteEdge(concept_id="regularization",      prerequisite_id="gradient_descent",    weight=0.7),

    # Gradient descent variants
    ConceptPrerequisiteEdge(concept_id="gradient_descent_variants", prerequisite_id="gradient_descent", weight=1.0),
    ConceptPrerequisiteEdge(concept_id="gradient_descent_variants", prerequisite_id="backpropagation",  weight=0.8),

    # Attention & Transformers
    ConceptPrerequisiteEdge(concept_id="attention_mechanisms", prerequisite_id="neural_networks",    weight=1.0),
    ConceptPrerequisiteEdge(concept_id="attention_mechanisms", prerequisite_id="linear_algebra",     weight=0.9),
    ConceptPrerequisiteEdge(concept_id="transformers",         prerequisite_id="attention_mechanisms", weight=1.0),
    ConceptPrerequisiteEdge(concept_id="transformers",         prerequisite_id="regularization",     weight=0.7),
]


def create_ml_concept_graph() -> ConceptGraph:
    """Factory: returns a ConceptGraph pre-seeded with the ML curriculum DAG."""
    store = InMemoryConceptGraph(concepts=ML_CONCEPTS, edges=ML_EDGES)
    return ConceptGraph(store=store)
