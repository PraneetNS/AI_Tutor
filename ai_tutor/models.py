from enum import Enum
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict, AliasChoices
import uuid


# ---------------------------------------------------------------------------
# Learning Event Schema
# ---------------------------------------------------------------------------

class LearningEventType(str, Enum):
    """All first-class event types emitted by the AI Tutor pipeline."""
    # Student interaction events
    MESSAGE_SENT        = "message_sent"         # Student sent a message
    RESPONSE_GENERATED  = "response_generated"   # Tutor replied to student
    HINT_REQUESTED      = "hint_requested"        # Student asked for a hint
    ANSWER_REVEALED     = "answer_revealed"       # Tutor gave a direct answer (rule 5)
    ANSWER_SUBMITTED    = "answer_submitted"      # Student submitted an answer/response

    # Pedagogy state transitions
    STUCK_DETECTED      = "stuck_detected"        # Student flagged as stuck
    TOPIC_CHANGED       = "topic_changed"         # Active topic shifted
    MODE_CHANGED        = "mode_changed"          # pedagogy_mode transitioned

    # Assessment signals
    CONCEPT_MASTERED    = "concept_mastered"      # Mastery threshold crossed (upward)
    MISCONCEPTION_FOUND = "misconception_found"   # Guardrails / classifier flagged a misconception
    QUIZ_SUBMITTED      = "quiz_submitted"        # Student submitted a quiz attempt

    # Content / retrieval events
    KNOWLEDGE_RETRIEVED = "knowledge_retrieved"   # RAG retrieval fired
    OFF_TOPIC_REDIRECT  = "off_topic_redirect"    # Student redirected after off-topic message

    # System / lifecycle events
    SESSION_STARTED     = "session_started"       # New session opened
    SESSION_ENDED       = "session_ended"         # Session closed / timed out


class LearningEvent(BaseModel):
    """
    Immutable record of a single learning interaction or state transition.

    Produced by any pipeline stage and consumed by downstream services
    (analytics, mastery tracker, review logger, recommendation engine).

    Fields
    ------
    event_id        : UUID4 string – globally unique, auto-generated.
    event_type      : One of ``LearningEventType`` — what happened.
    occurred_at     : ISO 8601 UTC timestamp — when it happened.
    student_id      : Opaque student identifier (int or string).
    session_id      : Conversation / session thread identifier.
    course_id       : Optional LMS course scope.
    lecture_id      : Optional LMS lecture scope.
    concept         : The concept / topic name relevant to this event, if any.
    mastery_score   : Float in [0, 1] — learner mastery at event time, if known.
    hint_level      : Pedagogy hint depth at event time.
    pedagogy_mode   : Active pedagogy mode at event time.
    payload         : Arbitrary JSON-serialisable extra data for the event type.
    """

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Globally unique UUID4 event identifier"
    )
    event_type: LearningEventType = Field(
        ...,
        description="Type of learning event that occurred"
    )
    occurred_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of when the event occurred"
    )

    # Who / where
    student_id: Optional[Union[int, str]] = Field(
        default=None,
        description="Opaque student identifier"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Conversation / session thread identifier"
    )
    course_id: Optional[int] = Field(
        default=None,
        description="LMS Course identifier"
    )
    lecture_id: Optional[int] = Field(
        default=None,
        description="LMS Lecture identifier"
    )

    # What was learned / attempted
    concept: Optional[str] = Field(
        default=None,
        description="Concept or topic name relevant to this event"
    )
    mastery_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Learner mastery score at event time (0.0 – 1.0)"
    )

    # Pedagogy context snapshot
    hint_level: Optional[int] = Field(
        default=None,
        ge=0,
        le=5,
        description="Hint scaffolding depth active at event time"
    )
    pedagogy_mode: Optional[str] = Field(
        default=None,
        description="Active pedagogy mode at event time (socratic/direct/off_topic)"
    )

    # Open-ended extra data
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary JSON-serialisable extra data specific to the event type"
    )

    model_config = ConfigDict(use_enum_values=True)   # store enums as strings for JSON serialisation


# ---------------------------------------------------------------------------
# Learner State Schemas
# ---------------------------------------------------------------------------

class BKTParams(BaseModel):
    """
    Bayesian Knowledge Tracing (BKT) 4-parameter configuration per concept.

    Parameters:
    - p_l0: Prior probability of already knowing the concept P(L0) [default: 0.3]
    - p_t:  Probability of learning it after one opportunity P(T) [default: 0.1]
    - p_g:  Probability of guessing right without knowing P(G) [default: 0.25]
    - p_s:  Probability of slipping (wrong despite knowing) P(S) [default: 0.1]
    """
    p_l0: float = Field(default=0.3, ge=0.0, le=1.0, validation_alias=AliasChoices("p_l0", "P_L0", "pL0", "P_l0"), description="Prior probability P(L0)")
    p_t: float = Field(default=0.1, ge=0.0, le=1.0, validation_alias=AliasChoices("p_t", "P_T", "pT", "P_t"), description="Learning transition probability P(T)")
    p_g: float = Field(default=0.25, ge=0.0, le=1.0, validation_alias=AliasChoices("p_g", "P_G", "pG", "P_g"), description="Guess probability P(G)")
    p_s: float = Field(default=0.1, ge=0.0, le=1.0, validation_alias=AliasChoices("p_s", "P_S", "pS", "P_s"), description="Slip probability P(S)")

    model_config = ConfigDict(populate_by_name=True)

    @property
    def P_L0(self) -> float:
        return self.p_l0

    @property
    def P_T(self) -> float:
        return self.p_t

    @property
    def P_G(self) -> float:
        return self.p_g

    @property
    def P_S(self) -> float:
        return self.p_s


class ConceptMastery(BaseModel):
    """Bayesian Knowledge Tracing mastery model for a specific concept."""
    concept: str = Field(..., description="Concept / skill name or key")
    mastery: float = Field(default=0.1, ge=0.0, le=1.0, description="Posterior probability of mastery P(L)")
    attempts: int = Field(default=0, ge=0, description="Total submission attempts for this concept")
    correct: int = Field(default=0, ge=0, description="Total correct submission attempts")
    last_updated: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of last mastery update"
    )


class Misconception(BaseModel):
    """Detected active misconception for a learner."""
    key: str = Field(..., description="Unique key/identifier for the misconception")
    description: str = Field(..., description="Human-readable description of the misconception")
    concept: str = Field(..., description="Associated concept/topic")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence level in this misconception")
    hit_count: int = Field(default=1, ge=1, description="Number of times matched in learner responses")
    detected_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp when first detected"
    )
    last_seen_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp when last observed"
    )


class BehaviorProfile(BaseModel):
    """Rolling behavioral metrics tracked across learning sessions."""
    hints_per_session: float = Field(default=0.0, ge=0.0, description="Rolling average of hints used per session")
    avg_persistence: float = Field(default=0.0, ge=0.0, description="Rolling average interaction turns before hint/reveal")
    engagement_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Rolling engagement ratio (0.0 to 1.0)")
    sessions_total: int = Field(default=0, ge=0, description="Total session count observed")
    sessions_active: int = Field(default=0, ge=0, description="Active sessions count (meaningful attempts made)")
    total_hints_used: int = Field(default=0, ge=0, description="Cumulative hints requested")
    total_turns: int = Field(default=0, ge=0, description="Cumulative user message turns")
    last_active_at: Optional[str] = Field(default=None, description="ISO 8601 UTC timestamp of latest interaction")


class LearnerState(BaseModel):
    """
    Comprehensive state representation of a student across concepts,
    misconceptions, and behavioral tendencies.
    """
    student_id: str = Field(..., description="Unique student identifier")
    concept_mastery: Dict[str, ConceptMastery] = Field(
        default_factory=dict,
        description="Concept mastery map keyed by concept name"
    )
    misconceptions: List[Misconception] = Field(
        default_factory=list,
        description="List of detected misconceptions"
    )
    behavior: BehaviorProfile = Field(
        default_factory=BehaviorProfile,
        description="Rolling behavioral tendencies"
    )
    schema_version: int = Field(default=1, description="Schema version for forward compatibility")
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of last state modification"
    )

    model_config = ConfigDict(use_enum_values=True)


# ---------------------------------------------------------------------------
# Knowledge Graph & Curriculum Position Schemas
# ---------------------------------------------------------------------------

class ConceptNode(BaseModel):
    """Concept entity corresponding to 'concepts' table."""
    concept_id: str = Field(..., description="Unique slug/identifier for the concept")
    domain: str = Field(default="general", description="Subject or domain area (e.g., machine_learning)")
    name: str = Field(..., description="Human-readable concept title")
    description: Optional[str] = Field(default=None, description="Explanation or scope of the concept")


class ConceptPrerequisiteEdge(BaseModel):
    """Prerequisite relationship corresponding to 'concept_prerequisites' table."""
    concept_id: str = Field(..., description="Target concept ID")
    prerequisite_id: str = Field(..., description="Required prerequisite concept ID")
    weight: float = Field(default=1.0, ge=0.0, description="Dependency strength/weight")


class CurriculumPosition(BaseModel):
    """
    Deterministic curriculum state calculated before LLM reasoning:
    Tracks mastered, in-progress, locked, and next-ready concepts.
    """
    current_concept: Optional[str] = Field(default=None, description="Active concept under instruction")
    mastered: List[str] = Field(default_factory=list, description="List of mastered concept IDs")
    in_progress: List[str] = Field(default_factory=list, description="List of in-progress concept IDs")
    locked: List[str] = Field(default_factory=list, description="Concepts locked due to unmastered prerequisites")
    next_ready: List[str] = Field(default_factory=list, description="Concepts unlocked whose prerequisites are all met")

    model_config = ConfigDict(use_enum_values=True)


class RootCauseDiagnosis(BaseModel):
    """
    Deterministic diagnostic root-cause gap computed by backward traversal
    over the concept prerequisite DAG.
    """
    struggling_concept: str = Field(..., description="Concept the learner is currently struggling with")
    likely_root_gap: Optional[str] = Field(default=None, description="Identified prerequisite root gap with lowest mastery")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score in the root-cause diagnosis")
    chain_analyzed: List[str] = Field(default_factory=list, description="Full prerequisite ancestry chain analyzed")

    model_config = ConfigDict(use_enum_values=True)


# ---------------------------------------------------------------------------
# Teaching Strategy Schemas
# ---------------------------------------------------------------------------

class StrategyAction(str, Enum):
    """Actionable teaching recommendation from ContextResolver."""
    HINT = "hint"             # Provide a progressive scaffolding hint
    EXPLAIN = "explain"       # Direct explanation after 2+ failed attempts
    QUIZ = "quiz"             # Formative check for understanding (mastery > 0.8)
    CHALLENGE = "challenge"   # Push to advanced problem / next concept (mastery > 0.9)
    GUIDE = "guide"           # Default Socratic diagnostic dialogue


class TeachingStrategy(BaseModel):
    """
    Resolved pedagogical instruction payload based on LearnerState,
    course/lesson scope, and historical performance.
    """
    recommendation: StrategyAction = Field(
        ...,
        description="Pedagogical action recommendation: 'hint', 'explain', 'quiz', 'challenge', or 'guide'"
    )
    target_concept: Optional[str] = Field(
        default=None,
        description="Active concept under instruction"
    )
    target_mastery: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Current estimated mastery for the target concept"
    )
    misconception_to_address: Optional[Misconception] = Field(
        default=None,
        description="Highest-confidence unresolved misconception for the learner"
    )
    hint_budget_remaining: int = Field(
        default=3,
        ge=0,
        description="Remaining hint allowances for this concept/session"
    )
    consecutive_failures: int = Field(
        default=0,
        ge=0,
        description="Number of consecutive incorrect attempts on target concept"
    )
    rationale: str = Field(
        ...,
        description="Reasoning explaining why this strategy was selected"
    )
    course_id: Optional[int] = Field(
        default=None,
        description="LMS course identifier"
    )
    lecture_id: Optional[int] = Field(
        default=None,
        description="LMS lecture or lesson identifier"
    )
    difficulty_adjustment: Optional[str] = Field(
        default=None,
        description="Target difficulty level ('easy', 'medium', 'hard' / 'foundational', 'intermediate', 'advanced')"
    )
    curriculum_position: Optional[CurriculumPosition] = Field(
        default=None,
        description="Computed deterministic curriculum position (mastered, in_progress, locked, next_ready)"
    )
    root_cause_diagnosis: Optional[RootCauseDiagnosis] = Field(
        default=None,
        description="Computed root cause gap when student is struggling"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional resolved context properties"
    )

    model_config = ConfigDict(use_enum_values=True)


class QuizQuestion(BaseModel):
    """Structured question generated by QuizAgent within token budget (500-1200 tokens)."""
    concept: str = Field(..., description="Target concept being assessed")
    difficulty: str = Field(default="medium", description="Difficulty level ('easy', 'medium', 'hard')")
    question: str = Field(..., description="Formatted question text within token budget")
    token_count: int = Field(default=0, description="Approximate token count of generated question")
    misconception_targeted: Optional[str] = Field(default=None, description="Misconception specifically targeted if any")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Auxiliary metadata")

    model_config = ConfigDict(use_enum_values=True)


class AssessmentGrade(BaseModel):
    """Structured evaluation result produced by AssessmentAgent, emitted as a learning event."""
    student_id: Optional[str] = Field(default=None, description="Student identifier")
    concept: str = Field(..., description="Concept evaluated")
    correct: bool = Field(..., description="Whether the answer is conceptually correct")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Score between 0.0 and 1.0")
    hints_used: int = Field(default=0, ge=0, description="Hints consumed prior to or during this attempt")
    feedback: str = Field(..., description="Formative feedback string explaining the grade")
    misconception_detected: Optional[str] = Field(default=None, description="Key or description of detected misconception")
    event_emitted: Optional[LearningEvent] = Field(default=None, description="The LearningEvent emitted to the event bus")

    model_config = ConfigDict(use_enum_values=True)





class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class PedagogyMode(str, Enum):
    SOCRATIC = "socratic"
    DIRECT = "direct"
    OFF_TOPIC = "off_topic"


class IntentLabel(str, Enum):
    CONCEPT = "CONCEPT"        # Deep conceptual understanding -> Socratic guidance
    FACTUAL = "FACTUAL"        # Quick factual / definitional / syntax -> Direct answer
    OFF_TOPIC = "OFF_TOPIC"    # Unrelated, chitchat, prompt injection -> Polite redirect


class Chunk(BaseModel):
    content: str = Field(..., description="Text content of the retrieved document chunk")
    source_title: str = Field(..., description="Title of the source lecture/document")
    source_id: Union[str, int] = Field(..., description="Unique ID of the source lecture/material")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Rich contextual metadata (course_id, lesson_id, lecture_id, chunk_id, score, etc.)"
    )


class SourceCitation(BaseModel):
    lecture_id: int = Field(..., description="Identifier of the referenced lecture")
    title: str = Field(..., description="Title of the referenced lecture/material")
    chunk_id: Optional[Union[str, int]] = Field(default=None, description="Specific chunk or segment ID")
    snippet: Optional[str] = Field(default=None, description="Extracted snippet or excerpt")
    relevance_score: Optional[float] = Field(default=None, description="Similarity or ranking score")


class ClassificationResult(BaseModel):
    label: IntentLabel = Field(..., description="Classified intent: CONCEPT, FACTUAL, or OFF_TOPIC")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence score between 0.0 and 1.0")
    rationale: Optional[str] = Field(default=None, description="Reasoning for the assigned label")
    flagged_for_review: bool = Field(default=False, description="True if OFF_TOPIC or confidence below threshold")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp"
    )


class ChatMessage(BaseModel):
    role: Role = Field(..., description="Role of the sender ('user', 'assistant', 'system')")
    content: str = Field(..., min_length=1, description="Message text content")
    timestamp: Optional[str] = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp"
    )


class PedagogyState(BaseModel):
    hint_level: int = Field(
        default=0,
        ge=0,
        le=5,
        description="Scaffolding hint level (0 = direct explanation/none, 1 = nudge, 3+ = near solution)"
    )
    topic: Optional[str] = Field(
        default=None,
        description="Detected core topic/concept discussed in the conversation"
    )
    stuck: bool = Field(
        default=False,
        description="Whether the student is struggling/stuck on the current concept"
    )
    pedagogy_mode: PedagogyMode = Field(
        default=PedagogyMode.DIRECT,
        description="Active pedagogical mode (socratic, direct, or off_topic)"
    )


# --- Step 1 API Contract Schemas ---

class AIChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Student's prompt or question")
    course_id: Optional[int] = Field(default=None, description="LMS Course identifier")
    lecture_id: Optional[int] = Field(default=None, description="LMS Lecture identifier")
    student_id: Optional[Union[int, str]] = Field(default=None, description="Student identifier")
    session_id: Optional[str] = Field(default=None, description="Session / conversation thread ID")
    pedagogy_mode: Optional[PedagogyMode] = Field(
        default=None,
        description="Desired teaching style: socratic, direct, or off_topic"
    )
    hint_level: Optional[int] = Field(
        default=None,
        ge=0,
        le=5,
        description="Desired hint scaffolding depth (0-5)"
    )
    conversation_history: Optional[List[ChatMessage]] = Field(
        default_factory=list,
        description="Prior dialogue turns within this session"
    )


class AIChatResponse(BaseModel):
    answer: str = Field(..., description="AI Tutor generated response")
    session_id: Optional[str] = Field(default=None, description="Session ID")
    pedagogy_mode: Optional[PedagogyMode] = Field(
        default=None,
        description="Pedagogy mode applied in this turn"
    )
    hint_level: Optional[int] = Field(
        default=None,
        description="Hint level applied in this turn"
    )
    knowledge_source_used: Optional[str] = Field(
        default=None,
        description="Source provider used for grounding citations"
    )
    sources: List[SourceCitation] = Field(
        default_factory=list,
        description="Referenced sources and chunks"
    )


# --- Internal Pipeline Structures ---

class TutorInput(BaseModel):
    message: str = Field(..., min_length=1, description="Latest student input message")
    conversation_history: List[ChatMessage] = Field(
        default_factory=list,
        description="Ordered list of prior conversation messages"
    )
    pedagogy_state: Optional[PedagogyState] = Field(
        default=None,
        description="Current pedagogy state prior to processing this turn (if any)"
    )
    course_id: Optional[int] = Field(default=None, description="LMS Course identifier")
    lecture_id: Optional[int] = Field(default=None, description="LMS Lecture identifier")


class TutorOutput(BaseModel):
    answer: str = Field(..., description="Tutor's generated textual response")
    pedagogy_state: PedagogyState = Field(
        ...,
        description="Updated pedagogical state after evaluating the turn"
    )
    classification: Optional[ClassificationResult] = Field(
        default=None,
        description="Intent classification result for this turn"
    )
    knowledge_source_used: Optional[str] = Field(
        default=None,
        description="Identifies the knowledge source used (e.g., 'MockKnowledgeSource')"
    )
    sources: List[SourceCitation] = Field(
        default_factory=list,
        description="Grounded sources/citations retrieved from KnowledgeSource"
    )


# ---------------------------------------------------------------------------
# Orchestrated Context Schemas
# ---------------------------------------------------------------------------

class SessionContext(BaseModel):
    """Short-term session memory: recent dialogue turns, mood, and pedagogy state."""
    session_id: str = Field(..., description="Unique conversation session ID")
    recent_messages: List[ChatMessage] = Field(default_factory=list, description="Recent conversation turns in order")
    pedagogy_state: PedagogyState = Field(default_factory=PedagogyState, description="Active pedagogy tracking state")
    detected_mood: str = Field(default="neutral", description="Detected student tone/mood (e.g., curious, confused, frustrated, confident)")
    turn_count: int = Field(default=0, ge=0, description="Total turns recorded in current session")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Auxiliary session flags")

    model_config = ConfigDict(use_enum_values=True)


class LearningContext(BaseModel):
    """Long-term student knowledge state, mastery, misconceptions, and strategy."""
    student_id: Optional[str] = Field(default=None, description="Student identifier")
    learner_state: Optional[LearnerState] = Field(default=None, description="Current full LearnerState snapshot")
    target_concept: Optional[str] = Field(default=None, description="Target concept being studied")
    target_mastery: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Estimated mastery score (0-1)")
    teaching_strategy: Optional[TeachingStrategy] = Field(default=None, description="Recommended teaching strategy")
    active_misconceptions: List[Misconception] = Field(default_factory=list, description="Active identified misconceptions")
    behavior_summary: Optional[Dict[str, Any]] = Field(default=None, description="Summary of behavioral tendencies")

    model_config = ConfigDict(use_enum_values=True)


class KnowledgeContext(BaseModel):
    """Retrieved course material and grounding citations."""
    chunks: List[Chunk] = Field(default_factory=list, description="Retrieved curriculum chunks")
    citations: List[SourceCitation] = Field(default_factory=list, description="Structured citations for UI rendering")
    knowledge_source_used: Optional[str] = Field(default=None, description="Name of knowledge source backend")
    query: Optional[str] = Field(default=None, description="Search query passed to retriever")

    model_config = ConfigDict(use_enum_values=True)


class OrchestratedContext(BaseModel):
    """
    Consolidated single context object merging short-term session memory,
    long-term learner model state, and retrieved curriculum knowledge.
    Passed directly to the Tutor Reasoner.
    """
    student_message: str = Field(..., description="Current student incoming prompt")
    session_context: SessionContext = Field(..., description="Short-term session memory")
    learning_context: LearningContext = Field(..., description="Long-term learner state & strategy")
    knowledge_context: KnowledgeContext = Field(..., description="Retrieved course material context")
    course_id: Optional[int] = Field(default=None, description="Active LMS course ID")
    lecture_id: Optional[int] = Field(default=None, description="Active LMS lecture ID")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of context orchestration"
    )

    model_config = ConfigDict(use_enum_values=True)

    def to_prompt_sections(self) -> Dict[str, str]:
        """
        Formats the unified context into named sections for BudgetManager prompt assembly.
        Curriculum position and root-cause diagnosis are pre-computed deterministically
        by ConceptGraph and injected here — the LLM receives answers, not raw history.
        """
        import json

        strategy = self.learning_context.teaching_strategy

        # 1. Mastery snapshot (top-N lowest-mastery concepts + active misconceptions)
        concepts_list = []
        if self.learning_context.learner_state:
            sorted_concepts = sorted(
                self.learning_context.learner_state.concept_mastery.items(),
                key=lambda kv: kv[1].mastery
            )
            for c_name, cm in sorted_concepts[:10]:  # budget-limited to top 10 weakest
                concepts_list.append({"name": c_name, "mastery": round(cm.mastery, 3)})

        misconceptions_list = [
            {"key": m.key, "description": m.description, "confidence": m.confidence}
            for m in self.learning_context.active_misconceptions
        ]

        # 2. Curriculum position (pre-computed, NOT inferred by LLM)
        curriculum_position_dict: Dict[str, Any] = {}
        if strategy and strategy.curriculum_position:
            cp = strategy.curriculum_position
            curriculum_position_dict = {
                "current_concept": cp.current_concept,
                "mastered": cp.mastered,
                "in_progress": cp.in_progress,
                "locked": cp.locked,
                "next_ready": cp.next_ready,
            }

        # 3. Root-cause diagnosis (pre-computed, NOT inferred by LLM)
        root_cause_dict: Dict[str, Any] = {}
        if strategy and strategy.root_cause_diagnosis:
            rcd = strategy.root_cause_diagnosis
            root_cause_dict = {
                "struggling_concept": rcd.struggling_concept,
                "likely_root_gap": rcd.likely_root_gap,
                "confidence": rcd.confidence,
                "chain_analyzed": rcd.chain_analyzed,
            }

        # 4. Behavioral profile
        behavioral: Dict[str, Any] = {}
        if self.learning_context.behavior_summary:
            bs = self.learning_context.behavior_summary
            behavioral = {
                "persistence": bs.get("avg_persistence", 0.0),
                "engagement": bs.get("engagement_score", 1.0),
                "avg_hints": bs.get("hints_per_session", 0.0),
                "sessions_total": bs.get("sessions_total", 0),
            }

        # Assemble the full learner state JSON handed to LLM
        learner_state_json = json.dumps({
            "student_id": self.learning_context.student_id,
            "course_id": self.course_id,
            "lecture_id": self.lecture_id,
            "mastery_snapshot": concepts_list,
            "misconceptions": misconceptions_list,
            "curriculum_position": curriculum_position_dict,
            "root_cause_diagnosis": root_cause_dict,
            "behavioral": behavioral,
            "target_concept": self.learning_context.target_concept,
            "recommended_strategy": strategy.recommendation if strategy else "guide",
        }, indent=None)

        # 5. Conversation history
        history_lines = [
            f"{msg.role.capitalize()}: {msg.content}"
            for msg in self.session_context.recent_messages
        ]
        history_text = "\n".join(history_lines) if history_lines else ""

        # 6. Session summary
        summary_text = (
            f"Student ({self.session_context.detected_mood} mood) is studying "
            f"'{self.learning_context.target_concept or 'concepts'}' "
            f"({len(self.session_context.recent_messages)} prior turns, "
            f"turn #{self.session_context.turn_count})."
        )

        # 7. RAG knowledge section
        rag_chunks = [
            f"[Source: {c.source_title}]: {c.content}"
            for c in self.knowledge_context.chunks
        ]
        rag_text = "\n---\n".join(rag_chunks) if rag_chunks else ""

        # 8. Teaching Strategy directive (explicit instruction to LLM)
        strategy_directive = ""
        if strategy:
            lines = [
                f"PEDAGOGICAL DIRECTIVE: {strategy.recommendation.upper()}",
                f"Rationale: {strategy.rationale}",
                f"Hints remaining: {strategy.hint_budget_remaining}",
            ]
            if strategy.root_cause_diagnosis and strategy.root_cause_diagnosis.likely_root_gap:
                rcd = strategy.root_cause_diagnosis
                lines.append(
                    f"ROOT CAUSE GAP: Student likely struggles with "
                    f"'{rcd.likely_root_gap}' (confidence {rcd.confidence:.0%}). "
                    f"Address this prerequisite before re-explaining '{rcd.struggling_concept}'."
                )
            if strategy.curriculum_position and strategy.curriculum_position.next_ready:
                lines.append(
                    f"NEXT READY CONCEPTS: {', '.join(strategy.curriculum_position.next_ready[:3])}"
                )
            strategy_directive = "\n".join(lines)

        return {
            "learner_state": learner_state_json,
            "rag_knowledge": rag_text,
            "teaching_strategy": strategy_directive,
            "conversation_history": history_text,
            "conversation_summary": summary_text,
        }


class ReasonerResult(BaseModel):
    """
    Output of the TutorReasoner 'decide -> teach/assess -> adapt' loop.
    """
    answer: str = Field(..., description="Generated textual response for the student")
    selected_agent: str = Field(..., description="The sub-agent that handled this turn ('TutorAgent', 'QuizAgent', 'AssessmentAgent')")
    pedagogy_state: PedagogyState = Field(..., description="Updated pedagogical state after turn adaptation")
    strategy_applied: TeachingStrategy = Field(..., description="Teaching strategy applied in this turn")
    sources: List[SourceCitation] = Field(default_factory=list, description="Grounding source citations")
    assessment_result: Optional[Dict[str, Any]] = Field(default=None, description="Assessment metadata if AssessmentAgent evaluated an answer")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Auxiliary execution metadata")

    model_config = ConfigDict(use_enum_values=True)


# ---------------------------------------------------------------------------
# Provider-Agnostic Gateway Response Schemas
# ---------------------------------------------------------------------------

class ModelUsage(BaseModel):
    """Token consumption metadata for LLM generations."""
    input_tokens: int = Field(default=0, ge=0, description="Count of prompt/input tokens")
    output_tokens: int = Field(default=0, ge=0, description="Count of completion/output tokens")

    model_config = ConfigDict(use_enum_values=True)


class ModelResponse(BaseModel):
    """
    Unified LLM response structure returned by ModelGateway and Provider Adapters.
    """
    content: str = Field(default="", description="Generated response content")
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted tool/function calls")
    finish_reason: str = Field(default="stop", description="Reason model stopped ('stop', 'length', 'tool_calls')")
    usage: Dict[str, int] = Field(
        default_factory=lambda: {"input_tokens": 0, "output_tokens": 0},
        description="Token usage stats (input_tokens, output_tokens)"
    )

    model_config = ConfigDict(use_enum_values=True)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


