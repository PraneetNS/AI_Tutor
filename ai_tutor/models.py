from enum import Enum
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict
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
        """Formats the unified context into named sections for BudgetManager prompt assembly."""
        import json

        # 1. Learner state section
        concepts_list = []
        if self.learning_context.learner_state:
            for c_name, cm in self.learning_context.learner_state.concept_mastery.items():
                concepts_list.append({"name": c_name, "mastery": cm.mastery})

        misconceptions_list = [
            m.description for m in self.learning_context.active_misconceptions
        ]
        learner_state_json = json.dumps({
            "concepts": concepts_list,
            "misconceptions": misconceptions_list,
            "target_concept": self.learning_context.target_concept,
            "recommended_strategy": self.learning_context.teaching_strategy.recommendation if self.learning_context.teaching_strategy else "guide"
        })

        # 2. Conversation history
        history_lines = [
            f"{msg.role.capitalize()}: {msg.content}"
            for msg in self.session_context.recent_messages
        ]
        history_text = "\n".join(history_lines) if history_lines else ""

        # Conversation summary fallback
        summary_text = f"Student is discussing {self.learning_context.target_concept or 'concepts'} ({len(self.session_context.recent_messages)} prior turns)."

        # 3. RAG knowledge section
        rag_chunks = [
            f"[Source: {c.source_title}]: {c.content}"
            for c in self.knowledge_context.chunks
        ]
        rag_text = "\n---\n".join(rag_chunks) if rag_chunks else ""

        # 4. Teaching Strategy directive
        strategy_directive = ""
        if self.learning_context.teaching_strategy:
            strategy_directive = (
                f"PEDAGOGICAL DIRECTIVE: {self.learning_context.teaching_strategy.recommendation.upper()}\n"
                f"Rationale: {self.learning_context.teaching_strategy.rationale}\n"
                f"Hints remaining: {self.learning_context.teaching_strategy.hint_budget_remaining}"
            )

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


