from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import uuid
import logging

from .models import (
    AIChatRequest,
    AIChatResponse,
    ChatMessage,
    Role,
    PedagogyMode,
    IntentLabel,
    ClassificationResult,
    PedagogyState,
    Chunk,
    SourceCitation,
    TutorOutput
)
from .classifier import IntentClassifier
from .knowledge_source import KnowledgeSource
from .llm_client import BaseLLMClient, OpenAILLMClient, MockLLMClient
from .session_store import BaseSessionStore, InMemorySessionStore
from .guardrails import ResponseGuardrail, GuardrailResult

logger = logging.getLogger("ai_tutor.pipeline")


# =====================================================================
# STAGE 1: ROUTER
# =====================================================================
class BaseRouter(ABC):
    @abstractmethod
    def route(
        self,
        request: AIChatRequest,
        history: List[ChatMessage]
    ) -> ClassificationResult:
        pass


class DefaultRouter(BaseRouter):
    def __init__(self, classifier: Optional[IntentClassifier] = None):
        self.classifier = classifier or IntentClassifier()

    def route(
        self,
        request: AIChatRequest,
        history: List[ChatMessage]
    ) -> ClassificationResult:
        return self.classifier.classify(
            student_message=request.message,
            conversation_history=history,
            session_id=request.session_id
        )


# =====================================================================
# STAGE 2: PEDAGOGY ENGINE
# =====================================================================
class BasePedagogyEngine(ABC):
    @abstractmethod
    def evaluate(
        self,
        request: AIChatRequest,
        classification: ClassificationResult,
        current_state: PedagogyState,
        history: List[ChatMessage]
    ) -> PedagogyState:
        pass


class DefaultPedagogyEngine(BasePedagogyEngine):
    def evaluate(
        self,
        request: AIChatRequest,
        classification: ClassificationResult,
        current_state: PedagogyState,
        history: List[ChatMessage]
    ) -> PedagogyState:
        if request.pedagogy_mode:
            target_mode = request.pedagogy_mode
        elif classification.label == IntentLabel.OFF_TOPIC:
            target_mode = PedagogyMode.OFF_TOPIC
        elif classification.label == IntentLabel.CONCEPT:
            target_mode = PedagogyMode.SOCRATIC
        else:
            target_mode = PedagogyMode.DIRECT

        msg_lower = request.message.lower()
        stuck_signals = ["stuck", "don't understand", "dont understand", "confused", "help", "hint"]
        is_stuck = any(s in msg_lower for s in stuck_signals)

        if request.hint_level is not None:
            new_hint_level = request.hint_level
        elif "hint" in msg_lower or is_stuck:
            new_hint_level = min(5, current_state.hint_level + 1)
        else:
            new_hint_level = current_state.hint_level

        return PedagogyState(
            hint_level=new_hint_level,
            topic=current_state.topic,
            stuck=is_stuck or current_state.stuck,
            pedagogy_mode=target_mode
        )


# =====================================================================
# STAGE 4: PROMPT ORCHESTRATOR
# =====================================================================
class BasePromptOrchestrator(ABC):
    @abstractmethod
    def build_prompt(
        self,
        request: AIChatRequest,
        pedagogy_state: PedagogyState,
        chunks: List[Chunk],
        history: List[ChatMessage]
    ) -> TupleTextPrompt:
        pass


class TupleTextPrompt:
    def __init__(self, system_prompt: str, messages: List[ChatMessage]):
        self.system_prompt = system_prompt
        self.messages = messages


class DefaultPromptOrchestrator(BasePromptOrchestrator):
    BASE_SYSTEM = """You are a Socratic AI tutor. Your job is to help the student understand concepts
by guiding them to the answer — never by lecturing them.

HARD RULES:
1. Never open a response to a concept question with a direct explanation or answer.
2. Start by asking a diagnostic question to find out what the student already knows,
   or by asking them to attempt the problem/concept themselves first.
3. If the student is wrong, don't correct directly — ask a question that exposes
   the gap in their reasoning.
4. Give hints progressively (small nudge -> bigger nudge -> partial reveal), never
   all at once.
5. Only give the direct answer if: the student has made 2+ genuine attempts and is
   still stuck, OR the student explicitly asks you to just explain it. Even then,
   explain the reasoning, don't just state the fact.
6. When course/lecture context is provided, ground your questions and hints in that
   material and mention the source lecture by name.
7. For quick factual lookups unrelated to a concept being taught, answer directly
   and briefly — don't force Socratic dialogue where it doesn't fit.

EXAMPLE 1
Student: "Explain supervised learning"
Bad: "Supervised learning is a type of machine learning where..."
Good: "Before I explain — have you worked with any dataset that had labeled examples, like emails marked spam/not spam? What do you think the model is 'learning' from in that case?"

EXAMPLE 2
Student: "Is this code right? [buggy loop]"
Bad: "No, line 3 has an off-by-one error."
Good: "Let's trace it — what value does your loop variable have on the very last iteration? Does that match what you expected?"

EXAMPLE 3
Student: "I don't know, just tell me."
Good: "Okay — supervised learning means training a model on labeled data, where each example already has the correct answer attached, so the model learns to map inputs to outputs. Here's why that matters for what we just discussed: [tie back to their earlier attempt]."

Never break character as a tutor. Never simply answer like a general assistant.
"""

    SOCRATIC_DIRECTIVE = """PEDAGOGICAL INSTRUCTION (SOCRATIC CONCEPT MODE):
- Strictly adhere to Rules 1, 2, 3, and 4.
- Open with a diagnostic question or request a student attempt.
- Progressively scaffold based on the current hint_level.
"""

    DIRECT_DIRECTIVE = """PEDAGOGICAL INSTRUCTION (FACTUAL LOOKUP MODE):
- Adhere to Rule 7: For quick factual lookups or definitions unrelated to deeper concept scaffolding, answer directly, accurately, and concisely.
"""

    OFF_TOPIC_DIRECTIVE = """PEDAGOGICAL INSTRUCTION (OFF-TOPIC REDIRECTION):
- Politely acknowledge the student's message and gently redirect them back to the active course learning goals.
"""

    def build_prompt(
        self,
        request: AIChatRequest,
        pedagogy_state: PedagogyState,
        chunks: List[Chunk],
        history: List[ChatMessage]
    ) -> TupleTextPrompt:
        if pedagogy_state.pedagogy_mode == PedagogyMode.SOCRATIC:
            directive = self.SOCRATIC_DIRECTIVE
        elif pedagogy_state.pedagogy_mode == PedagogyMode.OFF_TOPIC:
            directive = self.OFF_TOPIC_DIRECTIVE
        else:
            directive = self.DIRECT_DIRECTIVE

        scaffolding_info = (
            f"\nCURRENT STATE:\n"
            f"- Hint Level: {pedagogy_state.hint_level}\n"
            f"- Student is Stuck: {pedagogy_state.stuck}\n"
        )

        knowledge_context = ""
        if chunks:
            formatted_chunks = [
                f"[Source: {c.source_title} (ID {c.source_id})]: {c.content}"
                for c in chunks
            ]
            knowledge_context = "\n\nCOURSE KNOWLEDGE BASE CONTEXT:\n" + "\n".join(formatted_chunks) + "\n"

        system_content = f"{self.BASE_SYSTEM}\n{directive}\n{scaffolding_info}{knowledge_context}"
        active_messages = list(history) + [ChatMessage(role=Role.USER, content=request.message)]

        return TupleTextPrompt(system_prompt=system_content, messages=active_messages)


# =====================================================================
# STAGE 5: MODEL ADAPTER
# =====================================================================
class BaseModelAdapter(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: TupleTextPrompt,
        pedagogy_state: PedagogyState
    ) -> TutorOutput:
        pass


class DefaultModelAdapter(BaseModelAdapter):
    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self.llm_client = llm_client or OpenAILLMClient()

    def generate(
        self,
        prompt: TupleTextPrompt,
        pedagogy_state: PedagogyState
    ) -> TutorOutput:
        return self.llm_client.generate(
            system_prompt=prompt.system_prompt,
            messages=prompt.messages,
            current_state=pedagogy_state
        )


# =====================================================================
# STAGE 6: GUARDRAILS
# =====================================================================
class BaseGuardrails(ABC):
    @abstractmethod
    def validate_and_sanitize(
        self,
        raw_answer: str,
        pedagogy_state: PedagogyState,
        request: AIChatRequest,
        sources: Optional[List[SourceCitation]] = None,
        chunks: Optional[List[Chunk]] = None
    ) -> GuardrailResult:
        pass

    @abstractmethod
    def get_fallback_response(
        self,
        error: Optional[Exception] = None,
        request: Optional[AIChatRequest] = None
    ) -> str:
        pass


class DefaultGuardrails(BaseGuardrails):
    def __init__(self, guardrail_impl: Optional[ResponseGuardrail] = None):
        self.guardrail = guardrail_impl or ResponseGuardrail()

    def validate_and_sanitize(
        self,
        raw_answer: str,
        pedagogy_state: PedagogyState,
        request: AIChatRequest,
        sources: Optional[List[SourceCitation]] = None,
        chunks: Optional[List[Chunk]] = None
    ) -> GuardrailResult:
        return self.guardrail.validate_and_sanitize(
            raw_answer=raw_answer,
            pedagogy_state=pedagogy_state,
            request=request,
            sources=sources,
            chunks=chunks
        )

    def get_fallback_response(
        self,
        error: Optional[Exception] = None,
        request: Optional[AIChatRequest] = None
    ) -> str:
        return self.guardrail.get_fallback_response(error=error, request=request)


# =====================================================================
# END-TO-END TUTOR PIPELINE
# =====================================================================
class TutorPipeline:
    def __init__(
        self,
        router: Optional[BaseRouter] = None,
        pedagogy_engine: Optional[BasePedagogyEngine] = None,
        knowledge_source: Optional[KnowledgeSource] = None,
        prompt_orchestrator: Optional[BasePromptOrchestrator] = None,
        model_adapter: Optional[BaseModelAdapter] = None,
        guardrails: Optional[BaseGuardrails] = None,
        session_store: Optional[BaseSessionStore] = None
    ):
        self.router = router or DefaultRouter()
        self.pedagogy_engine = pedagogy_engine or DefaultPedagogyEngine()
        self.knowledge_source = knowledge_source
        self.prompt_orchestrator = prompt_orchestrator or DefaultPromptOrchestrator()
        self.model_adapter = model_adapter or DefaultModelAdapter()
        self.guardrails = guardrails or DefaultGuardrails()
        self.session_store = session_store or InMemorySessionStore()

    def process(self, request: AIChatRequest) -> AIChatResponse:
        session_id = request.session_id or f"sess_{uuid.uuid4().hex[:10]}"

        session_data = self.session_store.get_session(session_id)
        history = list(request.conversation_history or session_data.messages)
        current_state = session_data.pedagogy_state

        # STAGE 1: ROUTER
        classification = self.router.route(request=request, history=history)

        # STAGE 2: PEDAGOGY ENGINE
        updated_state = self.pedagogy_engine.evaluate(
            request=request,
            classification=classification,
            current_state=current_state,
            history=history
        )

        # STAGE 3: KNOWLEDGE INTERFACE
        retrieved_chunks: List[Chunk] = []
        knowledge_source_name: Optional[str] = None
        citations: List[SourceCitation] = []

        should_retrieve = (
            classification.label == IntentLabel.CONCEPT 
            and request.course_id is not None 
            and self.knowledge_source is not None
        )

        if should_retrieve:
            filters = {"course_id": request.course_id}
            if request.lecture_id is not None:
                filters["lecture_id"] = request.lecture_id

            try:
                retrieved_chunks = self.knowledge_source.retrieve(query=request.message, filters=filters)
                knowledge_source_name = type(self.knowledge_source).__name__

                for c in retrieved_chunks:
                    meta = c.metadata or {}
                    lec_id = meta.get("lecture_id", int(c.source_id) if isinstance(c.source_id, int) or (isinstance(c.source_id, str) and c.source_id.isdigit()) else (request.lecture_id or 0))
                    citations.append(
                        SourceCitation(
                            lecture_id=lec_id,
                            title=c.source_title,
                            chunk_id=meta.get("chunk_id"),
                            snippet=c.content[:150] + "..." if len(c.content) > 150 else c.content,
                            relevance_score=meta.get("relevance_score") or meta.get("hybrid_score")
                        )
                    )
            except Exception as e:
                logger.error(f"KnowledgeSource retrieval failed: {e}")

        # STAGE 4: PROMPT ORCHESTRATOR
        prompt = self.prompt_orchestrator.build_prompt(
            request=request,
            pedagogy_state=updated_state,
            chunks=retrieved_chunks,
            history=history
        )

        # STAGE 5: MODEL ADAPTER (With resilient error/timeout handling)
        try:
            raw_output = self.model_adapter.generate(prompt=prompt, pedagogy_state=updated_state)
            raw_answer = raw_output.answer
            final_state = raw_output.pedagogy_state or updated_state
        except Exception as e:
            # Model failure or timeout fallback
            raw_answer = self.guardrails.get_fallback_response(error=e, request=request)
            final_state = updated_state

        # STAGE 6: GUARDRAILS (Safety check, prompt leak scrub, RAG hallucination check)
        guardrail_result = self.guardrails.validate_and_sanitize(
            raw_answer=raw_answer,
            pedagogy_state=final_state,
            request=request,
            sources=citations,
            chunks=retrieved_chunks
        )
        safe_answer = guardrail_result.sanitized_answer

        # STAGE 7: PERSIST TO SESSION STORE
        self.session_store.append_message(session_id, Role.USER, request.message)
        self.session_store.append_message(session_id, Role.ASSISTANT, safe_answer)
        self.session_store.update_pedagogy_state(session_id, final_state)

        return AIChatResponse(
            answer=safe_answer,
            session_id=session_id,
            pedagogy_mode=final_state.pedagogy_mode,
            hint_level=final_state.hint_level,
            knowledge_source_used=knowledge_source_name,
            sources=citations
        )
