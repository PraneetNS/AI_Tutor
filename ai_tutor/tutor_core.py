from typing import List, Optional, Dict, Any
from .models import (
    ChatMessage,
    Role,
    PedagogyState,
    TutorInput,
    TutorOutput,
    SourceCitation,
    Chunk
)
from .llm_client import BaseLLMClient, OpenAILLMClient
from .knowledge_source import KnowledgeSource


DEFAULT_SYSTEM_PROMPT = """You are a Socratic AI tutor. Your job is to help the student understand concepts
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
Good: "Before I explain — have you worked with any dataset that had labeled examples,
like emails marked spam/not spam? What do you think the model is 'learning' from in
that case?"

EXAMPLE 2
Student: "Is this code right? [buggy loop]"
Bad: "No, line 3 has an off-by-one error."
Good: "Let's trace it — what value does your loop variable have on the very last
iteration? Does that match what you expected?"

EXAMPLE 3
Student: "I don't know, just tell me."
Good: "Okay — supervised learning means training a model on labeled data, where each
example already has the correct answer attached, so the model learns to map inputs
to outputs. Here's why that matters for what we just discussed: [tie back to their
earlier attempt]."

Never break character as a tutor. Never simply answer like a general assistant.

OUTPUT FORMAT:
Respond with a JSON object strictly matching this schema:
{
  "answer": "Your tutor response string to the student",
  "hint_level": 0,
  "topic": "Name of concept",
  "stuck": false,
  "pedagogy_mode": "socratic"
}
"""


class TutorCore:
    """
    Stateless Tutor Core Module.
    Accepts student message, history, and metadata filters.
    Calls the KnowledgeSource abstraction to retrieve relevant chunks (never touches LMS data directly),
    and produces the response with updated pedagogical state and source citations.
    """

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        knowledge_source: Optional[KnowledgeSource] = None,
        system_prompt: Optional[str] = None
    ):
        self.llm_client = llm_client or OpenAILLMClient()
        self.knowledge_source = knowledge_source
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    def set_system_prompt(self, system_prompt: str) -> None:
        """Dynamically update or override the base system prompt."""
        self.system_prompt = system_prompt

    def set_knowledge_source(self, knowledge_source: Optional[KnowledgeSource]) -> None:
        """Dynamically attach or detach a knowledge source."""
        self.knowledge_source = knowledge_source

    def generate(
        self,
        student_message: str,
        conversation_history: Optional[List[ChatMessage]] = None,
        current_state: Optional[PedagogyState] = None,
        system_prompt_override: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> TutorOutput:
        """
        Stateless generation method with optional knowledge retrieval.
        """
        history = list(conversation_history or [])
        filters = filters or {}

        # 1. Retrieve knowledge chunks via KnowledgeSource interface (if configured)
        retrieved_chunks: List[Chunk] = []
        knowledge_source_name: Optional[str] = None

        if self.knowledge_source:
            retrieved_chunks = self.knowledge_source.retrieve(query=student_message, filters=filters)
            knowledge_source_name = type(self.knowledge_source).__name__

        # 2. Build prompt context with knowledge chunks if available
        knowledge_context = ""
        citations: List[SourceCitation] = []

        if retrieved_chunks:
            chunk_texts = []
            for i, chunk in enumerate(retrieved_chunks, 1):
                chunk_texts.append(f"[Source {i} - Lecture: '{chunk.source_title}']: {chunk.content}")
                
                meta = chunk.metadata or {}
                lec_id = meta.get("lecture_id", int(chunk.source_id) if isinstance(chunk.source_id, int) or (isinstance(chunk.source_id, str) and chunk.source_id.isdigit()) else 0)
                
                citations.append(
                    SourceCitation(
                        lecture_id=lec_id,
                        title=chunk.source_title,
                        chunk_id=meta.get("chunk_id"),
                        snippet=chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content,
                        relevance_score=meta.get("relevance_score")
                    )
                )

            knowledge_context = f"\n\nCOURSE LECTURE RETRIEVAL CONTEXT:\n" + "\n".join(chunk_texts)

        # 3. Assemble active messages
        active_messages = history + [ChatMessage(role=Role.USER, content=student_message)]

        prompt = (system_prompt_override or self.system_prompt) + knowledge_context

        # 4. Call LLM
        output = self.llm_client.generate(
            system_prompt=prompt,
            messages=active_messages,
            current_state=current_state
        )

        # 5. Attach grounded citations and source metadata
        output.sources = citations
        output.knowledge_source_used = knowledge_source_name

        return output

    def process_turn(self, tutor_input: TutorInput) -> TutorOutput:
        """Convenience method accepting a Pydantic TutorInput object."""
        filters = {}
        if tutor_input.course_id is not None:
            filters["course_id"] = tutor_input.course_id
        if tutor_input.lecture_id is not None:
            filters["lecture_id"] = tutor_input.lecture_id

        return self.generate(
            student_message=tutor_input.message,
            conversation_history=tutor_input.conversation_history,
            current_state=tutor_input.pedagogy_state,
            filters=filters
        )
