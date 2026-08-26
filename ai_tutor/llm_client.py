import os
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from .models import ChatMessage, PedagogyState, PedagogyMode, TutorOutput


class LLMPayload(BaseModel):
    answer: str
    hint_level: int = 0
    topic: Optional[str] = None
    stuck: bool = False
    pedagogy_mode: PedagogyMode = PedagogyMode.DIRECT


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        messages: List[ChatMessage],
        current_state: Optional[PedagogyState] = None
    ) -> TutorOutput:
        pass


class OpenAILLMClient(BaseLLMClient):
    """
    Client for OpenAI / OpenAI-compatible managed AI APIs 
    (OpenAI, Gemini OpenAI-compatible, Azure, Groq, Ollama, LiteLLM).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None
    ):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", "dummy_key"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL")
        )
        self.model = model

    def generate(
        self,
        system_prompt: str,
        messages: List[ChatMessage],
        current_state: Optional[PedagogyState] = None
    ) -> TutorOutput:
        state_context = (
            f"\n\nCURRENT PEDAGOGY STATE:\n"
            f"- Current Hint Level: {current_state.hint_level if current_state else 0}\n"
            f"- Current Topic: {current_state.topic if current_state else 'General'}\n"
            f"- Current Stuck Status: {current_state.stuck if current_state else False}\n"
            f"- Current Pedagogy Mode: {current_state.pedagogy_mode.value if current_state else 'direct'}\n"
        )

        formatted_messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt + state_context}
        ]

        for msg in messages:
            formatted_messages.append({"role": msg.role.value, "content": msg.content})

        # Request structured JSON output
        response = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            response_format={"type": "json_object"},
            temperature=0.4
        )

        raw_content = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(raw_content)
            pedagogy_state = PedagogyState(
                hint_level=int(parsed.get("hint_level", current_state.hint_level if current_state else 0)),
                topic=parsed.get("topic", current_state.topic if current_state else None),
                stuck=bool(parsed.get("stuck", False)),
                pedagogy_mode=PedagogyMode(parsed.get("pedagogy_mode", PedagogyMode.DIRECT.value))
            )
            return TutorOutput(
                answer=parsed.get("answer", raw_content),
                pedagogy_state=pedagogy_state
            )
        except Exception:
            # Fallback if raw text returned
            return TutorOutput(
                answer=raw_content,
                pedagogy_state=current_state or PedagogyState()
            )


class MockLLMClient(BaseLLMClient):
    """
    Deterministic mock client for offline unit testing and development.
    """

    def generate(
        self,
        system_prompt: str,
        messages: List[ChatMessage],
        current_state: Optional[PedagogyState] = None
    ) -> TutorOutput:
        last_user_msg = messages[-1].content if messages else ""
        lower_msg = last_user_msg.lower()

        # Deterministic simulation heuristics
        stuck = any(k in lower_msg for k in ["hint", "stuck", "don't understand", "dont get it", "confused", "help"])
        prev_hint = current_state.hint_level if current_state else 0
        new_hint = prev_hint + 1 if ("hint" in lower_msg or stuck) else prev_hint

        topic = "General Knowledge"
        if "supervised" in lower_msg:
            topic = "Supervised Learning"
        elif "gradient" in lower_msg:
            topic = "Gradient Descent"
        elif "neural" in lower_msg:
            topic = "Neural Networks"
        elif current_state and current_state.topic:
            topic = current_state.topic

        mode = current_state.pedagogy_mode if current_state else (
            PedagogyMode.SOCRATIC if "socratic" in system_prompt.lower() else PedagogyMode.DIRECT
        )
        if "weather" in lower_msg or "joke" in lower_msg:
            mode = PedagogyMode.OFF_TOPIC

        answer = f"Mock Tutor response to: '{last_user_msg}'. (Hint tier {new_hint})"

        return TutorOutput(
            answer=answer,
            pedagogy_state=PedagogyState(
                hint_level=new_hint,
                topic=topic,
                stuck=stuck,
                pedagogy_mode=mode
            )
        )
