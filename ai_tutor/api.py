from fastapi import FastAPI, Depends, HTTPException
from typing import Optional
from .models import AIChatRequest, AIChatResponse
from .pipeline import TutorPipeline
from .knowledge_source import MockKnowledgeSource
from .llm_client import MockLLMClient, OpenAILLMClient
import os


def create_app(pipeline: Optional[TutorPipeline] = None) -> FastAPI:
    app = FastAPI(
        title="AI Tutor Service",
        description="Stateless pedagogical tutoring service with hybrid knowledge retrieval",
        version="1.0.0"
    )

    if pipeline is None:
        # Default production/dev pipeline wiring
        llm = OpenAILLMClient() if os.getenv("OPENAI_API_KEY") else MockLLMClient()
        ks = MockKnowledgeSource()
        pipeline = TutorPipeline(
            knowledge_source=ks,
            model_adapter=None  # uses default with llm
        )

    # Dependency helper
    def get_pipeline() -> TutorPipeline:
        return pipeline

    @app.post(
        "/api/ai/chat",
        response_model=AIChatResponse,
        summary="AI Chat Endpoint",
        description="Processes student question through the 6-stage pedagogical pipeline."
    )
    def chat_endpoint(
        request: AIChatRequest,
        pipe: TutorPipeline = Depends(get_pipeline)
    ) -> AIChatResponse:
        try:
            return pipe.process(request)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


# Default app instance
app = create_app()
