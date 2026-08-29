from fastapi import FastAPI, Depends, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List
import os
import time

from .models import AIChatRequest, AIChatResponse
from .pipeline import TutorPipeline
from .knowledge_source import MockKnowledgeSource
from .llm_client import MockLLMClient, OpenAILLMClient
from .concept_graph import create_ml_concept_graph, ML_CONCEPTS, ML_EDGES
from .telemetry import metrics


def create_app(pipeline: Optional[TutorPipeline] = None) -> FastAPI:
    app = FastAPI(
        title="AI Tutor Service",
        description="Stateless pedagogical tutoring service with hybrid knowledge retrieval & 3D visualization",
        version="1.0.0"
    )

    # Enable CORS for Vite frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if pipeline is None:
        llm = OpenAILLMClient() if os.getenv("OPENAI_API_KEY") else MockLLMClient()
        ks = MockKnowledgeSource()
        pipeline = TutorPipeline(
            knowledge_source=ks,
            model_adapter=None
        )

    ml_graph = create_ml_concept_graph()

    def get_pipeline() -> TutorPipeline:
        return pipeline

    @app.get(
        "/metrics",
        summary="Prometheus Metrics Endpoint",
        description="Returns Prometheus-formatted metrics on request latencies, counts, and guardrail stats."
    )
    def get_metrics():
        return Response(content=metrics.generate_prometheus_text(), media_type="text/plain; version=0.0.4")

    @app.get(
        "/api/concept-graph",
        summary="Get Knowledge Constellation Concept Graph",
        description="Returns curriculum nodes (mastered, in_progress, locked) and prerequisite edges."
    )
    def get_concept_graph() -> Dict[str, Any]:
        metrics.inc_counter("http_requests_total", labels={"endpoint": "/api/concept-graph"})
        # Formats the nodes and edges for KnowledgeConstellation
        nodes = []
        for c in ML_CONCEPTS:
            nodes.append({
                "id": c.concept_id,
                "name": c.name,
                "domain": c.domain,
                "status": "mastered" if c.concept_id in ["variables", "expressions", "functions", "linear_algebra"]
                          else "in_progress" if c.concept_id in ["loss_functions", "gradient_descent", "backpropagation"]
                          else "locked",
                "mastery": 0.9 if c.concept_id in ["variables", "expressions", "functions", "linear_algebra"]
                           else 0.55 if c.concept_id in ["loss_functions", "gradient_descent", "backpropagation"]
                           else 0.1,
            })

        edges = [
            {"source": e.prerequisite_id, "target": e.concept_id, "weight": e.weight}
            for e in ML_EDGES
        ]

        return {
            "domain": "machine_learning",
            "nodes": nodes,
            "edges": edges,
        }

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
        start_time = time.time()
        metrics.inc_counter("http_requests_total", labels={"endpoint": "/api/ai/chat"})
        try:
            res = pipe.process(request)
            duration = time.time() - start_time
            metrics.observe_latency("http_request_duration_seconds", duration, labels={"endpoint": "/api/ai/chat"})
            metrics.inc_counter("chat_turns_total", labels={"scaffolding": res.scaffolding_level.value})
            return res
        except Exception as e:
            metrics.inc_counter("http_errors_total", labels={"endpoint": "/api/ai/chat"})
            raise HTTPException(status_code=500, detail=str(e))

    return app


# Default app instance
app = create_app()

