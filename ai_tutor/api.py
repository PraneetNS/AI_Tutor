from fastapi import FastAPI, Depends, HTTPException, Response, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List
import os
import time

from .models import AIChatRequest, AIChatResponse, LearningEvent, LearningEventType
from .pipeline import TutorPipeline
from .knowledge_source import MockKnowledgeSource
from .llm_client import MockLLMClient, OpenAILLMClient
from .concept_graph import create_ml_concept_graph, ML_CONCEPTS, ML_EDGES
from .learner_model import LearnerModelEngine
from .learner_store import InMemoryLearnerStateStore
from .telemetry import metrics


def create_app(
    pipeline: Optional[TutorPipeline] = None,
    learner_engine: Optional[LearnerModelEngine] = None
) -> FastAPI:
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

    if learner_engine is None:
        learner_engine = LearnerModelEngine(store=InMemoryLearnerStateStore())

    ml_graph = create_ml_concept_graph()

    def get_pipeline() -> TutorPipeline:
        return pipeline

    def get_learner_engine() -> LearnerModelEngine:
        return learner_engine

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

    @app.get(
        "/api/learner/{student_id}/mastery",
        summary="Get Learner Profile & Mastery",
        description="Returns current Bayesian Knowledge Tracing mastery scores, misconceptions, and behavior profile."
    )
    def get_learner_mastery(
        student_id: str,
        engine: LearnerModelEngine = Depends(get_learner_engine)
    ) -> Dict[str, Any]:
        metrics.inc_counter("http_requests_total", labels={"endpoint": "/api/learner/mastery"})
        state = engine.get_learner_state(student_id)
        if not state:
            return {
                "student_id": student_id,
                "concept_mastery": {},
                "mastery": {},
                "misconceptions": [],
                "behavior": {
                    "hints_used_total": 0,
                    "engagement_score": 1.0,
                    "avg_persistence": 0.0
                }
            }

        mastery_dict = {k: v.model_dump() for k, v in state.concept_mastery.items()}
        return {
            "student_id": state.student_id,
            "concept_mastery": mastery_dict,
            "mastery": mastery_dict,
            "misconceptions": [m.model_dump() for m in state.misconceptions],
            "behavior": state.behavior.model_dump(),
            "updated_at": state.updated_at
        }

    @app.post(
        "/api/learner/{student_id}/interaction",
        summary="Record Learning Interaction",
        description="Records a learning event, updating BKT probability of mastery and detecting misconceptions."
    )
    def record_interaction(
        student_id: str,
        payload: Dict[str, Any] = Body(...),
        engine: LearnerModelEngine = Depends(get_learner_engine)
    ) -> Dict[str, Any]:
        metrics.inc_counter("http_requests_total", labels={"endpoint": "/api/learner/interaction"})
        concept = payload.get("concept", "general")
        correct = payload.get("correct", True)
        hints_used = payload.get("hints_used", 0)
        message = payload.get("message", "")

        event = LearningEvent(
            student_id=student_id,
            event_type=LearningEventType.ANSWER_SUBMITTED,
            concept=concept,
            payload={
                "concept": concept,
                "correct": correct,
                "hints_used": hints_used,
                "message": message,
                "response": message
            }
        )

        state = engine.process_event(event)
        updated_mastery = (
            state.concept_mastery[concept].mastery 
            if state and concept in state.concept_mastery 
            else None
        )
        return {
            "status": "success",
            "student_id": student_id,
            "concept": concept,
            "updated_mastery": updated_mastery
        }



    @app.post(
        "/api/learner/{student_id}/reset",
        summary="Reset Learner State",
        description="Clears all mastery, history, and behavioral stats for a learner."
    )
    def reset_learner_state(
        student_id: str,
        engine: LearnerModelEngine = Depends(get_learner_engine)
    ) -> Dict[str, Any]:
        metrics.inc_counter("http_requests_total", labels={"endpoint": "/api/learner/reset"})
        if hasattr(engine.store, "clear"):
            engine.store.clear()
        return {"status": "reset", "student_id": student_id}

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
            mode_str = res.pedagogy_mode.value if hasattr(res.pedagogy_mode, "value") else str(res.pedagogy_mode or "direct")
            metrics.inc_counter("chat_turns_total", labels={"pedagogy_mode": mode_str})
            return res

        except Exception as e:
            metrics.inc_counter("http_errors_total", labels={"endpoint": "/api/ai/chat"})
            raise HTTPException(status_code=500, detail=str(e))

    return app


# Default app instance
app = create_app()

