"""
tests/test_api_endpoints.py
----------------------------
Integration tests for the FastAPI API endpoints:
- GET /api/concept-graph (ML curriculum graph)
- POST /api/ai/chat (End-to-end pedagogical chat processing)
- CORS headers
"""

import pytest
from fastapi.testclient import TestClient
from ai_tutor.api import create_app
from ai_tutor.pipeline import TutorPipeline
from ai_tutor.knowledge_source import MockKnowledgeSource
from ai_tutor.llm_client import MockLLMClient


@pytest.fixture
def client() -> TestClient:
    ks = MockKnowledgeSource()
    llm = MockLLMClient()
    pipeline = TutorPipeline(knowledge_source=ks, model_adapter=None)
    app = create_app(pipeline=pipeline)
    return TestClient(app)


def test_get_concept_graph_returns_nodes_and_edges(client: TestClient):
    response = client.get("/api/concept-graph")
    assert response.status_code == 200
    data = response.json()

    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) >= 15
    assert len(data["edges"]) >= 15

    # Check node fields
    node_0 = data["nodes"][0]
    assert "id" in node_0
    assert "name" in node_0
    assert "status" in node_0
    assert "mastery" in node_0
    assert node_0["status"] in ["mastered", "in_progress", "locked"]


def test_post_chat_processes_student_question(client: TestClient):
    payload = {
        "message": "Can you explain the chain rule for backpropagation?",
        "course_id": 1,
        "lecture_id": 1,
    }
    response = client.post("/api/ai/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "answer" in data
    assert "session_id" in data
    assert "pedagogy_mode" in data
    assert len(data["answer"]) > 5


def test_post_chat_handles_socratic_guidance(client: TestClient):
    payload = {
        "message": "Give me a hint for gradient descent.",
        "course_id": 1,
        "lecture_id": 1,
    }
    response = client.post("/api/ai/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["hint_level"] >= 1
