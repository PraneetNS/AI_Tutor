"""
Tests for learner state and mastery API endpoints.
"""

from fastapi.testclient import TestClient
from ai_tutor.api import create_app
from ai_tutor.learner_model import LearnerModelEngine
from ai_tutor.learner_store import InMemoryLearnerStateStore


def test_learner_mastery_empty_profile():
    store = InMemoryLearnerStateStore()
    engine = LearnerModelEngine(store=store)
    app = create_app(learner_engine=engine)
    client = TestClient(app)

    response = client.get("/api/learner/student_101/mastery")
    assert response.status_code == 200
    data = response.json()
    assert data["student_id"] == "student_101"
    assert data["mastery"] == {}
    assert data["behavior"]["engagement_score"] == 1.0


def test_learner_record_interaction_and_query():
    store = InMemoryLearnerStateStore()
    engine = LearnerModelEngine(store=store)
    app = create_app(learner_engine=engine)
    client = TestClient(app)

    # 1. Record correct interaction on 'variables'
    res = client.post("/api/learner/student_101/interaction", json={
        "concept": "variables",
        "correct": True,
        "hints_used": 0,
        "message": "x = 5 assigns 5 to x"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["concept"] == "variables"
    assert data["updated_mastery"] is not None

    # 2. Query mastery
    res_mastery = client.get("/api/learner/student_101/mastery")
    assert res_mastery.status_code == 200
    mastery_data = res_mastery.json()
    assert "variables" in mastery_data["mastery"]
    assert mastery_data["mastery"]["variables"]["mastery"] > 0.3



def test_learner_reset_state():
    store = InMemoryLearnerStateStore()
    engine = LearnerModelEngine(store=store)
    app = create_app(learner_engine=engine)
    client = TestClient(app)

    # Record interaction
    client.post("/api/learner/student_101/interaction", json={
        "concept": "loss_functions",
        "correct": True
    })

    # Reset
    res_reset = client.post("/api/learner/student_101/reset")
    assert res_reset.status_code == 200
    assert res_reset.json()["status"] == "reset"

    # Query again
    res_mastery = client.get("/api/learner/student_101/mastery")
    assert res_mastery.json()["mastery"] == {}
