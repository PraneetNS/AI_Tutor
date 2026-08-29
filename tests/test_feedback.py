"""
Unit tests for learner feedback collector and analytics summary endpoints.
"""

from fastapi.testclient import TestClient
from ai_tutor.feedback_collector import FeedbackCollector, FeedbackItem
from ai_tutor.api import create_app


def test_feedback_collector_aggregation():
    collector = FeedbackCollector()
    collector.record(FeedbackItem(
        session_id="sess_1",
        student_id="student_1",
        rating=5,
        helpful=True,
        tags=["clear", "helpful_hint"]
    ))
    collector.record(FeedbackItem(
        session_id="sess_2",
        student_id="student_2",
        rating=3,
        helpful=False,
        tags=["too_fast"]
    ))

    summary = collector.get_summary()
    assert summary["total_count"] == 2
    assert summary["avg_rating"] == 4.0
    assert summary["helpful_percentage"] == 50.0
    assert summary["tag_counts"]["clear"] == 1
    assert summary["tag_counts"]["too_fast"] == 1


def test_api_feedback_endpoints():
    collector = FeedbackCollector()
    app = create_app(feedback_collector=collector)
    client = TestClient(app)

    # 1. Post feedback
    res = client.post("/api/feedback", json={
        "session_id": "sess_api_1",
        "student_id": "stu_99",
        "rating": 5,
        "helpful": True,
        "tags": ["great_analogy"],
        "comment": "The derivative hill-climbing visual was awesome."
    })
    assert res.status_code == 200
    assert res.json()["status"] == "success"

    # 2. Get summary
    res_summary = client.get("/api/feedback/summary")
    assert res_summary.status_code == 200
    data = res_summary.json()
    assert data["total_count"] == 1
    assert data["avg_rating"] == 5.0
    assert data["tag_counts"]["great_analogy"] == 1
