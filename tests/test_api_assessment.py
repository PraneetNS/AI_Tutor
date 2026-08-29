"""
Tests for quiz generation and assessment endpoints.
"""

from fastapi.testclient import TestClient
from ai_tutor.api import create_app
from ai_tutor.quiz_agent import QuizAgent
from ai_tutor.assessment_agent import AssessmentAgent
from ai_tutor.llm_client import MockLLMClient


def test_generate_quiz_endpoint():
    llm = MockLLMClient()
    quiz_agent = QuizAgent(llm_client=llm)
    app = create_app(quiz_agent=quiz_agent)
    client = TestClient(app)

    response = client.post("/api/assessment/generate-quiz", json={
        "concept": "Loss Functions",
        "difficulty": "medium"
    })
    assert response.status_code == 200
    data = response.json()
    assert "question" in data
    assert "concept" in data
    assert len(data["question"]) > 0



def test_grade_answer_endpoint_correct():
    assessment_agent = AssessmentAgent()
    app = create_app(assessment_agent=assessment_agent)
    client = TestClient(app)

    response = client.post("/api/assessment/grade-answer", json={
        "student_id": "student_42",
        "concept": "Loss Functions",
        "answer": "Loss functions quantify the difference between predicted and actual values.",
        "hints_used": 0
    })
    assert response.status_code == 200
    data = response.json()
    assert "correct" in data
    assert "feedback" in data
    assert "misconception_detected" in data
