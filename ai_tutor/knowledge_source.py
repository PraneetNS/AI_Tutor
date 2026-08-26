import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .models import Chunk


class KnowledgeSource(ABC):
    """
    Abstract interface for all Knowledge Retrieval backends.
    The Tutor Core calls this interface exclusively and never touches
    underlying LMS databases, vector stores, or APIs directly.
    """

    @abstractmethod
    def retrieve(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Retrieve relevant knowledge chunks matching the query and metadata filters.

        :param query: The search query or student question.
        :param filters: Optional key-value constraints (e.g., {'course_id': 101, 'lecture_id': 50}).
        :return: List of Chunk objects sorted by relevance.
        """
        pass


# Static LMS Course Data Fixtures (Course > Lesson > Lecture > Chunks)
MOCK_LMS_FIXTURES: List[Dict[str, Any]] = [
    {
        "course_id": 101,
        "course_title": "Machine Learning",
        "lesson_id": 10,
        "lesson_name": "ML Basics",
        "lecture_id": 50,
        "lecture_title": "Supervised Learning",
        "chunk_id": "chunk_101_50_1",
        "content": "Supervised learning is a machine learning paradigm where models are trained on labeled data pairs (x, y). The objective is to learn a mapping function f(x) ≈ y that generalizes to unseen test examples. Examples include linear regression for continuous targets and logistic regression or decision trees for discrete classification.",
        "type": "video"
    },
    {
        "course_id": 101,
        "course_title": "Machine Learning",
        "lesson_id": 10,
        "lesson_name": "ML Basics",
        "lecture_id": 50,
        "lecture_title": "Supervised Learning",
        "chunk_id": "chunk_101_50_2",
        "content": "In supervised learning, the model makes predictions on training instances and evaluates errors using a loss function like Mean Squared Error (MSE) for regression or Cross-Entropy for classification. Optimization algorithms adjust weights to minimize this loss.",
        "type": "video"
    },
    {
        "course_id": 101,
        "course_title": "Machine Learning",
        "lesson_id": 10,
        "lesson_name": "ML Basics",
        "lecture_id": 51,
        "lecture_title": "Unsupervised Learning",
        "chunk_id": "chunk_101_51_1",
        "content": "Unsupervised learning operates on unlabeled data {x_i}. The algorithm discovers hidden structure, clusters, or lower-dimensional representations without explicit target outputs. Key algorithms include K-Means clustering, Hierarchical clustering, and Principal Component Analysis (PCA).",
        "type": "video"
    },
    {
        "course_id": 101,
        "course_title": "Machine Learning",
        "lesson_id": 20,
        "lesson_name": "Optimization & Model Training",
        "lecture_id": 60,
        "lecture_title": "Gradient Descent & Cost Functions",
        "chunk_id": "chunk_101_60_1",
        "content": "Gradient descent is an iterative first-order optimization algorithm used to minimize a differentiable cost function J(θ). The update rule is θ ← θ - α ∇J(θ), where α is the learning rate. If α is too high, gradient descent may oscillate or diverge; if too low, convergence is slow.",
        "type": "video"
    },
    {
        "course_id": 101,
        "course_title": "Machine Learning",
        "lesson_id": 20,
        "lesson_name": "Optimization & Model Training",
        "lecture_id": 60,
        "lecture_title": "Gradient Descent & Cost Functions",
        "chunk_id": "chunk_101_60_2",
        "content": "Batch gradient descent calculates gradients over the entire dataset, Stochastic Gradient Descent (SGD) uses one sample per step, and Mini-batch SGD strikes a balance by computing gradients over small batches (e.g., 32 or 64 samples).",
        "type": "video"
    }
]


class MockKnowledgeSource(KnowledgeSource):
    """
    Mock implementation of KnowledgeSource backed by static LMS fixtures.
    Supports filtering by course_id, lecture_id, lesson_id, and lexical scoring.
    """

    def __init__(self, fixtures: Optional[List[Dict[str, Any]]] = None):
        self.fixtures = fixtures if fixtures is not None else MOCK_LMS_FIXTURES

    def retrieve(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        filters = filters or {}
        course_id_filter = filters.get("course_id")
        lecture_id_filter = filters.get("lecture_id")
        lesson_id_filter = filters.get("lesson_id")
        top_k = filters.get("top_k", 3)

        query_tokens = set(re.findall(r"\w+", query.lower()))

        scored_chunks = []

        for item in self.fixtures:
            # Apply exact filters if present
            if course_id_filter is not None and item.get("course_id") != course_id_filter:
                continue
            if lecture_id_filter is not None and item.get("lecture_id") != lecture_id_filter:
                continue
            if lesson_id_filter is not None and item.get("lesson_id") != lesson_id_filter:
                continue

            # Lexical overlap score
            content_tokens = set(re.findall(r"\w+", item["content"].lower()))
            title_tokens = set(re.findall(r"\w+", item["lecture_title"].lower()))

            overlap = len(query_tokens.intersection(content_tokens)) + (2 * len(query_tokens.intersection(title_tokens)))
            score = round(overlap / (len(query_tokens) + 1e-5), 3) if query_tokens else 0.5

            chunk = Chunk(
                content=item["content"],
                source_title=item["lecture_title"],
                source_id=item["lecture_id"],
                metadata={
                    "course_id": item.get("course_id"),
                    "course_title": item.get("course_title"),
                    "lesson_id": item.get("lesson_id"),
                    "lesson_name": item.get("lesson_name"),
                    "lecture_id": item.get("lecture_id"),
                    "chunk_id": item.get("chunk_id"),
                    "relevance_score": score
                }
            )
            scored_chunks.append((score, chunk))

        # Sort by relevance score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        # Return top_k matching chunks
        return [c for _, c in scored_chunks[:top_k]]
