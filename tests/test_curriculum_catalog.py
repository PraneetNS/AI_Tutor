"""
Unit tests for CurriculumCatalog and multi-domain graph retrieval.
"""

from fastapi.testclient import TestClient
from ai_tutor.curriculum_catalog import CurriculumCatalog
from ai_tutor.api import create_app


def test_curriculum_catalog_domains():
    domains = CurriculumCatalog.list_domains()
    assert "python_programming" in domains
    assert "data_structures" in domains


def test_curriculum_catalog_python_graph():
    graph = CurriculumCatalog.get_graph("python_programming")
    assert graph.get_concept("py_functions") is not None
    prereqs = graph.get_prerequisites("py_asyncio")
    prereq_ids = [p.concept_id for p in prereqs]
    assert "py_syntax" in prereq_ids or "py_generators" in prereq_ids


def test_curriculum_catalog_dsa_graph():
    graph = CurriculumCatalog.get_graph("data_structures")
    assert graph.get_concept("dsa_dp") is not None
    prereqs = graph.get_prerequisites("dsa_dp")
    prereq_ids = [p.concept_id for p in prereqs]
    assert "dsa_graphs" in prereq_ids or "dsa_graph_search" in prereq_ids


def test_api_multi_domain_concept_graph():
    app = create_app()
    client = TestClient(app)

    # 1. Default ML domain
    res_ml = client.get("/api/concept-graph")
    assert res_ml.status_code == 200
    assert res_ml.json()["domain"] == "machine_learning"

    # 2. Python programming domain
    res_py = client.get("/api/concept-graph?domain=python_programming")
    assert res_py.status_code == 200
    data_py = res_py.json()
    assert data_py["domain"] == "python_programming"
    assert any(n["id"] == "py_functions" for n in data_py["nodes"])

    # 3. Data structures domain
    res_dsa = client.get("/api/concept-graph?domain=data_structures")
    assert res_dsa.status_code == 200
    data_dsa = res_dsa.json()
    assert data_dsa["domain"] == "data_structures"
    assert any(n["id"] == "dsa_dp" for n in data_dsa["nodes"])
