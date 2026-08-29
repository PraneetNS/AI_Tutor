"""
Unit tests for curriculum graph export and chat transcript export.
"""

import json
from ai_tutor.concept_graph import create_ml_concept_graph
from ai_tutor.exporter import CurriculumExporter, TranscriptExporter
from ai_tutor.models import ChatMessage, Role


def test_curriculum_export_to_json_and_roundtrip():
    graph = create_ml_concept_graph()
    json_str = CurriculumExporter.to_json(graph)
    
    assert "machine_learning" in json_str or "linear_algebra" in json_str
    
    # Round-trip import
    imported_graph = CurriculumExporter.from_json(json_str)
    assert len(imported_graph.store.all_concepts()) == len(graph.store.all_concepts())
    assert imported_graph.get_concept("backpropagation") is not None


def test_curriculum_export_to_dot():
    graph = create_ml_concept_graph()
    dot_str = CurriculumExporter.to_dot(graph, graph_name="ML_Curriculum")
    
    assert 'digraph "ML_Curriculum"' in dot_str
    assert "->" in dot_str
    assert "gradient_descent" in dot_str


def test_curriculum_export_to_cytoscape():
    graph = create_ml_concept_graph()
    cy_elements = CurriculumExporter.to_cytoscape(graph)
    
    assert len(cy_elements) > 20
    assert any("source" in el["data"] for el in cy_elements)
    assert any("id" in el["data"] for el in cy_elements)


def test_transcript_export_markdown_and_json():
    messages = [
        ChatMessage(role=Role.USER, content="Can you explain what loss functions are?"),
        ChatMessage(role=Role.ASSISTANT, content="A loss function measures the discrepancy between predictions and targets.")
    ]
    
    md = TranscriptExporter.to_markdown(messages, title="Loss Function Session")
    assert "# Loss Function Session" in md
    assert "### User" in md
    assert "### Assistant" in md
    assert "Can you explain what loss functions are?" in md
    
    json_out = TranscriptExporter.to_json(messages, anonymize=True)
    parsed = json.loads(json_out)
    assert len(parsed) == 2
    assert parsed[0]["anonymized"] is True
