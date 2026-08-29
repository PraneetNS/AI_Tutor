"""
exporter.py
-----------
Utilities for exporting and importing curriculum dependency graphs,
learner transcripts, and mastery profiles into multiple industry standard formats
(JSON, Graphviz DOT, Cytoscape, Markdown).
"""

from __future__ import annotations

import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from .concept_graph import ConceptGraph, InMemoryConceptGraph
from .models import ConceptNode, ConceptPrerequisiteEdge, ChatMessage, LearnerState


class CurriculumExporter:
    """Exports and imports curriculum dependency DAGs."""

    @staticmethod
    def extract_nodes_and_edges(graph: ConceptGraph | InMemoryConceptGraph) -> tuple[List[ConceptNode], List[ConceptPrerequisiteEdge]]:
        """Extracts concept nodes and prerequisite edges from graph."""
        store = graph.store if isinstance(graph, ConceptGraph) else graph
        nodes = store.all_concepts()
        edges = []
        if isinstance(store, InMemoryConceptGraph):
            for concept_id, prereqs in store._prereq_edges.items():
                for prereq_id, weight in prereqs:
                    edges.append(ConceptPrerequisiteEdge(
                        concept_id=concept_id,
                        prerequisite_id=prereq_id,
                        weight=weight
                    ))
        return nodes, edges

    @classmethod
    def to_dict(cls, graph: ConceptGraph | InMemoryConceptGraph) -> Dict[str, Any]:
        """Serializes concept graph to dictionary representation."""
        nodes, edges = cls.extract_nodes_and_edges(graph)
        node_dicts = [
            {
                "concept_id": c.concept_id,
                "name": c.name,
                "domain": c.domain,
                "description": c.description
            }
            for c in nodes
        ]
        edge_dicts = [
            {
                "concept_id": e.concept_id,
                "prerequisite_id": e.prerequisite_id,
                "weight": e.weight
            }
            for e in edges
        ]
        return {
            "version": "1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "concepts_count": len(node_dicts),
            "prerequisites_count": len(edge_dicts),
            "nodes": node_dicts,
            "edges": edge_dicts,
        }

    @classmethod
    def to_json(cls, graph: ConceptGraph | InMemoryConceptGraph, indent: int = 2) -> str:
        """Serializes concept graph to formatted JSON string."""
        return json.dumps(cls.to_dict(graph), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConceptGraph:
        """Constructs ConceptGraph from dictionary representation."""
        concepts = [
            ConceptNode(
                concept_id=n["concept_id"],
                name=n["name"],
                domain=n.get("domain", "general"),
                description=n.get("description")
            )
            for n in data.get("nodes", [])
        ]
        edges = [
            ConceptPrerequisiteEdge(
                concept_id=e["concept_id"],
                prerequisite_id=e["prerequisite_id"],
                weight=e.get("weight", 1.0)
            )
            for e in data.get("edges", [])
        ]
        store = InMemoryConceptGraph(concepts=concepts, edges=edges)
        return ConceptGraph(store=store)

    @classmethod
    def from_json(cls, json_str: str) -> ConceptGraph:
        """Constructs ConceptGraph from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def to_dot(cls, graph: ConceptGraph | InMemoryConceptGraph, graph_name: str = "Curriculum") -> str:
        """Serializes concept graph to Graphviz DOT syntax."""
        nodes, edges = cls.extract_nodes_and_edges(graph)
        lines = [f'digraph "{graph_name}" {{', "  rankdir=LR;", "  node [shape=box, style=rounded];"]
        for c in nodes:
            lines.append(f'  "{c.concept_id}" [label="{c.name}"];')
        for e in edges:
            lines.append(f'  "{e.prerequisite_id}" -> "{e.concept_id}" [label="{e.weight}"];')
        lines.append("}")
        return "\n".join(lines)

    @classmethod
    def to_cytoscape(cls, graph: ConceptGraph | InMemoryConceptGraph) -> List[Dict[str, Any]]:
        """Converts graph to Cytoscape.js elements array."""
        nodes, edges = cls.extract_nodes_and_edges(graph)
        elements = []
        for c in nodes:
            elements.append({
                "data": {
                    "id": c.concept_id,
                    "label": c.name,
                    "domain": c.domain
                }
            })
        for idx, e in enumerate(edges):
            elements.append({
                "data": {
                    "id": f"edge_{e.prerequisite_id}_{e.concept_id}_{idx}",
                    "source": e.prerequisite_id,
                    "target": e.concept_id,
                    "weight": e.weight
                }
            })
        return elements


class TranscriptExporter:
    """Exports chat messages and learner summaries to JSON / Markdown."""

    @staticmethod
    def to_markdown(messages: List[ChatMessage], title: str = "Tutoring Session Transcript") -> str:
        """Generates formatted markdown transcript."""
        lines = [
            f"# {title}",
            f"*Generated on: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*",
            "",
            "---",
            ""
        ]
        for msg in messages:
            role_title = msg.role.value.capitalize() if hasattr(msg.role, "value") else str(msg.role).capitalize()
            ts = f" *({msg.timestamp})*" if msg.timestamp else ""
            lines.append(f"### {role_title}{ts}")
            lines.append("")
            lines.append(msg.content)
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def to_json(messages: List[ChatMessage], anonymize: bool = False) -> str:
        """Serializes chat history to JSON."""
        data = []
        for m in messages:
            role_val = m.role.value if hasattr(m.role, "value") else str(m.role)
            item = {
                "role": role_val,
                "content": m.content,
                "timestamp": m.timestamp
            }
            if anonymize and role_val == "user":
                item["anonymized"] = True
            data.append(item)
        return json.dumps(data, indent=2)
