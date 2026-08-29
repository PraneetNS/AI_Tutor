"""
curriculum_catalog.py
---------------------
Curriculum catalog repository with prerequisite dependency DAGs for:
- Machine Learning (Standard)
- Python Programming
- Data Structures & Algorithms
- Linear Algebra & Mathematics
"""

from __future__ import annotations

from typing import Dict, List
from .concept_graph import ConceptGraph, InMemoryConceptGraph
from .models import ConceptNode, ConceptPrerequisiteEdge


# =====================================================================
# 1. PYTHON PROGRAMMING CURRICULUM
# =====================================================================
PYTHON_CONCEPTS = [
    ConceptNode(concept_id="py_syntax", name="Syntax & Variables", domain="python_programming", description="Basic syntax, variable assignment, primitive types"),
    ConceptNode(concept_id="py_control_flow", name="Control Flow (If/Loops)", domain="python_programming", description="Conditionals, while loops, for-in iterations"),
    ConceptNode(concept_id="py_functions", name="Functions & Scope", domain="python_programming", description="Function definitions, parameters, return values, LEGB scoping"),
    ConceptNode(concept_id="py_data_structures", name="Lists, Dictionaries & Sets", domain="python_programming", description="Built-in collection types, indexing, slicing, comprehensions"),
    ConceptNode(concept_id="py_oop_basics", name="OOP Basics (Classes & Objects)", domain="python_programming", description="Class definitions, constructors (__init__), instance methods"),
    ConceptNode(concept_id="py_inheritance", name="Inheritance & Polymorphism", domain="python_programming", description="Subclassing, super(), method overriding, abstract base classes"),
    ConceptNode(concept_id="py_exceptions", name="Exception Handling", domain="python_programming", description="Try/except/finally blocks, custom exception classes"),
    ConceptNode(concept_id="py_generators", name="Iterators & Generators", domain="python_programming", description="Iterable protocol, yield keyword, generator expressions"),
    ConceptNode(concept_id="py_decorators", name="Decorators & Closures", domain="python_programming", description="First-class functions, closures, function & class decorators"),
    ConceptNode(concept_id="py_asyncio", name="Asynchronous Programming (asyncio)", domain="python_programming", description="Coroutines, event loops, async/await, Task concurrency"),
]

PYTHON_EDGES = [
    ConceptPrerequisiteEdge(concept_id="py_control_flow", prerequisite_id="py_syntax", weight=1.0),
    ConceptPrerequisiteEdge(concept_id="py_functions", prerequisite_id="py_control_flow", weight=1.0),
    ConceptPrerequisiteEdge(concept_id="py_data_structures", prerequisite_id="py_control_flow", weight=1.0),
    ConceptPrerequisiteEdge(concept_id="py_oop_basics", prerequisite_id="py_functions", weight=1.0),
    ConceptPrerequisiteEdge(concept_id="py_oop_basics", prerequisite_id="py_data_structures", weight=0.8),
    ConceptPrerequisiteEdge(concept_id="py_inheritance", prerequisite_id="py_oop_basics", weight=1.0),
    ConceptPrerequisiteEdge(concept_id="py_exceptions", prerequisite_id="py_functions", weight=0.9),
    ConceptPrerequisiteEdge(concept_id="py_generators", prerequisite_id="py_functions", weight=1.0),
    ConceptPrerequisiteEdge(concept_id="py_generators", prerequisite_id="py_data_structures", weight=0.8),
    ConceptPrerequisiteEdge(concept_id="py_decorators", prerequisite_id="py_functions", weight=1.0),
    ConceptPrerequisiteEdge(concept_id="py_asyncio", prerequisite_id="py_generators", weight=1.0),
]


# =====================================================================
# 2. DATA STRUCTURES & ALGORITHMS CURRICULUM
# =====================================================================
DSA_CONCEPTS = [
    ConceptNode(concept_id="dsa_arrays", name="Arrays & Memory Layout", domain="data_structures", description="Contiguous memory, O(1) random access, resizing overhead"),
    ConceptNode(concept_id="dsa_linked_lists", name="Linked Lists", domain="data_structures", description="Singly and doubly linked node structures, pointer manipulation"),
    ConceptNode(concept_id="dsa_stacks_queues", name="Stacks & Queues", domain="data_structures", description="LIFO/FIFO abstractions, deque implementations"),
    ConceptNode(concept_id="dsa_binary_trees", name="Binary Trees & Traversals", domain="data_structures", description="Hierarchical nodes, In-order, Pre-order, Post-order, BFS"),
    ConceptNode(concept_id="dsa_bst", name="Binary Search Trees", domain="data_structures", description="BST property, search, insertion, deletion, balancing intuition"),
    ConceptNode(concept_id="dsa_heaps", name="Heaps & Priority Queues", domain="data_structures", description="Binary min/max heaps, heapify, O(log N) priority operations"),
    ConceptNode(concept_id="dsa_graphs", name="Graph Representations (Adj List/Matrix)", domain="data_structures", description="Vertices, directed/undirected edges, adjacency representations"),
    ConceptNode(concept_id="dsa_graph_search", name="Graph Search (BFS / DFS)", domain="data_structures", description="Shortest path on unweighted graphs, cycle detection, topological sort"),
    ConceptNode(concept_id="dsa_dp", name="Dynamic Programming", domain="data_structures", description="Optimal substructure, overlapping subproblems, memoization vs tabulation"),
]

DSA_EDGES = [
    ConceptPrerequisiteEdge(concept_id="dsa_linked_lists", prerequisite_id="dsa_arrays", weight=0.9),
    ConceptPrerequisiteEdge(concept_id="dsa_stacks_queues", prerequisite_id="dsa_arrays", weight=0.8),
    ConceptPrerequisiteEdge(concept_id="dsa_stacks_queues", prerequisite_id="dsa_linked_lists", weight=0.8),
    ConceptPrerequisiteEdge(concept_id="dsa_binary_trees", prerequisite_id="dsa_linked_lists", weight=1.0),
    ConceptPrerequisiteEdge(concept_id="dsa_bst", prerequisite_id="dsa_binary_trees", weight=1.0),
    ConceptPrerequisiteEdge(concept_id="dsa_heaps", prerequisite_id="dsa_arrays", weight=0.9),
    ConceptPrerequisiteEdge(concept_id="dsa_heaps", prerequisite_id="dsa_binary_trees", weight=0.8),
    ConceptPrerequisiteEdge(concept_id="dsa_graphs", prerequisite_id="dsa_linked_lists", weight=0.9),
    ConceptPrerequisiteEdge(concept_id="dsa_graph_search", prerequisite_id="dsa_graphs", weight=1.0),
    ConceptPrerequisiteEdge(concept_id="dsa_graph_search", prerequisite_id="dsa_stacks_queues", weight=0.9),
    ConceptPrerequisiteEdge(concept_id="dsa_dp", prerequisite_id="dsa_graph_search", weight=0.8),
]


class CurriculumCatalog:
    """Catalog manager indexing all available subject curriculum graphs."""

    _DOMAINS: Dict[str, tuple[List[ConceptNode], List[ConceptPrerequisiteEdge]]] = {
        "python_programming": (PYTHON_CONCEPTS, PYTHON_EDGES),
        "data_structures": (DSA_CONCEPTS, DSA_EDGES),
    }

    @classmethod
    def list_domains(cls) -> List[str]:
        return list(cls._DOMAINS.keys())

    @classmethod
    def get_graph(cls, domain: str) -> ConceptGraph:
        if domain not in cls._DOMAINS:
            raise KeyError(f"Domain '{domain}' not found in catalog. Available: {cls.list_domains()}")
        concepts, edges = cls._DOMAINS[domain]
        store = InMemoryConceptGraph(concepts=concepts, edges=edges)
        return ConceptGraph(store=store)

    @classmethod
    def get_concepts_and_edges(cls, domain: str) -> tuple[List[ConceptNode], List[ConceptPrerequisiteEdge]]:
        if domain not in cls._DOMAINS:
            raise KeyError(f"Domain '{domain}' not found in catalog. Available: {cls.list_domains()}")
        return cls._DOMAINS[domain]
