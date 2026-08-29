"""
Unit tests for ExactResponseCache and SemanticCache.
"""

import time
from ai_tutor.semantic_cache import ExactResponseCache, SemanticCache
from ai_tutor.models import AIChatResponse, PedagogyMode


def test_exact_cache_hit_and_miss():
    cache = ExactResponseCache(max_size=5, ttl_seconds=60)
    
    resp1 = AIChatResponse(
        answer="A loss function measures error.",
        pedagogy_mode=PedagogyMode.DIRECT
    )
    
    # 1. Miss initially
    assert cache.get("What is a loss function?", course_id=1) is None
    assert cache.misses == 1
    assert cache.hits == 0
    
    # 2. Set
    cache.set("What is a loss function?", resp1, course_id=1)
    
    # 3. Hit (case and whitespace insensitive)
    cached = cache.get("  what is a loss function?  ", course_id=1)
    assert cached is not None
    assert cached.answer == "A loss function measures error."
    assert cache.hits == 1


def test_exact_cache_ttl_expiration():
    cache = ExactResponseCache(max_size=5, ttl_seconds=1)
    resp = AIChatResponse(answer="Quick answer", pedagogy_mode=PedagogyMode.DIRECT)
    
    cache.set("Hello", resp)
    assert cache.get("Hello") is not None
    
    time.sleep(1.1)
    assert cache.get("Hello") is None


def test_semantic_cache_fuzzy_match():
    cache = SemanticCache(similarity_threshold=0.6, ttl_seconds=60)
    resp = AIChatResponse(answer="Gradient descent optimizes weights iteratively.", pedagogy_mode=PedagogyMode.DIRECT)
    
    cache.set("Can you explain how gradient descent works in optimization?", resp)
    
    # Similar query
    matched = cache.get("Explain how gradient descent works in optimization algorithms")
    assert matched is not None
    assert matched.answer == "Gradient descent optimizes weights iteratively."
    
    # Unrelated query
    unrelated = cache.get("What is an operating system kernel process?")
    assert unrelated is None
