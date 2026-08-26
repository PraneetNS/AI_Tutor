import json
import pytest
from pathlib import Path
from ai_tutor import (
    IntentClassifier,
    FastRuleMatcher,
    IntentLabel,
    ReviewLogger,
    ChatMessage,
    Role
)


def test_fast_rule_matcher():
    # CONCEPT
    res_concept = FastRuleMatcher.match("Why does backpropagation calculate gradients backwards?")
    assert res_concept is not None
    label, conf, _ = res_concept
    assert label == IntentLabel.CONCEPT
    assert conf >= 0.85

    # FACTUAL
    res_factual = FastRuleMatcher.match("What is the formula for Cross Entropy Loss?")
    assert res_factual is not None
    label, conf, _ = res_factual
    assert label == IntentLabel.FACTUAL
    assert conf >= 0.85

    # OFF_TOPIC
    res_offtopic = FastRuleMatcher.match("Tell me a funny joke about cats")
    assert res_offtopic is not None
    label, conf, _ = res_offtopic
    assert label == IntentLabel.OFF_TOPIC
    assert conf >= 0.90


def test_classifier_and_review_logger(tmp_path):
    log_file = tmp_path / "test_review_logs.jsonl"
    logger = ReviewLogger(log_filepath=str(log_file))

    classifier = IntentClassifier(
        confidence_threshold=0.70,
        use_rules=True,
        review_logger=logger
    )

    # 1. Normal CONCEPT query -> Should NOT be flagged for review
    res1 = classifier.classify(
        student_message="Why is L1 regularization better for sparse feature selection?",
        session_id="sess_101"
    )
    assert res1.label == IntentLabel.CONCEPT
    assert res1.flagged_for_review is False

    # 2. OFF_TOPIC query -> MUST be flagged for review & logged
    res2 = classifier.classify(
        student_message="What's the weather in Seattle tomorrow?",
        session_id="sess_101"
    )
    assert res2.label == IntentLabel.OFF_TOPIC
    assert res2.flagged_for_review is True

    # 3. Ambiguous low-confidence query (force fallback or ambiguous input) -> MUST be flagged & logged
    res3 = classifier.classify(
        student_message="hmm maybe blue or 42",
        session_id="sess_101",
        force_model=False
    )
    assert res3.confidence < 0.70
    assert res3.flagged_for_review is True

    # Verify log file contents
    assert log_file.exists()
    with open(log_file, "r", encoding="utf-8") as f:
        log_lines = [json.loads(line) for line in f if line.strip()]

    assert len(log_lines) >= 2
    # Check that reasons are recorded
    reasons = [item["flag_reason"] for item in log_lines]
    assert any("OFF_TOPIC" in r for r in reasons)
    assert any("LOW_CONFIDENCE" in r for r in reasons)
