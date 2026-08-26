from ai_tutor import (
    IntentClassifier,
    ReviewLogger,
    ChatMessage,
    Role
)

def run_classifier_demo():
    print("=" * 65)
    print(" LIGHTWEIGHT INTENT CLASSIFIER DEMO (Rules + Cheap Model + Logging)")
    print("=" * 65)

    # Initialize review logger writing to 'logs/classifier_review.jsonl'
    logger = ReviewLogger(log_filepath="logs/classifier_review.jsonl")

    # Initialize hybrid classifier
    classifier = IntentClassifier(
        model="gpt-4o-mini",
        confidence_threshold=0.70,
        review_logger=logger
    )

    test_queries = [
        ("Why does L1 regularization cause sparsity while L2 doesn't?", "Deep Concept Question"),
        ("What is the formula for Mean Squared Error?", "Factual / Syntax Lookup"),
        ("What's the weather like in Paris right now?", "Off-Topic Query (Should log)"),
        ("tell me a joke about robots", "Off-Topic Chitchat (Should log)"),
        ("maybe 10 or 20?", "Ambiguous / Low Confidence (Should log)"),
        ("How does the learning rate affect gradient descent oscillations?", "Conceptual Trade-off")
    ]

    session_id = "demo_session_456"

    for query, desc in test_queries:
        print(f"\n[Scenario] {desc}")
        print(f"Student:  '{query}'")

        res = classifier.classify(
            student_message=query,
            session_id=session_id
        )

        flag_indicator = " [FLAGGED & LOGGED FOR REVIEW]" if res.flagged_for_review else ""
        print(f"Result:   Label={res.label.value} | Confidence={res.confidence:.2f}{flag_indicator}")
        print(f"Details:  {res.rationale}")

    print("\nReview log written to: logs/classifier_review.jsonl")

if __name__ == "__main__":
    run_classifier_demo()
