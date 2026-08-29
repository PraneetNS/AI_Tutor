"""
Unit tests for AI Tutor CLI commands.
"""

from ai_tutor.cli import main


def test_cli_graph_info(capsys):
    ret = main(["graph-info"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Curriculum Graph" in captured.out
    assert "Total Concepts" in captured.out


def test_cli_export_graph_json(capsys):
    ret = main(["export-graph", "--format", "json"])
    assert ret == 0
    captured = capsys.readouterr()
    assert '"version": "1.0"' in captured.out
    assert '"nodes"' in captured.out


def test_cli_simulate_bkt(capsys):
    ret = main(["simulate-bkt", "--answers", "1,1,0,1"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Step 01 [Correct  ]" in captured.out
    assert "Step 03 [Incorrect]" in captured.out


def test_cli_eval_guardrail_clean(capsys):
    ret = main(["eval-guardrail", "--text", "A derivative is the rate of change of a function."])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Safe: True" in captured.out



def test_cli_ask(capsys):
    ret = main(["ask", "--question", "What is gradient descent?", "--course-id", "1", "--lecture-id", "1"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "=== AI Tutor Response ===" in captured.out
