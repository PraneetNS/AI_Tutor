"""
Unit tests for telemetry metrics collector and Prometheus format generation.
"""

from fastapi.testclient import TestClient
from ai_tutor.telemetry import MetricsCollector, metrics
from ai_tutor.api import create_app


def test_metrics_collector_counters():
    collector = MetricsCollector()
    collector.inc_counter("test_requests_total", 1.0)
    collector.inc_counter("test_requests_total", 2.0)
    
    text = collector.generate_prometheus_text()
    assert "test_requests_total 3.0" in text


def test_metrics_collector_labels():
    collector = MetricsCollector()
    collector.inc_counter("test_api_calls", 1.0, labels={"endpoint": "/chat", "status": "200"})
    
    text = collector.generate_prometheus_text()
    assert 'test_api_calls{endpoint="/chat",status="200"} 1.0' in text


def test_metrics_collector_latency():
    collector = MetricsCollector()
    collector.observe_latency("request_latency", 0.05)
    collector.observe_latency("request_latency", 0.15)
    
    text = collector.generate_prometheus_text()
    assert "request_latency_count 2" in text
    assert "request_latency_sum 0.200000" in text
    assert "request_latency_avg 0.100000" in text


def test_metrics_endpoint_fastapi():
    app = create_app()
    client = TestClient(app)
    
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "# TYPE" in response.text
