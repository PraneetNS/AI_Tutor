"""
Prometheus and OpenTelemetry-compatible metrics instrumentation for AI Tutor service.
Tracks request latencies, token consumption, guardrail triggers, and pipeline stage execution times.
"""

from typing import Dict, Any, Optional
import time
import threading
from collections import defaultdict


class MetricsCollector:
    """Thread-safe in-memory Prometheus-style metrics collector."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._counters: Dict[str, float] = defaultdict(float)
        self._labeled_counters: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._histograms: Dict[str, list] = defaultdict(list)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._gauges["tutor_service_up"] = 1.0

        
    def inc_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        with self._lock:
            if labels:
                label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
                self._labeled_counters[name][label_str] += value
            else:
                self._counters[name] += value

    def observe_latency(self, name: str, duration_sec: float, labels: Optional[Dict[str, str]] = None):
        with self._lock:
            self._histograms[name].append(duration_sec)
            # Keep last 10,000 observations to avoid memory bloat
            if len(self._histograms[name]) > 10000:
                self._histograms[name] = self._histograms[name][-5000:]

    def set_gauge(self, name: str, value: float):
        with self._lock:
            self._gauges[name] = value

    def generate_prometheus_text(self) -> str:
        """Serializes current metrics into standard Prometheus text format."""
        lines = []
        with self._lock:
            # Counters
            for name, val in self._counters.items():
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {val}")
                
            # Labeled counters
            for name, label_dict in self._labeled_counters.items():
                lines.append(f"# TYPE {name} counter")
                for label_str, val in label_dict.items():
                    lines.append(f"{name}{{{label_str}}} {val}")
                    
            # Gauges
            for name, val in self._gauges.items():
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {val}")
                
            # Histograms (count and sum)
            for name, values in self._histograms.items():
                lines.append(f"# TYPE {name}_seconds summary")
                count = len(values)
                total = sum(values)
                avg = (total / count) if count > 0 else 0.0
                lines.append(f"{name}_count {count}")
                lines.append(f"{name}_sum {total:.6f}")
                lines.append(f"{name}_avg {avg:.6f}")
                
        return "\n".join(lines) + "\n"

    def reset(self):
        with self._lock:
            self._counters.clear()
            self._labeled_counters.clear()
            self._histograms.clear()
            self._gauges.clear()
            self._gauges["tutor_service_up"] = 1.0



# Global default metrics collector
metrics = MetricsCollector()
