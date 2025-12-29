"""Metrics collection for observability."""
import time
import asyncio
from collections import defaultdict
from statistics import quantiles
from functools import wraps
from typing import Callable, TypeVar, ParamSpec, Any

P = ParamSpec("P")
R = TypeVar("R")


class MetricsCollector:
    """Simple in-memory metrics for p95 latency tracking."""

    def __init__(self, max_samples: int = 1000) -> None:
        """
        Initialize metrics collector.

        Args:
            max_samples: Maximum samples to retain per component
        """
        self.latencies: defaultdict[str, list[float]] = defaultdict(list)
        self.counters: defaultdict[str, int] = defaultdict(int)
        self.max_samples = max_samples

    def record_latency(self, component: str, latency_ms: float) -> None:
        """
        Record a latency measurement.

        Args:
            component: Component name (e.g., "embedding", "reranking")
            latency_ms: Latency in milliseconds
        """
        self.latencies[component].append(latency_ms)
        if len(self.latencies[component]) > self.max_samples:
            self.latencies[component] = self.latencies[component][-self.max_samples :]

    def increment(self, counter: str) -> None:
        """
        Increment a counter.

        Args:
            counter: Counter name (e.g., "requests", "errors")
        """
        self.counters[counter] += 1

    def p95(self, component: str) -> float:
        """
        Calculate p95 latency for a component.

        Args:
            component: Component name

        Returns:
            p95 latency in milliseconds, or 0.0 if insufficient data
        """
        data = self.latencies[component]
        if len(data) < 20:
            return 0.0
        return quantiles(data, n=20)[18]  # 95th percentile

    def report(self) -> dict[str, dict[str, float | int]]:
        """
        Generate metrics report.

        Returns:
            Dict with latencies_p95 and counters
        """
        return {
            "latencies_p95": {comp: self.p95(comp) for comp in self.latencies},
            "counters": dict(self.counters),
        }


# Global metrics instance
metrics = MetricsCollector()


def timed(
    component: str, metrics_collector: MetricsCollector = metrics
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator to measure function execution time.

    Args:
        component: Component name for metrics
        metrics_collector: MetricsCollector instance to use

    Returns:
        Decorated function

    Example:
        @timed("embedding")
        async def embed_query(query: str):
            # Function implementation
            pass
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            t0 = time.perf_counter()
            try:
                result = await func(*args, **kwargs)  # type: ignore
                return result
            finally:
                latency = (time.perf_counter() - t0) * 1000
                metrics_collector.record_latency(component, latency)

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            t0 = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                latency = (time.perf_counter() - t0) * 1000
                metrics_collector.record_latency(component, latency)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper

    return decorator
