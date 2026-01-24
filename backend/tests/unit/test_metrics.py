"""Unit tests for metrics collector."""
import asyncio

import pytest

from app.observability.metrics import MetricsCollector, timed


@pytest.mark.unit
class TestMetricsCollector:
    """Test metrics collection and reporting."""

    def test_record_latency(self) -> None:
        """Test recording latency measurements."""
        metrics = MetricsCollector(max_samples=10)
        metrics.record_latency("embedding", 50.0)
        metrics.record_latency("embedding", 55.0)

        assert len(metrics.latencies["embedding"]) == 2
        assert metrics.latencies["embedding"][0] == 50.0
        assert metrics.latencies["embedding"][1] == 55.0

    def test_max_samples_limit(self) -> None:
        """Test that max_samples limit is enforced."""
        metrics = MetricsCollector(max_samples=5)

        for i in range(10):
            metrics.record_latency("test", float(i))

        assert len(metrics.latencies["test"]) == 5
        # Should keep the last 5 samples (5, 6, 7, 8, 9)
        assert metrics.latencies["test"][0] == 5.0

    def test_p95_calculation(self) -> None:
        """Test p95 percentile calculation."""
        metrics = MetricsCollector()

        # Add 100 samples (0 to 99)
        for i in range(100):
            metrics.record_latency("test", float(i))

        p95 = metrics.p95("test")
        # p95 of 0-99 should be around 94-95
        assert 94.0 <= p95 <= 96.0

    def test_p95_insufficient_samples(self) -> None:
        """Test p95 with insufficient samples returns 0."""
        metrics = MetricsCollector()
        metrics.record_latency("test", 50.0)

        p95 = metrics.p95("test")
        assert p95 == 0.0

    def test_p95_exact_minimum_samples(self) -> None:
        """Test p95 with exactly 20 samples."""
        metrics = MetricsCollector()

        # Add exactly 20 samples
        for i in range(20):
            metrics.record_latency("test", float(i))

        p95 = metrics.p95("test")
        # Should work with exactly 20 samples
        assert p95 > 0.0

    def test_increment_counter(self) -> None:
        """Test counter incrementation."""
        metrics = MetricsCollector()
        metrics.increment("requests")
        metrics.increment("requests")
        metrics.increment("errors")

        assert metrics.counters["requests"] == 2
        assert metrics.counters["errors"] == 1

    def test_report_generation(self) -> None:
        """Test metrics report generation."""
        metrics = MetricsCollector()

        # Add latency data
        for i in range(50):
            metrics.record_latency("embedding", float(i))

        metrics.increment("requests")
        metrics.increment("requests")

        report = metrics.report()

        assert "latencies_p95" in report
        assert "counters" in report
        assert "embedding" in report["latencies_p95"]
        assert report["counters"]["requests"] == 2

    def test_multiple_components(self) -> None:
        """Test tracking multiple components separately."""
        metrics = MetricsCollector()

        metrics.record_latency("embedding", 100.0)
        metrics.record_latency("reranking", 200.0)
        metrics.record_latency("embedding", 110.0)

        assert len(metrics.latencies["embedding"]) == 2
        assert len(metrics.latencies["reranking"]) == 1
        assert metrics.latencies["embedding"][0] == 100.0
        assert metrics.latencies["reranking"][0] == 200.0


@pytest.mark.unit
class TestTimedDecorator:
    """Test timed decorator for latency measurement."""

    @pytest.mark.asyncio
    async def test_timed_async_function(self) -> None:
        """Test timed decorator on async function."""
        metrics = MetricsCollector()

        @timed("test_component", metrics)
        async def async_func() -> str:
            await asyncio.sleep(0.01)  # ~10ms
            return "result"

        result = await async_func()

        assert result == "result"
        assert len(metrics.latencies["test_component"]) == 1
        # Should record some latency > 0
        assert metrics.latencies["test_component"][0] > 0.0

    def test_timed_sync_function(self) -> None:
        """Test timed decorator on sync function."""
        metrics = MetricsCollector()

        @timed("test_component", metrics)
        def sync_func() -> str:
            import time

            time.sleep(0.01)  # ~10ms
            return "result"

        result = sync_func()

        assert result == "result"
        assert len(metrics.latencies["test_component"]) == 1
        # Should record some latency > 0
        assert metrics.latencies["test_component"][0] > 0.0

    @pytest.mark.asyncio
    async def test_timed_with_arguments(self) -> None:
        """Test timed decorator with function arguments."""
        metrics = MetricsCollector()

        @timed("test_component", metrics)
        async def async_func_with_args(x: int, y: int) -> int:
            return x + y

        result = await async_func_with_args(5, 3)

        assert result == 8
        assert len(metrics.latencies["test_component"]) == 1

    @pytest.mark.asyncio
    async def test_timed_multiple_calls(self) -> None:
        """Test timed decorator records multiple calls."""
        metrics = MetricsCollector()

        @timed("test_component", metrics)
        async def async_func() -> int:
            return 42

        await async_func()
        await async_func()
        await async_func()

        assert len(metrics.latencies["test_component"]) == 3

    def test_timed_preserves_function_metadata(self) -> None:
        """Test that timed decorator preserves function name and docstring."""
        metrics = MetricsCollector()

        @timed("test_component", metrics)
        def test_function() -> str:
            """Test docstring."""
            return "test"

        assert test_function.__name__ == "test_function"
        assert test_function.__doc__ == "Test docstring."

    @pytest.mark.asyncio
    async def test_timed_handles_exceptions(self) -> None:
        """Test that timed decorator still records latency on exceptions."""
        metrics = MetricsCollector()

        @timed("test_component", metrics)
        async def failing_func() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await failing_func()

        # Should still record the latency even though function raised
        assert len(metrics.latencies["test_component"]) == 1

    def test_timed_with_default_global_metrics(self) -> None:
        """Test timed decorator uses global metrics by default."""
        from app.observability.metrics import metrics as global_metrics

        # Clear any existing data
        global_metrics.latencies.clear()
        global_metrics.counters.clear()

        @timed("test_component")
        def test_func() -> str:
            return "test"

        test_func()

        # Should record to global metrics
        assert len(global_metrics.latencies["test_component"]) == 1
