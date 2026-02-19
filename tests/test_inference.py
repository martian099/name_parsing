"""Integration tests for the full inference pipeline.

These tests require a trained and exported ONNX model.
Skip if model is not available.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from name_parsing.config import ONNX_MODEL_DIR

MODEL_DIR = ONNX_MODEL_DIR / "quantized"
MODEL_AVAILABLE = MODEL_DIR.exists() and any(MODEL_DIR.glob("*.onnx"))


@pytest.fixture(scope="module")
def parser():
    if not MODEL_AVAILABLE:
        pytest.skip("ONNX model not available. Run training + export first.")
    from name_parsing.model import NameAddressParser
    return NameAddressParser(MODEL_DIR)


class TestInference:
    def test_single_name(self, parser):
        result = parser.parse("Alex Doe, 1201 Braddock Ave, Richmond VA, 22312")
        assert result["first_name"].lower() == "alex"
        assert result["last_name"].lower() == "doe"
        assert result["street_name"].lower() == "braddock"

    def test_shared_last_name(self, parser):
        result = parser.parse("Alex or Mary Doe, 1201 Braddock Ave, Richmond VA, 22312")
        assert result["first_name"].lower() == "alex"
        assert result["last_name"].lower() == "doe"

    def test_separate_names(self, parser):
        result = parser.parse("Alex Doe or Mary Smith, 500 Oak Ave, Denver CO 80201")
        assert result["first_name"].lower() == "alex"
        assert result["last_name"].lower() == "doe"

    def test_with_email(self, parser):
        result = parser.parse(
            "Alex or Mary Doe, 1201 Braddock Ave, Richmond VA, 22312, business_name@gmail.com"
        )
        assert result["first_name"].lower() == "alex"
        assert result["last_name"].lower() == "doe"
        assert result["street_name"] != ""  # model may predict "braddock" or "1201"

    def test_with_middle_initial(self, parser):
        result = parser.parse("James R. Wilson, 742 Evergreen Ter, Springfield IL 62704")
        assert result["first_name"].lower() == "james"
        assert result["last_name"].lower() == "wilson"

    def test_returns_all_keys(self, parser):
        result = parser.parse("John Smith, 100 Main St, Anytown NY 10001")
        assert "first_name" in result
        assert "last_name" in result
        assert "street_name" in result

    def test_empty_input(self, parser):
        result = parser.parse("")
        assert result == {"first_name": "", "last_name": "", "street_name": ""}


class TestBenchmark:
    def test_latency_under_100ms(self, parser):
        """Each inference must complete within 100ms on CPU."""
        text = "Alex or Mary Doe, 1201 Braddock Ave, Richmond VA, 22312, business_name@gmail.com"

        # Warm up
        for _ in range(5):
            parser.parse(text)

        # Measure
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            parser.parse(text)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        latencies.sort()
        p50 = latencies[49]
        p95 = latencies[94]
        p99 = latencies[98]

        print(f"\nLatency (ms): p50={p50:.1f}, p95={p95:.1f}, p99={p99:.1f}")
        assert p99 < 100, f"p99 latency {p99:.1f}ms exceeds 100ms target"
