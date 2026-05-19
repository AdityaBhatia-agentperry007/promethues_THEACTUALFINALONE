import random

from backend.crypto.gc_bridge import NOTE, compare_private, quantize_risk


def test_compare_matches_quantized_plaintext_oracle():
    rng = random.Random(1337)
    for _ in range(500):
        a = rng.random()
        b = rng.random()
        result = compare_private(a, b)
        assert result["a_lower"] == (quantize_risk(a) < quantize_risk(b))


def test_compare_carries_honesty_note():
    result = compare_private(0.2, 0.8)
    assert result["note"] == NOTE
    assert "32-bit hash" in result["note"]

