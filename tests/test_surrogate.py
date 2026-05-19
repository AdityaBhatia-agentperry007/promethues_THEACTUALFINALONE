import math

from backend.surrogate import get_engine


def test_surrogate_predict_shape_finite_stable():
    result = get_engine().predict(1.5, 0.7, steps=12)
    assert len(result.frames) == 12
    assert len(result.frames[0]) == result.meta["resolution"][0]
    assert len(result.frames[0][0]) == result.meta["resolution"][1]
    assert 0.0 <= result.risk <= 1.0
    assert math.isfinite(result.risk)
    flat = [value for frame in result.frames for row in frame for value in row]
    assert all(math.isfinite(value) for value in flat)

