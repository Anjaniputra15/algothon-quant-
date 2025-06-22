import numpy as np
import pytest

try:
    import fastcalc
    FASTCALC_AVAILABLE = True
except ImportError:
    FASTCALC_AVAILABLE = False
    fastcalc = None

@pytest.mark.skipif(not FASTCALC_AVAILABLE, reason="fastcalc Rust extension not available")
def test_calc_position_changes():
    np.random.seed(42)
    prev = np.random.randn(10000, 50)
    newp = np.random.randn(10000, 50)
    expected = newp - prev
    result = fastcalc.calc_position_changes(prev, newp)
    np.testing.assert_allclose(result, expected, rtol=1e-8)

if not FASTCALC_AVAILABLE:
    def test_fallback():
        import warnings
        warnings.warn("fastcalc Rust extension not available; test skipped.") 