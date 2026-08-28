"""
Validation suite for bs.py.

Three classes of check, in descending order of importance:

1. **Finite-difference cross-validation.** Every analytic Greek is checked
   against a central difference of the function it claims to differentiate.
   Closed-form Greeks are easy to get wrong -- a sign, a dropped term, a
   missed chain rule -- and this is the only way to catch it on the spot.
   In production this is the minimum viable model validation.

2. **No-arbitrage and structural properties.** Put-call parity, price bounds,
   monotonicity in vol, Greek signs. These hold independently of the
   implementation, so a broken model violates them immediately.

3. **Degenerate and boundary cases.** T=0, sigma=0, deep ITM/OTM. Real market
   data contains all of these, and silently returning NaN is the single most
   common failure mode in a valuation system.

Run with:   pytest -q test_bs.py     or     python test_bs.py
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import norm

import bs

# Covers ITM / ATM / OTM, short and long dated, low and high vol, calls and puts.
CASES = list(
    itertools.product(
        [80.0, 100.0, 125.0],   # F
        [100.0],                # K
        [0.08, 1.0, 3.0],       # T
        [0.10, 0.30, 0.70],     # sigma
        [1, -1],                # cp
    )
)
DF = float(np.exp(-0.04))
TOL_FD = 2e-4          # relative tolerance for finite-difference agreement
TOL_EXACT = 1e-10


def _central(f, x, h):
    """Central difference. Truncation error O(h^2), an order cleaner than forward."""
    return (f(x + h) - f(x - h)) / (2.0 * h)


def _close(a, b, tol=TOL_FD, scale=1.0):
    """Relative comparison; `scale` floors the denominator to avoid false alarms
    when both quantities are near zero."""
    return abs(a - b) <= tol * max(abs(a), abs(b), scale)


# ------------------------------------- 1. finite-difference cross-validation

def test_delta_matches_fd():
    for F, K, T, s, cp in CASES:
        fd = _central(lambda x: float(bs.price(x, K, T, s, DF, cp)), F, 1e-4 * F)
        an = float(bs.delta(F, K, T, s, DF, cp))
        assert _close(fd, an), f"delta {F=} {T=} {s=} {cp=}: fd={fd:.8f} an={an:.8f}"


def test_gamma_matches_fd():
    for F, K, T, s, cp in CASES:
        # Gamma is very peaked for short-dated low-vol ATM options, so the step
        # must be smaller than for delta or the O(h^2 * f''') truncation error
        # exceeds tolerance.
        fd = _central(lambda x: float(bs.delta(x, K, T, s, DF, cp)), F, 1e-4 * F)
        an = float(bs.gamma(F, K, T, s, DF, cp))
        assert _close(fd, an, scale=1e-4), f"gamma {F=} {T=} {s=}: fd={fd:.8f} an={an:.8f}"


def test_vega_matches_fd():
    for F, K, T, s, cp in CASES:
        fd = _central(lambda x: float(bs.price(F, K, T, x, DF, cp)), s, 1e-4)
        an = float(bs.vega(F, K, T, s, DF, cp))
        assert _close(fd, an), f"vega {F=} {T=} {s=}: fd={fd:.8f} an={an:.8f}"


def test_theta_matches_fd():
    """
    Theta differentiates calendar time, so the derivative in T carries a minus
    sign -- and df must roll with T at a constant implied rate. Forgetting the
    second point is the most common theta implementation bug; this test exists
    specifically to catch it.
    """
    r = 0.04
    for F, K, T, s, cp in CASES:
        def v(t):
            return float(bs.price(F, K, t, s, np.exp(-r * t), cp))
        fd = -_central(v, T, 1e-4 * T)
        an = float(bs.theta(F, K, T, s, np.exp(-r * T), cp))
        assert _close(fd, an, scale=1e-3), f"theta {F=} {T=} {s=}: fd={fd:.8f} an={an:.8f}"


def test_rho_matches_fd():
    """Forward rho: F is held fixed, only df responds to r."""
    r = 0.04
    for F, K, T, s, cp in CASES:
        def v(rr):
            return float(bs.price(F, K, T, s, np.exp(-rr * T), cp))
        fd = _central(v, r, 1e-5)
        an = float(bs.rho(F, K, T, s, np.exp(-r * T), cp))
        assert _close(fd, an, scale=1e-3), f"rho {F=} {T=} {s=}: fd={fd:.8f} an={an:.8f}"


def test_vanna_matches_fd():
    for F, K, T, s, cp in CASES:
        fd = _central(lambda x: float(bs.delta(F, K, T, x, DF, cp)), s, 1e-4)
        an = float(bs.vanna(F, K, T, s, DF, cp))
        assert _close(fd, an, scale=1e-4), f"vanna {F=} {T=} {s=}: fd={fd:.8f} an={an:.8f}"


def test_volga_matches_fd():
    for F, K, T, s, cp in CASES:
        fd = _central(lambda x: float(bs.vega(F, K, T, x, DF, cp)), s, 1e-4)
        an = float(bs.volga(F, K, T, s, DF, cp))
        assert _close(fd, an, scale=1e-3), f"volga {F=} {T=} {s=}: fd={fd:.8f} an={an:.8f}"


def test_vanna_is_symmetric_cross_derivative():
    """Vanna is both d(delta)/dsigma and d(vega)/dF. Both routes must agree."""
    for F, K, T, s, _ in CASES:
        via_vega = _central(lambda x: float(bs.vega(x, K, T, s, DF, 1)), F, 1e-4 * F)
        an = float(bs.vanna(F, K, T, s, DF, 1))
        assert _close(via_vega, an, scale=1e-4)


# ------------------------------------- 2. no-arbitrage / structural properties

def test_put_call_parity():
    """C - P = df*(F - K). An identity, so the error should be at float precision."""
    for F, K, T, s, _ in CASES:
        c = float(bs.price(F, K, T, s, DF, 1))
        p = float(bs.price(F, K, T, s, DF, -1))
        assert abs((c - p) - DF * (F - K)) < 1e-10, f"parity broken at {F=} {T=} {s=}"


def test_delta_parity():
    """Differentiate parity in F: delta_call - delta_put = df."""
    for F, K, T, s, _ in CASES:
        dc = float(bs.delta(F, K, T, s, DF, 1))
        dp = float(bs.delta(F, K, T, s, DF, -1))
        assert abs((dc - dp) - DF) < 1e-10


def test_gamma_vega_identical_for_call_and_put():
    """The parity right-hand side has no sigma and is linear in F, so the
    second-order and vol Greeks must be identical for calls and puts."""
    for F, K, T, s, _ in CASES:
        for fn in (bs.gamma, bs.vega, bs.vanna, bs.volga):
            a = float(fn(F, K, T, s, DF, 1))
            b = float(fn(F, K, T, s, DF, -1))
            assert abs(a - b) < TOL_EXACT, f"{fn.__name__} differs between call and put"


def test_price_bounds():
    """df*max(cp*(F-K),0) <= V <= df*F for a call, df*K for a put."""
    for F, K, T, s, cp in CASES:
        v = float(bs.price(F, K, T, s, DF, cp))
        intrinsic = DF * max(cp * (F - K), 0.0)
        cap = DF * (F if cp == 1 else K)
        assert intrinsic - 1e-12 <= v <= cap + 1e-12, f"out of bounds {F=} {T=} {s=} {cp=}"


def test_monotonic_in_vol():
    """Vega > 0: raising vol makes both calls and puts more expensive."""
    for F, K, T, _, cp in CASES:
        vals = [float(bs.price(F, K, T, s, DF, cp)) for s in (0.05, 0.15, 0.30, 0.60)]
        assert all(b > a for a, b in zip(vals, vals[1:])), f"not monotonic {F=} {T=}"


def test_greek_signs():
    for F, K, T, s, cp in CASES:
        assert float(bs.gamma(F, K, T, s, DF, cp)) > 0        # long options own convexity
        assert float(bs.vega(F, K, T, s, DF, cp)) > 0
        assert 0 <= float(bs.delta(F, K, T, s, DF, 1)) <= DF
        assert -DF <= float(bs.delta(F, K, T, s, DF, -1)) <= 0


def test_atm_forward_symmetry():
    """At F=K, d1 = -d2 and the call equals the put (parity RHS is zero)."""
    F = K = 100.0
    for T, s in itertools.product([0.25, 1.0, 5.0], [0.1, 0.4]):
        d1, d2 = bs.d1_d2(F, K, T, s)
        assert abs(float(d1 + d2)) < 1e-12
        assert abs(float(d1) - 0.5 * s * np.sqrt(T)) < 1e-12
        c = float(bs.price(F, K, T, s, DF, 1))
        p = float(bs.price(F, K, T, s, DF, -1))
        assert abs(c - p) < 1e-12


def test_atm_approximation():
    """The mental-math approximation C ~ 0.4 * F * sigma * sqrt(T) should be
    within 1.5% for short-dated, moderate-vol options."""
    F = K = 100.0
    for T, s in [(0.25, 0.2), (1.0, 0.2), (0.5, 0.3)]:
        exact = float(bs.price(F, K, T, s, 1.0, 1))
        approx = F * 0.4 * s * np.sqrt(T)
        assert abs(exact - approx) / exact < 0.015


def test_matches_textbook_spot_bs():
    """
    Agreement with the spot-form Black-Scholes with continuous dividends
    demonstrates that Black-76 is not a different model -- it is the same model
    with the forward construction factored out of the pricing function.
    """
    S, r, q, K, T, s = 100.0, 0.04, 0.02, 95.0, 1.5, 0.25
    F, df = S * np.exp((r - q) * T), np.exp(-r * T)
    d1 = (np.log(S / K) + (r - q + 0.5 * s**2) * T) / (s * np.sqrt(T))
    d2 = d1 - s * np.sqrt(T)
    spot_call = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    assert abs(float(bs.price(F, K, T, s, df, 1)) - spot_call) < 1e-12


# ------------------------------------------ 3. degenerate and boundary cases

def test_expiry_returns_intrinsic():
    for F, cp in itertools.product([80.0, 100.0, 125.0], [1, -1]):
        v = float(bs.price(F, 100.0, 0.0, 0.2, 1.0, cp))
        assert abs(v - max(cp * (F - 100.0), 0.0)) < 1e-12


def test_zero_vol_returns_discounted_intrinsic():
    for F, cp in itertools.product([80.0, 100.0, 125.0], [1, -1]):
        v = float(bs.price(F, 100.0, 1.0, 0.0, DF, cp))
        assert abs(v - DF * max(cp * (F - 100.0), 0.0)) < 1e-12


def test_no_nans_anywhere():
    """A silent NaN is the most dangerous failure mode in a valuation system:
    it propagates downstream and surfaces as a wrong mark, not as an error."""
    grid = itertools.product(
        [1e-3, 50.0, 100.0, 1e4], [100.0], [0.0, 1e-6, 10.0], [0.0, 1e-6, 3.0], [1, -1]
    )
    for F, K, T, s, cp in grid:
        for k, v in bs.greeks(F, K, T, s, DF, cp).items():
            assert np.isfinite(float(v)), f"{k} is non-finite at {F=} {T=} {s=} {cp=}"


def test_deep_itm_call_behaves_like_forward():
    F, K, T, s = 1000.0, 100.0, 1.0, 0.2
    assert abs(float(bs.price(F, K, T, s, DF, 1)) - DF * (F - K)) < 1e-6
    assert abs(float(bs.delta(F, K, T, s, DF, 1)) - DF) < 1e-6


def test_deep_otm_call_worthless():
    F, K, T, s = 10.0, 100.0, 1.0, 0.2
    assert float(bs.price(F, K, T, s, DF, 1)) < 1e-10
    assert float(bs.delta(F, K, T, s, DF, 1)) < 1e-10


def test_vectorised_matches_scalar():
    """Array input must match scalar calls pointwise. Fitting a whole surface
    depends on this."""
    F = np.array([80.0, 100.0, 125.0])
    K = np.array([100.0, 100.0, 100.0])
    T = np.array([0.5, 1.0, 2.0])
    s = np.array([0.15, 0.25, 0.35])
    vec = bs.greeks(F, K, T, s, DF, 1)
    for i in range(3):
        sca = bs.greeks(float(F[i]), float(K[i]), float(T[i]), float(s[i]), DF, 1)
        for k in vec:
            assert abs(float(vec[k][i]) - float(sca[k])) < 1e-12


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}\n        {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
