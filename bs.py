"""
Black-76 pricing and analytic Greeks (Layer 1 of the vol surface fitter).

Design principle
----------------
The underlying is the **forward F**, not spot S. All carry -- rates, dividends,
borrow cost -- is absorbed into F upstream. This module knows nothing about them.

    forward.py  :  spot + dividends + borrow  ->  F      (Layer 2, not yet built)
    bs.py       :  F, K, T, sigma, df         ->  price  (this module)

The split is deliberate: swapping the dividend model requires neither touching
nor re-validating the pricing function.

Conventions
-----------
sigma : lognormal volatility as a decimal (0.20 = 20%). Equity uses lognormal
        vol, not the normal/bp vol convention of rates desks.
cp    : +1 for a call, -1 for a put
df    : discount factor to expiry, e^{-rT}, supplied by the caller
T     : time to expiry in years

Every function accepts scalars or numpy arrays, so the same code prices a
single quote or an entire surface.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = [
    "d1_d2",
    "price",
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "vanna",
    "volga",
    "greeks",
    "implied_rate",
]


def _sanitize(F, K, T, sigma):
    """Cast to float arrays and flag degenerate points (T <= 0 or sigma <= 0)."""
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    degenerate = (T <= 0.0) | (sigma <= 0.0)
    return F, K, T, sigma, degenerate


def d1_d2(F, K, T, sigma):
    """
    d1 = [ln(F/K) + 0.5 * sigma^2 * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    Degenerate points return +-inf so that N(d) collapses to 0 or 1 and the
    price falls back to intrinsic value.
    """
    F, K, T, sigma, degenerate = _sanitize(F, K, T, sigma)

    # Compute on safe substitutes first, then overwrite the degenerate entries.
    # This avoids divide-by-zero warnings without branching element by element.
    T_safe = np.where(degenerate, 1.0, T)
    sig_safe = np.where(degenerate, 1.0, sigma)

    vol_t = sig_safe * np.sqrt(T_safe)
    d1 = (np.log(F / K) + 0.5 * sig_safe**2 * T_safe) / vol_t
    d2 = d1 - vol_t

    moneyness = np.sign(F - K)
    fallback = np.where(moneyness > 0, np.inf, np.where(moneyness < 0, -np.inf, 0.0))
    d1 = np.where(degenerate, fallback, d1)
    d2 = np.where(degenerate, fallback, d2)
    return d1, d2


def implied_rate(df, T):
    """Back out the continuously compounded rate r = -ln(df)/T. Used by theta."""
    df = np.asarray(df, dtype=float)
    T = np.asarray(T, dtype=float)
    T_safe = np.where(T <= 0.0, 1.0, T)
    r = -np.log(df) / T_safe
    return np.where(T <= 0.0, 0.0, r)


# --------------------------------------------------------------------- price

def price(F, K, T, sigma, df, cp=1):
    """
    Black-76 European option price.

        V = df * cp * [ F*N(cp*d1) - K*N(cp*d2) ]

    cp = +1 expands to  df * (F*N(d1) - K*N(d2))        call
    cp = -1 expands to  df * (K*N(-d2) - F*N(-d1))      put
    """
    F, K, T, sigma, degenerate = _sanitize(F, K, T, sigma)
    cp = np.asarray(cp, dtype=float)
    df = np.asarray(df, dtype=float)

    d1, d2 = d1_d2(F, K, T, sigma)
    v = df * cp * (F * norm.cdf(cp * d1) - K * norm.cdf(cp * d2))

    intrinsic = df * np.maximum(cp * (F - K), 0.0)
    return np.where(degenerate, intrinsic, v)


# -------------------------------------------------------------------- Greeks
# All of the following are **forward** Greeks: derivatives with respect to F,
# not spot. Spot delta = forward delta * dF/dS = forward delta * e^{(r-q+b)T}.
# The two are close for short-dated options and diverge for long-dated,
# high-dividend, or hard-to-borrow names.

def delta(F, K, T, sigma, df, cp=1):
    """
    dV/dF = df * cp * N(cp*d1).

    N(d1) is the number of forwards in the replicating portfolio: the option is
    long N(d1) forwards and short a fixed bond position worth df*K*N(d2). The
    bond leg does not move with F, so all of the sensitivity comes from the
    first leg -- which is why N(d1) sits exactly where the hedge ratio belongs.
    """
    F, K, T, sigma, degenerate = _sanitize(F, K, T, sigma)
    cp = np.asarray(cp, dtype=float)
    d1, _ = d1_d2(F, K, T, sigma)
    val = df * cp * norm.cdf(cp * d1)
    edge = df * np.where(cp * (F - K) > 0, cp, 0.0)
    return np.where(degenerate, edge, val)


def gamma(F, K, T, sigma, df, cp=1):
    """d2V/dF2 = df * phi(d1) / (F * sigma * sqrt(T)). Same for calls and puts."""
    F, K, T, sigma, degenerate = _sanitize(F, K, T, sigma)
    d1, _ = d1_d2(F, K, T, sigma)
    T_safe = np.where(degenerate, 1.0, T)
    sig_safe = np.where(degenerate, 1.0, sigma)
    val = df * norm.pdf(d1) / (F * sig_safe * np.sqrt(T_safe))
    return np.where(degenerate, 0.0, val)


def vega(F, K, T, sigma, df, cp=1):
    """
    dV/dsigma = df * F * phi(d1) * sqrt(T). Same for calls and puts.

    Units are "per 1.0 of volatility"; divide by 100 for "per vol point".
    """
    F, K, T, sigma, degenerate = _sanitize(F, K, T, sigma)
    d1, _ = d1_d2(F, K, T, sigma)
    T_safe = np.where(degenerate, 1.0, T)
    val = df * F * norm.pdf(d1) * np.sqrt(T_safe)
    return np.where(degenerate, 0.0, val)


def theta(F, K, T, sigma, df, cp=1):
    """
    dV/dt as calendar time advances, annualised. Divide by 365 for daily theta.

        theta = r*V - df * F * phi(d1) * sigma / (2 * sqrt(T))

    The derivation uses the identity F*phi(d1) = K*phi(d2), which is why calls
    and puts share the same expression and differ only through V.

    Modelling assumption: F is held fixed and df rolls at a constant implied r.
    In practice F drifts along the dividend and borrow curves as time passes,
    and desks differ on the convention. Align with the official system before
    signing off marks, or P&L explain will carry an unexplained constant.
    """
    F, K, T, sigma, degenerate = _sanitize(F, K, T, sigma)
    d1, _ = d1_d2(F, K, T, sigma)
    r = implied_rate(df, T)
    v = price(F, K, T, sigma, df, cp)
    T_safe = np.where(degenerate, 1.0, T)
    decay = df * F * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T_safe))
    return np.where(degenerate, 0.0, r * v - decay)


def rho(F, K, T, sigma, df, cp=1):
    """
    dV/dr **holding F fixed** (forward rho) = -T * V.

    This is not the rho a trading desk quotes: the full rate sensitivity also
    flows through F = S*e^{(r-q+b)T}. Keeping the two channels separate is a
    direct consequence of the forward parameterisation, and arguably the honest
    treatment -- they are distinct risks and belong in distinct buckets.
    """
    _, _, T, _, degenerate = _sanitize(F, K, T, sigma)
    v = price(F, K, T, sigma, df, cp)
    return np.where(degenerate, 0.0, -T * v)


# Second-order cross Greeks. On an equity vol book these are primary daily P&L
# drivers, not optional extras: a P&L explain that stops at delta/gamma/vega/
# theta leaves an unexplained residual too large to defend.

def vanna(F, K, T, sigma, df, cp=1):
    """
    d(delta)/dsigma = d(vega)/dF = -df * phi(d1) * d2 / sigma.

    Uses dd1/dsigma = -d2/sigma. Same for calls and puts, since put delta
    differs from call delta by a constant. Vanishes at the ATM forward
    (d2 -> 0); it is the core exposure of a skew position.
    """
    F, K, T, sigma, degenerate = _sanitize(F, K, T, sigma)
    d1, d2 = d1_d2(F, K, T, sigma)
    d1 = np.where(degenerate, 0.0, d1)
    d2 = np.where(degenerate, 0.0, d2)
    sig_safe = np.where(degenerate, 1.0, sigma)
    val = -df * norm.pdf(d1) * d2 / sig_safe
    return np.where(degenerate, 0.0, val)


def volga(F, K, T, sigma, df, cp=1):
    """
    d(vega)/dsigma = vega * d1 * d2 / sigma. Also called vomma.

    Near the ATM forward d1*d2 < 0, so volga is slightly negative; it turns
    positive in both wings. That sign pattern is why selling the ATM and buying
    the wings -- a butterfly -- is a long-volga position.
    """
    F, K, T, sigma, degenerate = _sanitize(F, K, T, sigma)
    d1, d2 = d1_d2(F, K, T, sigma)
    d1 = np.where(degenerate, 0.0, d1)
    d2 = np.where(degenerate, 0.0, d2)
    sig_safe = np.where(degenerate, 1.0, sigma)
    val = vega(F, K, T, sigma, df, cp) * d1 * d2 / sig_safe
    return np.where(degenerate, 0.0, val)


def greeks(F, K, T, sigma, df, cp=1):
    """Compute price and all Greeks in one call. Returns a dict."""
    return {
        "price": price(F, K, T, sigma, df, cp),
        "delta": delta(F, K, T, sigma, df, cp),
        "gamma": gamma(F, K, T, sigma, df, cp),
        "vega": vega(F, K, T, sigma, df, cp),
        "theta": theta(F, K, T, sigma, df, cp),
        "rho": rho(F, K, T, sigma, df, cp),
        "vanna": vanna(F, K, T, sigma, df, cp),
        "volga": volga(F, K, T, sigma, df, cp),
    }


# ----------------------------------------------------------------------- demo

def _demo():
    """Textbook example plus three numerical experiments (see README)."""
    F, K, T, sigma, df = 100.0, 100.0, 1.0, 0.20, np.exp(-0.04)

    print("=" * 68)
    print("Black-76   F=100  K=100  T=1  sigma=20%  df=e^-0.04")
    print("=" * 68)
    d1, d2 = d1_d2(F, K, T, sigma)
    print(f"d1 = {float(d1):+.5f}   d2 = {float(d2):+.5f}   d1+d2 = {float(d1 + d2):+.1e}")
    for cp, name in ((1, "call"), (-1, "put")):
        print(f"\n{name}")
        for k, v in greeks(F, K, T, sigma, df, cp).items():
            print(f"  {k:<6} {float(v):>12.6f}")

    print("\n" + "=" * 68)
    print("1. The Ito correction term  -0.5*sigma^2*T")
    print("=" * 68)
    z = np.random.default_rng(0).standard_normal(400_000)
    with_ito = F * np.exp(sigma * np.sqrt(T) * z - 0.5 * sigma**2 * T)
    without = F * np.exp(sigma * np.sqrt(T) * z)
    print(f"  E[F_T] with correction    : {with_ito.mean():8.3f}   (theory = F = {F:.3f})")
    print(f"  E[F_T] without correction : {without.mean():8.3f}   "
          f"(theory = F*e^(0.5*sigma^2*T) = {F * np.exp(0.5 * sigma**2 * T):.3f})")
    print(f"  MC call, with correction  : {float(df * np.maximum(with_ito - K, 0).mean()):8.4f}"
          f"   (closed form = {float(price(F, K, T, sigma, df, 1)):.4f})")
    print(f"  MC call, without          : {float(df * np.maximum(without - K, 0).mean()):8.4f}"
          "   <- biased high")
    print("  The term exists solely to keep E[F_T] = F. Drop it and the forward is")
    print("  no longer a martingale: the distribution shifts right, calls are")
    print("  overpriced and puts underpriced, and put-call parity breaks.")

    print("\n" + "=" * 68)
    print("2. ATM forward symmetry, and price versus probability")
    print("=" * 68)
    print(f"{'sigma':>8} {'d1':>10} {'d2':>10} {'call':>10} {'N(d2) = P(ITM)':>16}")
    for s in (0.10, 0.20, 0.40, 0.80):
        a, b = d1_d2(F, K, T, s)
        print(f"{s:>8.0%} {float(a):>+10.4f} {float(b):>+10.4f} "
              f"{float(price(F, K, T, s, df, 1)):>10.4f} {float(norm.cdf(b)):>16.4f}")
    print("  At F=K the log term vanishes, so d1 = +0.5*sigma*sqrt(T) and")
    print("  d2 = -0.5*sigma*sqrt(T): symmetric about zero by construction.")
    print("  Note the tension: higher vol makes the option more expensive while")
    print("  making it LESS likely to finish ITM. The average payoff conditional")
    print("  on finishing ITM grows faster than the probability shrinks.")

    print("\n" + "=" * 68)
    print("3. How a forward error becomes a volatility error")
    print("=" * 68)
    S, r = 100.0, 0.04
    print("  Dividend forecast revised 2% -> 3% (S=100, r=4%).")
    print("  The pricing signature does not change; only F does.")
    for q in (0.02, 0.03):
        F_q = S * np.exp((r - q) * T)
        print(f"    q={q:.0%}   F={float(F_q):8.4f}   "
              f"call={float(price(F_q, K, T, sigma, df, 1)):7.4f}   "
              f"delta={float(delta(F_q, K, T, sigma, df, 1)):.4f}")
    F0, F1 = S * np.exp((r - 0.02) * T), S * np.exp((r - 0.03) * T)
    dC = float(price(F1, K, T, sigma, df, 1) - price(F0, K, T, sigma, df, 1))
    vol_point = float(vega(F0, K, T, sigma, df, 1)) / 100.0
    print(f"  Price moves {dC:+.4f}, i.e. {dC / vol_point:+.2f} vol points of systematic shift.")
    print("  A 1% forward error is worth roughly 1.5 vol points, and the sign is")
    print("  asymmetric: implied vol backed out of calls is pushed one way, puts")
    print("  the other. Call and put IVs disagreeing at the same strike is the")
    print("  signature of a wrong forward, not of volatility.")


if __name__ == "__main__":
    _demo()
