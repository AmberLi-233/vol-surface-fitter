"""
Forward construction (Layer 2 of the vol surface fitter).

Layer 1 takes the forward F as given. This module produces it.

There are two ways to get F, and they are not equally good:

    forward_from_carry   F = S*exp((r - q + b)T)
        Requires a dividend forecast q and a borrow cost b. Neither is
        observable. This is what you fall back on when no option quotes exist.

    forward_from_parity  regress (C - P) on K
        Requires only quoted option prices. The market's own view of dividends
        and borrow is already embedded in those quotes, so nothing has to be
        assumed. This is the preferred route whenever a chain is available.

The parity route works because put-call parity is a pure no-arbitrage identity:

    C - P = df * (F - K)

Holding a call and shorting a put of the same strike pays F_T - K at expiry
regardless of where the underlying lands, so the package is a forward contract
and must be worth df*(F - K) today. The argument never invokes a pricing model,
so **sigma does not appear**: every strike can carry its own implied volatility
and the relationship still holds exactly. That is what makes parity usable at
Layer 2, before any volatility is known.

Written as a regression of (C - P) on K, the identity is a straight line:

    C - P = df*F - df*K        slope = -df,  intercept = df*F

Linearity is therefore guaranteed by no arbitrage, not assumed. **Curvature in
the fit is evidence about the data, not about the model**, and the shape of the
deviation identifies the cause:

    parallel shift, slope intact   dividend or borrow assumption is off
    one end bending away           American early exercise (deep ITM puts)
    clean middle, noisy wings      bid-ask spread on illiquid strikes
    scatter with no structure      quotes captured at different timestamps

A 1% error in F is worth roughly 1.5 vol points of systematic shift in the
surface that Layer 3 backs out, so this layer decides the accuracy of every
layer above it.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "forward_from_carry",
    "forward_from_parity",
    "forward_from_parity_robust",
    "implied_carry",
    "plot_parity",
]

MAD_TO_SIGMA = 1.4826  # median absolute deviation -> standard deviation, for a normal


def _as_chain(K, C, P):
    """Coerce to float arrays and reject input that cannot support a regression."""
    K, C, P = (np.asarray(x, dtype=float).ravel() for x in (K, C, P))
    if not (K.shape == C.shape == P.shape):
        raise ValueError(f"K, C, P must be the same length, got {K.shape}, {C.shape}, {P.shape}")
    if K.size < 2:
        raise ValueError("need at least 2 strikes to identify both df and F")
    if not np.all(np.isfinite(K) & np.isfinite(C) & np.isfinite(P)):
        raise ValueError("K, C, P contain non-finite values; filter them out upstream")
    if np.unique(K).size < 2:
        raise ValueError("all strikes are identical; the regression is degenerate")
    return K, C, P


# --------------------------------------------------------------------- carry

def forward_from_carry(S, r, q, b, T):
    """
    F = S * exp((r - q + b) * T).

    Fallback only. q is a forecast and b is negotiated over the counter, so
    neither is observable; disagreement about them is the main reason two desks
    mark the same option differently. Prefer forward_from_parity whenever a
    liquid chain exists.
    """
    S, r, q, b, T = (np.asarray(x, dtype=float) for x in (S, r, q, b, T))
    return S * np.exp((r - q + b) * T)


# -------------------------------------------------------------------- parity

def forward_from_parity(K, C, P):
    """
    Least-squares fit of C - P = df*F - df*K across one expiry.

    Returns
    -------
    F     : market-implied forward
    df    : market-implied discount factor (no rate curve needed)
    resid : per-strike residual, (C - P) minus the fitted line

    All three quotes must come from the same expiry, the same underlying and
    the same snapshot. Strikes may carry different implied vols; parity does
    not care.

    The residual is the diagnostic output, not a by-product: its magnitude says
    whether the fit can be trusted and its shape says what is wrong when it
    cannot. Plot it before using F.
    """
    K, C, P = _as_chain(K, C, P)
    y = C - P
    slope, intercept = np.polyfit(K, y, 1)
    df = -slope
    if df <= 0:
        raise ValueError(
            f"fitted discount factor is {df:.6g}; the slope has the wrong sign, "
            "so C and P are probably swapped or the strikes are mismatched"
        )
    F = intercept / df
    resid = y - (slope * K + intercept)
    return F, df, resid


def forward_from_parity_robust(K, C, P, n_mad=4.0, floor=1e-8, min_points=4):
    """
    Iteratively drop outlying strikes, refitting after each removal.

    The cut-off is derived from the surviving residuals rather than hard-coded:

        tol = n_mad * 1.4826 * median(|resid - median(resid)|)

    The median absolute deviation is used instead of a standard deviation
    because a bad point inflates the standard deviation enough to hide behind
    it. The median is untouched by extreme values, so the threshold stays tight
    while the outlier is still in the sample.

    One point is removed per pass, never a batch: removing a point tilts the
    fit, which redistributes every other residual.

    `floor` guards the opposite failure. On synthetic or very clean data the
    residuals are floating-point noise (~1e-14), the MAD is the same order, and
    a purely relative threshold would start discarding perfectly good strikes.

    The method needs enough strikes to work. A single bad quote tilts the fit
    and so corrupts its neighbours' residuals as well; on a chain of five that
    is already two points out of five, which is enough to move the median and
    inflate the threshold past the outlier. No choice of `n_mad` rescues it --
    the outlier is only twice the MAD. Real chains carry 20-50 strikes and the
    routine is reliable there, but on a handful of points it will silently keep
    everything. Check `mask.sum()` rather than assuming a clean result.

    Returns
    -------
    F, df : as in forward_from_parity, computed on the surviving strikes
    mask  : boolean array over the original strikes, True where kept
    resid : residuals of the surviving strikes

    `mask` is part of the answer, not debugging output. This routine always
    converges to a tidy-looking fit, because it keeps discarding evidence until
    the residuals behave; dropping 1 strike of 20 and dropping 12 of 20 produce
    the same small residuals but deserve very different confidence. The caller
    has to see how much was thrown away.
    """
    K, C, P = _as_chain(K, C, P)
    if min_points < 2:
        raise ValueError("min_points must be at least 2")
    mask = np.ones(K.size, dtype=bool)

    while mask.sum() > min_points:
        _, _, resid = forward_from_parity(K[mask], C[mask], P[mask])
        mad = np.median(np.abs(resid - np.median(resid)))
        tol = max(n_mad * MAD_TO_SIGMA * mad, floor)
        if np.max(np.abs(resid)) <= tol:
            break
        worst = np.where(mask)[0][np.argmax(np.abs(resid))]
        mask[worst] = False

    F, df, resid = forward_from_parity(K[mask], C[mask], P[mask])
    return F, df, mask, resid


def implied_carry(F, S, r, T):
    """
    Net carry the market is pricing in: q - b = r - ln(F/S)/T.

    A market-implied forward and a spot price pin down only the *combination* of
    dividend yield and borrow cost, never the two separately. That single number
    is still the useful one: compare it with your own dividend forecast and the
    gap is what the market is charging to borrow the name.
    """
    F, S, r, T = (np.asarray(x, dtype=float) for x in (F, S, r, T))
    return r - np.log(F / S) / T


# ---------------------------------------------------------------- diagnostic

def plot_parity(K, C, P, mask=None, ax=None):
    """
    Two stacked panels: the parity line, and the residuals underneath it.

    The residual panel is the one that matters. On the raw (C - P) plot the
    slope is close to -1, so a 0.1 deviation on a line running from +14 to -14
    is invisible; flattening it out is what makes the diagnostic readable.

    Excluded strikes are drawn as red crosses rather than omitted. A plot of
    only the surviving points hides exactly the evidence needed to judge the
    fit.
    """
    import matplotlib.pyplot as plt

    K, C, P = _as_chain(K, C, P)
    y = C - P
    if mask is None:
        mask = np.ones(K.size, dtype=bool)
    mask = np.asarray(mask, dtype=bool)

    F, df, _ = forward_from_parity(K[mask], C[mask], P[mask])
    fitted = df * (F - K)          # evaluated at every strike, kept or not
    resid_all = y - fitted

    if ax is None:
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    else:
        ax1, ax2 = ax

    order = np.argsort(K)
    ax1.plot(K[order], fitted[order], "--", color="gray", lw=1, zorder=1)
    ax1.scatter(K[mask], y[mask], s=28, label="used", zorder=2)
    if (~mask).any():
        ax1.scatter(K[~mask], y[~mask], s=40, marker="x", color="red",
                    label="excluded", zorder=3)
    ax1.set_ylabel("C - P")
    ax1.legend()
    ax1.set_title(f"F = {F:.4f}   df = {df:.6f}   {mask.sum()}/{K.size} strikes used")

    ax2.axhline(0, color="gray", lw=0.5)
    ax2.scatter(K[mask], resid_all[mask], s=28)
    if (~mask).any():
        ax2.scatter(K[~mask], resid_all[~mask], s=40, marker="x", color="red")
    ax2.set_ylabel("residual")
    ax2.set_xlabel("K")

    plt.tight_layout()
    return F, df


# ---------------------------------------------------------------------- demo

def _demo():
    """Three synthetic chains: clean, wrong dividend, one bad quote."""
    import bs

    F_true, df_true, T = 100.0, float(np.exp(-0.04)), 1.0
    K = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    iv = np.array([0.25, 0.22, 0.20, 0.19, 0.18])   # deliberate skew

    def chain(F):
        return (bs.price(F, K, T, iv, df_true, 1),
                bs.price(F, K, T, iv, df_true, -1))

    print("=" * 70)
    print("1. Clean chain. Each strike carries a different implied vol.")
    print("=" * 70)
    C, P = chain(F_true)
    F, df, resid = forward_from_parity(K, C, P)
    print(f"  F  = {F:.10f}   (true {F_true})")
    print(f"  df = {df:.10f}   (true {df_true:.10f})")
    print(f"  max |residual| = {np.max(np.abs(resid)):.2e}")
    print("  Five different vols, exact recovery: parity contains no sigma.")

    print("\n" + "=" * 70)
    print("2. Same chain generated off F = 98, i.e. the dividend assumption is wrong.")
    print("=" * 70)
    C, P = chain(98.0)
    F, df, resid = forward_from_parity(K, C, P)
    print(f"  F  = {F:.10f}   <- shifted")
    print(f"  df = {df:.10f}   <- unchanged")
    print(f"  max |residual| = {np.max(np.abs(resid)):.2e}   <- still zero")
    print("  Slope intact, intercept wrong: the parallel-shift signature.")
    print("  Small residuals do NOT mean the forward is right.")

    print("\n" + "=" * 70)
    print("3. One corrupted call quote (+0.30), five strikes.")
    print("=" * 70)
    C, P = chain(F_true)
    C_noisy = C.copy()
    C_noisy[0] += 0.30
    F, df, resid = forward_from_parity(K, C_noisy, P)
    print(f"  plain fit   F = {F:.6f}   max |residual| = {np.max(np.abs(resid)):.4f}")
    print(f"  residuals   {np.array2string(resid, precision=3)}")
    print("  The damage is not confined to the bad strike: the fitted line tilts,")
    print("  so its neighbours pick up residuals too.")
    F, df, mask, resid = forward_from_parity_robust(K, C_noisy, P)
    print(f"  robust fit  F = {F:.6f}   kept {mask.sum()}/{K.size} -- nothing dropped.")
    print("  With two of five residuals corrupted, the median itself is contaminated")
    print("  and the threshold ends up above the outlier. No n_mad fixes this.")

    print("\n" + "=" * 70)
    print("4. The same corruption on a realistic chain (21 strikes).")
    print("=" * 70)
    K_big = np.arange(85.0, 116.0, 1.5)
    iv_big = 0.20 + 0.0015 * (100.0 - K_big)
    C_big = bs.price(F_true, K_big, T, iv_big, df_true, 1)
    P_big = bs.price(F_true, K_big, T, iv_big, df_true, -1)
    C_big = np.asarray(C_big).copy()
    C_big[3] += 0.30
    F, df, resid = forward_from_parity(K_big, C_big, P_big)
    print(f"  plain fit   F = {F:.6f}   max |residual| = {np.max(np.abs(resid)):.4f}")
    F, df, mask, resid = forward_from_parity_robust(K_big, C_big, P_big)
    print(f"  robust fit  F = {F:.6f}   kept {mask.sum()}/{K_big.size}"
          f"   dropped K = {K_big[~mask]}")
    print(f"  max |residual| = {np.max(np.abs(resid)):.2e}")
    print("  Enough points, and the outlier is isolated cleanly.")

    print("\n" + "=" * 70)
    print("5. What the market implies about carry")
    print("=" * 70)
    S, r = 100.0, 0.04
    for F_mkt in (100.0, 98.0):
        print(f"  F = {F_mkt:6.2f}  ->  q - b = {float(implied_carry(F_mkt, S, r, T)):+.4%}")
    print("  Only the combination is identified, never q and b separately.")


if __name__ == "__main__":
    _demo()
