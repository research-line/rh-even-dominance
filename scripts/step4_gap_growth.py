#!/usr/bin/env python3
"""
step4_gap_growth.py
====================
Design oracle for the three lemmas needed to close Step 4
(Gap grows unboundedly).

Lemma A: QW_11 <= -alpha * L^2
Lemma B: lambda_min(C) >= -beta * L
Lemma C: ||B||_op <= gamma * L

Combined: coupling = ||B||^2 / denom = O(1), gap ~ -alpha*L^2 -> -inf.
"""
import numpy as np
from scipy.linalg import eigh, norm
from sympy import primerange
import sys

LOG4PI_GAMMA_F = 3.2720532309274587

def build_basis_grid(N, t_grid, L, basis="cos"):
    phi = np.zeros((N, len(t_grid)))
    if basis == "cos":
        phi[0, :] = 1.0 / np.sqrt(2 * L)
        for n in range(1, N):
            phi[n, :] = np.cos(n * np.pi * t_grid / L) / np.sqrt(L)
    else:
        for n in range(N):
            phi[n, :] = np.sin((n + 1) * np.pi * t_grid / L) / np.sqrt(L)
    return phi

def build_shifted_basis(N, t_grid, L, shift, basis="cos"):
    ts = t_grid - shift
    mask = np.abs(ts) <= L
    phi = np.zeros((N, len(t_grid)))
    if basis == "cos":
        phi[0, mask] = 1.0 / np.sqrt(2 * L)
        for n in range(1, N):
            phi[n, mask] = np.cos(n * np.pi * ts[mask] / L) / np.sqrt(L)
    else:
        for n in range(N):
            phi[n, mask] = np.sin((n + 1) * np.pi * ts[mask] / L) / np.sqrt(L)
    return phi

def build_QW(lam, N, primes, basis="cos", n_quad=2000, n_int=1000):
    L = np.log(lam)
    t_grid = np.linspace(-L, L, n_quad)
    dt = t_grid[1] - t_grid[0]
    phi = build_basis_grid(N, t_grid, L, basis)
    W = LOG4PI_GAMMA_F * np.eye(N)
    s_max = min(2 * L, 12.0)
    s_grid = np.linspace(0.005, s_max, n_int)
    ds = s_grid[1] - s_grid[0]
    for s in s_grid:
        K = np.exp(s / 2) / (2.0 * np.sinh(s))
        if K < 1e-15:
            continue
        pp = build_shifted_basis(N, t_grid, L, s, basis)
        pm = build_shifted_basis(N, t_grid, L, -s, basis)
        Sp = (phi @ pp.T) * dt
        Sm = (phi @ pm.T) * dt
        W += K * (Sp + Sm - 2.0 * np.exp(-s / 2) * np.eye(N)) * ds
    for p in primes:
        logp = np.log(p)
        for m in range(1, 20):
            coeff = logp * p ** (-m / 2.0)
            shift = m * logp
            if shift >= 2 * L:
                break
            pp = build_shifted_basis(N, t_grid, L, shift, basis)
            pm = build_shifted_basis(N, t_grid, L, -shift, basis)
            Sp = (phi @ pp.T) * dt
            Sm = (phi @ pm.T) * dt
            W += coeff * (Sp + Sm)
    return W


if __name__ == "__main__":
    primes_all = [int(p) for p in primerange(2, 500)]
    lambdas = [30, 50, 100, 200, 500]
    N = 30
    K = 3

    print("=" * 75)
    print("STEP 4: DESIGN ORACLE -- Gap Growth via Block Bounds")
    print("=" * 75)
    sys.stdout.flush()

    # Build matrices
    cache = {}
    for lam in lambdas:
        L = np.log(lam)
        pu = [p for p in primes_all if p <= max(lam, 100)]
        Wc = build_QW(lam, N, pu, "cos")
        Ws = build_QW(lam, N, pu, "sin")
        cache[lam] = {"L": L, "Wc": Wc, "Ws": Ws}
        print("  Built lam=%d" % lam)
        sys.stdout.flush()

    # (a) Lemma A: QW_11 / L^2
    print("\n(a) LEMMA A: QW_11 / L^2  (target: -> -alpha < 0)")
    print("-" * 60)
    fmt = "%6d | %6.3f | %+10.4f | %+10.4f | %+10.4f"
    print("   lam |      L |     QW_11  |   QW_11/L  |  QW_11/L^2")
    for lam in lambdas:
        L = cache[lam]["L"]
        w11 = cache[lam]["Wc"][1, 1]
        print(fmt % (lam, L, w11, w11/L, w11/L**2))
    sys.stdout.flush()

    # (b) Lemma B: lmin(C) / L
    print("\n(b) LEMMA B: lmin(C) / L  (target: >= -beta)")
    print("-" * 60)
    print("   lam |      L |   lmin(C)  | lmin(C)/L  | Gersh_min  | Gersh/L")
    for lam in lambdas:
        L = cache[lam]["L"]
        C = cache[lam]["Wc"][K:, K:]
        Ce = np.sort(eigh(C, eigvals_only=True))[0]
        Cd = np.diag(C)
        Cr = np.array([np.sum(np.abs(C[i, :])) - np.abs(C[i, i]) for i in range(len(C))])
        gm = np.min(Cd - Cr)
        print("%6d | %6.3f | %+10.4f | %+10.4f | %+10.4f | %+10.4f" %
              (lam, L, Ce, Ce/L, gm, gm/L))
    sys.stdout.flush()

    # (c) Lemma C: ||B||_op / L
    print("\n(c) LEMMA C: ||B||_op / L  (target: <= gamma)")
    print("-" * 60)
    print("   lam |      L |  ||B||_op  |   ||B||/L  |  Schur_row |  Schur/L")
    for lam in lambdas:
        L = cache[lam]["L"]
        B = cache[lam]["Wc"][:K, K:]
        Bop = norm(B, 2)
        sr = np.max(np.sum(np.abs(B), axis=1))
        print("%6d | %6.3f | %10.4f | %10.4f | %10.4f | %10.4f" %
              (lam, L, Bop, Bop/L, sr, sr/L))
    sys.stdout.flush()

    # COMBINED
    print("\n" + "=" * 75)
    print("COMBINED: Schur Complement Gap Prediction")
    print("=" * 75)
    print("   lam |     QW_11 |    l1_sin |  gap_pred |    l1_cos | gap_actual |  coupling")
    for lam in lambdas:
        L = cache[lam]["L"]
        Wc = cache[lam]["Wc"]
        Ws = cache[lam]["Ws"]
        l1c = np.sort(eigh(Wc, eigvals_only=True))[0]
        l1s = np.sort(eigh(Ws, eigvals_only=True))[0]
        w11 = Wc[1, 1]
        C = Wc[K:, K:]
        B = Wc[:K, K:]
        Bop = norm(B, 2)
        lminC = np.sort(eigh(C, eigvals_only=True))[0]
        denom = lminC - w11
        coupling = Bop**2 / abs(denom) if abs(denom) > 0.01 else 9999
        gap_pred = w11 - l1s
        gap_actual = l1c - l1s
        print("%6d | %+9.4f | %+9.4f | %+9.4f | %+9.4f | %+9.4f | %9.4f" %
              (lam, w11, l1s, gap_pred, l1c, gap_actual, coupling))
    sys.stdout.flush()

    # Power law fits
    Ls = np.array([cache[l]["L"] for l in lambdas])

    # QW_11
    W11s = np.array([cache[l]["Wc"][1, 1] for l in lambdas])
    nw = -W11s[W11s < 0]
    nL = Ls[W11s < 0]
    if len(nw) > 2:
        c = np.polyfit(np.log(nL), np.log(nw), 1)
        print("\nFit: -QW_11 ~ %.4f * L^%.3f" % (np.exp(c[1]), c[0]))

    # Gap
    gaps = []
    for lam in lambdas:
        l1c = np.sort(eigh(cache[lam]["Wc"], eigvals_only=True))[0]
        l1s = np.sort(eigh(cache[lam]["Ws"], eigvals_only=True))[0]
        gaps.append(l1c - l1s)
    gaps = np.array(gaps)
    ng = [(Ls[i], -gaps[i]) for i in range(len(gaps)) if gaps[i] < 0]
    if len(ng) > 2:
        c2 = np.polyfit(np.log([x[0] for x in ng]), np.log([x[1] for x in ng]), 1)
        print("Fit: |Gap| ~ %.4f * L^%.3f" % (np.exp(c2[1]), c2[0]))

    # SIN sector QW_00 for comparison
    print("\nSIN sector W_sin[0,0] / L^2:")
    for lam in lambdas:
        L = cache[lam]["L"]
        w00s = cache[lam]["Ws"][0, 0]
        print("  lam=%5d: W_sin[0,0] = %+10.4f, /L^2 = %+10.4f" % (lam, w00s, w00s/L**2))

    # KEY: Why does Even go deeper than Odd?
    print("\nKEY COMPARISON: QW_11(cos) vs QW_00(sin)")
    for lam in lambdas:
        L = cache[lam]["L"]
        w11c = cache[lam]["Wc"][1, 1]
        w00s = cache[lam]["Ws"][0, 0]
        print("  lam=%5d: cos_11=%+.4f, sin_00=%+.4f, diff=%+.4f, diff/L^2=%+.6f" %
              (lam, w11c, w00s, w11c - w00s, (w11c - w00s)/L**2))

    print("\n" + "=" * 75)
    print("CONCLUSION:")
    print("  If QW_11/L^2 -> -alpha_cos, W_sin_00/L^2 -> -alpha_sin,")
    print("  and alpha_cos > alpha_sin (cos n=1 couples MORE to primes than sin n=1),")
    print("  then Delta ~ -(alpha_cos - alpha_sin) * L^2 -> -inf.")
    print("  This is the SHIFT PARITY LEMMA applied to the leading mode!")
    print("=" * 75)
