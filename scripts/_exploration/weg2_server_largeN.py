#!/usr/bin/env python3
"""
weg2_server_largeN.py
=====================
Large-N convergence with CORRECTED orthogonal basis.
For ellmos-services (2 vCPU, 8 GB RAM).

Computes l1(cos) and l1(sin) for lambda=100, 200 at N up to 80.
"""

import numpy as np
from scipy.linalg import eigh
from mpmath import euler as mp_euler, log as mplog, pi as mppi
import time
import json

LOG4PI_GAMMA = float(mplog(4 * mppi)) + float(mp_euler)


def shift_cos(n, m, s, L):
    """Corrected cos shift: cos(n*pi*t/L)/sqrt(L), orthogonal on [-L,L]."""
    if abs(s) > 2 * L:
        return 0.0
    a = max(-L, s - L)
    b = min(L, s + L)
    if a >= b:
        return 0.0
    if n == 0 and m == 0:
        norm = 1.0 / (2 * L)
    elif n == 0 or m == 0:
        norm = 1.0 / (L * np.sqrt(2))
    else:
        norm = 1.0 / L
    kn = n * np.pi / L
    km = m * np.pi / L
    result = 0.0
    for freq, phase in [(kn - km, km * s), (kn + km, -km * s)]:
        if abs(freq) < 1e-12:
            result += np.cos(phase) * (b - a) / 2
        else:
            result += (np.sin(freq * b + phase) - np.sin(freq * a + phase)) / (2 * freq)
    return norm * result


def shift_sin(n, m, s, L):
    """Corrected sin shift: sin((n+1)*pi*t/L)/sqrt(L), orthogonal on [-L,L]."""
    if abs(s) > 2 * L:
        return 0.0
    a = max(-L, s - L)
    b = min(L, s + L)
    if a >= b:
        return 0.0
    norm = 1.0 / L
    kn = (n + 1) * np.pi / L
    km = (m + 1) * np.pi / L
    result = 0.0
    for freq, phase, sign in [(kn - km, km * s, +1), (kn + km, -km * s, -1)]:
        if abs(freq) < 1e-12:
            result += sign * np.cos(phase) * (b - a) / 2
        else:
            result += sign * (np.sin(freq * b + phase) - np.sin(freq * a + phase)) / (2 * freq)
    return norm * result


def build_QW(lam, N, primes, basis='cos', n_int=None):
    """Build QW matrix with corrected basis."""
    L = np.log(lam)
    if n_int is None:
        n_int = max(2000, 30 * N)

    W = LOG4PI_GAMMA * np.eye(N)
    sf = shift_cos if basis == 'cos' else shift_sin

    s_max = min(2 * L, 12.0)
    s_grid = np.linspace(0.005, s_max, n_int)
    ds = s_grid[1] - s_grid[0]

    for s in s_grid:
        K = np.exp(s / 2) / (2.0 * np.sinh(s))
        if K < 1e-15:
            continue
        for i in range(N):
            for j in range(i, N):
                sp = sf(i, j, s, L)
                sm = sf(i, j, -s, L)
                reg = -2.0 * np.exp(-s / 2) * (1.0 if i == j else 0.0)
                val = K * (sp + sm + reg) * ds
                W[i, j] += val
                if i != j:
                    W[j, i] += val

    for p in primes:
        logp = np.log(p)
        for m_exp in range(1, 20):
            coeff = logp * p ** (-m_exp / 2.0)
            shift = m_exp * logp
            if shift >= 2 * L:
                break
            for i in range(N):
                for j in range(i, N):
                    sp = sf(i, j, shift, L)
                    sm = sf(i, j, -shift, L)
                    val = coeff * (sp + sm)
                    W[i, j] += val
                    if i != j:
                        W[j, i] += val
    return W


def main():
    from sympy import primerange
    primes = list(primerange(2, 200))

    results = {}

    for lam in [100, 200]:
        pu = [p for p in primes if p <= max(lam, 47)]
        data = []

        print(f"\nlambda={lam}:")
        print(f"  {'N':>4} | {'l1_cos':>14} | {'l1_sin':>14} | {'gap':>14} | {'time':>8}")
        print(f"  {'-'*62}")

        for N in [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80]:
            t0 = time.time()
            Wc = build_QW(lam, N, pu, 'cos')
            Ws = build_QW(lam, N, pu, 'sin')
            l1c = float(np.sort(eigh(Wc, eigvals_only=True))[0])
            l1s = float(np.sort(eigh(Ws, eigvals_only=True))[0])
            dt = time.time() - t0
            gap = l1c - l1s

            data.append({
                'N': N, 'l1_cos': l1c, 'l1_sin': l1s,
                'gap': gap, 'time': dt
            })

            print(f"  {N:4d} | {l1c:+14.8f} | {l1s:+14.8f} | {gap:+14.8f} | {dt:7.1f}s")
            # Flush output
            import sys
            sys.stdout.flush()

        results[str(lam)] = data

    # Save
    with open('largeN_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nErgebnisse in largeN_results.json")


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 70)
    print("LARGE-N CONVERGENCE (corrected basis)")
    print(f"Start: {time.strftime('%H:%M:%S')}")
    print("=" * 70)
    main()
    print(f"\nTotal: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)")
