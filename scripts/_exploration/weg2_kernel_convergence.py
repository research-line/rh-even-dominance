#!/usr/bin/env python3
"""
weg2_kernel_convergence.py
==========================
N-Konvergenztest fuer den korrigierten Kernel e^{u/2}/(2sinh(u))
bei kritischen lambda-Werten nahe dem Even/Odd-Crossover.
"""

import numpy as np
from scipy.linalg import eigh
from sympy import primerange
from mpmath import euler as mp_euler, log as mplog, pi as mppi
import time

LOG4PI_GAMMA = float(mplog(4 * mppi)) + float(mp_euler)


def make_basis(N, t_grid, L, basis_type='cos'):
    phi = np.zeros((N, len(t_grid)))
    if basis_type == 'cos':
        phi[0, :] = 1.0 / np.sqrt(2 * L)
        for n in range(1, N):
            phi[n, :] = np.cos(n * np.pi * t_grid / (2 * L)) / np.sqrt(L)
    else:
        for n in range(N):
            phi[n, :] = np.sin((n + 1) * np.pi * t_grid / (2 * L)) / np.sqrt(L)
    return phi


def make_shifted(N, t_grid, L, shift, basis_type='cos'):
    ts = t_grid - shift
    mask = np.abs(ts) <= L
    phi = np.zeros((N, len(t_grid)))
    if basis_type == 'cos':
        phi[0, mask] = 1.0 / np.sqrt(2 * L)
        for n in range(1, N):
            phi[n, mask] = np.cos(n * np.pi * ts[mask] / (2 * L)) / np.sqrt(L)
    else:
        for n in range(N):
            phi[n, mask] = np.sin((n + 1) * np.pi * ts[mask] / (2 * L)) / np.sqrt(L)
    return phi


def build_QW(lam, N, primes, M_terms=12, n_quad=None, n_int=None, basis='cos'):
    """QW mit korrektem Kernel e^{u/2}/(2sinh(u))."""
    L = np.log(lam)
    if n_quad is None:
        n_quad = max(800, 20 * N)
    if n_int is None:
        n_int = max(500, 12 * N)
    t_grid = np.linspace(-L, L, n_quad)
    dt = t_grid[1] - t_grid[0]

    phi = make_basis(N, t_grid, L, basis)
    W = LOG4PI_GAMMA * np.eye(N)

    s_max = min(2 * L, 8.0)
    s_grid = np.linspace(0.005, s_max, n_int)
    ds = s_grid[1] - s_grid[0]

    for s in s_grid:
        k = np.exp(s / 2) / (2.0 * np.sinh(s))
        if k < 1e-15:
            continue
        Sp = (phi @ make_shifted(N, t_grid, L, s, basis).T) * dt
        Sm = (phi @ make_shifted(N, t_grid, L, -s, basis).T) * dt
        W += (Sp + Sm - 2.0 * np.exp(-s / 2) * np.eye(N)) * k * ds

    for p in primes:
        logp = np.log(p)
        for m in range(1, M_terms + 1):
            coeff = logp * p**(-m / 2.0)
            shift = m * logp
            if shift >= 2 * L:
                break
            for sign in [1.0, -1.0]:
                S = (phi @ make_shifted(N, t_grid, L, sign * shift, basis).T) * dt
                W += coeff * S

    return W


def convergence_test():
    """N-Konvergenz fuer kritische lambda-Werte."""
    primes = list(primerange(2, 500))

    # Kritische lambda-Werte um den Crossover
    critical_lambdas = [16, 20, 22, 23, 24, 25, 30]
    N_values = [30, 40, 50, 60, 70, 80]

    print("=" * 85)
    print("N-KONVERGENZTEST: Korrekter Kernel e^{u/2}/(2sinh(u))")
    print("=" * 85)

    for lam in critical_lambdas:
        L = np.log(lam)
        primes_used = [p for p in primes if p <= max(lam, 100)]

        print(f"\n  lambda={lam} (L={L:.3f}, 2L={2*L:.3f}):")
        print(f"  {'N':>4} | {'l1_even':>14} | {'l1_odd':>14} | {'Sektor':>6} | "
              f"{'Sek.Gap':>10} | {'Gap Even':>10}")
        print(f"  {'-'*4}-+-{'-'*14}-+-{'-'*14}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}")

        for N in N_values:
            t0 = time.time()
            QW_e = build_QW(lam, N, primes_used, basis='cos')
            QW_o = build_QW(lam, N, primes_used, basis='sin')
            ev_e = np.sort(eigh(QW_e, eigvals_only=True))
            ev_o = np.sort(eigh(QW_o, eigvals_only=True))
            elapsed = time.time() - t0

            sector = "EVEN" if ev_e[0] < ev_o[0] else "ODD"
            gap = abs(ev_e[0] - ev_o[0])
            gap_even = ev_e[1] - ev_e[0]

            print(f"  {N:4d} | {ev_e[0]:+14.8e} | {ev_o[0]:+14.8e} | {sector:>6} | "
                  f"{gap:10.4e} | {gap_even:10.4e}  ({elapsed:.1f}s)")

    # Grosses lambda: Konvergenz bestaetigen
    print(f"\n\n{'='*85}")
    print("KONVERGENZ-CHECK: Grosse lambda")
    print(f"{'='*85}")

    for lam in [50, 100]:
        L = np.log(lam)
        primes_used = [p for p in primes if p <= max(lam, 100)]
        print(f"\n  lambda={lam}:")
        print(f"  {'N':>4} | {'l1_even':>14} | {'l1_odd':>14} | {'Sektor':>6} | {'Sek.Gap':>10}")
        print(f"  {'-'*4}-+-{'-'*14}-+-{'-'*14}-+-{'-'*6}-+-{'-'*10}")

        for N in [30, 50, 70]:
            QW_e = build_QW(lam, N, primes_used, basis='cos')
            QW_o = build_QW(lam, N, primes_used, basis='sin')
            ev_e = np.sort(eigh(QW_e, eigvals_only=True))
            ev_o = np.sort(eigh(QW_o, eigvals_only=True))
            sector = "EVEN" if ev_e[0] < ev_o[0] else "ODD"
            gap = abs(ev_e[0] - ev_o[0])
            print(f"  {N:4d} | {ev_e[0]:+14.8e} | {ev_o[0]:+14.8e} | {sector:>6} | {gap:10.4e}")


if __name__ == "__main__":
    convergence_test()
