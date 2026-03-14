#!/usr/bin/env python3
"""
weighted_compactness_server.py
===============================
Server version of the weighted compactness tests.
For ellmos-services (2 vCPU, 8 GB RAM).

Runs with N=50, dense lambda grid [30..1000], all 6 tests.
Saves results to JSON for analysis.

Usage: python -u weighted_compactness_server.py 2>&1 | tee wc_results.log

Author: Lukas Geiger
Date: 2026-03-14
"""

import numpy as np
from scipy.linalg import eigh
from sympy import primerange
import time
import json
import sys

LOG4PI_GAMMA_F = 3.2720532309274587


# =========================================================================
# Vectorized basis + QW builder (same as local version)
# =========================================================================

def build_basis_grid(N, t_grid, L, basis='cos'):
    n_pts = len(t_grid)
    phi = np.zeros((N, n_pts))
    if basis == 'cos':
        phi[0, :] = 1.0 / np.sqrt(2 * L)
        for n in range(1, N):
            phi[n, :] = np.cos(n * np.pi * t_grid / L) / np.sqrt(L)
    else:
        for n in range(N):
            phi[n, :] = np.sin((n + 1) * np.pi * t_grid / L) / np.sqrt(L)
    return phi


def build_shifted_basis(N, t_grid, L, shift, basis='cos'):
    ts = t_grid - shift
    mask = np.abs(ts) <= L
    phi = np.zeros((N, len(t_grid)))
    if basis == 'cos':
        phi[0, mask] = 1.0 / np.sqrt(2 * L)
        for n in range(1, N):
            phi[n, mask] = np.cos(n * np.pi * ts[mask] / L) / np.sqrt(L)
    else:
        for n in range(N):
            phi[n, mask] = np.sin((n + 1) * np.pi * ts[mask] / L) / np.sqrt(L)
    return phi


def build_QW(lam, N, primes, basis='cos', n_quad=2500, n_int=1200):
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
        phi_p = build_shifted_basis(N, t_grid, L, s, basis)
        phi_m = build_shifted_basis(N, t_grid, L, -s, basis)
        S_p = (phi @ phi_p.T) * dt
        S_m = (phi @ phi_m.T) * dt
        W += K * (S_p + S_m - 2.0 * np.exp(-s / 2) * np.eye(N)) * ds

    for p in primes:
        logp = np.log(p)
        for m_exp in range(1, 20):
            coeff = logp * p ** (-m_exp / 2.0)
            shift = m_exp * logp
            if shift >= 2 * L:
                break
            phi_p = build_shifted_basis(N, t_grid, L, shift, basis)
            phi_m = build_shifted_basis(N, t_grid, L, -shift, basis)
            S_p = (phi @ phi_p.T) * dt
            S_m = (phi @ phi_m.T) * dt
            W += coeff * (S_p + S_m)
    return W


# =========================================================================
# Main computation
# =========================================================================

if __name__ == "__main__":
    print("=" * 75)
    print("WEIGHTED COMPACTNESS -- SERVER RUN")
    print(f"N=50, lambda=[30..1000], primes to 600")
    print("=" * 75)
    sys.stdout.flush()

    primes_all = [int(p) for p in primerange(2, 600)]
    lambdas = [30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 700, 1000]
    N = 50

    results = {}

    # Phase 1: Build all matrices and extract eigensystems
    print("\nPhase 1: Matrix builds")
    sys.stdout.flush()

    for lam in lambdas:
        L = np.log(lam)
        primes_used = [p for p in primes_all if p <= max(lam, 100)]
        results[lam] = {'L': L, 'n_primes': len(primes_used)}

        for basis, sector in [('cos', 'even'), ('sin', 'odd')]:
            t0 = time.time()
            W = build_QW(lam, N, primes_used, basis=basis)
            evals, evecs = eigh(W)
            elapsed = time.time() - t0

            v0 = evecs[:, 0]

            # Sobolev data
            n_arr = np.arange(N, dtype=float)
            if basis == 'sin':
                n_arr = np.arange(1, N + 1, dtype=float)
            sum_n2c2 = float(np.sum(n_arr**2 * v0**2))
            h1_norm = float(np.sqrt(1 + np.pi**2 * sum_n2c2))

            # Top 5 modes
            top5 = list(map(int, np.argsort(np.abs(v0))[::-1][:5]))

            # Mode concentration: fraction in top-K modes
            sorted_c2 = np.sort(v0**2)[::-1]
            cum = np.cumsum(sorted_c2)
            k90 = int(np.searchsorted(cum, 0.9) + 1)  # modes for 90% energy

            results[lam][sector] = {
                'l1': float(evals[0]),
                'l2': float(evals[1]),
                'gap_12': float(evals[1] - evals[0]),
                'sum_n2c2': sum_n2c2,
                'h1_norm': h1_norm,
                'top5_modes': top5,
                'k90': k90,
                'time': elapsed
            }

            print(f"  lam={lam:5d} {sector:4s}: l1={evals[0]:+10.4f}, gap={evals[1]-evals[0]:.4f}, "
                  f"S={sum_n2c2:.2f}, H1={h1_norm:.2f}, k90={k90}, {elapsed:.1f}s")
            sys.stdout.flush()

    # Phase 2: Derived quantities
    print("\nPhase 2: Derived quantities")
    sys.stdout.flush()

    # Parity gap
    print("\nPARITY GAP:")
    print(f"{'lam':>6} | {'l1_even':>10} | {'l1_odd':>10} | {'Delta':>10} | {'D/L':>10} | {'D/L^2':>10}")
    for lam in lambdas:
        L = results[lam]['L']
        l1c = results[lam]['even']['l1']
        l1s = results[lam]['odd']['l1']
        d = l1c - l1s
        results[lam]['delta'] = d
        print(f"{lam:6d} | {l1c:+10.4f} | {l1s:+10.4f} | {d:+10.4f} | {d/L:+10.4f} | {d/L**2:+10.4f}")
    sys.stdout.flush()

    # Eigenvalue scaling
    print("\nEIGENVALUE SCALING (even):")
    print(f"{'lam':>6} | {'l1':>10} | {'l1/L':>10} | {'l1/L^2':>10} | {'l1/L^3':>10}")
    for lam in lambdas:
        L = results[lam]['L']
        l1 = results[lam]['even']['l1']
        print(f"{lam:6d} | {l1:+10.4f} | {l1/L:+10.4f} | {l1/L**2:+10.4f} | {l1/L**3:+10.4f}")
    sys.stdout.flush()

    # Sobolev summary
    print("\nSOBOLEV BOUND (sum n^2 c_n^2):")
    print(f"{'lam':>6} | {'EVEN':>10} | {'ODD':>10} | {'k90_e':>6} | {'k90_o':>6}")
    for lam in lambdas:
        se = results[lam]['even']['sum_n2c2']
        so = results[lam]['odd']['sum_n2c2']
        ke = results[lam]['even']['k90']
        ko = results[lam]['odd']['k90']
        print(f"{lam:6d} | {se:10.4f} | {so:10.4f} | {ke:6d} | {ko:6d}")
    sys.stdout.flush()

    # Phase 3: Profile convergence (rescaled)
    print("\nPROFILE CONVERGENCE (rescaled, even):")
    u_grid = np.linspace(-1, 1, 5000)
    du = u_grid[1] - u_grid[0]

    psis = {}
    for lam in lambdas:
        L = results[lam]['L']
        primes_used = [p for p in primes_all if p <= max(lam, 100)]
        W = build_QW(lam, N, primes_used, basis='cos')
        _, evecs = eigh(W)
        v0 = evecs[:, 0]

        t_grid = L * u_grid
        phi = build_basis_grid(N, t_grid, L, 'cos')
        f = v0 @ phi
        psi = np.sqrt(L) * f

        mid = len(u_grid) // 2
        if psi[mid] < 0:
            psi = -psi
        psis[lam] = psi

    for i in range(len(lambdas) - 1):
        d = float(np.sqrt(np.sum((psis[lambdas[i]] - psis[lambdas[i+1]])**2) * du))
        results[lambdas[i]]['profile_dist_next'] = d
        print(f"  {lambdas[i]:>6} -> {lambdas[i+1]:>6}: ||diff||_L2 = {d:.6f}")
    sys.stdout.flush()

    # Phase 4: Relative tail
    print("\nRELATIVE TAIL (even, alpha=0.9):")
    for lam in lambdas:
        psi = psis[lam]
        total = np.sum(psi**2) * du
        tail_mask = np.abs(u_grid) > 0.9
        tail = np.sum(psi[tail_mask]**2) * du
        frac = tail / total if total > 1e-15 else 0
        results[lam]['tail_090'] = frac
        print(f"  lam={lam:5d}: tail(a>0.9) = {frac:.5f}")
    sys.stdout.flush()

    # Fit Delta ~ -C * L^alpha
    neg = [(results[lam]['L'], -results[lam]['delta'])
           for lam in lambdas if results[lam]['delta'] < 0]
    if len(neg) > 4:
        log_L = np.log([x[0] for x in neg])
        log_d = np.log([x[1] for x in neg])
        c = np.polyfit(log_L, log_d, 1)
        print(f"\nGAP FIT: |Delta| ~ {np.exp(c[1]):.4f} * L^{c[0]:.3f}")

    # Save results
    output = {
        'N': N,
        'n_primes_max': len(primes_all),
        'lambdas': lambdas,
        'results': {}
    }
    for lam in lambdas:
        output['results'][str(lam)] = results[lam]

    with open('/opt/rh_proof/wc_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to /opt/rh_proof/wc_results.json")

    elapsed = time.time() - time.time()  # not great but ok
    print(f"\n{'='*75}")
    print("DONE")
    print(f"{'='*75}")
