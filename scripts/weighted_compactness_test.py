#!/usr/bin/env python3
"""
weighted_compactness_test.py
============================
Numerical verification of the weighted compactness ingredients for
the proof closure of Even Dominance for all large lambda.

VECTORIZED version -- uses NumPy broadcasting instead of Python loops.

Author: Lukas Geiger
Date: 2026-03-14
"""

import numpy as np
from scipy.linalg import eigh
from sympy import primerange
import time

LOG4PI_GAMMA_F = 3.2720532309274587  # log(4*pi) + gamma


# =========================================================================
# VECTORIZED basis + shift overlap (key speedup)
# =========================================================================

def build_basis_grid(N, t_grid, L, basis='cos'):
    """Build N x len(t_grid) basis function matrix."""
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
    """Build basis functions evaluated at t - shift, zero outside [-L,L]."""
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


def build_QW_vectorized(lam, N, primes, basis='cos', n_quad=2000, n_int=1500):
    """Build QW matrix using vectorized overlap integrals."""
    L = np.log(lam)
    t_grid = np.linspace(-L, L, n_quad)
    dt = t_grid[1] - t_grid[0]
    phi = build_basis_grid(N, t_grid, L, basis)  # (N, n_quad)

    W = LOG4PI_GAMMA_F * np.eye(N)

    # Archimedean integral
    s_max = min(2 * L, 12.0)
    s_grid = np.linspace(0.005, s_max, n_int)
    ds = s_grid[1] - s_grid[0]

    for s in s_grid:
        K = np.exp(s / 2) / (2.0 * np.sinh(s))
        if K < 1e-15:
            continue
        phi_p = build_shifted_basis(N, t_grid, L, s, basis)
        phi_m = build_shifted_basis(N, t_grid, L, -s, basis)
        # Overlap: (N,n_quad) @ (n_quad,N) * dt = (N,N)
        S_p = (phi @ phi_p.T) * dt
        S_m = (phi @ phi_m.T) * dt
        W += K * (S_p + S_m - 2.0 * np.exp(-s / 2) * np.eye(N)) * ds

    # Prime contributions
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


def reconstruct_on_grid(coeffs, t_grid, L, basis='cos'):
    """Reconstruct eigenfunction from coefficients on t_grid."""
    N = len(coeffs)
    phi = build_basis_grid(N, t_grid, L, basis)
    return coeffs @ phi  # (n_pts,)


# =========================================================================
# Cached matrix builder
# =========================================================================

_cache = {}

def get_eigensystem(lam, N, primes_all, basis='cos'):
    """Cached eigenvalue computation."""
    key = (lam, N, basis)
    if key not in _cache:
        primes_used = [p for p in primes_all if p <= max(lam, 100)]
        W = build_QW_vectorized(lam, N, primes_used, basis=basis)
        evals, evecs = eigh(W)
        _cache[key] = (evals, evecs)
    return _cache[key]


# =========================================================================
# TEST T4: Eigenvalue convergence rate
# =========================================================================

def test_eigenvalue_convergence(lambdas, N, primes_all):
    print("\n" + "=" * 75)
    print("TEST T4: Eigenvalue scaling with lambda")
    print("=" * 75)

    for basis, sector in [('cos', 'EVEN'), ('sin', 'ODD')]:
        print(f"\n  Sektor: {sector}")
        print(f"  {'lam':>6} | {'L':>6} | {'l1':>10} | {'l2':>10} | {'gap':>10} | {'l1/L':>10} | {'l1/L^2':>10}")
        print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

        for lam in lambdas:
            L = np.log(lam)
            evals, _ = get_eigensystem(lam, N, primes_all, basis)
            l1, l2 = evals[0], evals[1]
            gap = l2 - l1
            print(f"  {lam:6d} | {L:6.3f} | {l1:+10.4f} | {l2:+10.4f} | {gap:10.4f} | {l1/L:+10.4f} | {l1/L**2:+10.4f}")


# =========================================================================
# TEST T5: Even-Odd gap scaling
# =========================================================================

def test_parity_gap_scaling(lambdas, N, primes_all):
    print("\n" + "=" * 75)
    print("TEST T5: Parity gap Delta(lam) = l1(even) - l1(odd)")
    print("=" * 75)

    print(f"  {'lam':>6} | {'L':>6} | {'l1_even':>10} | {'l1_odd':>10} | {'Delta':>10} | {'Delta/L':>10} | {'Delta/L^2':>10}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    deltas, Ls = [], []
    for lam in lambdas:
        L = np.log(lam)
        ev_cos, _ = get_eigensystem(lam, N, primes_all, 'cos')
        ev_sin, _ = get_eigensystem(lam, N, primes_all, 'sin')
        l1c, l1s = ev_cos[0], ev_sin[0]
        delta = l1c - l1s
        print(f"  {lam:6d} | {L:6.3f} | {l1c:+10.4f} | {l1s:+10.4f} | {delta:+10.4f} | {delta/L:+10.4f} | {delta/L**2:+10.4f}")
        deltas.append(delta)
        Ls.append(L)

    neg = [(L, -d) for L, d in zip(Ls, deltas) if d < 0]
    if len(neg) > 3:
        log_L = np.log([x[0] for x in neg])
        log_d = np.log([x[1] for x in neg])
        c = np.polyfit(log_L, log_d, 1)
        print(f"\n  Fit |Delta| ~ C * L^alpha: alpha={c[0]:.3f}, C={np.exp(c[1]):.4f}")


# =========================================================================
# TEST B2: Uniform H^1 Sobolev bound (rescaled)
# =========================================================================

def test_uniform_sobolev(lambdas, N, primes_all):
    print("\n" + "=" * 75)
    print("TEST B2: Uniform H^1 bound of rescaled psi_lam on [-1,1]")
    print("=" * 75)

    print(f"  {'lam':>6} | {'L':>6} | {'sum c_n^2':>12} | {'sum n^2*c_n^2':>14} | {'||psi||_H1':>12} | {'top modes':>12}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*12}-+-{'-'*14}-+-{'-'*12}-+-{'-'*12}")

    for basis, sector in [('cos', 'EVEN'), ('sin', 'ODD')]:
        print(f"\n  Sektor: {sector}")
        for lam in lambdas:
            L = np.log(lam)
            _, evecs = get_eigensystem(lam, N, primes_all, basis)
            v0 = evecs[:, 0]

            # Fourier coefficients
            c2 = v0**2
            n_arr = np.arange(N, dtype=float)
            if basis == 'cos':
                n_arr[0] = 0  # constant mode
            else:
                n_arr = np.arange(1, N + 1, dtype=float)

            sum_c2 = np.sum(c2)
            sum_n2c2 = np.sum(n_arr**2 * c2)
            h1_norm = np.sqrt(1 + np.pi**2 * sum_n2c2)

            # Top 3 modes by |c_n|
            top3 = np.argsort(np.abs(v0))[::-1][:3]
            top_str = ",".join([f"{k}" for k in top3])

            print(f"  {lam:6d} | {L:6.3f} | {sum_c2:12.6f} | {sum_n2c2:14.4f} | {h1_norm:12.4f} | n={top_str}")


# =========================================================================
# TEST B3: Relative tail within support
# =========================================================================

def test_relative_tail(lambdas, N, primes_all):
    print("\n" + "=" * 75)
    print("TEST B3: Mass fraction in |u| > alpha on rescaled [-1,1]")
    print("=" * 75)

    alphas = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    u_grid = np.linspace(-1, 1, 5000)
    du = u_grid[1] - u_grid[0]

    for basis, sector in [('cos', 'EVEN'), ('sin', 'ODD')]:
        print(f"\n  Sektor: {sector}")
        header = f"  {'lam':>6} |"
        for a in alphas:
            header += f" {'a='+str(a):>7} |"
        print(header)
        print(f"  {'-'*6}-+" + ("-" * 9 + "-+") * len(alphas))

        for lam in lambdas:
            L = np.log(lam)
            _, evecs = get_eigensystem(lam, N, primes_all, basis)
            v0 = evecs[:, 0]

            # Reconstruct on u in [-1,1] (rescaled domain)
            t_grid = L * u_grid
            f = reconstruct_on_grid(v0, t_grid, L, basis)
            psi = np.sqrt(L) * f  # rescaled
            total = np.sum(psi**2) * du

            row = f"  {lam:6d} |"
            for alpha in alphas:
                tail = np.sum(psi[np.abs(u_grid) > alpha]**2) * du
                frac = tail / total if total > 1e-15 else 0
                row += f" {frac:7.4f} |"
            print(row)


# =========================================================================
# TEST T3: Profile convergence (rescaled)
# =========================================================================

def test_profile_convergence(lambdas, N, primes_all):
    print("\n" + "=" * 75)
    print("TEST T3: Rescaled eigenfunction profile convergence")
    print("=" * 75)

    u_grid = np.linspace(-1, 1, 5000)
    du = u_grid[1] - u_grid[0]

    for basis, sector in [('cos', 'EVEN'), ('sin', 'ODD')]:
        print(f"\n  Sektor: {sector}")
        psis = []

        for lam in lambdas:
            L = np.log(lam)
            _, evecs = get_eigensystem(lam, N, primes_all, basis)
            v0 = evecs[:, 0]

            t_grid = L * u_grid
            f = reconstruct_on_grid(v0, t_grid, L, basis)
            psi = np.sqrt(L) * f

            # Fix sign
            mid = len(u_grid) // 2
            if basis == 'cos':
                if psi[mid] < 0:
                    psi = -psi
            else:
                if psi[mid + 1] - psi[mid - 1] < 0:
                    psi = -psi

            psis.append(psi)

        # Distance matrix (upper triangle)
        print(f"\n  ||psi_i - psi_j||_L2:")
        print(f"  {'':>8}", end="")
        for lam in lambdas:
            print(f" {lam:>7}", end="")
        print()

        for i in range(len(lambdas)):
            print(f"  {lambdas[i]:>8}", end="")
            for j in range(len(lambdas)):
                if j >= i:
                    d = np.sqrt(np.sum((psis[i] - psis[j])**2) * du)
                    print(f" {d:7.4f}", end="")
                else:
                    print(f" {'':>7}", end="")
            print()

        # Successive differences
        print(f"\n  Successive L2 distances:")
        for i in range(len(lambdas) - 1):
            d = np.sqrt(np.sum((psis[i] - psis[i + 1])**2) * du)
            print(f"    {lambdas[i]:>6} -> {lambdas[i+1]:>6}: {d:.6f}")


# =========================================================================
# TEST T6: Mellin transform convergence
# =========================================================================

def test_mellin_convergence(lambdas, N, primes_all):
    print("\n" + "=" * 75)
    print("TEST T6: Mellin transform convergence (EVEN sector)")
    print("=" * 75)

    xi_grid = np.linspace(0, 60, 3000)
    dxi = xi_grid[1] - xi_grid[0]

    transforms = []
    for lam in lambdas:
        L = np.log(lam)
        n_pts = max(3000, int(150 * L))
        t_grid = np.linspace(-L, L, n_pts)
        dt_loc = t_grid[1] - t_grid[0]

        _, evecs = get_eigensystem(lam, N, primes_all, 'cos')
        v0 = evecs[:, 0]

        f = reconstruct_on_grid(v0, t_grid, L, 'cos')
        mid = len(t_grid) // 2
        if f[mid] < 0:
            f = -f

        f_w = f * np.exp(-t_grid / 2)

        # Vectorized FT
        # F(xi) = sum_j f_w(t_j) * cos(xi * t_j) * dt
        F_real = (f_w[np.newaxis, :] * np.cos(xi_grid[:, np.newaxis] * t_grid[np.newaxis, :])).sum(axis=1) * dt_loc

        peak = np.max(np.abs(F_real))
        if peak > 1e-15:
            F_real /= peak
        transforms.append(F_real)

    print(f"\n  Successive sup-norm differences (normalized):")
    for xi_max in [15, 30, 60]:
        mask = xi_grid <= xi_max
        print(f"\n  On [0, {xi_max}]:")
        for i in range(len(lambdas) - 1):
            diff = np.max(np.abs(transforms[i][mask] - transforms[i + 1][mask]))
            print(f"    {lambdas[i]:>6} -> {lambdas[i+1]:>6}: sup|diff| = {diff:.6f}")


# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":
    print("=" * 75)
    print("WEIGHTED COMPACTNESS VERIFICATION (vectorized)")
    print("Bausteine B2, B3, T3-T6")
    print("=" * 75)

    primes_all = [int(p) for p in primerange(2, 500)]
    lambdas = [30, 50, 80, 100, 150, 200, 300, 500]
    N = 30  # Reduced for speed; 30 modes are sufficient for convergence

    t_start = time.time()

    # Build all matrices first (cached)
    print("\n  Building matrices...")
    for lam in lambdas:
        for basis in ['cos', 'sin']:
            t0 = time.time()
            get_eigensystem(lam, N, primes_all, basis)
            print(f"    lam={lam:4d}, {basis}: {time.time()-t0:.1f}s")

    print(f"\n  Matrix-Build total: {time.time()-t_start:.1f}s")

    # Tests
    test_eigenvalue_convergence(lambdas, N, primes_all)
    test_parity_gap_scaling(lambdas, N, primes_all)
    test_uniform_sobolev(lambdas, N, primes_all)
    test_relative_tail(lambdas, N, primes_all)
    test_profile_convergence(lambdas, N, primes_all)
    test_mellin_convergence(lambdas, N, primes_all)

    elapsed = time.time() - t_start

    print(f"\n{'=' * 75}")
    print(f"Gesamtzeit: {elapsed:.1f}s")
    print(f"{'=' * 75}")

    print("""
ZUSAMMENFASSUNG:
================
B2: ||psi||_H1 beschraenkt? => sum n^2 c_n^2 Spalte pruefen
B3: Masse bei Rand klein? => alpha=0.9 Spalte pruefen (< 0.1 = gut)
T3: Profil konvergent? => Successive distances fallend?
T4: Eigenwert-Skalierung => l1/L oder l1/L^2 konvergent?
T5: Even-Odd Gap waechst? => Delta negativer mit lam?
T6: Mellin konvergent? => sup|diff| fallend auf Kompakta?
""")
