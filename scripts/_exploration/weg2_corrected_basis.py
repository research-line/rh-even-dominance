#!/usr/bin/env python3
"""
weg2_corrected_basis.py
========================
KORRIGIERTE Galerkin-Berechnung mit ORTHOGONALER Basis.

BUG im Original: cos(n*pi*t/(2L)) ist NICHT orthogonal auf [-L, L]!
Gram-Matrix hat ||M - I|| = 2.28, cond(M) = 10^7.
Alle Eigenwerte in den bisherigen Scripts sind FALSCH.

KORREKTE ORTHONORMALE BASIS:
  Even: psi_0(t) = 1/sqrt(2L), psi_n(t) = cos(n*pi*t/L)/sqrt(L) for n >= 1
  Odd:  psi_n(t) = sin(n*pi*t/L)/sqrt(L) for n >= 1

Verifizierung: <psi_n, psi_m> = delta_{nm} (orthonormal)

Product formulas (unchanged structure, different frequencies):
  cos(a)*cos(b) = [cos(a-b) + cos(a+b)] / 2
  sin(a)*sin(b) = [cos(a-b) - cos(a+b)] / 2
"""

import numpy as np
from scipy.linalg import eigh
from scipy.integrate import quad as scipy_quad
from mpmath import euler as mp_euler, log as mplog, pi as mppi
import time

LOG4PI_GAMMA = float(mplog(4 * mppi)) + float(mp_euler)


# ========== CORRECTED SHIFT ELEMENTS ==========

def shift_cos_correct(n, m, s, L):
    """
    Corrected cos shift: <psi_n, S_s psi_m> with psi_n = cos(n*pi*t/L)/sqrt(L)
    and psi_0 = 1/sqrt(2L).

    Integration: [max(-L, s-L), min(L, s+L)]
    """
    if abs(s) > 2 * L:
        return 0.0

    a = max(-L, s - L)
    b = min(L, s + L)
    if a >= b:
        return 0.0

    # Normalization
    if n == 0 and m == 0:
        norm = 1.0 / (2 * L)
    elif n == 0 or m == 0:
        norm = 1.0 / (L * np.sqrt(2))
    else:
        norm = 1.0 / L

    # CORRECTED frequencies: n*pi/L (not n*pi/(2L))
    kn = n * np.pi / L
    km = m * np.pi / L

    # Product formula: cos(kn*t) * cos(km*(t-s))
    # = (1/2)[cos((kn-km)*t + km*s) + cos((kn+km)*t - km*s)]
    result = 0.0
    for freq, phase in [(kn - km, km * s), (kn + km, -km * s)]:
        if abs(freq) < 1e-12:
            result += np.cos(phase) * (b - a) / 2
        else:
            result += (np.sin(freq * b + phase) - np.sin(freq * a + phase)) / (2 * freq)

    return norm * result


def shift_sin_correct(n, m, s, L):
    """
    Corrected sin shift: <psi_n, S_s psi_m> with psi_n = sin(n*pi*t/L)/sqrt(L).
    Note: n, m >= 1 (mode 0 doesn't exist for sin).
    In the code, we pass n=0,1,2,... meaning mode n+1.
    To match the old interface: sin((n+1)*pi*t/L)/sqrt(L) for n=0,1,2,...
    """
    if abs(s) > 2 * L:
        return 0.0

    a = max(-L, s - L)
    b = min(L, s + L)
    if a >= b:
        return 0.0

    norm = 1.0 / L

    # CORRECTED frequencies: (n+1)*pi/L (not (n+1)*pi/(2L))
    kn = (n + 1) * np.pi / L
    km = (m + 1) * np.pi / L

    # sin(kn*t) * sin(km*(t-s)) = (1/2)[cos((kn-km)*t + km*s) - cos((kn+km)*t - km*s)]
    result = 0.0
    for freq, phase, sign in [(kn - km, km * s, +1), (kn + km, -km * s, -1)]:
        if abs(freq) < 1e-12:
            result += sign * np.cos(phase) * (b - a) / 2
        else:
            result += sign * (np.sin(freq * b + phase) - np.sin(freq * a + phase)) / (2 * freq)

    return norm * result


# ========== BUILD QW MATRIX ==========

def build_QW_corrected(lam, N, primes, basis='cos', n_int=None, use_adaptive=False):
    """
    Build QW matrix with corrected orthogonal basis.

    If use_adaptive=True: use scipy.integrate.quad for archimedean kernel.
    Otherwise: trapezoidal rule with n_int quadrature points.
    """
    L = np.log(lam)

    if n_int is None:
        n_int = max(2000, 30 * N)

    W = LOG4PI_GAMMA * np.eye(N)

    shift_func = shift_cos_correct if basis == 'cos' else shift_sin_correct

    if use_adaptive:
        # Adaptive quadrature for each element
        for i in range(N):
            for j in range(i, N):
                is_diag = (i == j)

                def integrand(s, _i=i, _j=j, _diag=is_diag):
                    if s < 1e-15:
                        return 0.0
                    K = np.exp(s / 2) / (2.0 * np.sinh(s))
                    sp = shift_func(_i, _j, s, L)
                    sm = shift_func(_i, _j, -s, L)
                    reg = -2.0 * np.exp(-s / 2) if _diag else 0.0
                    return K * (sp + sm + reg)

                val, err = scipy_quad(integrand, 0, 2 * L, limit=200,
                                      epsabs=1e-12, epsrel=1e-10)
                W[i, j] += val
                if i != j:
                    W[j, i] += val
    else:
        # Trapezoidal quadrature
        s_max = min(2 * L, 12.0)
        s_grid = np.linspace(0.005, s_max, n_int)
        ds = s_grid[1] - s_grid[0]

        for s in s_grid:
            K = np.exp(s / 2) / (2.0 * np.sinh(s))
            if K < 1e-15:
                continue
            for i in range(N):
                for j in range(i, N):
                    sp = shift_func(i, j, s, L)
                    sm = shift_func(i, j, -s, L)
                    reg = -2.0 * np.exp(-s / 2) * (1.0 if i == j else 0.0)
                    val = K * (sp + sm + reg) * ds
                    W[i, j] += val
                    if i != j:
                        W[j, i] += val

    # Prime contributions (exact closed-form trig integrals)
    for p in primes:
        logp = np.log(p)
        for m_exp in range(1, 20):
            coeff = logp * p ** (-m_exp / 2.0)
            shift = m_exp * logp
            if shift >= 2 * L:
                break
            for i in range(N):
                for j in range(i, N):
                    sp = shift_func(i, j, shift, L)
                    sm = shift_func(i, j, -shift, L)
                    val = coeff * (sp + sm)
                    W[i, j] += val
                    if i != j:
                        W[j, i] += val

    return W


def verify_orthogonality():
    """Verify that the corrected basis is orthonormal."""
    L = np.log(100)
    N = 10

    # Numerical Gram matrix
    n_quad = 5000
    t_grid = np.linspace(-L, L, n_quad)
    dt = t_grid[1] - t_grid[0]

    # Cos basis
    M_cos = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            if i == 0:
                fi = np.ones_like(t_grid) / np.sqrt(2 * L)
            else:
                fi = np.cos(i * np.pi * t_grid / L) / np.sqrt(L)
            if j == 0:
                fj = np.ones_like(t_grid) / np.sqrt(2 * L)
            else:
                fj = np.cos(j * np.pi * t_grid / L) / np.sqrt(L)
            M_cos[i, j] = np.sum(fi * fj) * dt
            M_cos[j, i] = M_cos[i, j]

    # Sin basis
    M_sin = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            fi = np.sin((i + 1) * np.pi * t_grid / L) / np.sqrt(L)
            fj = np.sin((j + 1) * np.pi * t_grid / L) / np.sqrt(L)
            M_sin[i, j] = np.sum(fi * fj) * dt
            M_sin[j, i] = M_sin[i, j]

    print("ORTHOGONALITAETS-CHECK (korrigierte Basis):")
    print(f"  Cos: ||M - I||_F = {np.linalg.norm(M_cos - np.eye(N), 'fro'):.2e}")
    print(f"  Sin: ||M - I||_F = {np.linalg.norm(M_sin - np.eye(N), 'fro'):.2e}")
    print(f"  Cos cond(M) = {np.linalg.cond(M_cos):.2f}")
    print(f"  Sin cond(M) = {np.linalg.cond(M_sin):.2f}")

    return M_cos, M_sin


def compare_old_vs_new():
    """Compare eigenvalues with old (wrong) and new (correct) basis."""
    from sympy import primerange
    from weg2_analytic_even_odd import build_QW_analytic  # old code

    primes = list(primerange(2, 200))

    print("\n" + "=" * 80)
    print("VERGLEICH: Alte (falsche) vs korrigierte Basis")
    print("=" * 80)

    for lam in [30, 50, 100]:
        primes_used = [p for p in primes if p <= max(lam, 47)]
        N = 20

        # Old (wrong) basis
        W_cos_old = build_QW_analytic(lam, N, primes_used, 'cos')
        W_sin_old = build_QW_analytic(lam, N, primes_used, 'sin')
        l1c_old = np.sort(eigh(W_cos_old, eigvals_only=True))[0]
        l1s_old = np.sort(eigh(W_sin_old, eigvals_only=True))[0]

        # New (corrected) basis
        W_cos_new = build_QW_corrected(lam, N, primes_used, 'cos')
        W_sin_new = build_QW_corrected(lam, N, primes_used, 'sin')
        l1c_new = np.sort(eigh(W_cos_new, eigvals_only=True))[0]
        l1s_new = np.sort(eigh(W_sin_new, eigvals_only=True))[0]

        print(f"\nlambda={lam}, N={N}:")
        print(f"  ALT (falsch): l1_cos={l1c_old:+.6f}  l1_sin={l1s_old:+.6f}  "
              f"gap={l1c_old - l1s_old:+.6f}  {'EVEN' if l1c_old < l1s_old else 'ODD'}")
        print(f"  NEU (korrekt): l1_cos={l1c_new:+.6f}  l1_sin={l1s_new:+.6f}  "
              f"gap={l1c_new - l1s_new:+.6f}  {'EVEN' if l1c_new < l1s_new else 'ODD'}")


def main_analysis():
    """Hauptanalyse mit korrigierter Basis."""
    from sympy import primerange
    primes = list(primerange(2, 200))

    print("\n" + "=" * 80)
    print("KORRIGIERTE EVEN/ODD ANALYSE")
    print("=" * 80)

    print(f"\n{'lam':>5} | {'N':>3} | {'l1_cos':>12} | {'l1_sin':>12} | {'gap':>12} | dom")
    print("-" * 65)

    for lam in [10, 20, 30, 50, 100, 200]:
        primes_used = [p for p in primes if p <= max(lam, 47)]
        for N in [10, 20, 30]:
            W_cos = build_QW_corrected(lam, N, primes_used, 'cos')
            W_sin = build_QW_corrected(lam, N, primes_used, 'sin')
            l1c = np.sort(eigh(W_cos, eigvals_only=True))[0]
            l1s = np.sort(eigh(W_sin, eigvals_only=True))[0]
            gap = l1c - l1s
            dom = "EVEN" if gap < 0 else "ODD"
            print(f"  {lam:3d} | {N:3d} | {l1c:+12.6f} | {l1s:+12.6f} | {gap:+12.6f} | {dom}")


def adaptive_analysis():
    """Analyse mit adaptiver Quadratur (certified arch integral)."""
    from sympy import primerange
    primes = list(primerange(2, 200))

    print("\n" + "=" * 80)
    print("ADAPTIVE QUADRATUR (korrigierte Basis)")
    print("=" * 80)

    for lam in [30, 100]:
        primes_used = [p for p in primes if p <= max(lam, 47)]
        N = 10

        t0 = time.time()
        W_cos = build_QW_corrected(lam, N, primes_used, 'cos', use_adaptive=True)
        W_sin = build_QW_corrected(lam, N, primes_used, 'sin', use_adaptive=True)
        dt = time.time() - t0

        l1c = np.sort(eigh(W_cos, eigvals_only=True))[0]
        l1s = np.sort(eigh(W_sin, eigvals_only=True))[0]
        gap = l1c - l1s
        dom = "EVEN" if gap < 0 else "ODD"

        print(f"\nlambda={lam}, N={N} (adaptive, {dt:.1f}s):")
        print(f"  l1_cos = {l1c:+.10f}")
        print(f"  l1_sin = {l1s:+.10f}")
        print(f"  gap = {gap:+.10f} ({dom})")


if __name__ == "__main__":
    t_start = time.time()

    # Step 1: Verify orthogonality
    verify_orthogonality()

    # Step 2: Compare old vs new
    compare_old_vs_new()

    # Step 3: Main analysis with corrected basis
    main_analysis()

    # Step 4: Adaptive quadrature (small N)
    adaptive_analysis()

    print(f"\nTotal: {time.time() - t_start:.1f}s")
