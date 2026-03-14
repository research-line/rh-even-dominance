#!/usr/bin/env python3
"""
hellmann_feynman_gap.py
=======================
Hellmann-Feynman analysis of the parity gap d(Delta)/dL.

GOAL: Show that d/dL [lambda_1^+(L) - lambda_1^-(L)] < 0 for L >= L_0,
which would prove monotone deepening of even dominance.

METHOD:
  By Hellmann-Feynman:
    d/dL lambda_1^pm = <eta^pm(L), (dA/dL) eta^pm(L)>

  We compute dA/dL numerically via finite difference:
    dA/dL ~ [QW(L+h) - QW(L-h)] / (2h)

  on a FIXED basis (rescaled to [-1,1]), so the derivative is
  purely in the operator coefficients, not the domain.

  Then:
    d(Delta)/dL = <eta+, A' eta+> - <eta-, A' eta->

  If this is consistently negative for L >= L_0, we have
  monotone gap deepening.

RESCALING: We work on the physical domain [-L,L] with L=log(lam).
  The basis functions depend on L, so when L changes, both the
  operator AND the basis change. To get a clean derivative, we
  use the alternative approach:

  For each L, compute QW(L) and QW(L+dL), get eigenvectors at L,
  and compute d(lambda_1)/dL directly from the eigenvalue difference.
  Additionally decompose dA/dL into prime and archimedean contributions.

For ellmos-services (2 vCPU, 8 GB RAM).
"""

import numpy as np
from scipy.linalg import eigh, eigvalsh
from sympy import primerange
import sys
import time
import json

sys.path.insert(0, '/opt/rh_proof')
from weg2_rigorous_v3 import build_QW_float

primes = [int(p) for p in primerange(2, 600)]


def build_QW_components(lam, N, primes_used, basis='cos', n_int=2000):
    """
    Build QW decomposed into diagonal + archimedean + prime.
    Returns (W_diag, W_arch, W_prime, W_total).
    """
    L = np.log(lam)
    LOG4PI_GAMMA_F = 3.2720532309274587  # log(4*pi) + gamma

    sf = _get_shift_func(basis)

    W_diag = LOG4PI_GAMMA_F * np.eye(N)
    W_arch = np.zeros((N, N))
    W_prime = np.zeros((N, N))

    # Archimedean
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
                W_arch[i, j] += val
                if i != j:
                    W_arch[j, i] += val

    # Primes
    for p in primes_used:
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
                    W_prime[i, j] += val
                    if i != j:
                        W_prime[j, i] += val

    W_total = W_diag + W_arch + W_prime
    return W_diag, W_arch, W_prime, W_total


def _get_shift_func(basis):
    """Return the appropriate shift function."""
    if basis == 'cos':
        return _shift_cos_f
    else:
        return _shift_sin_f


def _shift_cos_f(n, m, s, L):
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


def _shift_sin_f(n, m, s, L):
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


def hellmann_feynman_analysis(lam, N, primes_used, dL_frac=0.001, n_int=2000):
    """
    Compute d(lambda_1)/dL for even and odd sectors via finite difference,
    plus Hellmann-Feynman decomposition into prime and archimedean contributions.

    Parameters
    ----------
    lam : float
        Lambda value.
    N : int
        Basis size.
    dL_frac : float
        Fractional step: dL = L * dL_frac.
    n_int : int
        Quadrature points.

    Returns
    -------
    dict with all results.
    """
    L = np.log(lam)
    dL = L * dL_frac

    # Central lambda values
    lam_minus = np.exp(L - dL)
    lam_plus = np.exp(L + dL)

    # Build QW at L (with decomposition)
    t0 = time.time()
    W_diag_c, W_arch_c, W_prime_c, W_cos = build_QW_components(
        lam, N, primes_used, 'cos', n_int)
    W_diag_s, W_arch_s, W_prime_s, W_sin = build_QW_components(
        lam, N, primes_used, 'sin', n_int)
    dt_center = time.time() - t0

    # Eigenvalues and eigenvectors at L
    evals_c, evecs_c = eigh(W_cos)
    evals_s, evecs_s = eigh(W_sin)
    l1_cos = evals_c[0]
    l1_sin = evals_s[0]
    eta_cos = evecs_c[:, 0]  # ground state eigenvector (even)
    eta_sin = evecs_s[:, 0]  # ground state eigenvector (odd)
    gap = l1_cos - l1_sin

    # Build QW at L-dL and L+dL (full matrices only)
    t0 = time.time()
    W_cos_m = build_QW_float(lam_minus, N, primes_used, 'cos', n_int)
    W_sin_m = build_QW_float(lam_minus, N, primes_used, 'sin', n_int)
    W_cos_p = build_QW_float(lam_plus, N, primes_used, 'cos', n_int)
    W_sin_p = build_QW_float(lam_plus, N, primes_used, 'sin', n_int)
    dt_fd = time.time() - t0

    # Finite-difference eigenvalues
    l1_cos_m = float(eigvalsh(W_cos_m)[0])
    l1_sin_m = float(eigvalsh(W_sin_m)[0])
    l1_cos_p = float(eigvalsh(W_cos_p)[0])
    l1_sin_p = float(eigvalsh(W_sin_p)[0])

    dl1_cos_dL = (l1_cos_p - l1_cos_m) / (2 * dL)
    dl1_sin_dL = (l1_sin_p - l1_sin_m) / (2 * dL)
    dgap_dL = dl1_cos_dL - dl1_sin_dL

    # Hellmann-Feynman decomposition: <eta, dA/dL eta>
    # dA/dL ~ (A(L+dL) - A(L-dL)) / (2dL)
    dW_cos = (W_cos_p - W_cos_m) / (2 * dL)
    dW_sin = (W_sin_p - W_sin_m) / (2 * dL)

    hf_cos = eta_cos @ dW_cos @ eta_cos  # <eta+, A' eta+>
    hf_sin = eta_sin @ dW_sin @ eta_sin  # <eta-, A' eta->
    hf_gap = hf_cos - hf_sin

    # Decomposed HF: prime vs archimedean contributions to dA/dL
    # Build components at L+dL and L-dL
    _, W_arch_c_p, W_prime_c_p, _ = build_QW_components(
        lam_plus, N, primes_used, 'cos', n_int)
    _, W_arch_c_m, W_prime_c_m, _ = build_QW_components(
        lam_minus, N, primes_used, 'cos', n_int)
    _, W_arch_s_p, W_prime_s_p, _ = build_QW_components(
        lam_plus, N, primes_used, 'sin', n_int)
    _, W_arch_s_m, W_prime_s_m, _ = build_QW_components(
        lam_minus, N, primes_used, 'sin', n_int)

    dW_arch_cos = (W_arch_c_p - W_arch_c_m) / (2 * dL)
    dW_prime_cos = (W_prime_c_p - W_prime_c_m) / (2 * dL)
    dW_arch_sin = (W_arch_s_p - W_arch_s_m) / (2 * dL)
    dW_prime_sin = (W_prime_s_p - W_prime_s_m) / (2 * dL)

    hf_arch_cos = eta_cos @ dW_arch_cos @ eta_cos
    hf_prime_cos = eta_cos @ dW_prime_cos @ eta_cos
    hf_arch_sin = eta_sin @ dW_arch_sin @ eta_sin
    hf_prime_sin = eta_sin @ dW_prime_sin @ eta_sin

    hf_arch_gap = hf_arch_cos - hf_arch_sin
    hf_prime_gap = hf_prime_cos - hf_prime_sin

    return {
        'lam': lam,
        'L': L,
        'N': N,
        'l1_cos': l1_cos,
        'l1_sin': l1_sin,
        'gap': gap,
        # Finite difference derivatives
        'dl1_cos_dL': dl1_cos_dL,
        'dl1_sin_dL': dl1_sin_dL,
        'dgap_dL': dgap_dL,
        # Hellmann-Feynman total
        'hf_cos': hf_cos,
        'hf_sin': hf_sin,
        'hf_gap': hf_gap,
        # HF decomposition
        'hf_arch_cos': hf_arch_cos,
        'hf_prime_cos': hf_prime_cos,
        'hf_arch_sin': hf_arch_sin,
        'hf_prime_sin': hf_prime_sin,
        'hf_arch_gap': hf_arch_gap,
        'hf_prime_gap': hf_prime_gap,
        # Ground state structure
        'eta_cos_norm_first4': float(np.sum(eta_cos[:4]**2)),
        'eta_sin_norm_first4': float(np.sum(eta_sin[:4]**2)),
        'dt_center': dt_center,
        'dt_fd': dt_fd,
    }


# ========== MAIN ==========

if __name__ == "__main__":
    # Lambda grid: focus on the regime where even dominance is robust
    lambdas_N60 = [55, 60, 70, 80, 90]
    lambdas_N80 = [100, 120, 140, 160, 200, 250, 300, 400, 500]

    print("HELLMANN-FEYNMAN GAP ANALYSE")
    print("=" * 120)
    print("Ziel: d(Delta)/dL < 0 => monotone Vertiefung der Even-Dominanz")
    print("=" * 120)

    header = (
        f"{'lam':>5} | {'N':>3} | {'L':>5} | {'gap':>10} | "
        f"{'dgap/dL':>10} | {'HF gap':>10} | "
        f"{'HF arch':>10} | {'HF prime':>10} | "
        f"{'eta+[0:4]':>9} | {'eta-[0:4]':>9} | {'time':>6}"
    )
    print(header)
    print("-" * 120)

    results = []

    for lam in lambdas_N60 + lambdas_N80:
        N = 80 if lam >= 100 else 60
        pu = [p for p in primes if p <= max(lam, 47)]

        t0 = time.time()
        r = hellmann_feynman_analysis(lam, N, pu, dL_frac=0.002, n_int=2000)
        dt_total = time.time() - t0

        sign_fd = "<0" if r['dgap_dL'] < -1e-6 else ">0" if r['dgap_dL'] > 1e-6 else "~0"
        sign_hf = "<0" if r['hf_gap'] < -1e-6 else ">0" if r['hf_gap'] > 1e-6 else "~0"

        print(
            f"{lam:5d} | {N:3d} | {r['L']:5.2f} | {r['gap']:+10.4f} | "
            f"{r['dgap_dL']:+10.4f} {sign_fd:>3} | {r['hf_gap']:+10.4f} {sign_hf:>3} | "
            f"{r['hf_arch_gap']:+10.4f} | {r['hf_prime_gap']:+10.4f} | "
            f"{r['eta_cos_norm_first4']:9.4f} | {r['eta_sin_norm_first4']:9.4f} | "
            f"{dt_total:5.0f}s"
        )
        sys.stdout.flush()
        results.append(r)

    # Summary
    print("\n" + "=" * 80)
    print("ZUSAMMENFASSUNG")
    print("=" * 80)

    print("\n1. Gap-Monotonie (d(Delta)/dL):")
    mono_fd = all(r['dgap_dL'] < 0 for r in results if r['lam'] >= 100)
    print(f"   d(Delta)/dL < 0 fuer ALLE lam >= 100: {mono_fd}")
    for r in results:
        if r['dgap_dL'] > 1e-6:
            print(f"   VERLETZUNG bei lam={r['lam']}: dgap/dL = {r['dgap_dL']:+.6f}")

    print("\n2. Hellmann-Feynman Dekomposition:")
    for r in results:
        if r['lam'] in [55, 100, 200, 500]:
            print(f"   lam={r['lam']:3d}: HF_gap={r['hf_gap']:+.4f} "
                  f"= arch({r['hf_arch_gap']:+.4f}) + prime({r['hf_prime_gap']:+.4f})")

    print("\n3. Treiber der Gap-Aenderung:")
    arch_dominant = sum(1 for r in results if abs(r['hf_arch_gap']) > abs(r['hf_prime_gap']))
    prime_dominant = len(results) - arch_dominant
    print(f"   Archimedisch dominiert: {arch_dominant}/{len(results)}")
    print(f"   Prime dominiert: {prime_dominant}/{len(results)}")

    print("\n4. Grundzustand-Konzentration (Anteil in ersten 4 Moden):")
    for r in results:
        if r['lam'] in [55, 100, 200, 500]:
            print(f"   lam={r['lam']:3d}: eta+ in [0:4] = {r['eta_cos_norm_first4']:.4f}, "
                  f"eta- in [0:4] = {r['eta_sin_norm_first4']:.4f}")

    # Asymptotic analysis: dgap/dL as function of L
    print("\n5. Asymptotik von d(Delta)/dL:")
    for r in results:
        if r['lam'] >= 100:
            # d(Delta)/dlam = d(Delta)/dL * dL/dlam = d(Delta)/dL * 1/lam
            dgap_dlam = r['dgap_dL'] / r['lam']
            print(f"   L={r['L']:.2f} (lam={r['lam']:3d}): "
                  f"d(Delta)/dL = {r['dgap_dL']:+.4f}, "
                  f"d(Delta)/dlam = {dgap_dlam:+.6f}")

    # Save
    outfile = '/opt/rh_proof/hellmann_feynman_results.json'
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nErgebnisse in {outfile}")
