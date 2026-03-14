#!/usr/bin/env python3
"""
weg2_monotonie_test.py
======================
Monotonie-Test: Wird |gamma_approx - gamma_true| kleiner mit wachsendem S?

Strategie:
  Wir arbeiten auf BEIDEN Seiten:
  (A) Gram-Matrix Q_zeros = V^T V  (Referenz, exakt PSD)
  (B) Explizitformel Q_S = Q_arch - Q_prime  (mit S Primzahlen)

  Fuer jedes S:
  1. Baue Q_S
  2. Finde kleinsten Eigenvektor von Q_S
  3. Rekonstruiere Funktion f_S(t) = sum c_n phi_n(t)
  4. Finde Nullstellen von f_S(t)
  5. Vergleiche mit echten gamma_k

  Zusaetzlich:
  - Eigenvector-Alignment: cos(angle) zwischen Kern-EVs von Q_S und Q_zeros
  - Subspace-Tracking: Hauptwinkel zwischen Kern-Raeumen
"""

import numpy as np
from mpmath import mp, im, zetazero, digamma, log, pi, euler
from sympy import primerange

mp.dps = 25

# ===========================================================================
# Basis und Nullstellen
# ===========================================================================

def load_zeros(K):
    gammas = []
    for k in range(1, K + 1):
        gammas.append(float(im(zetazero(k))))
    return np.array(gammas)

def cosine_basis(n, t, T):
    if n == 0:
        return 1.0 / np.sqrt(T)
    return np.sqrt(2.0 / T) * np.cos(n * np.pi * t / T)

def phi_hat_cos(n, xi, T):
    omega_n = n * np.pi / T
    if n == 0:
        if abs(xi) < 1e-15:
            return np.sqrt(T)
        return np.sin(xi * T) / (xi * np.sqrt(T))
    c = np.sqrt(2.0 / T)
    def sinc_term(w):
        if abs(w) < 1e-15:
            return T
        return np.sin(w * T) / w
    return c * 0.5 * (sinc_term(omega_n - xi) + sinc_term(omega_n + xi))

# ===========================================================================
# Q_zeros (Gram-Matrix Referenz)
# ===========================================================================

def build_QW_zeros(gammas, N_basis, T):
    K = len(gammas)
    V = np.zeros((K, N_basis))
    for k in range(K):
        for n in range(N_basis):
            V[k, n] = cosine_basis(n, gammas[k], T)
    return V.T @ V, V

# ===========================================================================
# Q_S (Explizitformel-Seite)
# ===========================================================================

def build_QW_primes(N_basis, T, primes, M_terms=5):
    # Prime contribution
    Q_prime = np.zeros((N_basis, N_basis))
    for p in primes:
        logp = np.log(p)
        for k in range(1, M_terms + 1):
            coeff = 2.0 * logp / p**(k / 2.0)
            xi = k * logp
            h_vals = np.array([phi_hat_cos(n, xi, T) for n in range(N_basis)])
            Q_prime += coeff * np.outer(h_vals, h_vals)

    # Archimedean contribution
    Q_arch = np.zeros((N_basis, N_basis))
    log4pi_half = float(log(4 * pi)) / 2
    euler_half = float(euler) / 2
    n_quad = 300
    t_quad = np.linspace(0.1, T - 0.1, n_quad)
    dt = t_quad[1] - t_quad[0]

    Phi_vals = np.zeros(n_quad)
    for i, t in enumerate(t_quad):
        Phi_vals[i] = float(digamma(0.25 + 1j * t / 2).real) + log4pi_half + euler_half

    phi_grid = np.zeros((N_basis, n_quad))
    for n in range(N_basis):
        for i, t in enumerate(t_quad):
            phi_grid[n, i] = cosine_basis(n, t, T)

    for n in range(N_basis):
        for m in range(n, N_basis):
            val = np.sum(phi_grid[n] * phi_grid[m] * Phi_vals) * dt / np.pi
            Q_arch[n, m] = val
            if m != n:
                Q_arch[m, n] = val

    return Q_arch - Q_prime, Q_arch, Q_prime

# ===========================================================================
# Nullstellen-Finder
# ===========================================================================

def find_zeros_from_evec(evec, N_basis, T, n_grid=8000):
    """Rekonstruiere Funktion und finde Nullstellen."""
    t_grid = np.linspace(0.5, T - 0.5, n_grid)
    f_vals = np.zeros(n_grid)
    for n in range(N_basis):
        for i, t in enumerate(t_grid):
            f_vals[i] += evec[n] * cosine_basis(n, t, T)

    zeros = []
    for i in range(len(f_vals) - 1):
        if f_vals[i] * f_vals[i+1] < 0:
            t0 = t_grid[i] - f_vals[i] * (t_grid[i+1] - t_grid[i]) / (f_vals[i+1] - f_vals[i])
            zeros.append(t0)
    return np.array(zeros)

def match_zeros(approx_zeros, true_gammas, tol=5.0):
    """Matche approximierte Nullstellen zu echten gammas."""
    matches = []
    for k, gk in enumerate(true_gammas):
        dists = np.abs(approx_zeros - gk)
        if len(dists) == 0:
            continue
        best_idx = np.argmin(dists)
        if dists[best_idx] < tol:
            matches.append((k, gk, approx_zeros[best_idx], dists[best_idx]))
    return matches

# ===========================================================================
# TEST 1: Nullstellen-Monotonie
# ===========================================================================

def test_zero_monotonie(gammas, N_basis, T):
    """Wird die Nullstellen-Approximation mit wachsendem S besser?"""
    print(f"\n{'='*75}")
    print(f"TEST 1: NULLSTELLEN-APPROXIMATION vs. |S|")
    print(f"  N_basis={N_basis}, T={T:.0f}, K_zeros={len(gammas)}")
    print(f"{'='*75}")

    # Referenz: Kern-Eigenvektoren von Q_zeros
    Q_ref, _ = build_QW_zeros(gammas, N_basis, T)
    evals_ref, evecs_ref = np.linalg.eigh(Q_ref)

    # Anzahl Kern-EVs der Referenz
    threshold = 1e-6
    n_kern_ref = np.sum(np.abs(evals_ref) < threshold)
    print(f"\n  Referenz: {n_kern_ref} Kern-EVs (threshold={threshold})")

    # Referenz-Nullstellen aus kleinstem EV
    ref_zeros = find_zeros_from_evec(evecs_ref[:, 0], N_basis, T)
    ref_matches = match_zeros(ref_zeros, gammas)
    if ref_matches:
        print(f"  Referenz-EV0 hat {len(ref_zeros)} Nullstellen, {len(ref_matches)} matchen")
        for k, gk, approx, err in ref_matches[:5]:
            print(f"    gamma_{k+1} = {gk:.6f}, approx = {approx:.6f}, err = {err:.6f}")

    # Wachsende Primmengen
    all_primes = list(primerange(2, 500))
    S_sizes = [2, 3, 5, 8, 12, 20, 30, 50, 75, 95]

    print(f"\n  {'|S|':>5} | {'p_max':>5} | {'lam_min':>12} | {'#zeros':>6} | "
          f"{'err_g1':>10} | {'err_g2':>10} | {'err_g3':>10} | {'mean_err':>10}")
    print(f"  {'-'*5}-+-{'-'*5}-+-{'-'*12}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    results = []
    for S_size in S_sizes:
        if S_size > len(all_primes):
            break
        primes_S = all_primes[:S_size]
        Q_S, _, _ = build_QW_primes(N_basis, T, primes_S)
        evals_S, evecs_S = np.linalg.eigh(Q_S)

        # Kleinster EV -> Nullstellen
        zeros_S = find_zeros_from_evec(evecs_S[:, 0], N_basis, T)
        matches = match_zeros(zeros_S, gammas[:10])  # erste 10 gammas

        errs = [None, None, None]
        for k, gk, approx, err in matches:
            if k < 3:
                errs[k] = err

        mean_err = np.mean([m[3] for m in matches]) if matches else float('inf')

        results.append({
            'S_size': S_size, 'lam_min': evals_S[0],
            'n_zeros': len(zeros_S), 'matches': matches,
            'errs': errs, 'mean_err': mean_err
        })

        err_strs = [f"{e:10.6f}" if e is not None else f"{'--':>10}" for e in errs]
        print(f"  {S_size:5d} | {primes_S[-1]:5d} | {evals_S[0]:+12.6e} | "
              f"{len(zeros_S):6d} | {err_strs[0]} | {err_strs[1]} | {err_strs[2]} | "
              f"{mean_err:10.6f}")

    # Monotonie-Check
    print(f"\n  MONOTONIE-CHECK:")
    for k in range(3):
        errs_k = [(r['S_size'], r['errs'][k]) for r in results if r['errs'][k] is not None]
        if len(errs_k) < 2:
            print(f"    gamma_{k+1}: zu wenig Daten")
            continue
        monotone = all(errs_k[i][1] >= errs_k[i+1][1] for i in range(len(errs_k)-1))
        trend = "MONOTON FALLEND" if monotone else "NICHT monoton"
        first_err = errs_k[0][1]
        last_err = errs_k[-1][1]
        print(f"    gamma_{k+1}: {trend}  ({first_err:.6f} -> {last_err:.6f})")

    return results

# ===========================================================================
# TEST 2: Eigenvektor-Alignment
# ===========================================================================

def test_evec_alignment(gammas, N_basis, T):
    """Konvergieren Kern-Eigenvektoren von Q_S gegen die von Q_zeros?"""
    print(f"\n{'='*75}")
    print(f"TEST 2: EIGENVEKTOR-ALIGNMENT Q_S vs. Q_zeros")
    print(f"{'='*75}")

    Q_ref, _ = build_QW_zeros(gammas, N_basis, T)
    evals_ref, evecs_ref = np.linalg.eigh(Q_ref)

    # Kern-Raeume der Referenz
    threshold = 1e-6
    kern_idx_ref = np.where(np.abs(evals_ref) < threshold)[0]
    n_kern = len(kern_idx_ref)
    V_kern_ref = evecs_ref[:, kern_idx_ref]  # N_basis x n_kern

    print(f"\n  Referenz Kern-Dimension: {n_kern}")

    all_primes = list(primerange(2, 500))
    S_sizes = [2, 5, 10, 20, 40, 75, 95]

    print(f"\n  {'|S|':>5} | {'cos(EV0,ref0)':>14} | {'cos(EV0,ref1)':>14} | "
          f"{'Kern-Proj':>10} | {'SubspAngle':>10}")
    print(f"  {'-'*5}-+-{'-'*14}-+-{'-'*14}-+-{'-'*10}-+-{'-'*10}")

    for S_size in S_sizes:
        if S_size > len(all_primes):
            break
        primes_S = all_primes[:S_size]
        Q_S, _, _ = build_QW_primes(N_basis, T, primes_S)
        evals_S, evecs_S = np.linalg.eigh(Q_S)

        v0_S = evecs_S[:, 0]  # kleinster EV von Q_S

        # Alignment mit Referenz-Kern-EVs
        cos0 = abs(np.dot(v0_S, evecs_ref[:, kern_idx_ref[0]])) if n_kern > 0 else 0
        cos1 = abs(np.dot(v0_S, evecs_ref[:, kern_idx_ref[1]])) if n_kern > 1 else 0

        # Projektion auf gesamten Kern-Raum
        proj = np.linalg.norm(V_kern_ref.T @ v0_S) if n_kern > 0 else 0

        # Hauptwinkel zwischen Kern-Raeumen
        # Q_S: nehme die n_kern kleinsten EVs
        V_kern_S = evecs_S[:, :n_kern]
        if n_kern > 0:
            M = V_kern_ref.T @ V_kern_S
            svs = np.linalg.svd(M, compute_uv=False)
            min_sv = svs[-1] if len(svs) > 0 else 0
            max_angle = np.arccos(np.clip(min_sv, -1, 1)) * 180 / np.pi
        else:
            max_angle = 90.0

        print(f"  {S_size:5d} | {cos0:14.6f} | {cos1:14.6f} | "
              f"{proj:10.6f} | {max_angle:10.2f} deg")

# ===========================================================================
# TEST 3: Spektralluecke der Primzahl-Seite
# ===========================================================================

def test_spectral_gap_detail(N_basis, T):
    """Detaillierte Spektralluecke: alle Eigenwerte von Q_S fuer verschiedene S."""
    print(f"\n{'='*75}")
    print(f"TEST 3: SPEKTRUM von Q_S (alle Eigenwerte)")
    print(f"{'='*75}")

    all_primes = list(primerange(2, 500))
    S_sizes = [3, 10, 30, 75, 95]

    for S_size in S_sizes:
        if S_size > len(all_primes):
            break
        primes_S = all_primes[:S_size]
        Q_S, Q_arch, Q_prime = build_QW_primes(N_basis, T, primes_S)
        evals_S = np.sort(np.linalg.eigvalsh(Q_S))

        # Trace-Decomposition
        tr_arch = np.trace(Q_arch)
        tr_prime = np.trace(Q_prime)
        tr_total = np.trace(Q_S)

        print(f"\n  |S|={S_size} (p_max={primes_S[-1]}):")
        print(f"    trace(Q_arch) = {tr_arch:+.6e}")
        print(f"    trace(Q_prime) = {tr_prime:+.6e}")
        print(f"    trace(Q_S) = {tr_total:+.6e}  (ratio arch/prime = {tr_arch/tr_prime:.4f})")
        print(f"    Eigenwerte: {np.array2string(evals_S, precision=4, separator=', ')}")
        n_neg = np.sum(evals_S < -1e-10)
        print(f"    n_negative = {n_neg}, lambda_min = {evals_S[0]:+.6e}, lambda_max = {evals_S[-1]:+.6e}")

# ===========================================================================
# HAUPTPROGRAMM
# ===========================================================================

if __name__ == "__main__":
    print("=" * 75)
    print("WEG 2: MONOTONIE-TEST")
    print("Wird |gamma_approx - gamma_true| kleiner mit wachsendem S?")
    print("=" * 75)

    K_ZEROS = 30
    N_BASIS = 15
    T = 120.0

    print(f"\n  Lade {K_ZEROS} Zeta-Nullstellen...")
    gammas = load_zeros(K_ZEROS)
    print(f"  gamma_1 = {gammas[0]:.6f}, gamma_30 = {gammas[-1]:.6f}")

    # Test 1: Nullstellen-Monotonie
    results = test_zero_monotonie(gammas, N_BASIS, T)

    # Test 2: Eigenvektor-Alignment
    test_evec_alignment(gammas, N_BASIS, T)

    # Test 3: Spektralluecke Detail
    test_spectral_gap_detail(N_BASIS, T)

    # FAZIT
    print(f"\n{'='*75}")
    print(f"FAZIT: MONOTONIE-TEST")
    print(f"{'='*75}")
    print(f"""
  KERNFRAGEN:
  1. Konvergieren die Nullstellen-Approximationen MONOTON gegen gamma_k?
     => Wenn ja: Starkes Indiz fuer Konvergenz S -> inf
     => Wenn nein: Fluktuationen koennen auf fehlende Terme hinweisen

  2. Konvergieren die Kern-Eigenvektoren (Subspace-Winkel)?
     => Wenn Subspace-Winkel -> 0: Kern stabilisiert sich
     => Schnelle Konvergenz waere ein starkes Argument

  3. Bleibt Q_S positiv semidefinit?
     => trace(Q_arch) vs trace(Q_prime): Ratio muss > 1 bleiben
     => Wenn Ratio -> 1: Grenzfall, erfordert Feinanalyse
""")
