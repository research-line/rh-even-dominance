#!/usr/bin/env python3
"""
weg2_gram_konvergenz.py
=======================
Gram-seitiger Konvergenz-Test.

Frage: Konvergiert der Kern von Q_zeros^K = V_K^T V_K wenn K -> inf?

Das ist die SAUBERE Seite (keine Normalisierungsprobleme).
Die Gram-Matrix V_K^T V_K ist automatisch PSD.

Tests:
  1. Kern-Dimension vs K (bei festem N, T)
  2. Rang-Saettigung: Ab welchem K ist rank = N?
  3. Spektralluecke: lambda_min(Q^K) vs K
  4. Kern-Vektor-Konvergenz: Stabilisieren sich die Eigenvektoren?
  5. Nullstellen-Rekonstruktion: Werden neue gammas "eingebaut"?
  6. Extrapolation: Was passiert fuer K >> N? (Ueberdeterminiert)
  7. T-Abhaengigkeit: Wie aendert sich Konvergenz mit T?
"""

import numpy as np
from mpmath import mp, im, zetazero
from itertools import combinations

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

def build_gram(gammas, N_basis, T):
    """Gram-Matrix Q = V^T V."""
    K = len(gammas)
    V = np.zeros((K, N_basis))
    for k in range(K):
        for n in range(N_basis):
            V[k, n] = cosine_basis(n, gammas[k], T)
    return V.T @ V, V

# ===========================================================================
# TEST 1: Rang und Kern vs K
# ===========================================================================

def test_rank_vs_K(all_gammas, N_basis, T):
    """Wie entwickeln sich Rang und Kern-Dimension mit wachsendem K?"""
    print(f"\n{'='*75}")
    print(f"TEST 1: RANG UND KERN VON Q^K = V_K^T V_K vs. K")
    print(f"  N_basis={N_basis}, T={T:.0f}")
    print(f"{'='*75}")

    K_max = len(all_gammas)
    K_vals = list(range(1, min(N_basis + 5, K_max + 1))) + \
             [N_basis + 10, N_basis + 20, 2 * N_basis, 3 * N_basis, K_max]
    K_vals = sorted(set([k for k in K_vals if k <= K_max]))

    print(f"\n  {'K':>5} | {'rank':>5} | {'kern':>5} | {'lam_min':>12} | "
          f"{'lam_max':>12} | {'cond':>12} | {'trace':>12}")
    print(f"  {'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    results = []
    for K in K_vals:
        Q, V = build_gram(all_gammas[:K], N_basis, T)
        evals = np.sort(np.linalg.eigvalsh(Q))

        threshold = 1e-10
        rank = np.sum(evals > threshold)
        kern = N_basis - rank

        lam_min_pos = evals[evals > threshold][0] if rank > 0 else 0
        cond = evals[-1] / lam_min_pos if lam_min_pos > 0 else float('inf')

        results.append({
            'K': K, 'rank': rank, 'kern': kern,
            'lam_min': evals[0], 'lam_max': evals[-1],
            'cond': cond, 'trace': np.trace(Q), 'evals': evals
        })

        cond_str = f"{cond:12.2e}" if cond < 1e15 else f"{'inf':>12}"
        print(f"  {K:5d} | {rank:5d} | {kern:5d} | {evals[0]:+12.6e} | "
              f"{evals[-1]:+12.6e} | {cond_str} | {np.trace(Q):12.4f}")

    return results

# ===========================================================================
# TEST 2: Eigenvektor-Stabilisierung
# ===========================================================================

def test_evec_stability(all_gammas, N_basis, T):
    """Stabilisieren sich die Eigenvektoren wenn K waechst?"""
    print(f"\n{'='*75}")
    print(f"TEST 2: EIGENVEKTOR-STABILITAET vs. K")
    print(f"{'='*75}")

    K_max = len(all_gammas)
    # Referenz: groesstes K
    Q_ref, _ = build_gram(all_gammas, N_basis, T)
    _, evecs_ref = np.linalg.eigh(Q_ref)

    K_vals = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, K_max]
    K_vals = [k for k in K_vals if k <= K_max]

    print(f"\n  Referenz: K={K_max}")
    print(f"  Messe: |cos(angle)| zwischen EV_i(K) und EV_i(K_max)")
    print(f"\n  {'K':>5} | {'EV_0':>8} | {'EV_1':>8} | {'EV_2':>8} | "
          f"{'EV_N-1':>8} | {'EV_N-2':>8} | {'SubspWinkel':>12}")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*12}")

    for K in K_vals:
        Q, _ = build_gram(all_gammas[:K], N_basis, T)
        evals, evecs = np.linalg.eigh(Q)

        # Alignment pro EV (absolute Werte wegen Vorzeichen-Ambiguitaet)
        cos0 = abs(np.dot(evecs[:, 0], evecs_ref[:, 0]))
        cos1 = abs(np.dot(evecs[:, 1], evecs_ref[:, 1]))
        cos2 = abs(np.dot(evecs[:, 2], evecs_ref[:, 2]))
        cosN1 = abs(np.dot(evecs[:, -1], evecs_ref[:, -1]))
        cosN2 = abs(np.dot(evecs[:, -2], evecs_ref[:, -2]))

        # Subspace-Winkel: 3 kleinste EVs
        n_sub = min(3, N_basis)
        V_sub = evecs[:, :n_sub]
        V_ref_sub = evecs_ref[:, :n_sub]
        M = V_sub.T @ V_ref_sub
        svs = np.linalg.svd(M, compute_uv=False)
        max_angle = np.arccos(np.clip(svs[-1], -1, 1)) * 180 / np.pi

        print(f"  {K:5d} | {cos0:8.4f} | {cos1:8.4f} | {cos2:8.4f} | "
              f"{cosN1:8.4f} | {cosN2:8.4f} | {max_angle:10.2f} deg")

# ===========================================================================
# TEST 3: Nullstellen-Rekonstruktion bei wachsendem K
# ===========================================================================

def test_zero_reconstruction(all_gammas, N_basis, T):
    """Wie gut rekonstruiert der kleinste EV die Nullstellen bei wachsendem K?"""
    print(f"\n{'='*75}")
    print(f"TEST 3: NULLSTELLEN-REKONSTRUKTION aus EV_0 vs. K")
    print(f"{'='*75}")

    K_vals = [5, 10, 15, 20, 30, 50, 75, len(all_gammas)]
    K_vals = [k for k in K_vals if k <= len(all_gammas)]

    t_grid = np.linspace(0.5, T - 0.5, 10000)

    print(f"\n  {'K':>5} | {'#zeros':>6} | {'Delta_g1':>10} | {'Delta_g2':>10} | "
          f"{'Delta_g3':>10} | {'median_D':>10} | {'phantom':>7}")
    print(f"  {'-'*5}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*7}")

    for K in K_vals:
        Q, _ = build_gram(all_gammas[:K], N_basis, T)
        evals, evecs = np.linalg.eigh(Q)
        v0 = evecs[:, 0]

        # Rekonstruiere Funktion
        f_vals = np.zeros(len(t_grid))
        for n in range(N_basis):
            for i, t in enumerate(t_grid):
                f_vals[i] += v0[n] * cosine_basis(n, t, T)

        # Finde Nullstellen
        zeros = []
        for i in range(len(f_vals) - 1):
            if f_vals[i] * f_vals[i+1] < 0:
                t0 = t_grid[i] - f_vals[i] * (t_grid[i+1] - t_grid[i]) / (f_vals[i+1] - f_vals[i])
                zeros.append(t0)
        zeros = np.array(zeros)

        # Matche mit echten gammas
        deltas = [None, None, None]
        matched = 0
        phantom = 0
        all_deltas = []

        for z in zeros:
            dists = np.abs(all_gammas[:K] - z)
            best_dist = np.min(dists)
            if best_dist < 3.0:
                matched += 1
                all_deltas.append(best_dist)
            else:
                phantom += 1

        for gk_idx in range(min(3, K)):
            if len(zeros) > 0:
                d = np.min(np.abs(zeros - all_gammas[gk_idx]))
                deltas[gk_idx] = d

        median_d = np.median(all_deltas) if all_deltas else float('inf')

        d_strs = [f"{d:10.6f}" if d is not None else f"{'--':>10}" for d in deltas]
        print(f"  {K:5d} | {len(zeros):6d} | {d_strs[0]} | {d_strs[1]} | "
              f"{d_strs[2]} | {median_d:10.6f} | {phantom:7d}")

# ===========================================================================
# TEST 4: T-Abhaengigkeit der Konvergenz
# ===========================================================================

def test_T_dependence(all_gammas, N_basis):
    """Wie aendert sich Konvergenz mit T?"""
    print(f"\n{'='*75}")
    print(f"TEST 4: T-ABHAENGIGKEIT")
    print(f"{'='*75}")

    T_vals = [50, 80, 100, 120, 150, 200]
    K = len(all_gammas)

    print(f"\n  K={K}, N_basis={N_basis}")
    print(f"\n  {'T':>6} | {'rank':>5} | {'kern':>5} | {'lam_min':>12} | "
          f"{'lam_max':>12} | {'cond':>12} | {'trace':>12}")
    print(f"  {'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for T in T_vals:
        # Nur gammas < T verwenden
        mask = all_gammas < T
        gammas_T = all_gammas[mask]
        if len(gammas_T) < 2:
            continue

        Q, _ = build_gram(gammas_T, N_basis, T)
        evals = np.sort(np.linalg.eigvalsh(Q))

        threshold = 1e-10
        rank = np.sum(evals > threshold)
        kern = N_basis - rank
        lam_min_pos = evals[evals > threshold][0] if rank > 0 else 0
        cond = evals[-1] / lam_min_pos if lam_min_pos > 0 else float('inf')

        cond_str = f"{cond:12.2e}" if cond < 1e15 else f"{'inf':>12}"
        print(f"  {T:6.0f} | {rank:5d} | {kern:5d} | {evals[0]:+12.6e} | "
              f"{evals[-1]:+12.6e} | {cond_str} | {np.trace(Q):12.4f}")

# ===========================================================================
# TEST 5: Konvergenzrate der Eigenwerte
# ===========================================================================

def test_eigenvalue_convergence(all_gammas, N_basis, T):
    """Wie schnell konvergieren die einzelnen Eigenwerte?"""
    print(f"\n{'='*75}")
    print(f"TEST 5: EIGENWERT-KONVERGENZ vs. K")
    print(f"{'='*75}")

    K_max = len(all_gammas)
    Q_ref, _ = build_gram(all_gammas, N_basis, T)
    evals_ref = np.sort(np.linalg.eigvalsh(Q_ref))

    K_vals = list(range(1, min(N_basis + 1, K_max + 1))) + \
             [N_basis + 5, N_basis + 10, 2 * N_basis, 3 * N_basis, K_max]
    K_vals = sorted(set([k for k in K_vals if k <= K_max]))

    print(f"\n  Referenz: K={K_max}, Eigenwerte:")
    print(f"    {np.array2string(evals_ref, precision=6, separator=', ')}")

    print(f"\n  {'K':>5} | {'||evals - ref||':>16} | {'max |diff|':>12} | "
          f"{'lam_0 diff':>12} | {'lam_N-1 diff':>12}")
    print(f"  {'-'*5}-+-{'-'*16}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for K in K_vals:
        Q, _ = build_gram(all_gammas[:K], N_basis, T)
        evals = np.sort(np.linalg.eigvalsh(Q))
        diff = evals - evals_ref
        norm_diff = np.linalg.norm(diff)
        max_diff = np.max(np.abs(diff))

        print(f"  {K:5d} | {norm_diff:16.8e} | {max_diff:12.6e} | "
              f"{diff[0]:+12.6e} | {diff[-1]:+12.6e}")

# ===========================================================================
# HAUPTPROGRAMM
# ===========================================================================

if __name__ == "__main__":
    print("=" * 75)
    print("WEG 2: GRAM-SEITIGER KONVERGENZ-TEST")
    print("Konvergiert der Kern von Q^K = V_K^T V_K fuer K -> inf?")
    print("=" * 75)

    K_MAX = 100
    N_BASIS = 15
    T = 120.0

    print(f"\n  Lade {K_MAX} Zeta-Nullstellen...")
    all_gammas = load_zeros(K_MAX)
    print(f"  gamma_1 = {all_gammas[0]:.6f}, gamma_{K_MAX} = {all_gammas[-1]:.6f}")
    print(f"  Nullstellen in [0,T={T}]: {np.sum(all_gammas < T)}")

    # Test 1: Rang und Kern
    results1 = test_rank_vs_K(all_gammas, N_BASIS, T)

    # Test 2: Eigenvektor-Stabilitaet
    test_evec_stability(all_gammas, N_BASIS, T)

    # Test 3: Nullstellen-Rekonstruktion
    test_zero_reconstruction(all_gammas, N_BASIS, T)

    # Test 4: T-Abhaengigkeit
    test_T_dependence(all_gammas, N_BASIS)

    # Test 5: Eigenwert-Konvergenz
    test_eigenvalue_convergence(all_gammas, N_BASIS, T)

    # FAZIT
    print(f"\n{'='*75}")
    print(f"FAZIT: GRAM-SEITIGER KONVERGENZ-TEST")
    print(f"{'='*75}")
    print(f"""
  ZENTRALE FRAGE: Konvergiert die Gram-Matrix Q^K fuer K -> inf?

  Da Q^K = V_K^T V_K (Summe von Rang-1-Matrizen), gilt:
    Q^{{K+1}} = Q^K + v_{{K+1}} v_{{K+1}}^T

  => Q^K ist MONOTON WACHSEND (im Loewner-Sinne)!
  => Alle Eigenwerte wachsen monoton mit K
  => Rang kann nur wachsen (oder gleich bleiben)

  Die Konvergenz-Frage wird damit:
  (a) Sind die Eigenwerte BESCHRAENKT? (Sonst divergiert Q^K -> inf)
  (b) Konvergiert der Kern-Raum (= Nullraum von Q^K)?

  Fuer (a): trace(Q^K) = sum_k ||phi(gamma_k)||^2 ~ K * (const/T)
            => Eigenwerte wachsen wie O(K) => NICHT beschraenkt
            => Q^K / K koennte konvergieren (Mittelwert)

  Fuer (b): kern(Q^K) = {{f : f(gamma_k)=0 fuer alle k=1..K}}
            => Kern SCHRUMPFT monoton (mehr Bedingungen)
            => Im Limes: kern = {{f : f(gamma)=0 fuer ALLE Nullstellen}}
            => Das ist GENAU die Funktion die RH liefert!
""")
