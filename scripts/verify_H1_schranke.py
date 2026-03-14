#!/usr/bin/env python3
"""
verify_H1_schranke.py
=====================
Numerische Verifikation der analytischen H^1-Schranke fuer rescalierte
Eigenfunktionen des Weil-Operators.

Prueft die Lemma-Kette aus BEWEISNOTIZ.md, Abschnitt "Analytische H^1-Schranke":
  Lemma 1: Spektralluecke lambda_min(C) > mu
  Lemma 2: Schur-Komplement c_high = (mu*I - C)^{-1} B^T c_low
  Lemma 3: Gewichtete Schranke sum w(n) c_n^2 via Resolvent
  Lemma 4: Uniforme H^1-Schranke sum n^2 c_n^2 <= M

Verwendet die QW-Matrix aus weighted_compactness_test.py.

Author: Lukas Geiger
Date: 2026-03-14
"""

import numpy as np
from scipy.linalg import eigh, eigvalsh
from sympy import primerange
import sys
import os

# Reuse matrix builder from weighted_compactness_test
sys.path.insert(0, os.path.dirname(__file__))
from weighted_compactness_test import build_QW_vectorized, get_eigensystem

LOG4PI_GAMMA = 3.2720532309274587


def analyze_diagonal_structure(lam, N, primes_all, basis='cos'):
    """Analysiere die Diagonalstruktur der QW-Matrix."""
    primes_used = [p for p in primes_all if p <= max(lam, 100)]
    W = build_QW_vectorized(lam, N, primes_used, basis=basis, n_quad=2000, n_int=1500)
    evals, evecs = eigh(W)

    L = np.log(lam)
    diag = np.diag(W)
    mu = evals[0]  # kleinster Eigenwert
    v0 = evecs[:, 0]  # Grundzustand

    return W, evals, evecs, diag, mu, v0, L


def test_lemma1_diagonal(lambdas, N, primes_all):
    """Lemma 1: Diagonale W_{nn} fuer grosse n."""
    print("\n" + "=" * 80)
    print("LEMMA 1: Diagonale W_{nn} -> LOG4PI_GAMMA fuer n -> inf")
    print(f"  LOG4PI_GAMMA = {LOG4PI_GAMMA:.6f}")
    print("=" * 80)

    for lam in lambdas:
        W, evals, evecs, diag, mu, v0, L = analyze_diagonal_structure(
            lam, N, primes_all
        )
        print(f"\n  lambda={lam}, L={L:.3f}, mu_1={mu:.4f}")
        print(f"  {'n':>4} | {'W_nn':>10} | {'W_nn - LOG4PI':>14} | {'|c_n|':>10} | {'n^2*c_n^2':>12}")
        print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*14}-+-{'-'*10}-+-{'-'*12}")

        for n in range(min(N, 20)):
            c_n = v0[n]
            print(f"  {n:4d} | {diag[n]:+10.4f} | {diag[n]-LOG4PI_GAMMA:+14.6f} | {abs(c_n):10.6f} | {n**2 * c_n**2:12.6f}")

        # Asymptotik
        if N > 10:
            high_diag = diag[10:]
            print(f"\n  n >= 10: min(W_nn) = {np.min(high_diag):.4f}, "
                  f"max(W_nn) = {np.max(high_diag):.4f}, "
                  f"mean(W_nn) = {np.mean(high_diag):.4f}")
            print(f"  Abweichung von LOG4PI_GAMMA: max|W_nn - 3.272| = "
                  f"{np.max(np.abs(high_diag - LOG4PI_GAMMA)):.6f}")


def test_lemma2_offdiagonal(lambdas, N, primes_all):
    """Lemma 2: Off-Diagonal-Abfall."""
    print("\n" + "=" * 80)
    print("LEMMA 2: Off-Diagonal-Abfall der QW-Matrix")
    print("=" * 80)

    for lam in lambdas:
        W, evals, evecs, diag, mu, v0, L = analyze_diagonal_structure(
            lam, N, primes_all
        )
        print(f"\n  lambda={lam}, L={L:.3f}")

        # Zeile-Summen der Betraege (Gershgorin)
        for n in [0, 1, 2, 5, 10, N-1]:
            if n >= N:
                continue
            row_sum = np.sum(np.abs(W[n, :])) - np.abs(W[n, n])
            print(f"  n={n:3d}: Gershgorin-Radius R_n = {row_sum:.4f}, "
                  f"W_nn = {W[n,n]:+.4f}, "
                  f"[W_nn - R_n, W_nn + R_n] = [{W[n,n]-row_sum:+.4f}, {W[n,n]+row_sum:+.4f}]")

        # Blockstruktur: Frobenius-Norm der Off-Diagonal-Bloecke
        for K in [3, 5, 8, 10]:
            if K >= N:
                continue
            B = W[:K, K:]  # Off-diagonal block
            C = W[K:, K:]  # High-high block
            A = W[:K, :K]  # Low-low block
            print(f"\n  Partition K={K}:")
            print(f"    ||A|| = {np.linalg.norm(A):.4f} ({K}x{K} low block)")
            print(f"    ||B||_F = {np.linalg.norm(B, 'fro'):.4f} (off-diagonal)")
            print(f"    ||B||_op = {np.linalg.norm(B, 2):.4f} (operator norm)")
            print(f"    lambda_min(C) = {eigvalsh(C)[0]:.4f} (high block min EV)")
            print(f"    lambda_max(C) = {eigvalsh(C)[-1]:.4f} (high block max EV)")
            print(f"    Tr(C)/(N-K) = {np.trace(C)/(N-K):.4f} (mean diagonal)")


def test_lemma3_schur_complement(lambdas, N, primes_all):
    """Lemma 3: Schur-Komplement => Kontrolle der hohen Moden."""
    print("\n" + "=" * 80)
    print("LEMMA 3: Schur-Komplement-Argument")
    print("  Aus W*c = mu*c und mu < 0 < lambda_min(C):")
    print("  c_high = (mu*I - C)^{-1} B^T c_low")
    print("  ||c_high||^2 <= ||B||^2 / (lambda_min(C) - mu)^2 * ||c_low||^2")
    print("=" * 80)

    for lam in lambdas:
        W, evals, evecs, diag, mu, v0, L = analyze_diagonal_structure(
            lam, N, primes_all
        )

        print(f"\n  lambda={lam}, L={L:.3f}, mu={mu:.4f}")

        for K in [3, 5, 8]:
            if K >= N:
                continue
            A = W[:K, :K]
            B = W[:K, K:]
            C = W[K:, K:]

            c_low = v0[:K]
            c_high = v0[K:]

            lam_min_C = eigvalsh(C)[0]
            B_op = np.linalg.norm(B, 2)
            B_fro = np.linalg.norm(B, 'fro')

            # Tatsaechliche Werte
            actual_c_high_sq = np.sum(c_high**2)
            actual_c_low_sq = np.sum(c_low**2)

            # Schur-Schranke
            if lam_min_C > mu:
                denominator = lam_min_C - mu
                schur_bound_op = (B_op / denominator)**2 * actual_c_low_sq
                schur_bound_fro = (B_fro / denominator)**2 * actual_c_low_sq

                # Gewichtete Schranke: sum_{n>K} n^2 c_n^2
                n_arr = np.arange(K, N, dtype=float)
                actual_weighted = np.sum(n_arr**2 * c_high**2)

                # Gewichtete Schur-Schranke via Gershgorin auf C
                C_diag = np.diag(C)
                C_offdiag_rowsum = np.array([
                    np.sum(np.abs(C[i, :])) - np.abs(C[i, i])
                    for i in range(len(C))
                ])
                gersh_lower = np.min(C_diag - C_offdiag_rowsum)

                # Feinere Schranke: (mu*I - C) ist invertierbar mit
                # ||(mu*I - C)^{-1}||_{n,n} <= 1/(C_nn - mu - R_n)
                # wobei R_n die Gershgorin-Radien von C sind

                print(f"\n  K={K}: Partition [low={K}, high={N-K}]")
                print(f"    ||c_low||^2  = {actual_c_low_sq:.6f}")
                print(f"    ||c_high||^2 = {actual_c_high_sq:.6f} (tatsaechlich)")
                print(f"    lambda_min(C) = {lam_min_C:.4f}")
                print(f"    mu = {mu:.4f}")
                print(f"    lambda_min(C) - mu = {denominator:.4f}")
                print(f"    ||B||_op = {B_op:.4f}")
                print(f"    ||B||_F  = {B_fro:.4f}")
                print(f"    Schur-Schranke (op):  ||c_high||^2 <= {schur_bound_op:.6f}")
                print(f"    Schur-Schranke (fro): ||c_high||^2 <= {schur_bound_fro:.6f}")
                print(f"    Ratio (op):  actual/bound = {actual_c_high_sq/schur_bound_op:.4f}" if schur_bound_op > 0 else "")
                print(f"    Ratio (fro): actual/bound = {actual_c_high_sq/schur_bound_fro:.4f}" if schur_bound_fro > 0 else "")

                # Gewichtete Version
                print(f"    sum_{{n>K}} n^2 c_n^2 = {actual_weighted:.6f} (tatsaechlich)")
                # Obere Schranke: sum n^2 c_n^2 <= ||c_high||^2 * max(n^2) (trivial)
                # Bessere Schranke via Resolvent-Abschaetzung
                max_n_sq = (N-1)**2
                weighted_bound = schur_bound_op * max_n_sq
                print(f"    Triviale gewichtete Schranke: <= {weighted_bound:.4f}")

                # Bessere Schranke: Gershgorin
                if gersh_lower > mu:
                    gersh_denom = gersh_lower - mu
                    gersh_bound = (B_fro / gersh_denom)**2 * actual_c_low_sq
                    print(f"    Gershgorin lower(C) = {gersh_lower:.4f}")
                    print(f"    Gershgorin-Schranke: ||c_high||^2 <= {gersh_bound:.6f}")
            else:
                print(f"\n  K={K}: WARNUNG: lambda_min(C) = {lam_min_C:.4f} <= mu = {mu:.4f}")
                print(f"    Schur-Komplement-Argument NICHT anwendbar!")


def test_lemma4_H1_bound(lambdas, N, primes_all):
    """Lemma 4: Resultierende H^1-Schranke."""
    print("\n" + "=" * 80)
    print("LEMMA 4: H^1-Schranke sum n^2 c_n^2")
    print("  = [low part] + [high part]")
    print("  <= K^2 * ||c_low||^2 + Schur-Schranke fuer high part")
    print("=" * 80)

    results = []
    for lam in lambdas:
        W, evals, evecs, diag, mu, v0, L = analyze_diagonal_structure(
            lam, N, primes_all
        )

        n_arr = np.arange(N, dtype=float)
        actual_sum_n2c2 = np.sum(n_arr**2 * v0**2)

        best_K = None
        best_bound = np.inf

        for K in range(2, min(N-2, 15)):
            A = W[:K, :K]
            B = W[:K, K:]
            C = W[K:, K:]
            c_low = v0[:K]
            c_high = v0[K:]

            lam_min_C = eigvalsh(C)[0]

            if lam_min_C <= mu:
                continue

            B_op = np.linalg.norm(B, 2)
            denominator = lam_min_C - mu

            # ||c_high||^2 <= (B_op / denom)^2 * ||c_low||^2
            c_high_bound = (B_op / denominator)**2 * np.sum(c_low**2)

            # Low contribution: sum_{n<K} n^2 c_n^2 <= K^2 (weil ||c||=1)
            low_part_actual = np.sum(n_arr[:K]**2 * c_low**2)
            low_part_bound = (K-1)**2  # max n^2 in low block

            # High contribution: sum_{n>=K} n^2 c_n^2
            # Brauche komponentenweise Kontrolle:
            # |c_n| <= ||B_{n,:}|| / (C_{nn} - mu) * ||c_low||  (Zeilen-Schranke)
            # Oder global: sum_{n>=K} n^2 c_n^2 <= (sum_{n>=K} n^2/(C_{nn}-mu)^2) * ||B||^2 * ||c_low||^2

            # Methode 1: Grobe Schranke mit max n^2
            high_max_n2 = (N-1)**2
            high_part_bound_crude = c_high_bound * high_max_n2

            # Methode 2: Feinere Schranke mit Resolvent-Gewichten
            # (mu*I - C) c_high = B^T c_low
            # c_n = sum_k [(mu*I - C)^{-1}]_{nk} [B^T c_low]_k
            # In Gershgorin: |(mu*I - C)^{-1}_{nn}| <= 1/(C_{nn} - mu - R_n(C))
            # Feinste Schranke: Berechne (mu*I - C)^{-1} direkt
            try:
                R = mu * np.eye(len(C)) - C
                R_inv = np.linalg.inv(R)
                c_high_pred = R_inv @ (B.T @ c_low)
                n_high = np.arange(K, N, dtype=float)
                high_part_bound_resolvent = np.sum(n_high**2 * c_high_pred**2)
            except np.linalg.LinAlgError:
                high_part_bound_resolvent = np.inf

            total_bound = low_part_bound + high_part_bound_resolvent
            if total_bound < best_bound:
                best_bound = total_bound
                best_K = K

        if best_K is not None:
            # Recompute best
            K = best_K
            A = W[:K, :K]
            B = W[:K, K:]
            C = W[K:, K:]
            c_low = v0[:K]
            lam_min_C = eigvalsh(C)[0]
            B_op = np.linalg.norm(B, 2)
            denominator = lam_min_C - mu
            c_high_bound = (B_op / denominator)**2 * np.sum(c_low**2)

            R = mu * np.eye(N-K) - C
            R_inv = np.linalg.inv(R)
            c_high_pred = R_inv @ (B.T @ c_low)
            n_high = np.arange(K, N, dtype=float)
            high_exact_resolvent = np.sum(n_high**2 * c_high_pred**2)
            low_actual = np.sum(n_arr[:K]**2 * c_low**2)

            print(f"\n  lambda={lam}, L={L:.3f}, mu={mu:.4f}")
            print(f"    Bester Cut: K={K}")
            print(f"    sum n^2 c_n^2 = {actual_sum_n2c2:.6f} (tatsaechlich)")
            print(f"    Low (n<{K}): {low_actual:.6f} (tatsaechlich), bound: {(K-1)**2}")
            print(f"    High (n>={K}): {np.sum(n_arr[K:]**2 * v0[K:]**2):.6f} (tatsaechlich)")
            print(f"    High via Resolvent: {high_exact_resolvent:.6f} (exakte Schur-Vorhersage)")
            print(f"    lambda_min(C) = {lam_min_C:.4f}")
            print(f"    ||B||_op = {B_op:.4f}")
            print(f"    (||B||/gap)^2 = {(B_op/denominator)**2:.6f}")
            print(f"    ||c_low||^2 = {np.sum(c_low**2):.6f}")

            results.append({
                'lam': lam, 'L': L, 'mu': mu, 'K': K,
                'actual': actual_sum_n2c2,
                'lam_min_C': lam_min_C,
                'B_op': B_op,
                'c_low_sq': np.sum(c_low**2),
            })

    # Zusammenfassung
    print("\n" + "=" * 80)
    print("ZUSAMMENFASSUNG: Uniforme H^1-Schranke")
    print("=" * 80)
    print(f"  {'lam':>6} | {'L':>6} | {'mu':>10} | {'K':>3} | {'sum n2c2':>10} | {'lam_min(C)':>10} | {'||B||':>8} | {'gap':>10}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*3}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}")
    for r in results:
        gap = r['lam_min_C'] - r['mu']
        print(f"  {r['lam']:6d} | {r['L']:6.3f} | {r['mu']:+10.4f} | {r['K']:3d} | {r['actual']:10.4f} | {r['lam_min_C']:+10.4f} | {r['B_op']:8.4f} | {gap:10.4f}")


def test_resolvent_accuracy(lambdas, N, primes_all):
    """Teste wie gut die Schur-Komplement-Vorhersage c_high reproduziert."""
    print("\n" + "=" * 80)
    print("RESOLVENT-GENAUIGKEIT: Wie gut sagt Schur c_high vorher?")
    print("  c_high^{pred} = (mu*I - C)^{-1} B^T c_low")
    print("  vs. c_high^{actual} = v0[K:]")
    print("=" * 80)

    for lam in lambdas:
        W, evals, evecs, diag, mu, v0, L = analyze_diagonal_structure(
            lam, N, primes_all
        )

        print(f"\n  lambda={lam}, L={L:.3f}, mu={mu:.4f}")

        for K in [3, 5]:
            if K >= N - 2:
                continue
            B = W[:K, K:]
            C = W[K:, K:]
            c_low = v0[:K]
            c_high_actual = v0[K:]

            lam_min_C = eigvalsh(C)[0]
            if lam_min_C <= mu:
                print(f"  K={K}: Schur nicht anwendbar (lam_min(C)={lam_min_C:.4f} <= mu)")
                continue

            R = mu * np.eye(N-K) - C
            c_high_pred = np.linalg.solve(R, B.T @ c_low)

            # Relativer Fehler
            rel_err = np.linalg.norm(c_high_pred - c_high_actual) / np.linalg.norm(c_high_actual) if np.linalg.norm(c_high_actual) > 1e-15 else np.inf

            print(f"\n  K={K}:")
            print(f"    ||c_high_actual|| = {np.linalg.norm(c_high_actual):.6f}")
            print(f"    ||c_high_pred||   = {np.linalg.norm(c_high_pred):.6f}")
            print(f"    ||pred - actual|| = {np.linalg.norm(c_high_pred - c_high_actual):.6f}")
            print(f"    Relativer Fehler  = {rel_err:.6f}")

            # Komponentenweiser Vergleich (erste 5)
            print(f"    n:      {'actual':>10}  {'predicted':>10}  {'ratio':>10}")
            for i in range(min(8, N-K)):
                n_idx = K + i
                if abs(c_high_actual[i]) > 1e-10:
                    ratio = c_high_pred[i] / c_high_actual[i]
                else:
                    ratio = float('nan')
                print(f"    {n_idx:4d}:  {c_high_actual[i]:+10.6f}  {c_high_pred[i]:+10.6f}  {ratio:+10.4f}")


def test_diagonal_asymptotics(N_large, primes_all):
    """Analysiere W_{nn} fuer grosse n bei verschiedenen lambda."""
    print("\n" + "=" * 80)
    print("DIAGONALE ASYMPTOTIK: W_{nn} vs n fuer verschiedene lambda")
    print("  Erwartung: W_{nn} -> LOG4PI_GAMMA + arch_diag(n) + prime_diag(n)")
    print("=" * 80)

    for lam in [100, 200, 500]:
        primes_used = [p for p in primes_all if p <= max(lam, 100)]
        W = build_QW_vectorized(lam, N_large, primes_used, basis='cos',
                                n_quad=2000, n_int=1500)
        L = np.log(lam)
        diag = np.diag(W)

        print(f"\n  lambda={lam}, L={L:.3f}, N={N_large}")
        print(f"  {'n':>4} | {'W_nn':>10} | {'W_nn - 3.272':>14}")
        print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*14}")
        for n in list(range(0, min(N_large, 10))) + list(range(10, N_large, 5)):
            if n >= N_large:
                continue
            print(f"  {n:4d} | {diag[n]:+10.4f} | {diag[n]-LOG4PI_GAMMA:+14.6f}")

        # Statistik fuer hohe n
        for thresh in [5, 10, 15]:
            if thresh < N_large:
                high = diag[thresh:]
                print(f"\n  n >= {thresh}: min={np.min(high):.4f}, max={np.max(high):.4f}, "
                      f"mean={np.mean(high):.4f}, std={np.std(high):.4f}")


if __name__ == "__main__":
    print("=" * 80)
    print("VERIFIKATION DER ANALYTISCHEN H^1-SCHRANKE")
    print("Schur-Komplement-Argument fuer uniforme Regularitaet")
    print("=" * 80)

    primes_all = [int(p) for p in primerange(2, 500)]
    lambdas = [50, 100, 200, 500]
    N = 25  # Schnelle Version

    import time
    t0 = time.time()

    test_lemma1_diagonal(lambdas, N, primes_all)
    test_lemma2_offdiagonal([100, 200], N, primes_all)
    test_lemma3_schur_complement(lambdas, N, primes_all)
    test_lemma4_H1_bound(lambdas, N, primes_all)
    test_resolvent_accuracy([100, 200], N, primes_all)
    test_diagonal_asymptotics(30, primes_all)

    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"Gesamtzeit: {elapsed:.1f}s")
    print(f"{'=' * 80}")

    print("""
BEWEIS-LOGIK:
=============
1. W_{nn} >= delta > 0 fuer n > K (numerisch: delta ~ 2.0 fuer K >= 5)
2. Eigenwert mu = l1^+(lambda) < 0 (strikt negativ)
3. Schur-Komplement: c_high = (mu*I - C)^{-1} B^T c_low
   wobei C = W[K:,K:] mit lambda_min(C) >= delta > 0 > mu
4. ||c_high||^2 <= (||B||/(delta - mu))^2 * ||c_low||^2
5. sum n^2 c_n^2 = [low: <= K^2] + [high: <= kontrolliert durch Schur]
6. Da ||c_low||^2 <= 1 (Normierung): uniforme Schranke!

KRITISCHER PUNKT:
- delta und ||B|| haengen von lambda ab
- Aber delta -> LOG4PI_GAMMA > 0 (lambda-unabhaengig)
- ||B|| muss lambda-unabhaengig beschraenkt sein (zu verifizieren)
- mu ~ -C * L^2 -> -inf, ABER (delta - mu) ~ |mu|, also
  ||B||/(delta - mu) ~ ||B||/|mu| -> 0 (falls ||B|| = o(|mu|))
""")
