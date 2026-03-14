#!/usr/bin/env python3
"""
weg2_kern_skalierung.py
=======================
Skalierungs-Test: Wie muss N_basis mit K wachsen, damit kern(Q^K) > 0 bleibt?

Connes' Argument: Die Kern-Dimension waechst PROPORTIONAL zu K,
wenn die Bandbreite L proportional zur letzten Nullstelle gamma_K skaliert.

Weil-Dichte: N(T) ~ T/(2*pi) * log(T/(2*pi)) - T/(2*pi)
Also: K Nullstellen bis gamma_K => N_basis ~ gamma_K * (const) Fourier-Moden

Kern-Dimension = N_basis - rank(V_K) = N_basis - min(K, eff_rank)
Wenn N_basis ~ c * K, dann kern ~ (c-1)*K fuer K >> 1.

Connes behauptet: kern(Q_W) hat Dimension ~ 50-100 (fuer endliches S).
Die Frage ist: Bleibt kern > 0 wenn S -> inf UND N_basis mitskaliert?
"""

import numpy as np
from mpmath import mp, im, zetazero

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
    K = len(gammas)
    V = np.zeros((K, N_basis))
    for k in range(K):
        for n in range(N_basis):
            V[k, n] = cosine_basis(n, gammas[k], T)
    return V.T @ V, V

# ===========================================================================
# TEST 1: Kern-Dimension bei mitskalierendem N
# ===========================================================================

def test_scaling(all_gammas):
    """Kern-Dimension wenn N_basis und T mit K skalieren."""
    print(f"\n{'='*75}")
    print(f"TEST 1: KERN-DIMENSION bei proportionaler Skalierung")
    print(f"{'='*75}")

    # Skalierungsregel: T = 1.2 * gamma_K, N_basis = floor(T * pi / pi) + 1
    # Paley-Wiener PW_L: Bandbreite L bestimmt N ~ L*T/pi
    # Connes' PW_{L/2}: L = log(cond), fuer zeta ist cond=1 => L=0 ???
    # Fuer Dirichlet-Chars: L = log(q), N ~ log(q) * T / pi

    # Wir testen verschiedene Skalierungen:
    # (A) N proportional zu K
    # (B) N proportional zu sqrt(K)
    # (C) N = fester Bruchteil von K

    K_vals = [5, 10, 15, 20, 30, 40, 50, 75, 100]
    K_vals = [k for k in K_vals if k <= len(all_gammas)]

    print(f"\n  Skalierung A: N = K (gleich viele Basis-Fkt wie Nullstellen)")
    print(f"  {'K':>5} | {'N':>5} | {'T':>8} | {'rank':>5} | {'kern':>5} | "
          f"{'kern/N':>7} | {'lam_min':>12} | {'cond':>12}")
    print(f"  {'-'*5}-+-{'-'*5}-+-{'-'*8}-+-{'-'*5}-+-{'-'*5}-+-{'-'*7}-+-{'-'*12}-+-{'-'*12}")

    for K in K_vals:
        gammas = all_gammas[:K]
        T = 1.3 * gammas[-1]  # T etwas groesser als letzte Nullstelle
        N = K  # gleich viele
        if N < 2:
            continue

        Q, V = build_gram(gammas, N, T)
        evals = np.sort(np.linalg.eigvalsh(Q))
        threshold = 1e-10
        rank = np.sum(evals > threshold)
        kern = N - rank
        lam_min_pos = evals[evals > threshold][0] if rank > 0 else 0
        cond = evals[-1] / lam_min_pos if lam_min_pos > 0 else float('inf')
        cond_str = f"{cond:12.2e}" if cond < 1e15 else f"{'inf':>12}"

        print(f"  {K:5d} | {N:5d} | {T:8.1f} | {rank:5d} | {kern:5d} | "
              f"{kern/N:7.3f} | {evals[0]:+12.6e} | {cond_str}")

    print(f"\n  Skalierung B: N = 2*K (doppelt so viele Basis-Fkt)")
    print(f"  {'K':>5} | {'N':>5} | {'T':>8} | {'rank':>5} | {'kern':>5} | "
          f"{'kern/N':>7} | {'lam_min':>12}")
    print(f"  {'-'*5}-+-{'-'*5}-+-{'-'*8}-+-{'-'*5}-+-{'-'*5}-+-{'-'*7}-+-{'-'*12}")

    for K in K_vals:
        gammas = all_gammas[:K]
        T = 1.3 * gammas[-1]
        N = 2 * K
        if N < 2:
            continue

        Q, V = build_gram(gammas, N, T)
        evals = np.sort(np.linalg.eigvalsh(Q))
        threshold = 1e-10
        rank = np.sum(evals > threshold)
        kern = N - rank

        print(f"  {K:5d} | {N:5d} | {T:8.1f} | {rank:5d} | {kern:5d} | "
              f"{kern/N:7.3f} | {evals[0]:+12.6e}")

    print(f"\n  Skalierung C: N = K + 10 (kleine Ueberbreite)")
    print(f"  {'K':>5} | {'N':>5} | {'T':>8} | {'rank':>5} | {'kern':>5} | "
          f"{'kern/N':>7} | {'lam_min':>12}")
    print(f"  {'-'*5}-+-{'-'*5}-+-{'-'*8}-+-{'-'*5}-+-{'-'*5}-+-{'-'*7}-+-{'-'*12}")

    for K in K_vals:
        gammas = all_gammas[:K]
        T = 1.3 * gammas[-1]
        N = K + 10
        if N < 2:
            continue

        Q, V = build_gram(gammas, N, T)
        evals = np.sort(np.linalg.eigvalsh(Q))
        threshold = 1e-10
        rank = np.sum(evals > threshold)
        kern = N - rank

        print(f"  {K:5d} | {N:5d} | {T:8.1f} | {rank:5d} | {kern:5d} | "
              f"{kern/N:7.3f} | {evals[0]:+12.6e}")

# ===========================================================================
# TEST 2: Weil-Dichte vs. Basis-Dimension
# ===========================================================================

def test_weil_density(all_gammas):
    """Vergleiche N(T) = #{gamma_k < T} mit der Basis-Dimension."""
    print(f"\n{'='*75}")
    print(f"TEST 2: WEIL-DICHTE N(T) vs. PALEY-WIENER DIMENSION")
    print(f"{'='*75}")

    # Weil-Dichte: N(T) ~ T/(2pi) * log(T/(2pi)) - T/(2pi) + 7/8 + O(1/T)
    T_vals = [20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250]

    print(f"\n  {'T':>6} | {'N(T) ist':>8} | {'N_Weyl':>8} | {'Diff':>6} | "
          f"{'T/(2pi)':>8} | {'N/T*2pi':>8}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}")

    for T in T_vals:
        N_actual = np.sum(all_gammas < T)
        # Weyl law
        if T > 2 * np.pi:
            N_weyl = T / (2*np.pi) * np.log(T / (2*np.pi)) - T / (2*np.pi) + 7/8
        else:
            N_weyl = 0
        ratio = N_actual / T * 2 * np.pi if T > 0 else 0

        print(f"  {T:6.0f} | {N_actual:8d} | {N_weyl:8.1f} | {N_actual - N_weyl:+6.1f} | "
              f"{T/(2*np.pi):8.2f} | {ratio:8.4f}")

    # Connes' Dimension: dim PW_L(0,T) = floor(L*T/pi) + 1
    # Fuer zeta: L = log(1) = 0 ???  -- Das kann nicht stimmen
    # Richtig: L haengt vom CUTOFF ab, nicht vom Conductor
    print(f"\n  PALEY-WIENER Dimension: dim PW_{{N*pi/T}}(0,T) = N")
    print(f"  D.h. N Cosinus-Basisfunktionen spannen PW_{{N*pi/T}} auf.")
    print(f"  Bandbreite L = N*pi/T")
    print(f"\n  Fuer Kern-Dimension > 0 brauchen wir N > K (mehr Basis als Nullstellen)")
    print(f"  D.h. Bandbreite L > K*pi/T = N(T)*pi/T ~ log(T/(2pi))/2")

# ===========================================================================
# TEST 3: Kern-Qualitaet bei Connes-Skalierung
# ===========================================================================

def test_connes_scaling(all_gammas):
    """Teste Connes' Skalierung: PW_{L/2} mit L = log(q)."""
    print(f"\n{'='*75}")
    print(f"TEST 3: CONNES-SKALIERUNG (PW mit L proportional zu log-Dichte)")
    print(f"{'='*75}")

    # Fuer die Riemann-Zeta (q=1) ist L=0 => PW_0 = Konstanten
    # Aber Connes arbeitet mit Dirichlet L-Funktionen (q > 1)
    # Wir simulieren: Wie gross muss L sein, damit kern(Q^K) proportional zu K ist?

    K_vals = [10, 20, 30, 50, 75, 100]
    K_vals = [k for k in K_vals if k <= len(all_gammas)]

    # Verschiedene L-Werte (Bandbreite in Fourier-Raum)
    print(f"\n  Frage: Fuer welches L gilt kern ~ K/2 ?")
    print(f"\n  {'K':>5} | {'gamma_K':>8} | {'T':>8} | {'L fuer kern=K/2':>16} | "
          f"{'N noetig':>8}")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*16}-+-{'-'*8}")

    for K in K_vals:
        gammas = all_gammas[:K]
        T = 1.3 * gammas[-1]

        # Suche N so dass kern(Q^K) ~ K/2
        target_kern = K // 2
        target_N = K + target_kern  # rank <= K, also kern >= N - K

        # Teste ob rank tatsaechlich K ist
        Q, V = build_gram(gammas, target_N, T)
        evals = np.sort(np.linalg.eigvalsh(Q))
        threshold = 1e-10
        rank = np.sum(evals > threshold)
        actual_kern = target_N - rank

        L = target_N * np.pi / T

        print(f"  {K:5d} | {gammas[-1]:8.2f} | {T:8.1f} | {L:16.4f} | "
              f"{target_N:8d}  (rank={rank}, kern={actual_kern})")

    # Erklaerung
    print(f"\n  INTERPRETATION:")
    print(f"  rank(V_K) = K solange K < N_basis (unterbestimmt)")
    print(f"  => kern = N - K genau dann wenn alle Zeilen von V_K lin. unabh.")
    print(f"  => Kern-Dimension waechst wie N - K")
    print(f"  => Fuer kern ~ K/2 brauchen wir N ~ 3K/2")
    print(f"  => Connes' grosse Kern-Dimensionen kommen von GROSSEM N/K-Verhaeltnis")

# ===========================================================================
# HAUPTPROGRAMM
# ===========================================================================

if __name__ == "__main__":
    print("=" * 75)
    print("WEG 2: KERN-SKALIERUNG")
    print("Wie muss N_basis mit K wachsen, damit kern(Q^K) > 0 bleibt?")
    print("=" * 75)

    K_MAX = 100
    print(f"\n  Lade {K_MAX} Zeta-Nullstellen...")
    all_gammas = load_zeros(K_MAX)
    print(f"  gamma_1 = {all_gammas[0]:.6f}, gamma_{K_MAX} = {all_gammas[-1]:.6f}")

    # Test 1: Kern bei verschiedenen Skalierungen
    test_scaling(all_gammas)

    # Test 2: Weil-Dichte
    test_weil_density(all_gammas)

    # Test 3: Connes-Skalierung
    test_connes_scaling(all_gammas)

    # FAZIT
    print(f"\n{'='*75}")
    print(f"FAZIT: KERN-SKALIERUNG")
    print(f"{'='*75}")
    print(f"""
  KERN-FORMEL: kern(Q^K) = N_basis - rank(V_K)

  Da V_K eine K x N Matrix ist:
  - rank(V_K) = min(K, N)  (generisch, wenn Zeilen lin. unabhaengig)
  - kern = N - K  fuer K < N  (unterdeterminiert)
  - kern = 0     fuer K >= N  (ueberdeterminiert)

  KONVERGENZ-FRAGE REFORMULIERT:
  Connes' Setting: L = log(q), T fest, N = floor(L*T/pi) + 1
  Fuer wachsendes q: N ~ log(q) * T / pi
  Die Nullstellen-Anzahl N(T) ist FEST (abhaengig von T, nicht q)
  => kern = N - N(T) ~ log(q) * T/pi - T/(2pi) * log(T/(2pi))
  => kern WAECHST mit log(q) !

  Fuer die Riemann-Zeta (q=1): L=0, N=1, kern=0 trivial.
  Connes' Trick: Er arbeitet mit L-Funktionen (q >> 1) wo der Kern GROSS ist.
  Die Konvergenz S->inf ist dann INNERHALB dieses grossen Kerns.

  Fuer FST-RH: Wir muessen entweder
  (a) mit Dirichlet-Chars arbeiten (Connes' Weg), oder
  (b) ein T-abhaengiges Argument finden (T->inf statt q->inf)
""")
