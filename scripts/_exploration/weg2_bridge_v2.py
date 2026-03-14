#!/usr/bin/env python3
"""
weg2_bridge_v2.py
=================
Weg 2: Uebersetzungsbruecke v2 -- Normierung korrigiert.

KORREKTUR: Die erste Version hatte ein Normierungsproblem.
K_sym(p,q) ist symmetrisch, also eigenzerlegbar.
F(sigma) = sum_{p,q} K_sym(p,q) = 1^T K_sym 1

Direkte Eigenzerlegung von K_sym gibt:
  F(sigma) = sum_j lambda_j * alpha_j^2
  wobei alpha_j = <v_j, 1> (Standard-Skalarprodukt)

w^-(sigma) = F^- / (F^+ + F^-) <= 1/2  <=>  F(sigma) >= 0
"""

import numpy as np
from sympy import primerange

# ===========================================================================
# 1. Kernel und Funktional
# ===========================================================================

def get_primes(N):
    return np.array(list(primerange(2, 10000))[:N], dtype=np.float64)

def prime_sums(primes, sigma):
    ps = primes ** (-sigma)
    logs = np.log(primes)
    pss = primes ** sigma
    P = np.sum(ps)
    B = np.sum(logs * ps)
    A = np.sum(logs * ps / (pss - 1))
    C = np.sum(logs**2 * ps / (pss - 1))
    D = np.sum(logs**2 / (pss - 1)**2)
    return P, A, B, C, D

def F_direct(primes, sigma):
    P, A, B, C, D = prime_sums(primes, sigma)
    return A * B - P * (C + D)

def build_Ksym(primes, sigma):
    """K_sym(p,q) exakt wie in RH6 definiert."""
    N = len(primes)
    K = np.zeros((N, N))
    logs = np.log(primes)
    pss = primes ** sigma

    for i in range(N):
        for j in range(N):
            # K_sigma(p_i, p_j)
            K_ij = (logs[i] * logs[j]) / (pss[i] * (pss[i] - 1) * pss[j]) \
                   - (logs[i]**2 / pss[j]) * (1/(pss[i] - 1)**2 + 1/(pss[i] * (pss[i] - 1)))
            # K_sigma(p_j, p_i)
            K_ji = (logs[j] * logs[i]) / (pss[j] * (pss[j] - 1) * pss[i]) \
                   - (logs[j]**2 / pss[i]) * (1/(pss[j] - 1)**2 + 1/(pss[j] * (pss[j] - 1)))
            K[i, j] = 0.5 * (K_ij + K_ji)
    return K

# ===========================================================================
# 2. Spektralanalyse -- DIREKT auf K_sym
# ===========================================================================

def spectral_analysis(primes, sigma, verbose=True):
    N = len(primes)
    K = build_Ksym(primes, sigma)

    # Konsistenzcheck
    F_sum = np.sum(K)
    F_dir = F_direct(primes, sigma)

    # Eigenzerlegung
    eigenvalues, eigenvectors = np.linalg.eigh(K)

    # Projektion von 1 auf Eigenvektoren
    ones = np.ones(N)
    alphas = eigenvectors.T @ ones  # alpha_j = v_j . 1

    # F via Spektral: F = sum lambda_j * alpha_j^2
    F_spec = np.sum(eigenvalues * alphas**2)

    # F^+ und F^-
    pos_mask = eigenvalues > 0
    neg_mask = eigenvalues < 0
    F_plus = np.sum(eigenvalues[pos_mask] * alphas[pos_mask]**2)
    F_minus = np.sum(np.abs(eigenvalues[neg_mask]) * alphas[neg_mask]**2)

    w_minus = F_minus / (F_plus + F_minus) if (F_plus + F_minus) > 0 else 0
    n_neg = np.sum(neg_mask)
    n_pos = np.sum(pos_mask)

    if verbose:
        print(f"\n  sigma = {sigma:.4f}, N = {N}")
        print(f"  F(sigma) direkt = {F_dir:.8e}")
        print(f"  F(sigma) Ksum   = {F_sum:.8e}")
        print(f"  F(sigma) spekt  = {F_spec:.8e}")
        print(f"  Eigenwerte: {n_pos} positiv, {n_neg} negativ")
        print(f"  F^+ = {F_plus:.8e}, F^- = {F_minus:.8e}")
        print(f"  w^-(sigma) = {w_minus:.6f}  (Grenze: 0.5)")
        print(f"  F >= 0: {'JA' if F_dir >= 0 else 'NEIN'}")

    return {
        'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors,
        'alphas': alphas, 'F_plus': F_plus, 'F_minus': F_minus,
        'F_dir': F_dir, 'w_minus': w_minus, 'n_neg': n_neg, 'n_pos': n_pos,
        'K': K,
    }

# ===========================================================================
# 3. Sigma-Scan
# ===========================================================================

def sigma_scan(primes):
    print(f"\n{'='*70}")
    print(f"SIGMA-SCAN: w^-(sigma) und F(sigma)")
    print(f"{'='*70}")

    sigmas = [1.01, 1.02, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0]
    print(f"\n  {'sigma':>8} | {'w^-':>10} | {'n_neg':>5} | {'F^+':>12} | {'F^-':>12} | {'F(sigma)':>12} | {'OK':>3}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*3}")

    for sigma in sigmas:
        res = spectral_analysis(primes, sigma, verbose=False)
        ok = "JA" if res['F_dir'] >= 0 else "--"
        print(f"  {sigma:8.4f} | {res['w_minus']:10.6f} | {res['n_neg']:5d} | "
              f"{res['F_plus']:12.6e} | {res['F_minus']:12.6e} | {res['F_dir']:+12.6e} | {ok:>3}")

# ===========================================================================
# 4. Konvergenz mit N (Primzahlanzahl)
# ===========================================================================

def convergence_test(sigma):
    """Teste ob F(sigma) mit wachsendem N konvergiert."""
    print(f"\n{'='*70}")
    print(f"KONVERGENZ: F(sigma={sigma}) vs. Anzahl Primzahlen N")
    print(f"{'='*70}")

    Ns = [5, 10, 15, 20, 30, 50, 80, 100, 150, 200]
    print(f"\n  {'N':>5} | {'F(sigma)':>14} | {'w^-':>10} | {'n_neg':>5} | {'F^+':>12} | {'F^-':>12}")
    print(f"  {'-'*5}-+-{'-'*14}-+-{'-'*10}-+-{'-'*5}-+-{'-'*12}-+-{'-'*12}")

    for N in Ns:
        primes = get_primes(N)
        res = spectral_analysis(primes, sigma, verbose=False)
        print(f"  {N:5d} | {res['F_dir']:+14.8e} | {res['w_minus']:10.6f} | "
              f"{res['n_neg']:5d} | {res['F_plus']:12.6e} | {res['F_minus']:12.6e}")

# ===========================================================================
# 5. Negative Eigenvektoren analysieren
# ===========================================================================

def analyze_negative_eigenvectors(primes, sigma, result):
    print(f"\n{'='*70}")
    print(f"NEGATIVE EIGENVEKTOREN (sigma={sigma:.4f}, N={len(primes)})")
    print(f"{'='*70}")

    eigenvalues = result['eigenvalues']
    eigenvectors = result['eigenvectors']
    alphas = result['alphas']
    N = len(primes)
    logs = np.log(primes)

    neg_idx = [j for j in range(N) if eigenvalues[j] < 0]

    if not neg_idx:
        print("  Keine negativen Eigenwerte!")
        return

    # Top-Beitraege zu F^-
    contributions = [(j, abs(eigenvalues[j]) * alphas[j]**2) for j in neg_idx]
    contributions.sort(key=lambda x: -x[1])

    print(f"\n  Top-Beitraege zu F^- (absteigend):")
    print(f"  {'j':>3} | {'lambda_j':>12} | {'alpha_j':>10} | {'|lam|*a^2':>12} | {'%F^-':>8}")
    total_neg = result['F_minus']
    for j, c in contributions[:10]:
        pct = c / total_neg * 100 if total_neg > 0 else 0
        print(f"  {j:3d} | {eigenvalues[j]:+12.6e} | {alphas[j]:+10.4f} | {c:12.6e} | {pct:7.1f}%")

    # Dominanter negativer Eigenvektor: Korrelation mit Primzahl-Funktionen
    j_dom = contributions[0][0]
    v_dom = eigenvectors[:, j_dom]
    v_n = v_dom / np.linalg.norm(v_dom)
    ones_n = np.ones(N) / np.sqrt(N)
    logs_n = logs / np.linalg.norm(logs)

    print(f"\n  Dominanter neg. EV (j={j_dom}, lambda={eigenvalues[j_dom]:.4e}):")
    print(f"    |<v, 1>|/||1||  = {abs(np.dot(v_n, ones_n)):.6f}")
    print(f"    |<v, log p>|    = {abs(np.dot(v_n, logs_n)):.6f}")

    # Vorzeichen-Struktur des dominanten EV
    print(f"    Erste 10 Komponenten: {['%.3f' % v_dom[k] for k in range(min(10, N))]}")
    sign_changes = sum(1 for k in range(N-1) if v_dom[k]*v_dom[k+1] < 0)
    print(f"    Vorzeichenwechsel: {sign_changes}")

# ===========================================================================
# 6. No-Coordination Test
# ===========================================================================

def no_coordination_test(sigma):
    """
    Vergleiche echte Primzahlen mit Pseudo-Primzahlen
    (geometrisch = rationale log-Verhaeltnisse).
    """
    print(f"\n{'='*70}")
    print(f"NO-COORDINATION TEST (sigma={sigma:.4f})")
    print(f"{'='*70}")

    N = 30
    primes = get_primes(N)

    # Geometrisch: p_k = 2^k (log-Verhaeltnisse alle rational!)
    pseudo_geom = np.array([2.0**k for k in range(1, N+1)])

    # "Fast-Primzahlen": zufaellige Zahlen mit aehnlicher Verteilung
    rng = np.random.RandomState(42)
    pseudo_rand = np.sort(2 + rng.exponential(scale=10.0, size=N).cumsum())

    datasets = [
        ("Echte Primzahlen", primes),
        ("Geometrisch 2^k", pseudo_geom),
        ("Zufaellig (exp)", pseudo_rand),
    ]

    print(f"\n  {'Basis':>20} | {'F(sigma)':>14} | {'w^-':>10} | {'n_neg':>5} | {'F >= 0?':>8}")
    print(f"  {'-'*20}-+-{'-'*14}-+-{'-'*10}-+-{'-'*5}-+-{'-'*8}")

    for name, data in datasets:
        res = spectral_analysis(data, sigma, verbose=False)
        ok = "JA" if res['F_dir'] >= 0 else "NEIN"
        print(f"  {name:>20} | {res['F_dir']:+14.6e} | {res['w_minus']:10.6f} | {res['n_neg']:5d} | {ok:>8}")

# ===========================================================================
# 7. Die Uebersetzungsbruecke: Welche xi-Eigenschaft kontrolliert w^-?
# ===========================================================================

def bridge_analysis(primes, sigma):
    """
    Zentrale Frage: Welche Eigenschaft von xi(s) entspricht
    "1 fast orthogonal zu V^-(sigma)" in RH6?

    Hypothesen testen:
    H1: Die Euler-Positivitaet (alle Koeffizienten positiv) kontrolliert w^-
    H2: Die Funktionalgleichung (Spiegelsymmetrie) kontrolliert w^-
    H3: Die arithmetische Unabhaengigkeit (No-Coordination) kontrolliert w^-
    """
    print(f"\n{'='*70}")
    print(f"UEBERSETZUNGSBRUECKE (sigma={sigma:.4f})")
    print(f"{'='*70}")

    N = len(primes)
    res = spectral_analysis(primes, sigma, verbose=False)
    eigenvalues = res['eigenvalues']
    eigenvectors = res['eigenvectors']
    K = res['K']

    # H1: Zerlegung von K in positiven und negativen Teil
    # K(p,q) = K_cross(p,q) + K_self(p,q)
    # K_cross: Wechselwirkungsterme (p != q)
    # K_self: Selbstwechselwirkung (Diagonale)
    diag = np.diag(np.diag(K))
    offdiag = K - diag

    print(f"\n  H1: Zerlegung Diagonal vs. Off-Diagonal:")
    print(f"    sum(K_diag) = {np.sum(diag):+.6e}")
    print(f"    sum(K_off)  = {np.sum(offdiag):+.6e}")
    print(f"    sum(K)      = {np.sum(K):+.6e}")
    print(f"    => Diagonale dominiert: {'JA' if abs(np.sum(diag)) > abs(np.sum(offdiag)) else 'NEIN'}")

    # Eigenwerte der Diagonale vs. Offdiagonale
    evals_diag = np.linalg.eigvalsh(diag)
    evals_off = np.linalg.eigvalsh(offdiag)
    print(f"    K_diag: {sum(1 for e in evals_diag if e < 0)} neg. EW")
    print(f"    K_off:  {sum(1 for e in evals_off if e < 0)} neg. EW")

    # H2: Rolle der Primzahl p=2 (dominiert den Kernel)
    # Entferne p=2 und teste
    K_no2 = K[1:, 1:]
    primes_no2 = primes[1:]
    ones_no2 = np.ones(N-1)
    F_no2 = ones_no2 @ K_no2 @ ones_no2
    print(f"\n  H2: Ohne p=2:")
    print(f"    F(sigma) mit p=2: {res['F_dir']:+.6e}")
    print(f"    F(sigma) ohne p=2: {F_no2:+.6e}")

    # H3: Rang-1-Update durch Euler-Korrekturen
    # Die Euler-Produkt-Positivitaet sagt: log zeta = sum log(1-p^{-s})^{-1} > 0
    # Das liefert eine POSITIVE definite Korrektur zum Kernel?
    logs = np.log(primes)
    pss = primes ** sigma

    # Euler-Korrektur: e_i = log(p_i) / (p_i^sigma - 1)
    euler = logs / (pss - 1)
    euler_outer = np.outer(euler, euler)

    # K + c * euler . euler^T: ab welchem c wird K PSD?
    # Brauchen: lambda_min(K) + c * ||euler_proj_auf_min_EV||^2 >= 0
    min_eval = eigenvalues[0]
    min_evec = eigenvectors[:, 0]
    euler_proj = np.dot(euler / np.linalg.norm(euler), min_evec)**2

    if min_eval < 0 and euler_proj > 1e-15:
        c_crit = abs(min_eval) / (euler_proj * np.linalg.norm(euler)**2)
        print(f"\n  H3: Euler-Rang-1-Korrektur:")
        print(f"    lambda_min = {min_eval:.6e}")
        print(f"    |<euler, v_min>|^2 = {euler_proj:.6f}")
        print(f"    Kritisches c fuer PSD: {c_crit:.6e}")
    else:
        print(f"\n  H3: Euler-Korrektur nicht anwendbar (min_eval={min_eval:.6e})")

    # Die entscheidende Beobachtung
    print(f"\n  ENTSCHEIDENDE BEOBACHTUNG:")
    print(f"  K_sym hat {res['n_neg']} negative Eigenwerte.")
    print(f"  Die KONSTANTE Funktion 1 projiziert mit w^- = {res['w_minus']:.6f} auf V^-.")
    if res['w_minus'] > 0.5:
        print(f"  => F(sigma) < 0: Die Projektion auf V^- ueberwiegt!")
        print(f"  => ABER: Mit endlich vielen Primzahlen (N={N}) fehlen die")
        print(f"     kleinen Eigenwerte, die bei N->inf entstehen.")
        print(f"  => Konvergenz-Test noetig!")

# ===========================================================================
# 8. Analytischer Check: F(sigma) fuer sigma >> 1
# ===========================================================================

def analytical_check():
    """
    Fuer sigma >> 1 dominiert p=2:
    F(sigma) ~ A(sigma)*B(sigma) - P(sigma)*(C(sigma) + D(sigma))
    mit A ~ log(2)/(2^sigma(2^sigma-1)), B ~ log(2)*2^{-sigma}, etc.
    """
    print(f"\n{'='*70}")
    print(f"ANALYTISCHER CHECK: F(sigma) fuer grosse sigma")
    print(f"{'='*70}")

    for sigma in [2.0, 3.0, 5.0, 10.0, 20.0]:
        # Exakt mit allen Primzahlen (asymptotisch nur p=2 relevant)
        s2 = 2.0 ** sigma
        l2 = np.log(2.0)

        A_approx = l2 / (s2 * (s2 - 1))
        B_approx = l2 / s2
        P_approx = 1 / s2
        C_approx = l2**2 / (s2 * (s2 - 1))
        D_approx = l2**2 / (s2 - 1)**2

        F_approx = A_approx * B_approx - P_approx * (C_approx + D_approx)

        # Mit mehr Primzahlen
        primes = get_primes(50)
        F_50 = F_direct(primes, sigma)

        print(f"  sigma={sigma:5.1f}: F_approx(p=2) = {F_approx:+.6e}, F(50 Primz.) = {F_50:+.6e}")

    # Fuer sigma -> infinity:
    # A*B ~ (log 2)^2 / (2^{2sigma} * (2^sigma-1)) ~ (log 2)^2 * 2^{-3sigma}
    # P*C ~ (log 2)^2 / (2^{2sigma} * (2^sigma-1)) ~ (log 2)^2 * 2^{-3sigma}
    # P*D ~ (log 2)^2 / (2^sigma * (2^sigma-1)^2) ~ (log 2)^2 * 2^{-3sigma}
    # Also F ~ (log 2)^2 * 2^{-3sigma} * [1 - 1 - 1] < 0 !
    print(f"\n  ASYMPTOTIK: Fuer sigma >> 1 ist F(sigma) ~ -(log 2)^2 * 2^{{-3sigma}} < 0")
    print(f"  Das heisst: F(sigma) < 0 ist NICHT nur ein finites-N-Artefakt!")
    print(f"  F(sigma) < 0 fuer alle sigma > 1 (mit endlich vielen oder allen Primzahlen).")

# ===========================================================================
# 9. Reinterpretation: Was sagt RH6 wirklich?
# ===========================================================================

def reinterpretation(primes, sigma):
    print(f"\n{'='*70}")
    print(f"REINTERPRETATION: Was sagt RH6 wirklich?")
    print(f"{'='*70}")

    P, A, B, C, D = prime_sums(primes, sigma)
    F = A*B - P*(C+D)

    # m_1(sigma) = -zeta'(sigma)/zeta(sigma) - sigma/(sigma-1)
    # m_1'(sigma) <= 0 <=> F(sigma) >= 0
    # Aber m_1'(sigma) hat eine bestimmte Normierung!

    # Schauen wir uns die Terme einzeln an:
    print(f"\n  sigma = {sigma:.4f}")
    print(f"  P = {P:.8e}")
    print(f"  A = {A:.8e}")
    print(f"  B = {B:.8e}")
    print(f"  C = {C:.8e}")
    print(f"  D = {D:.8e}")
    print(f"  A*B = {A*B:.8e}")
    print(f"  P*(C+D) = {P*(C+D):.8e}")
    print(f"  F = A*B - P*(C+D) = {F:.8e}")

    # Vorzeichen-Analyse:
    # A*B > 0 immer (beide positiv)
    # P*(C+D) > 0 immer
    # F = A*B - P*(C+D): Differenz zweier positiver Groessen
    # Cauchy-Schwarz: A*B <= sqrt(?) * sqrt(?)

    # Der Kern der Sache: Ist A*B >= P*(C+D) ?
    ratio = (A*B) / (P*(C+D)) if P*(C+D) > 0 else float('inf')
    print(f"\n  A*B / P*(C+D) = {ratio:.8f}")
    print(f"  F >= 0 <=> Verhaeltnis >= 1")

    if ratio < 1:
        print(f"\n  VERHAELTNIS < 1: F(sigma) < 0.")
        print(f"  Die 'Kreuzterme' A*B (Wechselwirkung verschiedener Primzahlen)")
        print(f"  sind KLEINER als die 'Selbstterme' P*(C+D).")
        print(f"  => Die Primzahlen-Selbstwechselwirkung dominiert!")

# ===========================================================================
# HAUPTPROGRAMM
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("WEG 2 v2: ANALYTIZITAET ALS STABILISIERUNG")
    print("=" * 70)

    N_PRIMES = 50
    primes = get_primes(N_PRIMES)
    sigma_test = 1.5

    # 1. Konsistenz-Check
    print(f"\n{'='*70}")
    print(f"TEIL 1: Konsistenz-Check und Spektralanalyse")
    print(f"{'='*70}")
    result = spectral_analysis(primes, sigma_test, verbose=True)

    # 2. Sigma-Scan
    sigma_scan(primes)

    # 3. Konvergenz
    convergence_test(1.5)
    convergence_test(1.01)

    # 4. Negative Eigenvektoren
    analyze_negative_eigenvectors(primes, sigma_test, result)

    # 5. No-Coordination
    no_coordination_test(sigma_test)

    # 6. Uebersetzungsbruecke
    bridge_analysis(primes, sigma_test)

    # 7. Analytischer Check fuer grosse sigma
    analytical_check()

    # 8. Reinterpretation
    reinterpretation(primes, sigma_test)

    # FAZIT
    print(f"\n{'='*70}")
    print(f"FAZIT")
    print(f"{'='*70}")
    print(f"""
  ENTSCHEIDENDES ERGEBNIS:

  F(sigma) = A*B - P*(C+D) < 0 fuer ALLE getesteten sigma > 1.
  Das ist KEIN numerisches Artefakt -- es gilt auch fuer sigma >> 1.

  Das bedeutet: m_1'(sigma) > 0, d.h. m_1 ist NICHT monoton fallend.

  ABER WARTE: Das Paper sagt F(sigma) >= 0 <=> RH.
  Wenn F(sigma) < 0, dann ist RH NICHT bewiesen durch diesen Zugang.

  Die Frage ist: Ist F(sigma) < 0 tatsaechlich WAHR (dann waere RH
  durch diesen Zugang widerlegt!), oder gibt es einen Fehler in der
  Berechnung?

  MOEGLICHE ERKLAERUNGEN:
  1. F(sigma) in RH6 gilt fuer sigma im KRITISCHEN STREIFEN (1/2 < sigma < 1),
     nicht fuer sigma > 1. Die Fortsetzung von m_1 nach sigma < 1 aendert
     die Vorzeichen.
  2. Die Normierung von F in RH6 beinhaltet Terme, die ich hier weglasse
     (z.B. Gamma-Funktions-Beitraege aus xi).
  3. F(sigma) >= 0 gilt NUR unter Annahme von RH (und ist dann trivial),
     nicht als aequivalente Bedingung.

  NAECHSTER SCHRITT: RH6-Paper genau lesen, welche sigma-Bereiche
  die Aussage F(sigma) >= 0 betrifft, und ob es einen Korrekturfaktor gibt.
""")
