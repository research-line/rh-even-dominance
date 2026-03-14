#!/usr/bin/env python3
"""
weg2_analyticity_bridge.py
==========================
Weg 2: Analytizitaet als Stabilisierung -- Uebersetzungsbruecke

Ziel: Verbinde die analytische Struktur von xi (ganz + FE + Euler)
      mit F(sigma) >= 0 bzw. w^-(sigma) <= 1/2 aus RH6.

Kernfragen:
  1. Wie sieht die Spektralzerlegung von T_sigma aus?
  2. Welche Eigenvektoren spannen V^-(sigma) auf?
  3. Warum ist 1 "fast orthogonal" zu V^-(sigma)?
  4. Welche Rolle spielt die arithmetische Unabhaengigkeit (log p / log q irrational)?

Definitionen aus RH6:
  K_sigma(p,q) = [log(p)*log(q)] / [p^s*(p^s-1)*q^s]
                 - [log(p)^2 / q^s] * [1/(p^s-1)^2 + 1/(p^s*(p^s-1))]
  K_sym(p,q) = (K_sigma(p,q) + K_sigma(q,p)) / 2
  mu_sigma(p) = p^{-sigma} / P(sigma)
  T_sigma: (T_sigma f)(p) = sum_q K_sym(p,q) * mu_sigma(q) * f(q)
  F(sigma) / P(sigma)^2 = <1, T_sigma 1>_sigma
  w^-(sigma) = F^- / (F^+ + F^-)  <=  1/2  <=>  F(sigma) >= 0
"""

import numpy as np
from sympy import primerange
import sys

# ===========================================================================
# 1. Prime sums and kernel
# ===========================================================================

def get_primes(N):
    """Erste N Primzahlen."""
    primes = list(primerange(2, 10000))[:N]
    return np.array(primes, dtype=np.float64)

def prime_sums(primes, sigma):
    """Berechne P, A, B, C, D aus RH6."""
    ps = primes ** (-sigma)
    logs = np.log(primes)

    P = np.sum(ps)
    B = np.sum(logs * ps)
    A = np.sum(logs * ps / (primes**sigma - 1))
    C = np.sum(logs**2 * ps / (primes**sigma - 1))
    D = np.sum(logs**2 / (primes**sigma - 1)**2)

    return P, A, B, C, D

def F_direct(primes, sigma):
    """F(sigma) = A*B - P*(C+D) direkt."""
    P, A, B, C, D = prime_sums(primes, sigma)
    return A * B - P * (C + D)

def build_kernel_matrix(primes, sigma):
    """Baue K_sym(p,q) als Matrix."""
    N = len(primes)
    K = np.zeros((N, N))
    logs = np.log(primes)

    for i in range(N):
        p = primes[i]
        ps_i = p ** sigma
        for j in range(N):
            q = primes[j]
            qs_j = q ** sigma

            # K_sigma(p,q)
            K_ij = (logs[i] * logs[j]) / (ps_i * (ps_i - 1) * qs_j) \
                   - (logs[i]**2 / qs_j) * (1/(ps_i - 1)**2 + 1/(ps_i * (ps_i - 1)))

            # K_sigma(q,p)
            K_ji = (logs[j] * logs[i]) / (qs_j * (qs_j - 1) * ps_i) \
                   - (logs[j]**2 / ps_i) * (1/(qs_j - 1)**2 + 1/(qs_j * (qs_j - 1)))

            K[i, j] = 0.5 * (K_ij + K_ji)

    return K

def build_T_matrix(primes, sigma):
    """
    T_sigma als Matrix auf ell^2(PP, mu_sigma).
    (T f)(p_i) = sum_j K_sym(p_i, p_j) * mu_sigma(p_j) * f(p_j)

    In Matrix-Form: T_ij = K_sym(p_i, p_j) * mu_sigma(p_j)
    Das Skalarprodukt ist <f,g> = sum_i f(p_i) * g(p_i) * mu_sigma(p_i).

    Fuer symmetrische Matrix bzgl. <.,.>_sigma:
    T_sym_ij = sqrt(mu_i) * K_sym_ij * sqrt(mu_j)  (aehnlichkeitstransformation)
    """
    N = len(primes)
    K_sym = build_kernel_matrix(primes, sigma)

    ps = primes ** (-sigma)
    P = np.sum(ps)
    mu = ps / P  # Gibbs-Mass

    # Symmetrische Form: M_ij = sqrt(mu_i) * K_sym_ij * sqrt(mu_j)
    sqrt_mu = np.sqrt(mu)
    M = np.outer(sqrt_mu, sqrt_mu) * K_sym

    return M, K_sym, mu

# ===========================================================================
# 2. Spektralanalyse
# ===========================================================================

def spectral_analysis(primes, sigma, verbose=True):
    """Volle Spektralanalyse von T_sigma."""
    N = len(primes)
    M, K_sym, mu = build_T_matrix(primes, sigma)

    # Eigenzerlegung
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    # eigh gibt aufsteigend sortiert zurueck

    # Konstante Funktion im transformierten Raum:
    # 1 -> sqrt(mu) * 1 = sqrt(mu)
    sqrt_mu = np.sqrt(mu)
    one_transformed = sqrt_mu / np.linalg.norm(sqrt_mu)

    # alpha_j = <1_transformed, v_j>
    alphas = eigenvectors.T @ one_transformed  # shape (N,)

    # F^+ und F^-
    F_plus = sum(eigenvalues[j] * alphas[j]**2 for j in range(N) if eigenvalues[j] > 0)
    F_minus = sum(abs(eigenvalues[j]) * alphas[j]**2 for j in range(N) if eigenvalues[j] < 0)

    P = np.sum(primes ** (-sigma))
    F_val = (F_plus - F_minus) * P**2

    w_minus = F_minus / (F_plus + F_minus) if (F_plus + F_minus) > 0 else 0

    n_neg = sum(1 for ev in eigenvalues if ev < 0)
    n_pos = sum(1 for ev in eigenvalues if ev > 0)

    if verbose:
        print(f"\n  sigma = {sigma:.4f}, N_primes = {N}")
        print(f"  Eigenwerte: {n_pos} positiv, {n_neg} negativ")
        print(f"  F^+ = {F_plus:.8e}, F^- = {F_minus:.8e}")
        print(f"  F(sigma) = {F_val:.8e}  (direkt: {F_direct(primes, sigma):.8e})")
        print(f"  w^-(sigma) = {w_minus:.6f}  (Grenze: 0.5)")
        print(f"  F >= 0: {'JA' if F_val >= 0 else 'NEIN!!!'}")

    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'alphas': alphas,
        'F_plus': F_plus,
        'F_minus': F_minus,
        'F_val': F_val,
        'w_minus': w_minus,
        'mu': mu,
        'n_neg': n_neg,
        'n_pos': n_pos,
    }

# ===========================================================================
# 3. Analyse der negativen Eigenvektoren
# ===========================================================================

def analyze_negative_eigenvectors(primes, sigma, result):
    """
    Untersuche die Struktur der negativen Eigenvektoren.
    Kernfrage: Nutzen sie Primzahl-Korrelationen aus?
    """
    print(f"\n{'='*70}")
    print(f"STRUKTUR DER NEGATIVEN EIGENVEKTOREN (sigma={sigma:.4f})")
    print(f"{'='*70}")

    eigenvalues = result['eigenvalues']
    eigenvectors = result['eigenvectors']
    alphas = result['alphas']
    mu = result['mu']
    sqrt_mu = np.sqrt(mu)

    neg_indices = [j for j in range(len(eigenvalues)) if eigenvalues[j] < 0]

    if not neg_indices:
        print("  Keine negativen Eigenwerte!")
        return

    print(f"\n  {len(neg_indices)} negative Eigenwerte:")
    print(f"  {'j':>3} | {'lambda_j':>12} | {'alpha_j':>10} | {'|lam|*a^2':>12} | {'Beitrag zu F^-':>14}")
    print(f"  {'-'*3}-+-{'-'*12}-+-{'-'*10}-+-{'-'*12}-+-{'-'*14}")

    for j in neg_indices:
        contrib = abs(eigenvalues[j]) * alphas[j]**2
        pct = contrib / result['F_minus'] * 100 if result['F_minus'] > 0 else 0
        print(f"  {j:3d} | {eigenvalues[j]:+12.6e} | {alphas[j]:+10.6f} | {contrib:12.6e} | {pct:13.1f}%")

    # Ruecktransformation: v_j im Originalraum
    # v_orig_j(p) = v_j(p) / sqrt(mu(p))
    print(f"\n  Eigenvektoren im Originalraum (erste 8 Primzahlen):")
    print(f"  {'j':>3} | {'lam':>10} | " +
          " | ".join(f"{'p='+str(int(p)):>8}" for p in primes[:8]))

    for j in neg_indices[:5]:
        v_orig = eigenvectors[:, j] / sqrt_mu
        v_norm = v_orig / np.max(np.abs(v_orig))  # normiert auf max=1
        row = f"  {j:3d} | {eigenvalues[j]:+10.4e} | "
        row += " | ".join(f"{v_norm[k]:+8.4f}" for k in range(min(8, len(primes))))
        print(row)

    # Oszillationsanalyse: Vorzeichenwechsel
    print(f"\n  Oszillationsanalyse (Vorzeichenwechsel):")
    for j in neg_indices[:5]:
        v_orig = eigenvectors[:, j] / sqrt_mu
        sign_changes = sum(1 for k in range(len(v_orig)-1) if v_orig[k] * v_orig[k+1] < 0)
        print(f"  j={j}: {sign_changes} Vorzeichenwechsel in {len(primes)} Primzahlen")

    # No-Coordination Test: Korrelation mit log(p)
    print(f"\n  Korrelation der neg. Eigenvektoren mit verschiedenen Funktionen:")
    logs = np.log(primes)
    for j in neg_indices[:5]:
        v_orig = eigenvectors[:, j] / sqrt_mu
        v_orig_n = v_orig / np.linalg.norm(v_orig)

        # Korrelation mit 1 (= alpha_j nach Normierung)
        corr_const = abs(alphas[j])

        # Korrelation mit log(p)
        log_n = logs / np.linalg.norm(logs)
        corr_log = abs(np.dot(v_orig_n, log_n))

        # Korrelation mit p^{-sigma/2}
        half = primes ** (-sigma/2)
        half_n = half / np.linalg.norm(half)
        corr_half = abs(np.dot(v_orig_n, half_n))

        print(f"  j={j}: |<v,1>| = {corr_const:.6f}, "
              f"|<v,log p>| = {corr_log:.6f}, "
              f"|<v,p^{{-s/2}}>| = {corr_half:.6f}")


# ===========================================================================
# 4. No-Coordination Lemma Verbindung
# ===========================================================================

def no_coordination_test(primes, sigma):
    """
    Teste das No-Coordination Lemma (aus RH2):
    log(p)/log(q) irrational fuer p != q Primzahlen
    => Die Phasen p^{-it} und q^{-it} koennen nicht synchronisieren.

    Frage: Kontrolliert diese arithmetische Unabhaengigkeit w^-(sigma)?

    Methode: Ersetze Primzahlen durch "Pseudo-Primzahlen" mit
    rationalen log-Verhaeltnissen und vergleiche w^-(sigma).
    """
    print(f"\n{'='*70}")
    print(f"NO-COORDINATION TEST: Arithmetische vs. rationale Basen")
    print(f"{'='*70}")

    N = len(primes)

    # Echte Primzahlen
    result_real = spectral_analysis(primes, sigma, verbose=False)
    w_real = result_real['w_minus']

    # Pseudo-Primzahlen: geometrische Reihe (alle log-Verhaeltnisse rational!)
    base = 2.0
    pseudo_geom = np.array([base ** k for k in range(1, N+1)])
    result_geom = spectral_analysis(pseudo_geom, sigma, verbose=False)
    w_geom = result_geom['w_minus']

    # Pseudo-Primzahlen: arithmetische Reihe
    pseudo_arith = np.array([2.0 + 3*k for k in range(N)])
    result_arith = spectral_analysis(pseudo_arith, sigma, verbose=False)
    w_arith = result_arith['w_minus']

    # Pseudo-Primzahlen: zufaellig
    rng = np.random.RandomState(42)
    pseudo_rand = np.sort(rng.uniform(2, primes[-1], N))
    result_rand = spectral_analysis(pseudo_rand, sigma, verbose=False)
    w_rand = result_rand['w_minus']

    print(f"\n  sigma = {sigma:.4f}, N = {N}")
    print(f"  {'Basis':>20} | {'w^-(sigma)':>10} | {'n_neg':>5} | {'F(sigma)':>12} | {'Stabil?':>8}")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*5}-+-{'-'*12}-+-{'-'*8}")
    for name, res in [("Echte Primzahlen", result_real),
                      ("Geometrisch (2^k)", result_geom),
                      ("Arithmetisch (2+3k)", result_arith),
                      ("Zufaellig", result_rand)]:
        stable = "JA" if res['F_val'] >= 0 else "NEIN"
        print(f"  {name:>20} | {res['w_minus']:10.6f} | {res['n_neg']:5d} | {res['F_val']:+12.6e} | {stable:>8}")

    print(f"\n  INTERPRETATION:")
    if w_real < w_geom:
        print(f"  Echte Primzahlen: w^- = {w_real:.6f} < {w_geom:.6f} = w^-(geometrisch)")
        print(f"  => Arithmetische Unabhaengigkeit REDUZIERT w^-(sigma)!")
        print(f"  => No-Coordination hilft bei Stabilitaet.")
    else:
        print(f"  Echte Primzahlen: w^- = {w_real:.6f} >= {w_geom:.6f} = w^-(geometrisch)")
        print(f"  => Arithmetische Unabhaengigkeit allein reicht NICHT.")


# ===========================================================================
# 5. Sigma-Abhaengigkeit: Kritischer Bereich sigma -> 1/2
# ===========================================================================

def sigma_scan(primes):
    """Scanne w^-(sigma) fuer sigma von 1.01 bis 5."""
    print(f"\n{'='*70}")
    print(f"SIGMA-SCAN: w^-(sigma) ueber den kritischen Bereich")
    print(f"{'='*70}")

    sigmas = [1.01, 1.02, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0]
    print(f"\n  {'sigma':>8} | {'w^-(sigma)':>10} | {'n_neg':>5} | {'F^+':>12} | {'F^-':>12} | {'F(sigma)':>12} | {'OK?':>4}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*4}")

    for sigma in sigmas:
        res = spectral_analysis(primes, sigma, verbose=False)
        ok = "JA" if res['F_val'] >= 0 else "NEIN"
        print(f"  {sigma:8.4f} | {res['w_minus']:10.6f} | {res['n_neg']:5d} | "
              f"{res['F_plus']:12.6e} | {res['F_minus']:12.6e} | {res['F_val']:+12.6e} | {ok:>4}")

    # Feinerer Scan nahe sigma=1
    print(f"\n  Feiner Scan nahe sigma = 1:")
    fine_sigmas = np.linspace(1.001, 1.1, 20)
    for sigma in fine_sigmas:
        res = spectral_analysis(primes, sigma, verbose=False)
        ok = "JA" if res['F_val'] >= 0 else "NEIN"
        print(f"  {sigma:8.5f} | {res['w_minus']:10.6f} | {res['n_neg']:5d} | {res['F_val']:+12.6e} | {ok:>4}")


# ===========================================================================
# 6. Euler-Produkt Constraint: Multiplikative Struktur
# ===========================================================================

def euler_product_constraint(primes, sigma):
    """
    Teste, ob die multiplikative Struktur des Euler-Produkts
    die Projektion von 1 auf V^-(sigma) kontrolliert.

    Euler: zeta(s) = prod_p (1-p^{-s})^{-1}
    => log zeta(s) = sum_p sum_m (1/m) p^{-ms}
    => Alle Koeffizienten POSITIV

    Hypothese: Die Positivitaet der Euler-Koeffizienten
    erzwingt eine Struktur in T_sigma, die w^-(sigma) <= 1/2 garantiert.
    """
    print(f"\n{'='*70}")
    print(f"EULER-PRODUKT CONSTRAINT (sigma={sigma:.4f})")
    print(f"{'='*70}")

    result = spectral_analysis(primes, sigma, verbose=False)
    eigenvalues = result['eigenvalues']
    eigenvectors = result['eigenvectors']
    mu = result['mu']
    sqrt_mu = np.sqrt(mu)
    N = len(primes)

    # Die "Euler-Funktion": e(p) = log(p) * p^{-sigma} / (1 - p^{-sigma})
    # = d/dsigma [log(1 - p^{-sigma})] (mit Vorzeichen)
    euler_func = np.log(primes) * primes**(-sigma) / (1 - primes**(-sigma))
    euler_transformed = euler_func * sqrt_mu
    euler_norm = euler_transformed / np.linalg.norm(euler_transformed)

    # Projektion der Euler-Funktion auf Eigenraeume
    euler_alphas = eigenvectors.T @ euler_norm

    print(f"\n  Projektion der Euler-Funktion e(p) = log(p)*p^{{-s}}/(1-p^{{-s}}) auf Eigenraeume:")
    print(f"  Anteil in V^+: {sum(euler_alphas[j]**2 for j in range(N) if eigenvalues[j] > 0):.6f}")
    print(f"  Anteil in V^-: {sum(euler_alphas[j]**2 for j in range(N) if eigenvalues[j] < 0):.6f}")

    # Vergleich: 1 vs Euler-Funktion
    one_transformed = sqrt_mu / np.linalg.norm(sqrt_mu)
    one_alphas = eigenvectors.T @ one_transformed

    print(f"\n  Vergleich Projektion auf V^-:")
    print(f"  Konstante 1: {sum(one_alphas[j]**2 for j in range(N) if eigenvalues[j] < 0):.6f}")
    print(f"  Euler e(p):  {sum(euler_alphas[j]**2 for j in range(N) if eigenvalues[j] < 0):.6f}")

    # Winkel zwischen 1 und Euler-Funktion
    cos_angle = np.dot(one_transformed / np.linalg.norm(one_transformed),
                       euler_norm)
    print(f"\n  Winkel zwischen 1 und e(p): {np.degrees(np.arccos(np.clip(cos_angle, -1, 1))):.2f} Grad")

    # Test: Ist e(p) - <1,e>*1 in V^-?
    projection_on_1 = np.dot(euler_norm, one_transformed) * one_transformed
    euler_perp = euler_norm - projection_on_1
    if np.linalg.norm(euler_perp) > 1e-10:
        euler_perp = euler_perp / np.linalg.norm(euler_perp)
        perp_alphas = eigenvectors.T @ euler_perp
        print(f"  e(p) senkrecht zu 1:")
        print(f"    Anteil in V^+: {sum(perp_alphas[j]**2 for j in range(N) if eigenvalues[j] > 0):.6f}")
        print(f"    Anteil in V^-: {sum(perp_alphas[j]**2 for j in range(N) if eigenvalues[j] < 0):.6f}")


# ===========================================================================
# 7. Uebersetzungsbruecke: Zusammenfassung
# ===========================================================================

def translation_bridge_summary(primes, sigma):
    """
    Fasse die Ergebnisse zusammen und formuliere die Uebersetzungsbruecke.
    """
    print(f"\n{'='*70}")
    print(f"UEBERSETZUNGSBRUECKE: Analytizitaet -> F(sigma) >= 0")
    print(f"{'='*70}")

    result = spectral_analysis(primes, sigma, verbose=True)

    # Zentrale Frage beantworten
    print(f"\n  ZENTRALE FRAGE: Warum ist w^-(sigma) <= 1/2?")
    print(f"  Aktuell: w^-(sigma) = {result['w_minus']:.6f}")
    print(f"  F^+/F^- Verhaeltnis: {result['F_plus']/result['F_minus']:.4f}" if result['F_minus'] > 0 else "")

    # Hypothetische Bruecke
    print(f"""
  HYPOTHETISCHE UEBERSETZUNGSBRUECKE:

  xi-Eigenschaft                    RH6-Entsprechung
  ================                  ================
  xi ganze Funktion Ordnung 1   =>  T_sigma trace-class (endlich viele neg. EW)
  xi(s) = xi(1-s) (FE)         =>  Symmetrie im Kernel K_sym
  Euler-Produkt (Positivitaet)  =>  ???  -> Kontrolle von alpha_j fuer V^-?
  log p / log q irrational      =>  ???  -> Keine Synchronisation in V^-?

  OFFENE FRAGEN:
  1. Kann man zeigen, dass die Euler-Positivitaet w^-(sigma) <= 1/2 impliziert?
  2. Gibt es eine operatorielle Formulierung der No-Coordination?
  3. Was passiert bei sigma -> 1/2+ ? (kritischer Grenzwert)
""")


# ===========================================================================
# HAUPTPROGRAMM
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("WEG 2: ANALYTIZITAET ALS STABILISIERUNG")
    print("Uebersetzungsbruecke: xi-Struktur -> F(sigma) >= 0")
    print("=" * 70)

    N_PRIMES = 50
    primes = get_primes(N_PRIMES)
    sigma_test = 1.5

    # 1. Spektralanalyse
    print(f"\n{'='*70}")
    print(f"TEIL 1: Spektralanalyse von T_sigma")
    print(f"{'='*70}")
    result = spectral_analysis(primes, sigma_test, verbose=True)

    # 2. Negative Eigenvektoren
    analyze_negative_eigenvectors(primes, sigma_test, result)

    # 3. No-Coordination Test
    no_coordination_test(primes, sigma_test)

    # 4. Sigma-Scan
    sigma_scan(primes)

    # 5. Euler-Produkt Constraint
    euler_product_constraint(primes, sigma_test)

    # 6. Zusammenfassung
    translation_bridge_summary(primes, sigma_test)
