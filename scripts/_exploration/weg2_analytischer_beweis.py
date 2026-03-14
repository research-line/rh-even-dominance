#!/usr/bin/env python3
"""
weg2_analytischer_beweis.py
============================
Analytischer Beweis-Ansatz fuer gap > 0.

STRATEGIE:
  1. Berechne Diagonal-Elemente Q_{nn} = <phi_n|QW|phi_n> als Funktion von n
  2. Zeige: Q_{nn} hat isoliertes MINIMUM bei n_res (omega_{n_res} ~ gamma_1)
  3. Berechne Off-Diagonal-Elemente |Q_{nm}|
  4. Perturbationstheorie: gap >= diag_gap - 2*sum |Q_{nm}|^2 / diag_gap
  5. Wenn diag_gap >> off-diag-coupling: gap > 0 bewiesen

ANALYTISCHE FORMELN:
  Basis: phi_n(t) = cos(n*pi*t/(2L)) / sqrt(L)  (n>=1), phi_0 = 1/sqrt(2L)

  W_arch Diagonal:
    <phi_n|W_arch|phi_n> = (log 4pi + gamma) + integral_term(n)

  W_p Diagonal:
    <phi_n|W_p|phi_n> = (log p) sum_m p^{-m/2} * 2 * <phi_n, L_{p^m} phi_n>
    = (log p) sum_m p^{-m/2} * 2 * integral phi_n(t) phi_n(t - m*log(p)) dt
    = (log p) sum_m p^{-m/2} * 2 * cos(n*pi*m*log(p)/(2L)) * overlap(m*log(p), L)

  Der Cosinus-Faktor cos(omega_n * m * log(p)) ist der SCHLUESSEL:
  Bei omega_n ~ gamma_k resoniert er mit der Weil-Formel!
"""

import numpy as np
from scipy.linalg import eigh
from sympy import primerange
from mpmath import euler as mp_euler, log as mplog, pi as mppi, im, zetazero, mp
import time

mp.dps = 25
LOG4PI_GAMMA = float(mplog(4 * mppi)) + float(mp_euler)

# ===========================================================================
# Analytische Diagonal-Elemente
# ===========================================================================

def overlap_factor(shift, L):
    """Overlap-Faktor: integral_{-L}^{L} phi_n(t) phi_n(t-s) dt / ||phi_n||^2
    Fuer die Cosinus-Basis (n>=1): Overlap = max(0, 1 - |s|/(2L)) * cos(...)
    Genauer: Fuer phi_n(t) = cos(omega_n * t):
      integral phi_n(t) phi_n(t-s) dt = (L - |s|/2) * cos(omega_n * s)  fuer |s| < 2L
    Normierung: integral phi_n^2 dt = L
    => Overlap = (1 - |s|/(2L)) * cos(omega_n * s)
    """
    if abs(shift) >= 2 * L:
        return 0.0
    return max(0.0, 1.0 - abs(shift) / (2 * L))

def diagonal_Wp(n, p, L, M_terms=20):
    """Diagonal-Element <phi_n|W_p|phi_n> (analytisch).

    W_p Beitrag zum Diagonal-Element:
    = (log p) sum_{m=1}^M p^{-m/2} * 2 * overlap(m*logp, L) * cos(omega_n * m * logp)

    wobei omega_n = n*pi/(2L) und overlap = max(0, 1 - m*logp/(2L))
    """
    logp = np.log(p)
    omega_n = n * np.pi / (2 * L)
    result = 0.0

    for m in range(1, M_terms + 1):
        shift = m * logp
        if shift >= 2 * L:
            break
        coeff = logp * p**(-m / 2.0)
        ov = overlap_factor(shift, L)
        # Faktor 2 weil beide Richtungen (+shift und -shift) gleich beitragen
        result += 2.0 * coeff * ov * np.cos(omega_n * shift)

    return result

def diagonal_Warch(n, L, n_int=500):
    """Diagonal-Element <phi_n|W_arch|phi_n> (numerisch)."""
    omega_n = n * np.pi / (2 * L)

    # Erster Term: (log 4pi + gamma)
    result = LOG4PI_GAMMA

    # Integral-Term: integral_0^{2L} K(s) * (2*ov(s)*cos(omega*s) - 2*e^{-s/2}) ds
    # K(s) = 1/(2*sinh(s))
    s_max = min(2 * L, 8.0)
    s_grid = np.linspace(0.005, s_max, n_int)
    ds = s_grid[1] - s_grid[0]

    for s in s_grid:
        kernel = 1.0 / (2.0 * np.sinh(s))
        if kernel < 1e-15:
            continue
        ov = overlap_factor(s, L)
        # Beitrag von shift +s und -s ist gleich (gerade Funktion)
        result += 2.0 * (ov * np.cos(omega_n * s) - np.exp(-s/2)) * kernel * ds

    return result

def diagonal_QW(n, L, primes, M_terms=20, n_int=500):
    """Gesamtes Diagonal-Element Q_{nn}."""
    d_arch = diagonal_Warch(n, L, n_int)
    d_prime = sum(diagonal_Wp(n, p, L, M_terms) for p in primes)
    return d_arch + d_prime, d_arch, d_prime

# ===========================================================================
# Off-Diagonal-Elemente (analytisch)
# ===========================================================================

def offdiag_Wp(n, m_idx, p, L, M_terms=20):
    """Off-Diagonal-Element <phi_n|W_p|phi_m> (analytisch).

    Fuer Cosinus-Basis:
    <phi_n|L_s phi_m> = integral cos(omega_n * t) cos(omega_m * (t-s)) dt
    = 0.5 * integral [cos((omega_n - omega_m)*t + omega_m*s)
                     + cos((omega_n + omega_m)*t - omega_m*s)] dt

    Fuer das Integral ueber [-L, L]:
    = 0.5 * [sinc-term(omega_n - omega_m) * cos(omega_m*s)
           + sinc-term(omega_n + omega_m) * cos(omega_m*s)]  (ungefaehr)

    Genauer (ohne Sinc-Approximation):
    = 0.5 * overlap_{n-m}(s) * cos((omega_n+omega_m)*s/2) * ...
    """
    # Numerisch berechnen (analytisch ist komplex wegen Randeffekte)
    logp = np.log(p)
    omega_n = n * np.pi / (2 * L)
    omega_m = m_idx * np.pi / (2 * L)

    result = 0.0
    n_quad = 400
    t_grid = np.linspace(-L, L, n_quad)
    dt = t_grid[1] - t_grid[0]

    # Basis-Werte
    if n == 0:
        pn = np.ones(n_quad) / np.sqrt(2 * L)
    else:
        pn = np.cos(omega_n * t_grid) / np.sqrt(L)

    if m_idx == 0:
        pm_base = np.ones(n_quad) / np.sqrt(2 * L)
    else:
        pm_base = np.cos(omega_m * t_grid) / np.sqrt(L)

    for k in range(1, M_terms + 1):
        shift = k * logp
        if shift >= 2 * L:
            break
        coeff = logp * p**(-k / 2.0)

        for sign in [1.0, -1.0]:
            s = sign * shift
            ts = t_grid - s
            mask = np.abs(ts) <= L
            pm_shifted = np.zeros(n_quad)
            if m_idx == 0:
                pm_shifted[mask] = 1.0 / np.sqrt(2 * L)
            else:
                pm_shifted[mask] = np.cos(omega_m * ts[mask]) / np.sqrt(L)

            result += coeff * np.sum(pn * pm_shifted) * dt

    return result

# ===========================================================================
# TEST 1: Diagonal-Landschaft
# ===========================================================================

def test_diagonal_landscape(lam, primes):
    """Zeige Q_{nn} als Funktion von n."""
    L = np.log(lam)
    N_max = int(4 * L * 15 / np.pi) + 5  # Bis omega ~ 60 (gamma_10 ~ 50)

    print(f"\n{'='*75}")
    print(f"DIAGONAL-LANDSCHAFT Q_{{nn}} (lambda={lam}, L={L:.4f})")
    print(f"  N_max = {N_max}, omega_max = {N_max * np.pi / (2*L):.2f}")
    print(f"{'='*75}")

    # Zeta-Nullstellen
    gammas = [float(im(zetazero(k))) for k in range(1, 11)]

    print(f"\n  {'n':>4} | {'omega_n':>10} | {'Q_nn':>12} | {'W_arch':>12} | {'W_prime':>12} | Resonanz")
    print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+----------")

    diag_vals = []
    min_val = float('inf')
    min_n = -1

    for n in range(N_max + 1):
        omega = n * np.pi / (2 * L)
        q_total, q_arch, q_prime = diagonal_QW(n, L, primes)
        diag_vals.append((n, omega, q_total, q_arch, q_prime))

        if q_total < min_val:
            min_val = q_total
            min_n = n

        # Resonanz-Check
        resonance = ""
        for k, gk in enumerate(gammas):
            if abs(omega - gk) < 0.5:
                resonance = f"<= gamma_{k+1} (Delta={abs(omega-gk):.3f})"
                break

        if n <= 5 or resonance or n == min_n or (n % 5 == 0 and n <= 50):
            print(f"  {n:4d} | {omega:10.4f} | {q_total:+12.6e} | {q_arch:+12.6e} | "
                  f"{q_prime:+12.6e} | {resonance}")

    # Zusammenfassung
    print(f"\n  MINIMUM: n={min_n}, omega={min_n*np.pi/(2*L):.4f}, Q_nn={min_val:+.8e}")

    # Zweitsleinstes
    sorted_diag = sorted(diag_vals, key=lambda x: x[2])
    second = sorted_diag[1]
    print(f"  ZWEIT:   n={second[0]}, omega={second[1]:.4f}, Q_nn={second[2]:+.8e}")
    print(f"  DIAG-LUECKE: {second[2] - min_val:.8e}")

    return diag_vals, gammas

# ===========================================================================
# TEST 2: Off-Diagonal Staerke
# ===========================================================================

def test_offdiag_strength(lam, primes, n_res):
    """Wie stark sind die Off-Diagonal-Elemente im Vergleich zur Diag-Luecke?"""
    L = np.log(lam)

    print(f"\n{'='*75}")
    print(f"OFF-DIAGONAL STAERKE (lambda={lam}, n_res={n_res})")
    print(f"{'='*75}")

    # Berechne Q_{n_res, m} fuer alle m
    N_test = min(60, int(4 * L * 15 / np.pi) + 5)

    offdiag = []
    for m in range(N_test + 1):
        if m == n_res:
            continue
        val = sum(offdiag_Wp(n_res, m, p, L) for p in primes[:10])
        offdiag.append((m, abs(val), val))

    # Sortiere nach Staerke
    offdiag.sort(key=lambda x: -x[1])

    print(f"\n  Staerkste Off-Diagonal-Elemente |Q_{{n_res, m}}|:")
    print(f"  {'m':>4} | {'omega_m':>10} | {'|Q_{n,m}|':>12} | {'Q_{n,m}':>12}")
    print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}")
    for m, absval, val in offdiag[:15]:
        omega_m = m * np.pi / (2 * L)
        print(f"  {m:4d} | {omega_m:10.4f} | {absval:12.6e} | {val:+12.6e}")

    # Summe der quadrierten Off-Diagonal-Elemente
    sum_sq = sum(v**2 for _, _, v in offdiag)
    max_offdiag = offdiag[0][1] if offdiag else 0
    sum_abs = sum(absv for _, absv, _ in offdiag)

    print(f"\n  sum |Q_{{n_res,m}}|^2 = {sum_sq:.6e}")
    print(f"  max |Q_{{n_res,m}}|   = {max_offdiag:.6e}")
    print(f"  sum |Q_{{n_res,m}}|   = {sum_abs:.6e}")

    return offdiag, sum_sq

# ===========================================================================
# TEST 3: Perturbations-Schranke fuer Gap
# ===========================================================================

def test_perturbation_bound(lam, primes):
    """Berechne untere Schranke fuer Gap via Perturbationstheorie."""
    L = np.log(lam)

    print(f"\n{'='*75}")
    print(f"PERTURBATIONS-SCHRANKE (lambda={lam})")
    print(f"{'='*75}")

    # Diagonal-Landschaft
    N_max = int(4 * L * 15 / np.pi) + 5
    diag = []
    for n in range(N_max + 1):
        q_total, _, _ = diagonal_QW(n, L, primes)
        diag.append((n, q_total))

    # Sortiere nach Wert
    diag.sort(key=lambda x: x[1])
    n_res = diag[0][0]
    d_min = diag[0][1]
    d_second = diag[1][1]
    diag_gap = d_second - d_min

    print(f"\n  n_res = {n_res} (omega = {n_res*np.pi/(2*L):.4f})")
    print(f"  d_min = {d_min:+.8e}")
    print(f"  d_2nd = {d_second:+.8e}")
    print(f"  diag_gap = {diag_gap:.8e}")

    # Off-Diagonal
    N_test = min(50, N_max)
    primes_used = primes[:10]

    # Frobenius-Norm der Off-Diagonal-Zeile n_res
    row_sum_sq = 0.0
    for m in range(N_test + 1):
        if m == n_res:
            continue
        val = sum(offdiag_Wp(n_res, m, p, L) for p in primes_used)
        row_sum_sq += val**2

    row_norm = np.sqrt(row_sum_sq)

    # Gershgorin: |lambda_min - d_min| <= sum_m |Q_{n_res,m}|
    # => lambda_min >= d_min - row_sum_abs
    # => gap >= diag_gap - 2 * row_sum_abs  (grob)

    row_sum_abs = 0.0
    for m in range(N_test + 1):
        if m == n_res:
            continue
        val = sum(offdiag_Wp(n_res, m, p, L) for p in primes_used)
        row_sum_abs += abs(val)

    # Weyl-Perturbation (schaerfer):
    # |lambda_k - d_k| <= ||Off-Diag||_F
    # Also: gap >= diag_gap - 2 * ||Off-Diag-Zeile||_2

    gap_lower_gershgorin = diag_gap - 2 * row_sum_abs
    gap_lower_weyl = diag_gap - 2 * row_norm

    print(f"\n  Off-Diagonal-Zeile n_res:")
    print(f"    ||row||_2 (Frobenius) = {row_norm:.6e}")
    print(f"    sum |Q_{n_res,m}|     = {row_sum_abs:.6e}")

    print(f"\n  SCHRANKEN:")
    print(f"    Gershgorin: gap >= {gap_lower_gershgorin:+.6e}")
    print(f"    Weyl:       gap >= {gap_lower_weyl:+.6e}")

    # Numerischer Vergleich
    from scipy.linalg import eigh as sp_eigh

    def make_basis_grid(N, t_grid, L):
        phi = np.zeros((N, len(t_grid)))
        phi[0, :] = 1.0 / np.sqrt(2 * L)
        for n in range(1, N):
            phi[n, :] = np.cos(n * np.pi * t_grid / (2 * L)) / np.sqrt(L)
        return phi

    def make_shifted(N, t_grid, L, shift):
        ts = t_grid - shift
        mask = np.abs(ts) <= L
        phi = np.zeros((N, len(t_grid)))
        phi[0, mask] = 1.0 / np.sqrt(2 * L)
        for nn in range(1, N):
            phi[nn, mask] = np.cos(nn * np.pi * ts[mask] / (2 * L)) / np.sqrt(L)
        return phi

    N_num = min(50, N_max)
    n_quad = 800
    t_grid = np.linspace(-L, L, n_quad)
    dt = t_grid[1] - t_grid[0]
    phi = make_basis_grid(N_num, t_grid, L)

    W = LOG4PI_GAMMA * np.eye(N_num)
    s_max = min(2 * L, 8.0)
    s_grid = np.linspace(0.005, s_max, 400)
    ds = s_grid[1] - s_grid[0]
    for s in s_grid:
        k = 1.0 / (2.0 * np.sinh(s))
        if k < 1e-15:
            continue
        Sp = (phi @ make_shifted(N_num, t_grid, L, s).T) * dt
        Sm = (phi @ make_shifted(N_num, t_grid, L, -s).T) * dt
        W += (Sp + Sm - 2.0 * np.exp(-s/2) * np.eye(N_num)) * k * ds

    for p in primes:
        logp = np.log(p)
        for mm in range(1, 13):
            coeff = logp * p**(-mm / 2.0)
            shift = mm * logp
            if shift >= 2 * L:
                break
            for sign in [1.0, -1.0]:
                S = (phi @ make_shifted(N_num, t_grid, L, sign * shift).T) * dt
                W += coeff * S

    evals = np.sort(sp_eigh(W, eigvals_only=True))
    num_gap = evals[1] - evals[0]

    print(f"\n  NUMERISCH (N={N_num}):")
    print(f"    lambda_min = {evals[0]:+.8e}")
    print(f"    lambda_2   = {evals[1]:+.8e}")
    print(f"    gap        = {num_gap:.8e}")

    print(f"\n  VERGLEICH:")
    print(f"    diag_gap       = {diag_gap:.6e}")
    print(f"    num_gap        = {num_gap:.6e}")
    print(f"    ratio num/diag = {num_gap/diag_gap:.4f}" if diag_gap > 0 else "")

    return diag_gap, num_gap, gap_lower_weyl

# ===========================================================================
# TEST 4: Resonanz-Formel
# ===========================================================================

def test_resonance_formula(primes):
    """Leite eine analytische Formel fuer den Resonanz-Dip her."""
    print(f"\n{'='*75}")
    print(f"RESONANZ-FORMEL")
    print(f"{'='*75}")

    gammas = [float(im(zetazero(k))) for k in range(1, 11)]
    gamma1 = gammas[0]

    print(f"\n  gamma_1 = {gamma1:.6f}")
    print(f"\n  Fuer jedes lambda gibt es n_res = round(gamma_1 * 2L / pi)")
    print(f"  Dann: omega_{'{n_res}'} = n_res * pi / (2L) ~ gamma_1")
    print(f"  Fehler: |omega - gamma_1| <= pi/(4L)")

    lambdas = [5, 8, 13, 20, 30, 50, 80, 100, 200]

    print(f"\n  {'lambda':>6} | {'L':>8} | {'n_res':>5} | {'omega':>10} | "
          f"{'Delta':>10} | {'Q_nn(res)':>12} | {'Q_nn(res-1)':>12} | {'diag_gap':>12}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for lam in lambdas:
        L = np.log(lam)
        n_res = round(gamma1 * 2 * L / np.pi)
        omega_res = n_res * np.pi / (2 * L)
        delta = abs(omega_res - gamma1)

        q_res, _, _ = diagonal_QW(n_res, L, primes)
        q_prev, _, _ = diagonal_QW(max(0, n_res - 1), L, primes)
        q_next, _, _ = diagonal_QW(n_res + 1, L, primes)

        # Diagonal-Gap: Differenz zum naechsten Nicht-Resonanz-Element
        neighbors = [q_prev, q_next]
        diag_gap = min(neighbors) - q_res

        print(f"  {lam:6d} | {L:8.4f} | {n_res:5d} | {omega_res:10.4f} | "
              f"{delta:10.6f} | {q_res:+12.6e} | {q_prev:+12.6e} | {diag_gap:+12.6e}")

    # Analytische Abschaetzung des Dips
    print(f"\n  ANALYTISCHE ABSCHAETZUNG:")
    print(f"  Q_nn(omega) ~ const + sum_p (log p) * sum_m p^{{-m/2}} * cos(omega * m * log p)")
    print(f"  Bei omega ~ gamma_1: Die Weil-Formel sagt")
    print(f"    sum_p W_p(delta_gamma_1) = -1 + O(Arch-Terme)")
    print(f"  => Q_nn(gamma_1) liegt UNTER dem Durchschnitt")
    print(f"  => Die 'Tiefe' des Dips ist proportional zur 'Staerke' der Nullstelle")

# ===========================================================================
# HAUPTPROGRAMM
# ===========================================================================

if __name__ == "__main__":
    print("=" * 75)
    print("ANALYTISCHER BEWEIS-ANSATZ: gap > 0")
    print("=" * 75)

    primes = list(primerange(2, 100))

    # Test 1: Diagonal-Landschaft fuer lambda=20
    diag_vals, gammas = test_diagonal_landscape(20, primes)

    # Test 2: Off-Diagonal
    L = np.log(20)
    n_res_20 = round(gammas[0] * 2 * L / np.pi)
    offdiag, sum_sq = test_offdiag_strength(20, primes, n_res_20)

    # Test 3: Perturbations-Schranke
    results = {}
    for lam in [13, 20, 30]:
        print(f"\n  === lambda = {lam} ===")
        dg, ng, lb = test_perturbation_bound(lam, primes)
        results[lam] = (dg, ng, lb)

    # Test 4: Resonanz-Formel
    test_resonance_formula(primes)

    # FAZIT
    print(f"\n{'='*75}")
    print(f"FAZIT: ANALYTISCHER BEWEIS-ANSATZ")
    print(f"{'='*75}")
    print(f"""
  BEWEIS-SKIZZE:

  Theorem (Gap > 0): Fuer alle lambda > lambda_0 gilt:
    gap(A_lambda) >= c / L  (mit L = log lambda, c > 0)

  Beweis-Idee:
  1. DIAGONAL-DOMINANZ: Die Matrix QW ist diagonal-dominant in der
     Cosinus-Basis, mit Diagonal-Elementen Q_nn(omega_n).

  2. RESONANZ-DIP: Q_nn hat ein lokales Minimum bei n_res wo
     omega_{{n_res}} ~ gamma_1. Die Tiefe des Dips ist O(1)
     (unabhaengig von L), weil die Weil-Formel die Beitraege
     bei Zeta-Nullstellen zu einem festen Wert summiert.

  3. ISOLATION: Der Dip ist isoliert weil gamma_1 die EINZIGE
     Nullstelle in einer Umgebung der Groesse O(1) ist
     (Nullstellen-Abstand ~ 2*pi/log(gamma) ~ 6.3).

  4. OFF-DIAGONAL: Die Off-Diagonal-Elemente |Q_nm| fallen mit
     |n-m| (weil die Translations-Operatoren L_s bei grossem s
     schwachen Overlap haben). Die Gershgorin-Summe ist O(1).

  5. KOMBINIERT: diag_gap ~ O(1), off_diag ~ O(1),
     aber diag_gap > 2 * off_diag fuer lambda > lambda_0.

  OFFENER PUNKT: Die Schranke diag_gap > 2*off_diag muss
  quantitativ verifiziert werden. Bei lambda=20 (wo gap minimal ist)
  ist dies der engste Punkt.
""")
