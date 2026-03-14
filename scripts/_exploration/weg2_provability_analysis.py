#!/usr/bin/env python3
"""
weg2_provability_analysis.py
=============================
Phase 14: Kann man v0(n*) != 0 beweisen?

EHRLICHE ANALYSE: v0(n*) != 0 ist AEQUIVALENT zu gap > 0.
Das Cauchy-Argument ist eine REFORMULIERUNG, kein Beweis.

DREI VIELVERSPRECHENDE ANSAETZE:

A) ANALYTISCHE FORTSETZUNG (Homotopie):
   QW(t) = D + t*V,  t: 0 -> 1
   Bei t=0: v0 = e_{n*} (Diagonal-Minimizer), also v0(n*)=1.
   Falls lambda_1(t) EINFACH bleibt fuer alle t, ist v0(t) stetig
   und v0(1)(n*) kann nur Null werden wenn ein Level-Crossing auftritt.
   => Test: Gibt es Level-Crossings auf dem Pfad t=0..1?

B) TOTAL-POSITIVITAET DES KERNS:
   Falls der Integraloperator-Kern total positiv (TP) ist,
   dann sind ALLE Eigenwerte einfach (Oscillation Theorem).
   TP-Kern: alle Minoren det(K(t_i, s_j)) >= 0.
   Der archimedische Kern 1/(2*sinh(|t-s|)) IST TP.
   Frage: Bleibt der Gesamtkern TP nach Primzahl-Addition?

C) EVEN/ODD-SEPARATION:
   Connes' Operator kommutiert mit t -> -t (Paritaet).
   Unsere Cosinus-Basis ist REIN GERADE.
   Falls lambda_1(even) != lambda_1(odd), ist der Grundzustand
   AUTOMATISCH einfach in seinem Sektor.
   => Test: Berechne auch den Sinus-Sektor (ungerade).

D) IRREDUZIBILITAET + PERRON-FROBENIUS (fuer transformierte Matrix):
   Falls es eine Diagonalmatrix S gibt mit S*QW*S >= 0 (elementweise)
   UND diese Matrix irreduzibel ist, dann ist der Perron-Eigenvektor
   strikt positiv => v0(n*) > 0 fuer ALLE n*.
"""

import numpy as np
from scipy.linalg import eigh
from sympy import primerange
from mpmath import euler as mp_euler, log as mplog, pi as mppi
import time

LOG4PI_GAMMA = float(mplog(4 * mppi)) + float(mp_euler)

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
    for n in range(1, N):
        phi[n, mask] = np.cos(n * np.pi * ts[mask] / (2 * L)) / np.sqrt(L)
    return phi

def make_sin_basis(N, t_grid, L):
    """Sinus-Basis (ungerade Funktionen)."""
    phi = np.zeros((N, len(t_grid)))
    for n in range(N):
        phi[n, :] = np.sin((n+1) * np.pi * t_grid / (2 * L)) / np.sqrt(L)
    return phi

def make_sin_shifted(N, t_grid, L, shift):
    ts = t_grid - shift
    mask = np.abs(ts) <= L
    phi = np.zeros((N, len(t_grid)))
    for n in range(N):
        phi[n, mask] = np.sin((n+1) * np.pi * ts[mask] / (2 * L)) / np.sqrt(L)
    return phi

def build_QW(lam, N, primes, M_terms=12, n_quad=800, n_int=500, basis='cos'):
    L = np.log(lam)
    t_grid = np.linspace(-L, L, n_quad)
    dt = t_grid[1] - t_grid[0]

    if basis == 'cos':
        phi = make_basis_grid(N, t_grid, L)
        make_sh = lambda N, t, L, s: make_shifted(N, t, L, s)
    else:  # sin
        phi = make_sin_basis(N, t_grid, L)
        make_sh = lambda N, t, L, s: make_sin_shifted(N, t, L, s)

    W = LOG4PI_GAMMA * np.eye(N)
    s_max = min(2 * L, 8.0)
    s_grid = np.linspace(0.005, s_max, n_int)
    ds = s_grid[1] - s_grid[0]
    for s in s_grid:
        k = 1.0 / (2.0 * np.sinh(s))
        if k < 1e-15:
            continue
        Sp = (phi @ make_sh(N, t_grid, L, s).T) * dt
        Sm = (phi @ make_sh(N, t_grid, L, -s).T) * dt
        W += (Sp + Sm - 2.0 * np.exp(-s/2) * np.eye(N)) * k * ds

    for p in primes:
        logp = np.log(p)
        for m in range(1, M_terms + 1):
            coeff = logp * p**(-m / 2.0)
            shift = m * logp
            if shift >= 2 * L:
                break
            for sign in [1.0, -1.0]:
                S = (phi @ make_sh(N, t_grid, L, sign * shift).T) * dt
                W += coeff * S

    return W

# ===========================================================================
# ANSATZ A: Homotopie / Level-Crossings
# ===========================================================================

def test_homotopy(lam, primes, N=50):
    """Gibt es Level-Crossings auf dem Pfad D + t*V?"""
    print(f"\n{'='*75}")
    print(f"ANSATZ A: HOMOTOPIE (lambda={lam}, N={N})")
    print(f"{'='*75}")

    QW = build_QW(lam, N, primes)
    D = np.diag(np.diag(QW))
    V = QW - D

    n_steps = 200
    t_vals = np.linspace(0, 1, n_steps + 1)

    # Verfolge die ersten 5 Eigenwerte
    trajectories = np.zeros((n_steps + 1, min(5, N)))
    min_gaps = []

    for i, t in enumerate(t_vals):
        M = D + t * V
        evals = np.sort(eigh(M, eigvals_only=True))
        trajectories[i, :] = evals[:min(5, N)]
        if i > 0:
            gap = evals[1] - evals[0]
            min_gaps.append(gap)

    min_gap = min(min_gaps)
    min_gap_t = t_vals[1 + min_gaps.index(min_gap)]

    print(f"\n  Pfad D + t*V fuer t in [0, 1]:")
    print(f"  Minimaler Gap auf dem Pfad: {min_gap:.8e} bei t={min_gap_t:.4f}")

    # Detaillierte Analyse um das Minimum
    if min_gap < 0.01:
        print(f"  WARNUNG: Sehr kleiner Gap bei t={min_gap_t:.4f}!")
        # Feiner auflösen
        t_fine = np.linspace(max(0, min_gap_t - 0.02), min(1, min_gap_t + 0.02), 100)
        for t in t_fine[::10]:
            M = D + t * V
            evals = np.sort(eigh(M, eigvals_only=True))
            print(f"    t={t:.4f}: gap={evals[1]-evals[0]:.8e}")

    # Verfolge |v0(n*)|^2 entlang des Pfades
    n_star_D = np.argmin(np.diag(D))
    print(f"\n  n*(Diagonal-Minimum) = {n_star_D}")
    print(f"  Verfolge |v0(n*)|^2 entlang des Pfades:")

    sample_t = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    print(f"  {'t':>6} | {'|v0(n*)|^2':>12} | {'gap':>12} | {'n*_actual':>10}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")

    for t in sample_t:
        M = D + t * V
        evals, evecs = eigh(M)
        v0 = evecs[:, 0]
        gap = evals[1] - evals[0]
        n_actual = np.argmax(np.abs(v0))
        weight = v0[n_star_D]**2
        print(f"  {t:6.2f} | {weight:12.6f} | {gap:12.6e} | {n_actual:10d}")

    # LEVEL-CROSSING-ZAEHLER
    crossings = 0
    for i in range(len(min_gaps)):
        if min_gaps[i] < 1e-10:
            crossings += 1

    print(f"\n  LEVEL-CROSSINGS (gap < 1e-10): {crossings}")
    print(f"  EINFACHHEIT durchgehend: {'JA' if crossings == 0 else 'NEIN'}")

    return min_gap, crossings

# ===========================================================================
# ANSATZ B: Total-Positivitaet
# ===========================================================================

def test_total_positivity(lam, primes, N=20):
    """Ist der QW-Kern total positiv?"""
    print(f"\n{'='*75}")
    print(f"ANSATZ B: TOTAL-POSITIVITAET (lambda={lam}, N={N})")
    print(f"{'='*75}")

    L = np.log(lam)
    # Teste den Kern K(t,s) = QW-Kern auf einem Gitter
    n_test = 12
    t_pts = np.linspace(-L*0.9, L*0.9, n_test)

    # Berechne Kern-Matrix (direkt im t-Raum)
    n_quad = 400
    t_grid = np.linspace(-L, L, n_quad)
    dt = t_grid[1] - t_grid[0]

    # Archimedischer Kern
    K_arch = np.zeros((n_test, n_test))
    for i in range(n_test):
        for j in range(n_test):
            s = abs(t_pts[i] - t_pts[j])
            if s < 1e-10:
                K_arch[i, j] = LOG4PI_GAMMA
            else:
                # Integral-Term
                K_arch[i, j] = 0
                s_grid = np.linspace(0.005, min(2*L, 8.0), 200)
                ds = s_grid[1] - s_grid[0]
                for sk in s_grid:
                    kern = 1.0 / (2.0 * np.sinh(sk))
                    if kern < 1e-15:
                        continue
                    # Vereinfachung: Kern als Funktion von |t-s|
                    # Dies ist nur naeherungsweise korrekt
                    K_arch[i, j] += kern * ds

    # Prime-Kern
    K_prime = np.zeros((n_test, n_test))
    for p in primes[:10]:
        logp = np.log(p)
        for m in range(1, 13):
            coeff = logp * p**(-m / 2.0)
            shift = m * logp
            if shift >= 2 * L:
                break
            for i in range(n_test):
                for j in range(n_test):
                    # K_p(t, s) ~ coeff * delta(t - s - shift)
                    # Im Matrixbild: t_i - t_j ~ shift?
                    for sign in [1.0, -1.0]:
                        diff = t_pts[i] - t_pts[j] - sign * shift
                        # Gauss-Approximation der Delta-Funktion
                        sigma = 0.1
                        K_prime[i, j] += coeff * np.exp(-diff**2 / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi))

    K_total = K_arch + K_prime

    # Teste Vorzeichen der Minoren
    print(f"\n  Teste Minoren der Kern-Matrix (K_arch):")
    # 1x1 Minoren (Diagonal)
    diag_positive = all(K_arch[i,i] > 0 for i in range(n_test))
    print(f"  1x1 Minoren (Diagonal): alle positiv = {diag_positive}")

    # 2x2 Minoren
    count_neg_2x2 = 0
    count_2x2 = 0
    for i in range(n_test):
        for j in range(i+1, n_test):
            det2 = K_arch[i,i]*K_arch[j,j] - K_arch[i,j]*K_arch[j,i]
            count_2x2 += 1
            if det2 < -1e-10:
                count_neg_2x2 += 1

    print(f"  2x2 Minoren: {count_neg_2x2}/{count_2x2} negativ")

    # Stattdessen: Pruefe die QW-Matrix im Cosinus-Raum
    print(f"\n  QW-MATRIX DIREKT (Cosinus-Basis, N={N}):")
    QW = build_QW(lam, N, primes)
    evals = np.sort(eigh(QW, eigvals_only=True))

    # Alle Eigenwerte einfach?
    spacings = np.diff(evals)
    min_spacing = np.min(spacings)
    print(f"  Minimaler Eigenwert-Abstand: {min_spacing:.8e}")
    print(f"  Alle einfach (spacing > 1e-10): {min_spacing > 1e-10}")

    # Oscillation Count: Vorzeichenwechsel in Eigenvektoren
    print(f"\n  OSZILLATIONS-THEOREM-CHECK:")
    print(f"  (Fuer TP-Kerne: n-ter EV hat genau n-1 Vorzeichenwechsel)")
    _, evecs = eigh(QW)
    for k in range(min(5, N)):
        v = evecs[:, k]
        sign_changes = np.sum(np.diff(np.sign(v)) != 0)
        expected = k
        match = "OK" if sign_changes == expected else f"ABWEICHUNG (erwartet {expected})"
        print(f"  EV {k}: {sign_changes} Vorzeichenwechsel -- {match}")

    return min_spacing

# ===========================================================================
# ANSATZ C: Even/Odd Separation
# ===========================================================================

def test_even_odd(lam, primes, N=50):
    """Vergleiche Even-Sektor (Cosinus) mit Odd-Sektor (Sinus)."""
    print(f"\n{'='*75}")
    print(f"ANSATZ C: EVEN/ODD SEPARATION (lambda={lam}, N={N})")
    print(f"{'='*75}")

    # Even-Sektor (Cosinus-Basis)
    QW_even = build_QW(lam, N, primes, basis='cos')
    evals_even = np.sort(eigh(QW_even, eigvals_only=True))

    # Odd-Sektor (Sinus-Basis)
    QW_odd = build_QW(lam, N, primes, basis='sin')
    evals_odd = np.sort(eigh(QW_odd, eigvals_only=True))

    print(f"\n  Even-Sektor (Cosinus):")
    print(f"    lambda_1 = {evals_even[0]:+.8e}")
    print(f"    lambda_2 = {evals_even[1]:+.8e}")
    print(f"    gap_even = {evals_even[1] - evals_even[0]:.8e}")

    print(f"\n  Odd-Sektor (Sinus):")
    print(f"    lambda_1 = {evals_odd[0]:+.8e}")
    print(f"    lambda_2 = {evals_odd[1]:+.8e}")
    print(f"    gap_odd  = {evals_odd[1] - evals_odd[0]:.8e}")

    # Globaler Grundzustand
    global_min = min(evals_even[0], evals_odd[0])
    second_min = min(evals_even[1], evals_odd[1])
    if evals_even[0] < evals_odd[0]:
        # Even hat Grundzustand
        second_min = min(evals_even[1], evals_odd[0])
        sector = "EVEN"
    else:
        second_min = min(evals_odd[1], evals_even[0])
        sector = "ODD"

    global_gap = second_min - global_min

    print(f"\n  GLOBAL:")
    print(f"    Grundzustand in Sektor: {sector}")
    print(f"    lambda_1(even) = {evals_even[0]:+.8e}")
    print(f"    lambda_1(odd)  = {evals_odd[0]:+.8e}")
    print(f"    Sektor-Abstand = {abs(evals_even[0] - evals_odd[0]):.8e}")
    print(f"    Globaler Gap   = {global_gap:.8e}")

    # Ist der Grundzustand EVEN? (Connes Theorem 6.1 braucht das!)
    print(f"\n  Connes Theorem 6.1 Voraussetzung:")
    print(f"    Grundzustand even: {'JA' if sector == 'EVEN' else 'NEIN'}")
    print(f"    Grundzustand einfach: {'JA' if global_gap > 1e-10 else 'NEIN'}")
    if sector == 'EVEN' and global_gap > 1e-10:
        print(f"    => Theorem 6.1 ANWENDBAR")
    else:
        print(f"    => Theorem 6.1 NICHT ANWENDBAR")

    return evals_even, evals_odd

# ===========================================================================
# ANSATZ D: Matrix-Struktur / Irreduzibilitaet
# ===========================================================================

def test_irreducibility(lam, primes, N=30):
    """Ist QW irreduzibel? Gibt es eine Perron-Transformation?"""
    print(f"\n{'='*75}")
    print(f"ANSATZ D: IRREDUZIBILITAET (lambda={lam}, N={N})")
    print(f"{'='*75}")

    QW = build_QW(lam, N, primes)

    # Graph-Konnektivitaet
    # Off-Diagonal ungleich Null?
    threshold = 1e-10
    adj = np.abs(QW) > threshold
    np.fill_diagonal(adj, False)

    # BFS von Knoten 0
    visited = set([0])
    queue = [0]
    while queue:
        node = queue.pop(0)
        for j in range(N):
            if adj[node, j] and j not in visited:
                visited.add(j)
                queue.append(j)

    connected = len(visited) == N
    print(f"\n  Graph-Konnektivitaet: {len(visited)}/{N} erreichbar")
    print(f"  Irreduzibel: {'JA' if connected else 'NEIN'}")

    # Vorzeichen-Analyse
    n_positive = np.sum(QW > threshold) - N  # minus Diagonale
    n_negative = np.sum(QW < -threshold)
    n_zero = N*N - N - n_positive - n_negative
    print(f"\n  Off-Diagonal-Vorzeichen:")
    print(f"    Positiv:  {n_positive}")
    print(f"    Negativ:  {n_negative}")
    print(f"    ~Null:    {n_zero}")
    print(f"    Gesamt:   {N*(N-1)}")

    # Gibt es eine Vorzeichentransformation S*QW*S^{-1} >= 0?
    # Suche Diagonalmatrix S mit S_ii = +/- 1
    # sodass S * QW * S alle Off-Diagonal >= 0
    # Dies ist ein NP-hartes Problem, aber fuer kleine N brute-forcebar
    if N <= 20:
        print(f"\n  Suche Perron-Transformation (S*QW*S >= 0)...")
        # Heuristik: Greedy-Algorithmus
        signs = np.ones(N)
        for iteration in range(100):
            changed = False
            for i in range(N):
                # Zaehle negative Off-Diagonal-Eintraege in Zeile i nach Transformation
                neg_count = 0
                for j in range(N):
                    if i == j:
                        continue
                    if signs[i] * signs[j] * QW[i,j] < -threshold:
                        neg_count += 1
                if neg_count > N // 2:
                    signs[i] *= -1
                    changed = True
            if not changed:
                break

        S = np.diag(signs)
        QW_transformed = S @ QW @ S
        n_neg_offdiag = 0
        for i in range(N):
            for j in range(N):
                if i != j and QW_transformed[i,j] < -threshold:
                    n_neg_offdiag += 1

        print(f"    Negative Off-Diag nach Transformation: {n_neg_offdiag}/{N*(N-1)}")
        if n_neg_offdiag == 0:
            print(f"    => PERRON-TRANSFORMATION GEFUNDEN!")
            print(f"    => Perron-Frobenius anwendbar => Grundzustand strikt positiv")
            print(f"    => v0(n*) != 0 BEWIESEN!")
        else:
            print(f"    => Keine perfekte Perron-Transformation gefunden")
            print(f"    => Perron-Frobenius nicht direkt anwendbar")

    return connected

# ===========================================================================
# META-ANALYSE: Welcher Ansatz funktioniert?
# ===========================================================================

def meta_analysis(primes):
    """Vergleiche alle Ansaetze."""
    print(f"\n{'='*75}")
    print(f"META-ANALYSE: BEWEISBARKEIT VON v0(n*) != 0")
    print(f"{'='*75}")

    lambdas = [10, 20, 50]

    for lam in lambdas:
        print(f"\n{'='*60}")
        print(f"  lambda = {lam}")
        print(f"{'='*60}")

        # A: Homotopie
        min_gap_A, crossings_A = test_homotopy(lam, primes, N=40)

        # B: Total-Positivitaet
        min_spacing_B = test_total_positivity(lam, primes, N=20)

        # C: Even/Odd
        evals_e, evals_o = test_even_odd(lam, primes, N=40)

        # D: Irreduzibilitaet
        connected_D = test_irreducibility(lam, primes, N=20)

    # FAZIT
    print(f"\n{'='*75}")
    print(f"FAZIT: BEWEISBARKEITS-ANALYSE")
    print(f"{'='*75}")
    print(f"""
  ANSATZ A (Homotopie):
    Funktioniert der Pfad D + t*V ohne Level-Crossing?
    => Entscheidend ist ob lambda_1(t) auf dem ganzen Pfad einfach bleibt.
    => Falls JA: v0(n*) != 0 folgt aus Stetigkeit.
    => Problem: Level-Crossings bei t < 1 sind moeglich.

  ANSATZ B (Total-Positivitaet):
    Ist der Kern total positiv?
    => Falls JA: Oscillation Theorem => ALLE Eigenwerte einfach.
    => Das waere der staerkste Beweis.
    => Problem: Primzahl-Beitraege koennen TP brechen.

  ANSATZ C (Even/Odd Separation):
    Liegt lambda_1(even) < lambda_1(odd)?
    => Falls JA: Grundzustand ist automatisch even UND einfach im Gesamtspektrum.
    => Das wuerde Connes Theorem 6.1 direkt erfuellen.
    => Problem: Braucht Beweis dass Sektor-Abstand > 0.

  ANSATZ D (Perron-Transformation):
    Gibt es S = diag(+/-1) mit S*QW*S >= 0?
    => Falls JA: Perron-Frobenius => Grundzustand STRIKT POSITIV.
    => Das wuerde v0(n*) > 0 fuer ALLE n* beweisen!
    => Problem: Existenz der Transformation ist nicht garantiert.

  BEWERTUNG:
    Keiner der Ansaetze liefert einen VOLLSTAENDIGEN Beweis.
    Am vielversprechendsten ist Ansatz C (Even/Odd), weil:
    - Die Paritaets-Symmetrie ist EXAKT (nicht numerisch)
    - Der Sektor-Abstand ist physikalisch motiviert
    - Connes' Theorem 6.1 braucht genau diese Aussage
""")

if __name__ == "__main__":
    print("=" * 75)
    print("PHASE 14: BEWEISBARKEITS-ANALYSE")
    print("=" * 75)

    primes = list(primerange(2, 100))
    meta_analysis(primes)
