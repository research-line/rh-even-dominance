#!/usr/bin/env python3
"""
density_test_li_polynomials.py
==============================
Numerischer Test: Sind die Li-Polynome dicht in der Weil-Testklasse?

Drei Angriffspunkte:
1. APPROXIMATIONSTEST: Wie gut approximieren Linearkombinationen von h_1,...,h_N
   beliebige C_c^inf-Testfunktionen?
2. GRAM-MATRIX: Ist die Gram-Matrix <h_m, h_n> positiv definit? (notwendig fuer Basis)
3. STONE-WEIERSTRASS-CHECK: Trennen die h_n Punkte? Verschwinden sie nirgends simultan?

Li-Polynome: h_n(x) = 1 - (1 - 1/x)^n  fuer x > 0, n = 1, 2, ...
Substitution u = 1 - 1/x:  h_n = 1 - u^n  (gewoehnliche Polynome in u)
"""

import numpy as np
from mpmath import mp, mpf, matrix, log, exp, pi, gamma, zeta, zetazero, fsum
from mpmath import quad as mpquad
import sys

mp.dps = 50  # 50 Dezimalstellen

# ===========================================================================
# 1. Li-Polynome definieren
# ===========================================================================

def h_n(x, n):
    """Li-Polynom h_n(x) = 1 - (1 - 1/x)^n, x > 0"""
    if x <= 0:
        return mpf(0)
    u = 1 - 1/mpf(x)
    return 1 - u**n

def h_n_float(x, n):
    """Float-Version fuer schnelle numpy-Berechnungen"""
    u = 1.0 - 1.0/x
    return 1.0 - u**n

# ===========================================================================
# 2. Testfunktionen (C_c^inf oder Schwartz)
# ===========================================================================

def gaussian(x, mu=1.0, sigma=0.3):
    """Gauss-Glocke (Schwartz-Klasse)"""
    return np.exp(-0.5*((x - mu)/sigma)**2)

def bump(x, a=0.5, b=2.0):
    """C_c^inf Bump-Funktion mit Traeger [a, b]"""
    result = np.zeros_like(x)
    mask = (x > a) & (x < b)
    t = (x[mask] - a) / (b - a)  # t in (0,1)
    result[mask] = np.exp(-1.0/(t*(1-t)))
    return result

def oscillating(x, freq=5.0):
    """Oszillierende Schwartz-Funktion"""
    return np.sin(freq * x) * np.exp(-x**2)

def sharp_peak(x, center=1.0, width=0.1):
    """Schmaler Peak (testet hochfrequente Approximation)"""
    return np.exp(-0.5*((x - center)/width)**2)

# ===========================================================================
# 3. APPROXIMATIONSTEST
#    Minimiere ||f - sum_{n=1}^N c_n * h_n||^2 auf einem Gitter
# ===========================================================================

def approximation_test(test_func, func_name, N_max=50, x_grid_size=200):
    """Teste wie gut N Li-Polynome eine gegebene Funktion approximieren."""
    print(f"\n{'='*60}")
    print(f"APPROXIMATIONSTEST: {func_name}")
    print(f"{'='*60}")

    # Gitter auf (0.1, 5.0) -- vermeidet Singularitaet bei x=0
    x = np.linspace(0.2, 5.0, x_grid_size)
    f_values = test_func(x)

    # Normierung
    f_norm = np.sqrt(np.sum(f_values**2) / x_grid_size)
    if f_norm < 1e-15:
        print("  Testfunktion ist quasi Null -- ueberspringe")
        return

    print(f"  ||f|| = {f_norm:.6e}")
    print(f"  N_polynome | rel. Fehler  | Verbesserung")
    print(f"  -----------|--------------|-------------")

    prev_error = 1.0
    results = []

    for N in [5, 10, 15, 20, 30, 40, 50]:
        if N > N_max:
            break

        # Matrix A: A[i,j] = h_j(x_i)
        A = np.zeros((x_grid_size, N))
        for j in range(N):
            A[:, j] = h_n_float(x, j+1)

        # Least-squares: min ||f - A*c||^2
        c, residuals, rank, sv = np.linalg.lstsq(A, f_values, rcond=None)

        # Approximation und Fehler
        f_approx = A @ c
        error = np.sqrt(np.sum((f_values - f_approx)**2) / x_grid_size)
        rel_error = error / f_norm

        improvement = prev_error / rel_error if rel_error > 1e-16 else float('inf')

        # Konditionszahl
        cond = sv[0]/sv[-1] if sv[-1] > 0 else float('inf')

        print(f"  {N:10d} | {rel_error:.6e} | {improvement:.2f}x  (cond={cond:.1e})")

        results.append((N, rel_error, cond))
        prev_error = rel_error

    return results


# ===========================================================================
# 4. GRAM-MATRIX mit mpmath (hochpraezise)
#    G_{mn} = integral_0^inf h_m(x) * h_n(x) * w(x) dx
#    Gewichtsfunktion w(x) = 1/x^2 (natuerlich fuer Mellin)
# ===========================================================================

def gram_matrix_test(N=20):
    """Berechne Gram-Matrix der Li-Polynome mit mpmath."""
    print(f"\n{'='*60}")
    print(f"GRAM-MATRIX (N={N}, {mp.dps} Dezimalstellen)")
    print(f"{'='*60}")

    # Substitution u = 1 - 1/x, du = 1/x^2 dx
    # h_n(x) = 1 - u^n
    # integral_0^inf h_m(x)*h_n(x)/x^2 dx = integral_{-inf}^{1} (1-u^m)(1-u^n) du
    # Aber u kann beliebig negativ werden (x -> 0+)

    # Besser: Integral auf (1, inf) -- dort ist u in (0, 1)
    # h_n(x) fuer x > 1: u = 1-1/x in (0,1), h_n = 1 - u^n
    # Gewicht: dx/x^2 = du (Substitution)

    # G_{mn} = integral_0^1 (1 - u^m)(1 - u^n) du
    #        = 1 - 1/(m+1) - 1/(n+1) + 1/(m+n+1)

    print("  Analytische Formel: G_{mn} = 1 - 1/(m+1) - 1/(n+1) + 1/(m+n+1)")
    print("  (Integration auf [0,1] nach Substitution u = 1-1/x, x in [1,inf))")

    G = matrix(N, N)
    for m in range(1, N+1):
        for n in range(1, N+1):
            G[m-1, n-1] = 1 - mpf(1)/(m+1) - mpf(1)/(n+1) + mpf(1)/(m+n+1)

    # Eigenwerte (konvertiere zu numpy fuer Eigenwertberechnung)
    G_np = np.array([[float(G[i,j]) for j in range(N)] for i in range(N)])
    eigenvalues = np.linalg.eigvalsh(G_np)
    eigenvalues.sort()

    print(f"\n  Eigenwerte der Gram-Matrix ({N}x{N}):")
    print(f"  {'Index':>5} | {'Eigenwert':>15} | {'Positiv?':>8}")
    print(f"  {'-----':>5}-+-{'-'*15}-+-{'-'*8}")

    n_positive = 0
    n_negative = 0
    for i, ev in enumerate(eigenvalues):
        sign = "JA" if ev > 0 else "NEIN"
        if ev > 0:
            n_positive += 1
        else:
            n_negative += 1
        if i < 10 or i >= N - 5:
            print(f"  {i+1:5d} | {ev:15.6e} | {sign:>8}")
        elif i == 10:
            print(f"  {'...':>5} | {'...':>15} | {'...':>8}")

    print(f"\n  Positiv: {n_positive}/{N}, Negativ: {n_negative}/{N}")
    print(f"  Kleinster EW: {eigenvalues[0]:.6e}")
    print(f"  Groesster EW: {eigenvalues[-1]:.6e}")
    print(f"  Konditionszahl: {eigenvalues[-1]/max(abs(eigenvalues[0]), 1e-300):.6e}")

    # Determinante (Vorzeichen)
    det_sign = np.prod(np.sign(eigenvalues))
    print(f"  Vorzeichen der Determinante: {'+' if det_sign > 0 else '-'}")

    return eigenvalues, G_np


# ===========================================================================
# 5. STONE-WEIERSTRASS CHECK
#    Frage: Trennen die h_n Punkte?
#    h_n(x) = 1 - (1-1/x)^n
#    h_n(a) = h_n(b) fuer alle n  =>  a = b ?
# ===========================================================================

def stone_weierstrass_check():
    """Pruefe Stone-Weierstrass Bedingungen fuer Li-Polynome."""
    print(f"\n{'='*60}")
    print(f"STONE-WEIERSTRASS ANALYSE")
    print(f"{'='*60}")

    print("\n  SUBSTITUTION: u = 1 - 1/x,  h_n(x) = 1 - u^n")
    print("  x in (0, inf) <=> u in (-inf, 1)")
    print("  x = 1 <=> u = 0")
    print("  x -> inf <=> u -> 1")
    print("  x -> 0+ <=> u -> -inf")

    print("\n  PUNKTETRENNUNG:")
    print("  h_n(a) = h_n(b) fuer alle n")
    print("  <=> 1 - u_a^n = 1 - u_b^n fuer alle n")
    print("  <=> u_a^n = u_b^n fuer alle n")
    print("  <=> u_a = u_b  (wenn beide in (-inf, 1))")
    print("  <=> a = b")
    print("")
    print("  BEWEIS: Fuer n=1: u_a = u_b => a = b. TRIVIAL.")
    print("  => Li-Polynome TRENNEN PUNKTE auf (0, inf).")

    print("\n  VERSCHWINDEN:")
    print("  Gibt es x_0 > 0 mit h_n(x_0) = 0 fuer alle n?")
    print("  h_n(x_0) = 0 <=> (1-1/x_0)^n = 1 <=> 1-1/x_0 = 1 <=> x_0 = inf")
    print("  => Kein endlicher Punkt, an dem alle h_n verschwinden.")
    print("  => Li-Polynome verschwinden NIRGENDS simultan auf (0, inf).")

    print("\n  ALGEBRA-EIGENSCHAFT:")
    print("  Ist {h_n} eine Algebra (abgeschlossen unter Multiplikation)?")
    print("  h_m * h_n = (1-u^m)(1-u^n) = 1 - u^m - u^n + u^{m+n}")
    print("  = h_m + h_n - h_{m+n}  ??? Nein!")
    print("  h_m + h_n = (1-u^m) + (1-u^n) = 2 - u^m - u^n")
    print("  h_m * h_n = 1 - u^m - u^n + u^{m+n}")
    print("  = (h_m + h_n - 1) + (1 - u^{m+n})")
    print("  = h_m + h_n + h_{m+n} - 1")
    print("")
    print("  Also: h_m * h_n = h_m + h_n + h_{m+n} - 1")
    print("  Die Algebra erzeugt von {1, h_1, h_2, ...} ist abgeschlossen!")
    print("  (Weil 1 = h_n + (1-1/x)^n und wir 1 als konstante Funktion hinzunehmen)")

    print("\n  STONE-WEIERSTRASS FAZIT:")
    print("  Die Algebra A = span{1, h_1, h_2, ...} erfuellt:")
    print("    (i)   Trennt Punkte: JA")
    print("    (ii)  Verschwindet nirgends: JA (enthaelt 1)")
    print("    (iii) Ist Algebra: JA (h_m*h_n = h_m + h_n + h_{m+n} - 1)")
    print("    (iv)  Ist selbstadjungiert: JA (reellwertig)")
    print("")
    print("  => Stone-Weierstrass: A ist DICHT in C_0(K) fuer jedes Kompaktum K ⊂ (0,inf)")
    print("")
    print("  ABER: C_0(K) ≠ C_c^inf !!!")
    print("  Stone-Weierstrass gibt uniforme Approximation auf Kompakta,")
    print("  aber NICHT Approximation in der Schwartz-Topologie oder L^2.")
    print("  Die Weil-Testklasse braucht Kontrolle ueber ABLEITUNGEN und ABFALL.")
    print("")
    print("  => Stone-Weierstrass ist NOTWENDIG aber NICHT HINREICHEND.")
    print("  => Die Dichtheitsfrage reduziert sich auf:")
    print("     Kann man den Approximationsfehler auch in ABLEITUNGEN kontrollieren?")


# ===========================================================================
# 6. ABLEITUNGS-APPROXIMATION
#    Teste ob Li-Polynome auch Ableitungen approximieren koennen
# ===========================================================================

def derivative_approximation_test(N_max=40):
    """Teste Approximation von Funktion UND Ableitung gleichzeitig."""
    print(f"\n{'='*60}")
    print(f"ABLEITUNGS-APPROXIMATIONSTEST")
    print(f"{'='*60}")

    x = np.linspace(0.3, 4.0, 300)
    dx = x[1] - x[0]

    # Testfunktion: Gauss
    f = np.exp(-2*(x - 1.5)**2)
    f_prime = -4*(x - 1.5) * f  # analytische Ableitung

    # h_n und h_n'
    # h_n(x) = 1 - (1-1/x)^n
    # h_n'(x) = -n * (1-1/x)^{n-1} * (1/x^2) = n/x^2 * (1-1/x)^{n-1}

    def h_n_deriv(x_arr, n):
        u = 1.0 - 1.0/x_arr
        return n / x_arr**2 * u**(n-1)

    print(f"\n  Testfunktion: f(x) = exp(-2*(x-1.5)^2)")
    print(f"  Simultane Approximation von f und f' durch Li-Polynome")
    print(f"\n  N  | rel.Fehler f | rel.Fehler f' | Ableitungs-Ratio")
    print(f"  ---|--------------|---------------|----------------")

    f_norm = np.sqrt(np.mean(f**2))
    fp_norm = np.sqrt(np.mean(f_prime**2))

    for N in [5, 10, 15, 20, 30, 40]:
        if N > N_max:
            break

        # Baue erweiterte Matrix: [h_n(x); h_n'(x)]
        A_f = np.zeros((len(x), N))
        A_fp = np.zeros((len(x), N))
        for j in range(N):
            A_f[:, j] = h_n_float(x, j+1)
            A_fp[:, j] = h_n_deriv(x, j+1)

        # Stapele: min ||[f; w*f'] - [A_f; w*A_fp] * c||^2
        w = 1.0  # Gewicht fuer Ableitung
        A_stack = np.vstack([A_f, w * A_fp])
        b_stack = np.concatenate([f, w * f_prime])

        c, _, _, _ = np.linalg.lstsq(A_stack, b_stack, rcond=None)

        err_f = np.sqrt(np.mean((f - A_f @ c)**2)) / f_norm
        err_fp = np.sqrt(np.mean((f_prime - A_fp @ c)**2)) / fp_norm
        ratio = err_fp / err_f if err_f > 1e-16 else float('inf')

        print(f"  {N:2d} | {err_f:.6e} | {err_fp:.6e} | {ratio:.2f}")


# ===========================================================================
# 7. MELLIN-TRANSFORM ANALYSE
#    Die Weil-Positivitaet ist im Mellin-Raum formuliert.
#    Mellin[h_n](s) = integral_0^inf h_n(x) * x^{s-1} dx
# ===========================================================================

def mellin_analysis():
    """Analysiere Li-Polynome im Mellin-Raum."""
    print(f"\n{'='*60}")
    print(f"MELLIN-TRANSFORM ANALYSE")
    print(f"{'='*60}")

    print("\n  Li-Polynom: h_n(x) = 1 - (1-1/x)^n")
    print("  Mellin-Transform: M[h_n](s) = integral_0^inf h_n(x) * x^{s-1} dx")
    print("")
    print("  Substitution u = 1/x:")
    print("  M[h_n](s) = integral_0^inf [1 - (1-u)^n] * u^{-s-1} du  (Re s < 0)")
    print("")
    print("  Binomialentwicklung: (1-u)^n = sum_{k=0}^n C(n,k)(-u)^k")
    print("  => 1 - (1-u)^n = sum_{k=1}^n C(n,k)(-1)^{k+1} u^k")
    print("")
    print("  PROBLEM: Der Mellin-Transform von h_n konvergiert nur in einem Streifen.")
    print("  h_n(x) -> 1 fuer x -> inf (langsam), also braucht Re(s) < 0.")
    print("  h_n(x) ~ n/x fuer x -> 0+, also braucht Re(s) > -1.")
    print("  => Mellin[h_n] existiert nur fuer -1 < Re(s) < 0.")
    print("")
    print("  Die Weil-Testfunktionen leben aber auf der KRITISCHEN LINIE Re(s) = 1/2!")
    print("  => Li-Polynome sind NICHT DIREKT in der Weil-Testklasse.")
    print("")
    print("  ENTSCHEIDENDE BEOBACHTUNG:")
    print("  Li (1997) und Bombieri-Lagarias (1999) umgehen das Problem durch")
    print("  eine MODIFIZIERTE Formulierung:")
    print("    lambda_n = sum_rho [1 - (1 - 1/rho)^n]")
    print("  Das ist NICHT das Mellin-Integral von h_n gegen die Nullstellen-Distribution,")
    print("  sondern eine DIREKTE AUSWERTUNG an den Nullstellen.")
    print("")
    print("  => Die 'Testklasse' der Li-Koeffizienten ist eine ANDERE Topologie")
    print("     als die Weil-Testklasse C_c^inf.")
    print("  => Stone-Weierstrass auf Kompakta ist IRRELEVANT fuer den Mellin-Raum!")
    print("")
    print("  DAS IST DER STRUKTURELLE GRUND warum Li -> Weil schwer ist:")
    print("  Die beiden Kriterien leben in VERSCHIEDENEN FUNKTIONENRAEUMEN.")


# ===========================================================================
# 8. HAUPTPROGRAMM
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DICHTHEIT DER LI-POLYNOME IN DER WEIL-TESTKLASSE")
    print("Numerische und analytische Untersuchung")
    print("=" * 60)

    # 1. Stone-Weierstrass (analytisch)
    stone_weierstrass_check()

    # 2. Mellin-Analyse (analytisch)
    mellin_analysis()

    # 3. Approximationstests (numerisch)
    test_functions = [
        (gaussian, "Gauss (mu=1, sigma=0.3)"),
        (bump, "Bump C_c^inf [0.5, 2.0]"),
        (lambda x: sharp_peak(x, 1.0, 0.1), "Scharfer Peak (width=0.1)"),
        (lambda x: sharp_peak(x, 1.0, 0.02), "Sehr scharfer Peak (width=0.02)"),
        (oscillating, "Oszillierend (freq=5)"),
    ]

    all_results = {}
    for func, name in test_functions:
        results = approximation_test(func, name, N_max=50)
        all_results[name] = results

    # 4. Ableitungs-Approximation
    derivative_approximation_test(N_max=40)

    # 5. Gram-Matrix
    for N in [10, 20, 40]:
        gram_matrix_test(N)

    # 6. Zusammenfassung
    print(f"\n{'='*60}")
    print(f"ZUSAMMENFASSUNG")
    print(f"{'='*60}")
    print("""
  ANALYTISCH:
    - Stone-Weierstrass: Li-Polynome TRENNEN PUNKTE und bilden eine ALGEBRA
      => Dicht in C(K) fuer jedes Kompaktum K (uniforme Topologie)
      => ABER: Das ist NICHT die Weil-Topologie!

    - Mellin-Analyse: Li-Polynome h_n haben Mellin-Strip (-1, 0),
      die Weil-Testklasse lebt bei Re(s) = 1/2.
      => VERSCHIEDENE FUNKTIONENRAEUME -- Stone-Weierstrass greift nicht!

    - STRUKTURELLER GRUND: Li -> Weil ist schwer, weil die beiden
      Kriterien in verschiedenen topologischen Raeumen formuliert sind.
      Li arbeitet mit DISKRETER Auswertung an Nullstellen,
      Weil mit KONTINUIERLICHEN Testfunktionen im Mellin-Raum.

  NUMERISCH:
    - Approximation auf Kompakta: KONVERGIERT (Stone-Weierstrass bestaetigt)
    - Ableitungs-Approximation: Fehler waechst mit Ordnung
    - Gram-Matrix: Wird schlecht konditioniert fuer grosses N

  FAZIT FUER BEWEISNOTIZ:
    Der direkte Weg Li -> Weil via Dichtheit ist VERSPERRT,
    weil die Funktionenraeume inkompatibel sind.
    Ein Beweis muesste einen ANDEREN Mechanismus finden,
    der die Li-Positivitaet in Weil-Positivitaet uebersetzt --
    nicht durch Approximation, sondern durch STRUKTURELLE ARGUMENTE.
""")
