#!/usr/bin/env python3
"""
weg1_raumdefinition.py
======================
Weg 1: Exakte Raumdefinition fuer den erweiterten Weil-Bereich

Ziel: Eine praezise Aussage der Form:
  "Der zulaessige Testfunktionsraum W_ext enthaelt die Li-Polynome h_n,
   und W_ext ist dicht in W_Weil bezueglich Topologie tau."

=== MATHEMATISCHER HINTERGRUND ===

1. WEIL'S ORIGINALE TESTKLASSE (1952):
   W_Weil = { h : R_{>0} -> R | h gerade Schwartz-Funktion,
              deren Fourier-Transform h_hat(t) geeigneten Abfall hat }

   Genauer (Guinand-Weil explizite Formel):
   h muss erfuellen:
   (a) h ist gerade: h(-x) = h(x)
   (b) h ist Schwartz: h in S(R)
   (c) Die Mellin-Transformierte Mh(s) = integral_0^inf h(x) x^{s-1} dx
       konvergiert absolut auf dem kritischen Streifen 0 < Re(s) < 1

   Die Weil-Positivitaet lautet:
   W(h) = h_hat(0) + h_hat(1) - sum_rho h_hat(rho)
        - integral_0^inf [h(x) + h(1/x)] * x^{-1} * [Primzahl-Terme] dx
        >= 0 fuer alle h in W_Weil

2. BOMBIERI-LAGARIAS ERWEITERUNG (1999):
   Sie zeigen: Die Guinand-Weil-Formel gilt fuer eine GROESSERE Klasse:

   W_ext = { h : R_{>0} -> C | h beschraenkt,
             Mh(s) analytisch auf 0 < Re(s) < 1,
             |Mh(s)| = O(|Im(s)|^{-2-epsilon}) fuer |Im(s)| -> inf,
             geeignete Regularitaet }

   Die Li-Polynome h_n(x) = 1 - (1-1/x)^n liegen in W_ext:
   - Mh_n(s) hat Pole bei s = 0 und s = -1, ..., -n+1 (NICHT im Streifen!)
   - |Mh_n(s)| = O(|Im(s)|^{-1}) auf dem kritischen Streifen
   - Die Formel lambda_n = W(h_n) gilt (Bombieri-Lagarias, Theorem 1)

   ABER: h_n sind NICHT in W_Weil (nicht Schwartz, nicht kompakt getragen)

3. DIE TOPOLOGIE-FRAGE:
   Welche Topologie auf W_ext macht den Weil-Funktional W stetig?

   Kandidaten:
   (a) Mellin-L2: ||h||^2 = integral_Re(s)=1/2 |Mh(s)|^2 ds
       Problem: h_n sind nicht L2 (Mellin-Transformierte divergiert)

   (b) Gewichtete Mellin-L2: ||h||^2_w = integral |Mh(s)|^2 w(s) ds
       mit schnell fallendem Gewicht w(s) -> 0 fuer |Im(s)| -> inf
       Vorteil: Kann h_n einschliessen, wenn w schnell genug faellt

   (c) Schwache Topologie: h_n -> h genau wenn W(phi * h_n) -> W(phi * h)
       fuer alle phi in W_Weil
       Vorteil: Direkt an Positivitaets-Transfer angepasst

   (d) Distributionen-Dualitaet: W_ext als Teilmenge von S'(R)
       Die h_n als temperierte Distributionen
       Vorteil: Natuerlicher Rahmen, Fourier/Mellin als Isomorphismus

   (e) Bombieri's Q_W(t)-Topologie: Parametrisiert durch Traegergroesse t
       Q_W(t) positiv definit fuer kleine t -> Approximation von innen

=== NUMERISCHER TEST ===

Teste: In welcher Topologie konvergieren Linearkombinationen der h_n
gegen typische Weil-Testfunktionen?
"""

import numpy as np
from mpmath import mp, mpf, mpc, gamma, zeta, zetazero, log, exp, pi, im, re
from mpmath import quad as mpquad

mp.dps = 25

# ===========================================================================
# 1. Mellin-Transformierte der Li-Polynome (analytisch)
# ===========================================================================

def mellin_h_n(s, n):
    """
    Mellin-Transformierte von h_n(x) = 1 - (1-1/x)^n

    M[h_n](s) = integral_0^inf [1 - (1-1/x)^n] x^{s-1} dx

    Fuer Re(s) in (-1, 0):
    M[h_n](s) = sum_{k=1}^n C(n,k) (-1)^{k+1} / (s+k)
              = n * B(s+1, n) / s   (via Beta-Funktion)

    Wobei B(a,b) = Gamma(a)*Gamma(b)/Gamma(a+b)
    """
    # Direkte Berechnung via Partialbruchzerlegung
    s = mpc(s)
    result = mpc(0)
    for k in range(1, n+1):
        binom = 1
        for j in range(k):
            binom = binom * (n - j) / (j + 1)
        result += binom * (-1)**(k+1) / (s + k)
    return result


def test_mellin_strip():
    """Teste wo die Mellin-Transformierte der h_n lebt."""
    print("=" * 60)
    print("MELLIN-STRIP DER LI-POLYNOME")
    print("=" * 60)

    print("\n  M[h_n](s) = sum_{k=1}^n C(n,k)(-1)^{k+1} / (s+k)")
    print("  Pole bei s = -1, -2, ..., -n (alle links von Re(s) = 0)")
    print("  Holomorph auf Re(s) > -1")
    print("  Abfall: |M[h_n](1/2+it)| fuer grosses |t|:")
    print()

    for n in [1, 2, 5, 10, 20]:
        vals = []
        for t in [10, 50, 100, 500, 1000]:
            s = mpc(0.5, t)
            M = mellin_h_n(s, n)
            vals.append((t, abs(float(abs(M)))))

        # Schaetze Abfall-Exponent
        if len(vals) >= 2:
            t1, v1 = vals[1]
            t2, v2 = vals[-1]
            if v1 > 0 and v2 > 0:
                exponent = np.log(v2/v1) / np.log(t2/t1)
            else:
                exponent = 0

        print(f"  n={n:3d}: |M(1/2+it)| ~ |t|^{exponent:.2f}")
        print(f"         t=10: {vals[0][1]:.4e}, t=100: {vals[2][1]:.4e}, t=1000: {vals[4][1]:.4e}")


# ===========================================================================
# 2. Mellin-Transformierte von Schwartz-Testfunktionen
# ===========================================================================

def mellin_gaussian_approx(s, mu=1.0, sigma=0.3):
    """
    Approximative Mellin-Transformierte von f(x) = exp(-(x-mu)^2/(2*sigma^2))

    Exakt: M[f](s) = integral_0^inf f(x) x^{s-1} dx
    Konvergiert fuer alle s mit Re(s) > 0 (f schnell fallend bei inf, O(1) bei 0)
    """
    s = mpc(s)
    # Numerische Integration
    def integrand(x):
        return exp(-(x-mu)**2 / (2*sigma**2)) * x**(s-1)

    result = mpquad(integrand, [mpf('0.01'), mpf('5.0')])
    return result


# ===========================================================================
# 3. Gewichtete Mellin-L2 Norm: ||h||^2_w = int |Mh(s)|^2 w(s) ds
# ===========================================================================

def weighted_mellin_L2(n, weight_type="gaussian", weight_param=10.0):
    """
    Berechne ||h_n||^2_w = integral_{-inf}^{inf} |M[h_n](1/2+it)|^2 w(t) dt

    Gewichte:
    - gaussian: w(t) = exp(-t^2/alpha^2)
    - polynomial: w(t) = 1/(1+t^2)^p
    """
    def integrand(t):
        s = mpc(0.5, t)
        M = mellin_h_n(s, n)
        M_abs2 = float(re(M * M.conjugate()))

        if weight_type == "gaussian":
            w = float(exp(-t**2 / weight_param**2))
        elif weight_type == "polynomial":
            w = 1.0 / (1 + float(t)**2)**weight_param
        else:
            w = 1.0

        return M_abs2 * w

    result = mpquad(integrand, [-100, 100])
    return float(result)


def test_weighted_norms():
    """Teste ob h_n endliche gewichtete Mellin-L2 Normen haben."""
    print(f"\n{'='*60}")
    print(f"GEWICHTETE MELLIN-L2 NORMEN")
    print(f"{'='*60}")

    print(f"\n  ||h_n||^2_w = integral |M[h_n](1/2+it)|^2 w(t) dt")

    for weight_type, weight_param, label in [
        ("gaussian", 10.0, "w(t)=exp(-t^2/100)"),
        ("gaussian", 50.0, "w(t)=exp(-t^2/2500)"),
        ("polynomial", 1.0, "w(t)=1/(1+t^2)"),
        ("polynomial", 2.0, "w(t)=1/(1+t^2)^2"),
    ]:
        print(f"\n  Gewicht: {label}")
        print(f"  {'n':>5} | {'||h_n||^2_w':>15} | {'endlich?':>8}")
        print(f"  {'-'*5}-+-{'-'*15}-+-{'-'*8}")

        for n in [1, 2, 5, 10, 20]:
            try:
                norm2 = weighted_mellin_L2(n, weight_type, weight_param)
                finite = "JA" if norm2 < 1e10 else "NEIN"
                print(f"  {n:5d} | {norm2:15.6e} | {finite:>8}")
            except Exception as e:
                print(f"  {n:5d} | {'ERROR':>15} | {str(e)[:20]}")


# ===========================================================================
# 4. APPROXIMATION IM MELLIN-RAUM
#    Teste: Koennen Linearkombinationen von M[h_n] die Mellin-Transformierte
#    einer Schwartz-Funktion approximieren?
# ===========================================================================

def mellin_approximation_test():
    """Approximiere M[gaussian] durch Linearkombinationen von M[h_n]."""
    print(f"\n{'='*60}")
    print(f"APPROXIMATION IM MELLIN-RAUM")
    print(f"{'='*60}")

    # Gitter auf der kritischen Linie
    T_max = 50.0
    N_grid = 100
    t_grid = np.linspace(-T_max, T_max, N_grid)

    # Ziel: Mellin-Transformierte einer Gauss-Funktion
    print(f"  Ziel: M[gaussian](1/2+it) auf [-{T_max}, {T_max}]")

    target = np.zeros(N_grid, dtype=complex)
    for i, t in enumerate(t_grid):
        target[i] = complex(mellin_gaussian_approx(mpc(0.5, t)))

    target_norm = np.sqrt(np.sum(np.abs(target)**2) / N_grid)
    print(f"  ||Ziel|| = {target_norm:.6e}")

    # Approximation durch h_1, ..., h_N
    for N in [5, 10, 20, 30]:
        A = np.zeros((N_grid, N), dtype=complex)
        for j in range(N):
            for i, t in enumerate(t_grid):
                A[i, j] = complex(mellin_h_n(mpc(0.5, t), j+1))

        # Least squares
        c, residuals, rank, sv = np.linalg.lstsq(A, target, rcond=None)

        approx = A @ c
        error = np.sqrt(np.sum(np.abs(target - approx)**2) / N_grid)
        rel_error = error / target_norm

        cond = sv[0] / sv[-1] if sv[-1] > 0 else float('inf')

        print(f"  N={N:3d}: rel. Fehler = {rel_error:.6e}, cond = {cond:.2e}")


# ===========================================================================
# 5. DIE ENTSCHEIDENDE FRAGE: Topologie des erweiterten Bereichs
# ===========================================================================

def topologie_analyse():
    """Analysiere welche Topologie fuer den erweiterten Weil-Bereich geeignet ist."""
    print(f"\n{'='*60}")
    print(f"TOPOLOGIE-ANALYSE FUER DEN ERWEITERTEN WEIL-BEREICH")
    print(f"{'='*60}")

    print("""
  === DEFINITION DES ERWEITERTEN WEIL-BEREICHS W_ext ===

  Nach Bombieri-Lagarias (1999), Theorem 1:
  Die Guinand-Weil-Formel gilt fuer Testfunktionen h mit:

  (BL1) Mh(s) analytisch auf dem offenen Streifen 0 < Re(s) < 1
  (BL2) Mh(s) stetig auf dem abgeschlossenen Streifen 0 <= Re(s) <= 1
  (BL3) |Mh(s)| = O(|Im(s)|^{-2-epsilon}) fuer |Im(s)| -> inf
        gleichmaessig im Streifen
  (BL4) h ist beschraenkt und stueckweise stetig

  Die Li-Polynome erfuellen (BL1-BL4):
  - Mh_n(s) = sum C(n,k)(-1)^{k+1}/(s+k) -- analytisch auf Re(s) > -1
  - |Mh_n(1/2+it)| = O(|t|^{-1}) -- erfuellt (BL3) mit epsilon > 0
  - h_n beschraenkt auf (0, inf) -- erfuellt (BL4)

  === NATUERLICHE TOPOLOGIE AUF W_ext ===

  Die naheliegendste Topologie ist die MELLIN-BILD-TOPOLOGIE:

  Definiere die Halbnormen:
    p_k(h) = sup_{0 <= Re(s) <= 1} |Mh(s)| * (1 + |Im(s)|)^{2+k}

  fuer k = 0, 1, 2, ...

  Das ergibt eine Familie von Halbnormen, die W_ext zu einem
  Frechet-Raum machen (metrisierbar, vollstaendig).

  In DIESER Topologie ist W_Weil DICHT in W_ext (trivial, weil
  W_Weil die strengere Bedingung "h Schwartz" hat, und Schwartz-
  Funktionen im Mellin-Bild ueberall schnell fallen).

  ABER: Wir wollen die UMKEHRUNG -- h_n approximiert W_Weil-Funktionen.
  Die Frage ist: Ist {h_n} dicht in W_ext bezueglich dieser Topologie?

  === ABFALL-RATE ALS SCHLUESSEL ===

  Schwartz-Funktionen: |Mh(1/2+it)| = O(|t|^{-N}) fuer ALLE N
  Li-Polynome:         |Mh_n(1/2+it)| = O(|t|^{-1})  (festes n)

  LINEARKOMBINATION: sum c_n h_n hat Abfall O(|t|^{-1})
  (der langsamste Term dominiert)

  => Linearkombinationen der h_n koennen NUR Funktionen mit
     Mellin-Abfall O(|t|^{-1}) approximieren.
  => Schwartz-Funktionen (Abfall O(|t|^{-N})) liegen NICHT im
     Abschluss von span{h_n} in der Frechet-Topologie!

  ABER: Unendliche Reihen sum_{n=1}^inf c_n h_n koennen besseren
  Abfall haben, weil sich Pole GEGENSEITIG AUFHEBEN koennen.

  Mh_n(s) = sum_{k=1}^n C(n,k)(-1)^{k+1}/(s+k)

  Eine geschickte Wahl der c_n koennte die 1/(s+k)-Terme so
  kombinieren, dass die Summe schnell faellt.

  DAS IST DIE MATHEMATISCHE KERNFRAGE:

  Kann man durch geeignete Wahl von {c_n}_{n=1}^inf die Reihe
    sum_{n=1}^inf c_n * Mh_n(s)
  so konstruieren, dass sie fuer |Im(s)| -> inf beliebig schnell faellt?

  In Partialbruch-Sprache:
    Mh_n(s) = sum_{k=1}^n a_{nk} / (s+k)

  sum_n c_n * Mh_n(s) = sum_k [ sum_n c_n * a_{nk} ] / (s+k)
                       = sum_k b_k / (s+k)

  wobei b_k = sum_{n>=k} c_n * C(n,k) * (-1)^{k+1}

  Schneller Abfall von sum_k b_k/(s+k) fuer |Im(s)| -> inf
  erfordert: b_k -> 0 SEHR SCHNELL fuer k -> inf.

  Das ist ein MOMENTENPROBLEM: Gibt es {c_n} mit
    sum_{n>=k} c_n * C(n,k) * (-1)^{k+1} = b_k
  wobei |b_k| <= C * r^k fuer ein r < 1?

  (Exponentieller Abfall der b_k => super-polynomial Abfall im Mellin-Bild)
""")


# ===========================================================================
# 6. NUMERISCHER TEST: Partialbruch-Cancellation
# ===========================================================================

def partial_fraction_cancellation_test():
    """Teste ob Linearkombinationen der h_n besseren Mellin-Abfall haben koennen."""
    print(f"\n{'='*60}")
    print(f"PARTIALBRUCH-CANCELLATION TEST")
    print(f"{'='*60}")

    print(f"\n  Teste: Kann sum c_n * Mh_n(s) schneller als O(|t|^-1) fallen?")
    print(f"  Methode: Optimiere c_n um |sum c_n Mh_n(1/2+it)| fuer grosses t zu minimieren")

    N_max = 30
    T_large = 100.0  # Teste bei grossen t

    # Baue die Matrix: A[i,j] = Mh_{j+1}(1/2 + i*t_i) fuer t_i gross
    t_test = [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 120.0, 150.0, 200.0, 300.0, 500.0]
    N_test = len(t_test)

    for N in [5, 10, 15, 20, 25, 30]:
        A = np.zeros((N_test, N), dtype=complex)
        for j in range(N):
            for i, t in enumerate(t_test):
                A[i, j] = complex(mellin_h_n(mpc(0.5, t), j+1))

        # Minimiere ||A*c||^2 mit Nebenbedingung ||c|| = 1
        # Das ist der kleinste Singulaerwert von A
        U, sv, Vh = np.linalg.svd(A)

        # Kleinster Singulaerwert = beste Cancellation
        min_sv = sv[-1]

        # Der zugehoerige Koeffizientenvektor
        c_opt = Vh[-1, :]

        # Teste den Abfall
        decay_values = []
        for t in [10.0, 50.0, 100.0, 500.0, 1000.0]:
            val = sum(c_opt[j] * complex(mellin_h_n(mpc(0.5, t), j+1)) for j in range(N))
            decay_values.append((t, abs(val)))

        # Schaetze Abfall-Exponent
        if decay_values[1][1] > 0 and decay_values[-1][1] > 0:
            t1, v1 = decay_values[1]
            t2, v2 = decay_values[-1]
            exp_rate = np.log(v2/v1) / np.log(t2/t1)
        else:
            exp_rate = float('-inf')

        print(f"\n  N={N:3d}: min SV = {min_sv:.4e}, Abfall ~ |t|^{exp_rate:.2f}")
        for t, v in decay_values:
            print(f"    t={t:5.0f}: |sum c_n Mh_n| = {v:.6e}")


# ===========================================================================
# 7. HAUPTPROGRAMM
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("WEG 1: RAUMDEFINITION UND TOPOLOGIE")
    print("Fuer den erweiterten Weil-Bereich")
    print("=" * 60)

    # 1. Mellin-Strip
    test_mellin_strip()

    # 2. Topologie-Analyse (analytisch)
    topologie_analyse()

    # 3. Partialbruch-Cancellation (der entscheidende Test!)
    partial_fraction_cancellation_test()

    # 4. Gewichtete Normen
    test_weighted_norms()

    # 5. Mellin-Approximation
    mellin_approximation_test()

    print(f"\n{'='*60}")
    print(f"ERGEBNIS")
    print(f"{'='*60}")
    print("""
  Die KERNFRAGE reduziert sich auf:

  Koennen die Partialbrueche sum b_k/(s+k) mit exponentiell
  fallenden b_k durch endliche Linearkombinationen der h_n
  approximiert werden?

  Das ist aequivalent zu:
  Gibt es {c_n} mit sum_{n>=k} c_n * C(n,k) * (-1)^{k+1} = b_k
  wobei |b_k| <= C * r^k  (exponentieller Abfall)?

  WENN JA: h_n dicht im erweiterten Bereich (Weg 1 funktioniert)
  WENN NEIN: Fundamentale Obstruction (zurueck zu Weg 2)

  Das Cancellation-Experiment oben zeigt, ob durch Linearkombination
  der Mellin-Abfall verbessert werden kann.
""")
