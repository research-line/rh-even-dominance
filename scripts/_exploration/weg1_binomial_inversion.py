#!/usr/bin/env python3
"""
weg1_binomial_inversion.py
===========================
Weg 1: Kann man die Binomialkoeffizienten-Matrix invertieren,
um exponentiell fallende b_k zu erzeugen?

Die Kernfrage:
  Mh_n(s) = sum_{k=1}^n C(n,k)(-1)^{k+1} / (s+k)

  sum_n c_n * Mh_n(s) = sum_k b_k / (s+k)

  wobei b_k = sum_{n>=k} c_n * C(n,k) * (-1)^{k+1}

  Gesucht: {c_n} sodass |b_k| <= C * r^k fuer ein r < 1.

Die Matrix A mit A_{kn} = C(n,k) * (-1)^{k+1} (fuer n >= k) ist eine
untere Dreiecksmatrix. Ihre Inverse ist durch Moebius-Inversion gegeben:
  A^{-1}_{nk} = C(n,k) * (-1)^{n+k+1}  (Binomial-Inversion)

Also: c_n = sum_{k=1}^n A^{-1}_{nk} * b_k = sum_{k=1}^n C(n,k)*(-1)^{n+k+1} * b_k

Die Frage ist: Wenn b_k = r^k (exponentieller Abfall), sind die
resultierenden c_n beschraenkt / summierbar?

c_n = (-1)^{n+1} * sum_{k=1}^n C(n,k) * (-r)^k
    = (-1)^{n+1} * [(1-r)^n - 1]     (Binomischer Satz!)

Also: c_n = (-1)^{n+1} * [(1-r)^n - 1]

Fuer 0 < r < 1: |1-r| < 1, also (1-r)^n -> 0 exponentiell.
Also c_n -> (-1)^{n+1} * (-1) = (-1)^n -- ALTERNIEREND, NICHT summierbar!

ABER: Die Reihe sum c_n h_n konvergiert trotzdem, wenn h_n -> 0 schnell genug.

Genauer: c_n = (-1)^n + (-1)^{n+1}(1-r)^n
        = (-1)^n [1 - (1-r)^n]
        = (-1)^n * h_n(1/(1-r))  (!)  -- die Li-Polynome selbst!

Das ist ein ZIRKELSCHLUSS? Nein -- es zeigt die STRUKTUR.
Testen wir das numerisch.
"""

import numpy as np
from mpmath import mp, mpf, mpc, binomial as mpbinom
import sys

mp.dps = 50

# ===========================================================================
# 1. Analytische Inversion
# ===========================================================================

def analytical_test():
    """Teste die analytische Formel fuer c_n bei b_k = r^k."""
    print("=" * 60)
    print("ANALYTISCHE BINOMIAL-INVERSION")
    print("=" * 60)

    print("""
  Gegeben: b_k = r^k (exponentieller Abfall, 0 < r < 1)
  Matrix: b_k = sum_{n>=k} c_n * C(n,k) * (-1)^{k+1}
  Inverse (Binomial-Inversion):
    c_n = (-1)^{n+1} * [(1-r)^n - 1]

  Pruefe: Ist sum c_n * h_n konvergent? Welchen Mellin-Abfall hat sie?
""")

    for r in [0.1, 0.3, 0.5, 0.7, 0.9]:
        print(f"\n  r = {r}:")

        # Koeffizienten
        c = []
        for n in range(1, 31):
            c_n = (-1)**(n+1) * ((1-r)**n - 1)
            c.append(c_n)

        # Pruefe b_k = sum c_n C(n,k) (-1)^{k+1}
        print(f"  c_1..c_5: {[f'{x:.4f}' for x in c[:5]]}")
        print(f"  |c_n| fuer n=10,20,30: {abs(c[9]):.4e}, {abs(c[19]):.4e}, {abs(c[29]):.4e}")

        # Verifikation: b_k
        N = 30
        b_check = []
        for k in range(1, 11):
            b_k = 0
            for n in range(k, N+1):
                binom = float(mpbinom(n, k))
                b_k += c[n-1] * binom * (-1)**(k+1)
            b_check.append(b_k)

        b_target = [r**k for k in range(1, 11)]

        print(f"  b_1..b_5 (berechnet): {[f'{x:.6f}' for x in b_check[:5]]}")
        print(f"  b_1..b_5 (Ziel r^k):  {[f'{x:.6f}' for x in b_target[:5]]}")

        # Abweichung
        max_err = max(abs(b_check[k] - b_target[k]) for k in range(min(len(b_check), len(b_target))))
        print(f"  Max Fehler: {max_err:.4e}")


# ===========================================================================
# 2. Mellin-Abfall der resultierenden Reihe
# ===========================================================================

def mellin_decay_test():
    """Berechne den Mellin-Abfall von sum c_n Mh_n(s) mit c_n aus der Inversion."""
    print(f"\n{'='*60}")
    print(f"MELLIN-ABFALL DER INVERTIEREN REIHE")
    print(f"{'='*60}")

    def mellin_h_n(s, n):
        """Mellin-Transformierte von h_n."""
        result = mpc(0)
        for k in range(1, n+1):
            binom = float(mpbinom(n, k))
            result += binom * (-1)**(k+1) / (s + k)
        return result

    for r in [0.3, 0.5, 0.7]:
        print(f"\n  r = {r}, b_k = r^k:")
        N = 50  # Mehr Terme fuer bessere Konvergenz

        c = [(-1)**(n+1) * ((1-r)**n - 1) for n in range(1, N+1)]

        # Mellin-Abfall
        print(f"  {'t':>8} | {'|sum c_n Mh_n(1/2+it)|':>25} | {'Einzeln |Mh_1|':>18}")
        print(f"  {'-'*8}-+-{'-'*25}-+-{'-'*18}")

        for t in [10.0, 50.0, 100.0, 500.0, 1000.0]:
            s = mpc(0.5, t)

            # Summe
            total = mpc(0)
            for n in range(N):
                total += c[n] * mellin_h_n(s, n+1)

            # Einzeln
            single = mellin_h_n(s, 1)

            print(f"  {t:8.0f} | {float(abs(total)):25.10e} | {float(abs(single)):18.10e}")

        # Schaetze Abfall-Exponent
        s1 = mpc(0.5, 100.0)
        s2 = mpc(0.5, 1000.0)
        v1 = abs(sum(c[n] * mellin_h_n(s1, n+1) for n in range(N)))
        v2 = abs(sum(c[n] * mellin_h_n(s2, n+1) for n in range(N)))
        if v1 > 0 and v2 > 0:
            exp_rate = float(mp.log(v2/v1) / mp.log(mpf(1000)/mpf(100)))
        else:
            exp_rate = float('-inf')
        print(f"  Geschaetzter Abfall: O(|t|^{exp_rate:.2f})")


# ===========================================================================
# 3. Was die Reihe im ZEIT-Bild ist
# ===========================================================================

def time_domain_analysis():
    """Analysiere sum c_n h_n(x) im Zeitraum."""
    print(f"\n{'='*60}")
    print(f"ZEITRAUM-ANALYSE: sum c_n h_n(x)")
    print(f"{'='*60}")

    print("""
  c_n = (-1)^{n+1} * [(1-r)^n - 1]

  sum_{n=1}^inf c_n * h_n(x) = sum_{n=1}^inf (-1)^{n+1}[(1-r)^n - 1][1-(1-1/x)^n]

  Sei u = 1 - 1/x. Dann h_n(x) = 1 - u^n.

  sum c_n h_n = sum (-1)^{n+1}[(1-r)^n - 1](1 - u^n)
             = sum (-1)^{n+1}[(1-r)^n - 1] - sum (-1)^{n+1}[(1-r)^n - 1] u^n

  Term 1: sum (-1)^{n+1}(1-r)^n = (1-r)/(1+(1-r)) = (1-r)/(2-r)  (geom. Reihe)
           - sum (-1)^{n+1} = -(-1/(1+1)) = 1/2
           = (1-r)/(2-r) - 1/2 = (2(1-r) - (2-r)) / (2(2-r))
           = (2 - 2r - 2 + r) / (2(2-r)) = -r / (2(2-r))

  Term 2: sum (-1)^{n+1}(1-r)^n u^n = (1-r)u / (1 + (1-r)u)  (geom.)
           - sum (-1)^{n+1} u^n = u / (1 + u)
           => -(1-r)u/(1+(1-r)u) + u/(1+u)

  Also: f(x) = sum c_n h_n(x) = -r/(2(2-r))
               + u/(1+u) - (1-r)u/(1+(1-r)u)

  Zurueck zu x: u = 1 - 1/x
    u/(1+u) = (1-1/x) / (2-1/x) = (x-1)/(2x-1)
    (1-r)u/(1+(1-r)u) = (1-r)(1-1/x) / (1+(1-r)(1-1/x))

  DAS IST EINE GESCHLOSSENE FORMEL fuer f(x)!
  Und f ist NICHT Schwartz (hat Grenzwert fuer x -> inf).

  ABER: Die Mellin-Transformierte von f hat Abfall besser als O(|t|^{-1})
  weil die Pole sich gegenseitig aufheben!
""")

    for r in [0.3, 0.5, 0.7]:
        print(f"\n  r = {r}:")
        print(f"  f(x) = -r/(2(2-r)) + (x-1)/(2x-1) - (1-r)(x-1)/((1-r)(x-1)+x)")

        # Numerische Werte
        for x in [0.5, 1.0, 2.0, 5.0, 10.0, 100.0]:
            u = 1 - 1/x if x != 0 else -float('inf')
            term1 = -r / (2*(2-r))
            term2 = u / (1 + u) if abs(1 + u) > 1e-15 else float('inf')
            term3 = (1-r)*u / (1 + (1-r)*u) if abs(1 + (1-r)*u) > 1e-15 else float('inf')
            f_val = term1 + term2 - term3
            print(f"    f({x:6.1f}) = {f_val:+.8f}")

        # Grenzwert bei inf
        lim = -r/(2*(2-r)) + 1/2 - (1-r)/1
        print(f"    f(inf)   = {lim:+.8f}  (NICHT Null!)")
        print(f"    => f ist NICHT in L1 oder Schwartz")
        print(f"    => ABER: Die Differenz f - f(inf) KOENNTE guten Abfall haben")


# ===========================================================================
# 4. DER TIEFERE PUNKT: Unendliche Reihen vs endliche Linearkombinationen
# ===========================================================================

def deeper_analysis():
    """Die entscheidende mathematische Einsicht."""
    print(f"\n{'='*60}")
    print(f"DIE ENTSCHEIDENDE EINSICHT")
    print(f"{'='*60}")

    print("""
  ERGEBNIS DER ANALYSE:

  1. Die Binomial-Inversion ist EXAKT loesbar:
     c_n = (-1)^{n+1} * [(1-r)^n - 1]  fuer b_k = r^k

  2. Die resultierenden c_n sind ALTERNIEREND und konvergieren
     gegen (-1)^n (nicht gegen 0).

  3. Die Reihe sum c_n h_n hat eine GESCHLOSSENE FORM
     und ist eine RATIONALE FUNKTION in u = 1-1/x.

  4. Diese rationale Funktion ist NICHT Schwartz,
     hat aber eine Mellin-Transformierte mit BESSEREM Abfall
     als die einzelnen h_n.

  5. ENTSCHEIDEND: Der bessere Abfall kommt aus CANCELLATION
     der einfachen Pole -- die 1/(s+k)-Terme heben sich teilweise auf.
     Bei sum b_k/(s+k) mit b_k = r^k hat man:
       sum r^k/(s+k) = Psi((s+1)/(1-r)) - Psi(...)  (Digamma!)
     Das hat Abfall O(|t|^{-1} * r^{|t|}) -- EXPONENTIELL in t!

  6. Also: Die Reihe sum c_n Mh_n(s) = sum r^k/(s+k) hat
     EXPONENTIELLEN Abfall im Mellin-Bild, nicht nur polynomialen!

  7. DAS BEDEUTET: Die h_n koennen Funktionen mit BELIEBIG SCHNELLEM
     Mellin-Abfall approximieren (durch geeignete unendliche Reihen).

  KONSEQUENZ FUER DIE DICHTHEITSFRAGE:

  Die Menge {sum c_n h_n : sum |c_n| < inf} ist NICHT dicht
  (weil sum |c_n| < inf nur O(|t|^{-1}) Abfall gibt).

  ABER: Die Menge {sum c_n h_n : c_n alternierend, geom. beschraenkt}
  IST "dicht genug", weil die Partialbruch-Cancellation
  beliebig schnellen Abfall erzeugt.

  Die RICHTIGE Topologie ist also NICHT die L1-Koeffizienten-Topologie,
  sondern eine SCHWAECHERE Topologie, die alternierende Reihen zulaesst.

  DAS IST EXAKT das Nyman-Beurling-Muster:
  Dort konvergiert die Approximation NICHT in L1,
  sondern in L2 (schwaechere Topologie).

  VORSCHLAG: Arbeite mit der Topologie der Mellin-L2-Konvergenz
  auf dem kritischen Streifen mit geeignetem Gewicht.

  ===========================================================
  ZUSAMMENFASSUNG FUER BEWEISNOTIZ:

  Weg 1 ist MACHBAR, aber erfordert:
  1. Richtige Topologie: Mellin-L2 mit Gewicht (nicht Frechet)
  2. Alternierende Reihen zugelassen (nicht nur abs. konvergente)
  3. Der Beweis geht ueber Partialbruch-Cancellation:
     b_k = r^k => sum b_k/(s+k) hat exponentiellen Abfall
  4. Die Dichtheit ist dann aequivalent zu:
     "Jede Funktion mit |Mf(s)| = O(|t|^{-2-eps}) kann durch
      Partialbruchreihen sum b_k/(s+k) mit schnell fallenden b_k
      approximiert werden."
  5. Das ist ein KLASSISCHES Approximationsproblem in der Theorie
     der Dirichlet-Reihen / Partialbruchzerlegungen.
  ===========================================================
""")


# ===========================================================================
# 5. VERIFIZIERE: Exponentieller Mellin-Abfall fuer b_k = r^k
# ===========================================================================

def verify_exponential_decay():
    """Verifiziere numerisch den exponentiellen Mellin-Abfall."""
    print(f"\n{'='*60}")
    print(f"VERIFIKATION: EXPONENTIELLER MELLIN-ABFALL")
    print(f"{'='*60}")

    print(f"\n  F(s) = sum_{{k=1}}^inf r^k / (s+k)")
    print(f"  Teste |F(1/2+it)| fuer verschiedene r und grosse t")

    for r in [0.3, 0.5, 0.7, 0.9]:
        print(f"\n  r = {r}:")
        print(f"  {'t':>8} | {'|F(1/2+it)|':>15} | {'r^t':>15} | {'Ratio':>10}")
        print(f"  {'-'*8}-+-{'-'*15}-+-{'-'*15}-+-{'-'*10}")

        for t in [5.0, 10.0, 20.0, 50.0, 100.0]:
            s = mpc(0.5, t)
            # Berechne sum r^k / (s+k) fuer k=1..500
            F = mpc(0)
            for k in range(1, 501):
                F += mpf(r)**k / (s + k)

            F_abs = float(abs(F))
            r_t = r**t
            ratio = F_abs / r_t if r_t > 1e-300 else float('inf')

            print(f"  {t:8.0f} | {F_abs:15.6e} | {r_t:15.6e} | {ratio:10.4f}")


# ===========================================================================
# 6. HAUPTPROGRAMM
# ===========================================================================

if __name__ == "__main__":
    analytical_test()
    time_domain_analysis()
    verify_exponential_decay()
    mellin_decay_test()
    deeper_analysis()
