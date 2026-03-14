#!/usr/bin/env python3
"""
herglotz_log_xi.py
==================
Teste: Ist log xi(s) eine Herglotz-Funktion (Nevanlinna-Funktion)?

Herglotz-Eigenschaft: Im[f(z)] >= 0 fuer Im(z) > 0
Nevanlinna-Eigenschaft: Im[f(z)] >= 0 fuer Im(z) > 0 (aequivalent)

Fuer xi(s) = xi(1-s), reell auf Re(s) = 1/2:
  xi(1/2 + it) ist reell fuer reelles t (Funktionalgleichung + Hermitizitaet)

Wenn alle Nullstellen auf Re(s) = 1/2 liegen (RH), dann hat
  xi(s) / xi(0) = prod_rho (1 - s/rho)
nur REELLE "Nullstellen" in der Variable t = Im(s - 1/2).

Frage: Ist xi'/xi (logarithmische Ableitung) eine Herglotz-Funktion
in der OBEREN HALBEBENE von s?

Zusammenhang Li <-> Herglotz:
  Wenn xi'/xi Herglotz ist, dann sind die Taylor-Koeffizienten von
  log xi um s=1 (verwandt mit Li-Koeffizienten) positiv -- das IST
  das Li-Kriterium!

Wir testen mehrere Varianten:
1. xi'/xi als Funktion von s
2. log xi(1/2 + z) als Funktion von z (zentriert auf kritischer Linie)
3. Z(t) = e^{i*theta(t)} * zeta(1/2 + it) (Hardy Z-Funktion)
"""

from mpmath import (mp, mpf, mpc, log, exp, pi, gamma, zeta, zetazero,
                    diff, im, re, arg, loggamma, sqrt, inf, fsum)
import numpy as np

mp.dps = 30

# ===========================================================================
# 1. xi-Funktion und log xi
# ===========================================================================

def xi_func(s):
    """Riemann xi: xi(s) = (1/2) s(s-1) pi^{-s/2} Gamma(s/2) zeta(s)"""
    return mpf('0.5') * s * (s - 1) * exp(-s/2 * log(pi)) * gamma(s/2) * zeta(s)

def log_xi(s):
    """log xi(s) -- direkt berechnet"""
    return log(mpf('0.5')) + log(s) + log(s-1) + (-s/2)*log(pi) + loggamma(s/2) + log(zeta(s))

def xi_prime_over_xi(s):
    """xi'/xi via numerischer Differentiation"""
    return diff(lambda z: log_xi(z), s)


# ===========================================================================
# 2. Test: Im[log xi(s)] fuer Im(s) > 0
# ===========================================================================

def test_log_xi_herglotz():
    """Teste Herglotz-Eigenschaft von log xi in der oberen Halbebene."""
    print("=" * 60)
    print("TEST: Ist log xi(s) eine Herglotz-Funktion?")
    print("=" * 60)
    print("Herglotz: Im[log xi(s)] >= 0 fuer Im(s) > 0")
    print()

    # Testpunkte in der oberen Halbebene
    test_points = []

    # Entlang der kritischen Linie, leicht nach oben verschoben
    for t in [0.1, 0.5, 1.0, 5.0, 10.0, 14.0, 14.5, 15.0, 20.0, 25.0, 30.0]:
        test_points.append(("krit. Linie", mpc(0.5, t)))

    # Rechts der kritischen Linie
    for sigma in [0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]:
        for t in [1.0, 10.0, 14.5]:
            test_points.append((f"sigma={sigma}", mpc(sigma, t)))

    # Links der kritischen Linie
    for sigma in [0.1, 0.2, 0.3, 0.4]:
        for t in [1.0, 10.0, 14.5]:
            test_points.append((f"sigma={sigma}", mpc(sigma, t)))

    # Nahe der ersten Nullstelle (rho_1 ~ 0.5 + 14.135i)
    rho1_t = float(im(zetazero(1)))
    for eps in [0.01, 0.1, 0.5, 1.0]:
        test_points.append(("nahe rho1", mpc(0.5, rho1_t + eps)))
        test_points.append(("nahe rho1", mpc(0.5, rho1_t - eps)))
        test_points.append(("nahe rho1", mpc(0.5 + eps, rho1_t)))

    print(f"  {'Bereich':>15} | {'s':>25} | {'Im[log xi(s)]':>18} | Herglotz?")
    print(f"  {'-'*15}-+-{'-'*25}-+-{'-'*18}-+-{'-'*9}")

    violations = 0
    total = 0

    for label, s in test_points:
        if im(s) <= 0:
            continue
        total += 1
        try:
            val = log_xi(s)
            im_val = float(im(val))
            ok = "JA" if im_val >= 0 else "NEIN !!!"
            if im_val < 0:
                violations += 1
            print(f"  {label:>15} | {str(s):>25} | {im_val:>18.10f} | {ok}")
        except Exception as e:
            print(f"  {label:>15} | {str(s):>25} | {'ERROR':>18} | {str(e)[:20]}")

    print(f"\n  Verletzungen: {violations}/{total}")
    if violations == 0:
        print("  => log xi KOENNTE Herglotz sein (kein Gegenbeispiel gefunden)")
    else:
        print("  => log xi ist KEINE Herglotz-Funktion")


# ===========================================================================
# 3. Test: Im[xi'/xi(s)] fuer Im(s) > 0
# ===========================================================================

def test_xi_log_deriv_herglotz():
    """Teste Herglotz-Eigenschaft der logarithmischen Ableitung xi'/xi."""
    print("\n" + "=" * 60)
    print("TEST: Ist xi'/xi(s) eine Herglotz-Funktion?")
    print("=" * 60)
    print("Fuer Funktionen mit nur reellen Nullstellen (Laguerre-Polya):")
    print("  f'/f = sum_n 1/(s - rho_n)  mit rho_n reell")
    print("  => Im[f'/f(s)] = sum_n Im(s) / |s - rho_n|^2 >= 0 fuer Im(s) > 0")
    print()
    print("Fuer xi: Nullstellen auf Re(s) = 1/2 (RH), NICHT auf reeller Achse.")
    print("  xi'/xi = sum_rho 1/(s - rho)")
    print("  rho = 1/2 + i*gamma_n")
    print("  Im[1/(s-rho)] = -Im(s-rho)/|s-rho|^2 = -(Im(s) - gamma_n)/|s-rho|^2")
    print()
    print("  => Fuer Im(s) > max(gamma_n): Im[xi'/xi] < 0 (VERLETZUNG!)")
    print("  => Fuer Im(s) zwischen Nullstellen: Vorzeichen WECHSELT")
    print("  => xi'/xi ist KEINE standard Herglotz-Funktion.")
    print()

    # Numerische Verifikation
    test_s = [mpc(0.5, 5), mpc(0.5, 10), mpc(0.5, 14), mpc(0.5, 15),
              mpc(0.5, 20), mpc(0.5, 25), mpc(0.5, 50), mpc(0.5, 100),
              mpc(0.7, 10), mpc(0.3, 10), mpc(1.0, 14.5)]

    print(f"  {'s':>20} | {'Im[xi_prime/xi]':>18} | Herglotz?")
    print(f"  {'-'*20}-+-{'-'*18}-+-{'-'*9}")

    for s in test_s:
        try:
            val = xi_prime_over_xi(s)
            im_val = float(im(val))
            ok = "JA" if im_val >= 0 else "NEIN"
            print(f"  {str(s):>20} | {im_val:>18.10f} | {ok}")
        except Exception as e:
            print(f"  {str(s):>20} | {'ERROR':>18} | {str(e)[:20]}")


# ===========================================================================
# 4. ROTIERTE Herglotz: f(z) = log xi(1/2 + iz)
#    Substitution: s = 1/2 + iz, z in oberer Halbebene => Im(s) = Re(z)
#    Nullstellen von xi bei s = 1/2 + i*gamma => z = gamma (reell!)
#    => In der z-Variable HAT xi nur reelle Nullstellen (unter RH)!
# ===========================================================================

def test_rotated_herglotz():
    """Teste: Ist log xi(1/2 + iz) Herglotz in z?"""
    print("\n" + "=" * 60)
    print("TEST: ROTIERTE VARIABLE z, s = 1/2 + iz")
    print("=" * 60)
    print("Substitution: s = 1/2 + iz")
    print("  z in obere Halbebene (Im z > 0) => Re(s) > 1/2")
    print("  Nullstellen: rho = 1/2 + i*gamma => z = gamma (REELL unter RH!)")
    print()
    print("In der z-Variable ist xi eine Funktion mit NUR REELLEN Nullstellen.")
    print("Fuer solche Funktionen gilt:")
    print("  log f(z) ist Herglotz <=> alle Nullstellen reell + Wachstumsbedingung")
    print("  (Laguerre-Polya Klasse)")
    print()

    # g(z) = xi(1/2 + iz) / xi(1/2)
    xi_half = xi_func(mpf('0.5'))
    print(f"  xi(1/2) = {xi_half}")

    def g(z):
        s = mpc(0.5, 0) + mpc(0, 1) * z
        return xi_func(s) / xi_half

    def log_g(z):
        return log(g(z))

    # Teste in der oberen z-Halbebene
    test_z = []
    for x in [-20, -10, -5, 0, 5, 10, 14, 14.5, 15, 20, 25, 30, 50]:
        for y in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
            test_z.append(mpc(x, y))

    print(f"\n  {'z':>20} | {'s = 1/2 + iz':>20} | {'Im[log g(z)]':>18} | Herglotz?")
    print(f"  {'-'*20}-+-{'-'*20}-+-{'-'*18}-+-{'-'*9}")

    violations = 0
    total = 0

    for z in test_z:
        if im(z) <= 0:
            continue
        total += 1
        try:
            val = log_g(z)
            im_val = float(im(val))
            s = mpc(0.5, 0) + mpc(0, 1) * z
            ok = "JA" if im_val >= -1e-20 else "NEIN"
            if im_val < -1e-20:
                violations += 1
            # Nur interessante Punkte ausgeben
            if abs(im_val) > 0.01 or im_val < 0 or abs(re(z)) < 0.1:
                print(f"  {str(z):>20} | {str(s):>20} | {im_val:>18.10f} | {ok}")
        except Exception as e:
            if "not zero-free" not in str(e):
                print(f"  {str(z):>20} | {'':>20} | {'ERROR':>18} | {str(e)[:30]}")

    print(f"\n  Verletzungen: {violations}/{total}")
    if violations == 0:
        print("  => log xi(1/2 + iz) KOENNTE Herglotz in z sein!")
        print("  => Das waere AEQUIVALENT zu: alle Nullstellen von xi reell in z")
        print("  => Das waere AEQUIVALENT zu RH!")
        print()
        print("  ABER VORSICHT: log xi(1/2+iz) IST Herglotz <=> RH")
        print("  Das ist eine UMFORMULIERUNG, kein Beweis.")
        print("  Der Wert liegt darin, RH in die SPRACHE der Herglotz-Funktionen zu bringen,")
        print("  wo maechtige Werkzeuge existieren (Integraldarstellungen, Positivitaet).")
    else:
        print("  => log xi(1/2 + iz) ist KEINE Herglotz-Funktion")
        print("  => Entweder: numerischer Fehler nahe Nullstelle, oder")
        print("  => Die rotierte Variable loest das Problem NICHT")


# ===========================================================================
# 5. Verbindung zu Li-Koeffizienten
# ===========================================================================

def li_herglotz_connection():
    """Zeige die Verbindung zwischen Herglotz-Eigenschaft und Li-Koeffizienten."""
    print("\n" + "=" * 60)
    print("VERBINDUNG: Herglotz <-> Li-Koeffizienten")
    print("=" * 60)

    print("""
  Die entscheidende Verbindung:

  Definiere: phi(z) = log[xi(1/2 + iz) / xi(1/2)]
             (log xi in der rotierten Variable, normiert)

  Wenn phi Herglotz ist (Im phi >= 0 fuer Im z > 0), dann hat phi
  eine INTEGRALDARSTELLUNG (Herglotz/Nevanlinna):

    phi(z) = a + bz + integral_{-inf}^{inf} [1/(t-z) - t/(1+t^2)] dmu(t)

  wobei mu ein POSITIVES Mass ist, b >= 0, a reell.

  Andererseits, die Li-Koeffizienten sind Taylor-Koeffizienten
  einer verwandten Funktion. Genauer:

    log[xi(s)/xi(0)] hat Taylorentwicklung um s = 1:
    = sum_{n=1}^inf (-1)^{n+1} * lambda_n / n * (s-1)^n

  Die Li-Positivitaet lambda_n >= 0 ist eine Bedingung an diese
  Taylor-Koeffizienten.

  ZUSAMMENHANG:
  - Herglotz-Eigenschaft => Alle Nullstellen reell (in z-Variable) => RH
  - Herglotz-Eigenschaft => Integraldarstellung mit positivem Mass mu
  - mu positiv => bestimmte Positivitaetsbedingungen an Momente
  - Diese Momente sind VERWANDT mit den Li-Koeffizienten

  ABER: Die Taylor-Entwicklung von phi um z=0 und die Li-Koeffizienten
  um s=1 sind in VERSCHIEDENEN Variablen (z vs s) und um VERSCHIEDENE
  Punkte entwickelt.

  Die Substitution s = 1/2 + iz => z = (s - 1/2)/i = i(1/2 - s):
    s = 1 => z = i(1/2 - 1) = -i/2  (NICHT im Entwicklungsbereich!)

  => Die direkte Uebersetzung ist NICHT trivial.
""")

    # Numerisch: Berechne Momente des Herglotz-Masses
    print("  Numerische Pruefung: Taylor-Koeffizienten von phi(z) um z=0")
    print("  phi(z) = log[xi(1/2 + iz) / xi(1/2)]")
    print()

    xi_half = xi_func(mpf('0.5'))

    def phi(z):
        s = mpc(0.5, 0) + mpc(0, 1) * z
        return log(xi_func(s) / xi_half)

    # Taylor-Koeffizienten via numerische Differentiation
    print(f"  {'n':>3} | {'phi^(n)(0)/n!':>20} | {'Re':>12} | {'Im':>12}")
    print(f"  {'-'*3}-+-{'-'*20}-+-{'-'*12}-+-{'-'*12}")

    for n in range(1, 11):
        try:
            coeff = diff(phi, mpf(0), n) / mp.factorial(n)
            print(f"  {n:3d} | {'':>20} | {float(re(coeff)):>12.6e} | {float(im(coeff)):>12.6e}")
        except Exception as e:
            print(f"  {n:3d} | ERROR: {str(e)[:40]}")


# ===========================================================================
# 6. HAUPTPROGRAMM
# ===========================================================================

if __name__ == "__main__":
    test_log_xi_herglotz()
    test_xi_log_deriv_herglotz()
    test_rotated_herglotz()
    li_herglotz_connection()

    print("\n" + "=" * 60)
    print("GESAMTFAZIT")
    print("=" * 60)
    print("""
  1. log xi(s) ist KEINE Herglotz-Funktion in s
     (Nullstellen nicht auf der reellen Achse in s-Variable)

  2. xi'/xi(s) ist KEINE Herglotz-Funktion
     (Vorzeichenwechsel zwischen Nullstellen)

  3. log xi(1/2 + iz) SOLLTE Herglotz in z sein GENAU DANN WENN RH gilt
     (Rotation bringt Nullstellen auf reelle Achse in z)
     => AEQUIVALENTE UMFORMULIERUNG von RH, kein Beweis

  4. Die Verbindung Li <-> Herglotz ist INDIREKT:
     Beide folgen aus RH, aber der direkte Weg
     (Herglotz-Mass -> Li-Positivitaet oder umgekehrt)
     erfordert Kontrolle ueber die Substitution s <-> z
     und Taylorentwicklung um VERSCHIEDENE Punkte.

  STRATEGISCHE BEWERTUNG:
  Die Herglotz-Umformulierung bringt RH in einen NEUEN RAHMEN,
  aber liefert keinen Beweis. Der Mehrwert liegt in der
  Integraldarstellung: Wenn man zeigen koennte, dass phi(z)
  eine Herglotz-Darstellung HAT (ohne RH vorauszusetzen),
  waere das ein Beweis.
""")
