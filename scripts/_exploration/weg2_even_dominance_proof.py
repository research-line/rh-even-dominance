#!/usr/bin/env python3
"""
weg2_even_dominance_proof.py
============================
Strukturelles Argument fuer Even-Dominanz bei grossem lambda.

IDEE: Der Kernel e^{u/2}/(2sinh(u)) hat bei u>0 einen Faktor e^{u/2},
der die Asymmetrie between even und odd erklaert.

ANALYSE:
1. Zerlegung: e^{u/2}/(2sinh(u)) = 1/(2(1 - e^{-2u})) = sum_{k=0}^inf e^{-2ku}/2
   => geometrische Reihe! Der Kernel ist die ERZEUGENDE Funktion von e^{-2ku}.

   Verifizierung: e^{u/2}/(2sinh(u)) = e^{u/2}/(e^u - e^{-u})
                = 1/(e^{u/2} - e^{-3u/2})... nein.

   Besser: e^{u/2}/(2sinh(u)) = e^{u/2}/(e^u - e^{-u})
         = 1/(e^{u/2} - e^{-3u/2})

   Oder: = (1/2) * e^{u/2}/(e^u/2 * (1 - e^{-2u}))
   Hmm. Lass es einfacher machen:

   e^{u/2}/(2sinh(u)) = e^{u/2}/(e^u - e^{-u})

   Fuer u >> 0:  ~ e^{u/2}/e^u = e^{-u/2}
   Fuer u -> 0+: ~ e^0/(2u) = 1/(2u)

2. Der KEY-Unterschied: Fuer den even-Sektor existiert die Konstant-Funktion
   phi_0(t) = 1/sqrt(2L), die ALLE Shift-Operatoren maximal ueberlappt:
   <T_u phi_0, phi_0> = (2L - |u|)/(2L) fuer |u| < 2L.

   Fuer den odd-Sektor ist phi_0 = sin(pi*t/2L)/sqrt(L), und
   <T_u phi_1, phi_1> = integral von sin(pi*t/2L) * sin(pi*(t-u)/2L) dt / L
   was kleiner ist und schneller mit u abfaellt.

3. FORMALES ARGUMENT:
   Sei phi_0 die Konstant-Mode im even-Sektor.
   QW(phi_0, phi_0) = (log4pi+gamma) + integral_terme + prim_terme

   Die Integral-Terme fuer phi_0 sind:
   int_0^s_max [<T_u phi_0, phi_0> + <T_{-u} phi_0, phi_0> - 2e^{-u/2}] * k(u) du

   wobei <T_u phi_0, phi_0> = (2L-u)/(2L) fuer u < 2L.

   Also: 2*(2L-u)/(2L) - 2e^{-u/2} = (2L-u)/L - 2e^{-u/2}

   Fuer u << L: ~ 2 - u/L - 2 + u = u(1 - 1/L) > 0 fuer L > 1.
   Fuer u -> 2L: ~ 0 - 2e^{-L} < 0 aber exponentiell klein.

   Frage: Ist QW(phi_0, phi_0) < lambda_1(odd)?
"""

import numpy as np
from scipy.linalg import eigh
from sympy import primerange
from mpmath import euler as mp_euler, log as mplog, pi as mppi
from scipy.integrate import quad

LOG4PI_GAMMA = float(mplog(4 * mppi)) + float(mp_euler)


def analyze_constant_mode(lam, primes, n_points=10000):
    """Berechne QW(phi_0, phi_0) analytisch fuer die Konstant-Mode."""
    L = np.log(lam)
    norm = 1.0 / (2 * L)  # phi_0 = 1/sqrt(2L), phi_0^2 = 1/(2L)

    # 1. Diagonal: (log 4pi + gamma) * ||phi_0||^2 = log4pi_gamma
    diag = LOG4PI_GAMMA

    # 2. Archimedean integral
    # <T_u phi_0, phi_0> = integral_{-L}^{L} phi_0(t+u) phi_0(t) dt
    # = (1/(2L)) * max(0, 2L - |u|) = (1 - |u|/(2L)) fuer |u| < 2L

    def overlap(u):
        if abs(u) >= 2 * L:
            return 0.0
        return 1.0 - abs(u) / (2 * L)

    def integrand_arch(u):
        k = np.exp(u / 2) / (2 * np.sinh(u))
        ov = overlap(u) + overlap(-u)  # = 2 * overlap(u) da overlap ist gerade
        return (ov - 2 * np.exp(-u / 2)) * k

    # Numerische Integration (Singularitaet bei u=0 ist integrabel: ~1/(2u) * 2u = endlich)
    s_max = min(2 * L, 8.0)
    u_grid = np.linspace(0.005, s_max, n_points)
    du = u_grid[1] - u_grid[0]
    arch = sum(integrand_arch(u) * du for u in u_grid)

    # 3. Primzahl-Terme
    # Fuer phi_0: <T_{m*logp} phi_0, phi_0> = overlap(m*logp)
    prime_sum = 0.0
    for p in primes:
        logp = np.log(p)
        for m in range(1, 20):
            shift = m * logp
            if shift >= 2 * L:
                break
            coeff = logp * p**(-m / 2.0)
            prime_sum += coeff * 2 * overlap(shift)  # +shift und -shift

    total = diag + arch + prime_sum

    print(f"  lambda={lam}: L={L:.3f}")
    print(f"    Diag (log4pi+gamma):  {diag:+.6f}")
    print(f"    Archimedean integral: {arch:+.6f}")
    print(f"    Prime sum:            {prime_sum:+.6f}")
    print(f"    QW(phi_0, phi_0):     {total:+.6f}")

    return total


def compare_test_functions(lambdas, primes):
    """Vergleiche QW(phi_0, phi_0) mit lambda_1(odd)."""
    from weg2_kernel_convergence import build_QW

    print("=" * 75)
    print("EVEN-DOMINANZ ANALYSE: QW(phi_0) vs lambda_1(odd)")
    print("=" * 75)
    print("\n  Wenn QW(phi_0, phi_0) < lambda_1(odd), dann ist der Even-Sektor")
    print("  garantiert tiefer (Variationsprinzip).")

    for lam in lambdas:
        primes_used = [p for p in primes if p <= max(lam, 100)]

        # QW(phi_0, phi_0) analytisch
        qw00 = analyze_constant_mode(lam, primes_used)

        # lambda_1(odd) numerisch
        N = 60
        QW_odd = build_QW(lam, N, primes_used, basis='sin')
        l1_odd = np.sort(eigh(QW_odd, eigvals_only=True))[0]

        # Vergleich
        is_lower = qw00 < l1_odd
        print(f"    lambda_1(odd):        {l1_odd:+.6f}")
        print(f"    QW(phi_0) < l1(odd)?  {is_lower}  "
              f"(Differenz: {qw00 - l1_odd:+.6f})")
        print()

    # Asymptotisches Argument
    print("=" * 75)
    print("ASYMPTOTISCHE ANALYSE: Warum Even fuer grosse lambda dominiert")
    print("=" * 75)

    print("\n  Archim. Integral der Konstant-Mode:")
    print("  int_0^{2L} [2(1-u/2L) - 2e^{-u/2}] * e^{u/2}/(2sinh(u)) du")
    print()

    for lam in [10, 30, 50, 100, 200, 500, 1000]:
        L = np.log(lam)

        def integrand(u):
            if u < 1e-10:
                return 0.0
            k = np.exp(u / 2) / (2 * np.sinh(u))
            bracket = 2 * (1 - u / (2 * L)) - 2 * np.exp(-u / 2)
            return bracket * k

        s_max = min(2 * L, 15.0)
        u_grid = np.linspace(0.005, s_max, 5000)
        du = u_grid[1] - u_grid[0]
        val = sum(integrand(u) * du for u in u_grid)

        print(f"  lambda={lam:5d} (L={L:.2f}): arch_integral = {val:+.4f}")


if __name__ == "__main__":
    primes = list(primerange(2, 500))
    lambdas = [10, 20, 25, 30, 40, 50, 100]
    compare_test_functions(lambdas, primes)
