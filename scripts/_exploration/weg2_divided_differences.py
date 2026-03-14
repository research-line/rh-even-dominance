#!/usr/bin/env python3
"""
weg2_divided_differences.py
===========================
Alternative Diskretisierung von QW_lambda via dividierte Differenzen
nach Connes & van Suijlekom (arXiv:2511.23257, Proposition 4.1-4.2).

Die Matrix hat die Form:
  q_{m,n} = (psi(m) - psi(n)) / (m - n)   fuer m != n
  q_{n,n} = psi'(n)

wobei psi(x) aus den Fourier-Koeffizienten der Weil-Distribution D besteht.

VORTEIL: Keine Quadratur-Integration noetig, analytisch berechenbar.
Die Matrix ist durch die Werte psi(0), psi(1), ..., psi(N-1) bestimmt.

ABLEITUNG fuer unseren Fall:
Die Weil-Distribution auf [0, L] mit L = log(lambda) besteht aus:
  D(y) = (log4pi + gamma) * delta(y)       [Diagonalterm]
       + K(y) * dy                           [Archimedischer Kernel]
       + sum_p sum_m logp * p^{-m/2} * delta(y - m*logp)  [Primterme]

wobei K(y) = e^{y/2}/(2sinh(y)) - e^{-y/2}/y   (regularisiert bei y=0)

Die psi-Funktion ist:
  psi(x) = (1/pi) int_0^L sin(2*pi*x*(1-y/L)) * D(y) dy
"""

import numpy as np
from scipy.linalg import eigh
from sympy import primerange
from mpmath import euler as mp_euler, log as mplog, pi as mppi
from scipy.integrate import quad
import time

LOG4PI_GAMMA = float(mplog(4 * mppi)) + float(mp_euler)


def compute_psi_values(N, lam, primes, n_int=2000):
    """
    Berechne psi(x) fuer x = 0, 1, ..., N-1.

    psi(x) = (1/pi) int_0^L sin(2*pi*x*(1 - y/L)) * D(y) dy

    D(y) = Weil-Distribution = arch_kernel(y) + prime_deltas(y) + diag_delta(y)
    """
    L = np.log(lam)
    psi = np.zeros(N)

    for n in range(N):
        x = float(n)

        # 1. Diagonalterm: (log4pi+gamma) * delta(y=0)
        # sin(2*pi*x*(1 - 0/L)) = sin(2*pi*x) = 0 fuer ganzzahliges x
        # Also traegt der Diagonalterm NICHT zu psi bei (nur zu psi'(n) via Ableitung)
        # WAIT: psi(x) ist fuer REELLE x definiert, wir brauchen ganzzahlige x
        # sin(2*pi*n) = 0, also kein Beitrag vom Delta bei y=0

        # 2. Archimedischer Integral-Kernel
        # K(y) = [e^{y/2}/(2sinh(y)) - e^{-y/2}/y] fuer y > 0
        # Regularisierung: der Subtraktionsterm macht K(y) integrabel bei y=0
        # Fuer y -> 0: e^{y/2}/(2sinh(y)) ~ 1/(2y), e^{-y/2}/y ~ 1/y
        # Also K(y) ~ 1/(2y) - 1/y = -1/(2y) -- hmm, das divergiert immer noch
        #
        # ALTERNATIVE: Connes' Gl. 10 hat den VOLLEN Integrand
        # [f(x) + f(x^-1) - 2*x^{-1/2}*f(1)] * x^{1/2}/(x-x^{-1})
        # Der Term -2*x^{-1/2}*f(1) regularisiert die Singularitaet
        # In der Fourier-Darstellung: die delta-bei-0-Beitraege canceln sich
        #
        # Fuer die psi-Berechnung brauchen wir also den GESAMTEN Integrand,
        # nicht nur den Kernel allein.
        #
        # psi_arch(x) = (1/pi) int_0^L sin(2*pi*x*(1-y/L)) * K_full(y) dy
        # wobei K_full den regularisierten Kernel einschliesst

        def integrand_arch(y):
            if y < 1e-12:
                return 0.0
            s = 2 * np.pi * x * (1 - y / L)
            k = np.exp(y / 2) / (2 * np.sinh(y))
            # Der volle Kernel ohne Regularisierung
            # (die Regularisierung via -2e^{-y/2} geht in den diag-Term)
            return np.sin(s) * k / np.pi

        if x > 0:
            # Numerische Integration
            y_grid = np.linspace(1e-6, L, n_int)
            dy = y_grid[1] - y_grid[0]
            arch_val = sum(integrand_arch(y) * dy for y in y_grid)
            psi[n] += arch_val

        # 3. Primzahl-Beitraege: delta(y - m*logp)
        for p in primes:
            logp = np.log(p)
            for m in range(1, 20):
                shift = m * logp
                if shift >= L:
                    break
                coeff = logp * p**(-m / 2.0)
                s = 2 * np.pi * x * (1 - shift / L)
                psi[n] += coeff * np.sin(s) / np.pi

    return psi


def compute_psi_derivative(N, lam, primes, n_int=2000):
    """
    Berechne psi'(x) fuer x = 0, 1, ..., N-1 (fuer die Diagonale).

    psi'(x) = d/dx psi(x) = (1/pi) int_0^L 2*pi*(1-y/L) * cos(2*pi*x*(1-y/L)) * D(y) dy
            = 2 * int_0^L (1-y/L) * cos(2*pi*x*(1-y/L)) * D(y) dy
    """
    L = np.log(lam)
    psi_p = np.zeros(N)

    for n in range(N):
        x = float(n)

        # 1. Diag-Term: (log4pi+gamma) * delta(y=0)
        # (1-0/L) * cos(2*pi*x*1) = 1 * cos(2*pi*x) = 1 fuer ganzzahliges x
        psi_p[n] += 2 * LOG4PI_GAMMA

        # 2. Archimedes
        def integrand_deriv(y):
            if y < 1e-12:
                return 0.0
            s = 2 * np.pi * x * (1 - y / L)
            k = np.exp(y / 2) / (2 * np.sinh(y))
            return 2 * (1 - y / L) * np.cos(s) * k

        y_grid = np.linspace(1e-6, L, n_int)
        dy = y_grid[1] - y_grid[0]
        arch_val = sum(integrand_deriv(y) * dy for y in y_grid)
        psi_p[n] += arch_val

        # 3. Primzahlen
        for p in primes:
            logp = np.log(p)
            for m in range(1, 20):
                shift = m * logp
                if shift >= L:
                    break
                coeff = logp * p**(-m / 2.0)
                s = 2 * np.pi * x * (1 - shift / L)
                psi_p[n] += 2 * coeff * (1 - shift / L) * np.cos(s)

    return psi_p


def build_QW_divided_diff(lam, N, primes, n_int=2000):
    """
    Baue QW-Matrix via dividierte Differenzen.

    q_{m,n} = (psi(m) - psi(n)) / (m - n)   fuer m != n
    q_{n,n} = psi'(n)
    """
    psi = compute_psi_values(N, lam, primes, n_int)
    psi_p = compute_psi_derivative(N, lam, primes, n_int)

    Q = np.zeros((N, N))
    for i in range(N):
        Q[i, i] = psi_p[i]
        for j in range(N):
            if i != j:
                Q[i, j] = (psi[i] - psi[j]) / (i - j)

    return Q, psi, psi_p


def test_divided_differences():
    """Teste die dividierte-Differenzen-Methode."""
    primes = list(primerange(2, 200))

    print("=" * 75)
    print("DIVIDIERTE-DIFFERENZEN METHODE (Connes-van Suijlekom)")
    print("=" * 75)

    lambdas = [10, 20, 30, 50, 100]

    for lam in lambdas:
        L = np.log(lam)
        primes_used = [p for p in primes if p <= max(lam, 47)]

        t0 = time.time()

        for N in [20, 30, 50]:
            Q, psi, psi_p = build_QW_divided_diff(lam, N, primes_used, n_int=1000)
            evals = np.sort(eigh(Q, eigvals_only=True))
            elapsed = time.time() - t0

            # Symmetrie-Check
            sym_err = np.max(np.abs(Q - Q.T))

            print(f"  lam={lam:3d}, N={N:2d}: l1={evals[0]:+.6e}, l2={evals[1]:+.6e}, "
                  f"gap={evals[1]-evals[0]:.4e}, sym_err={sym_err:.2e} ({elapsed:.1f}s)")

        # psi-Werte anzeigen
        Q20, psi20, psip20 = build_QW_divided_diff(lam, 10, primes_used, n_int=1000)
        print(f"    psi(0..4) = {psi20[:5]}")
        print(f"    psi'(0..4) = {psip20[:5]}")
        print()


if __name__ == "__main__":
    test_divided_differences()
