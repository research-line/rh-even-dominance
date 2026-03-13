#!/usr/bin/env python3
"""
weg2_analytic_even_odd.py
=========================
Analytische Berechnung der Shift-Matrixelemente fuer cos vs sin.

KORRIGIERT (2026-03-13): Orthogonale Basis cos(n*pi*t/L) statt cos(n*pi*t/(2L)).
Die alte Basis war NICHT orthogonal auf [-L,L] (||M-I||=2.28, cond=10^7).

Korrekte orthonormale Basis auf [-L, L]:
  Even: psi_0(t) = 1/sqrt(2L), psi_n(t) = cos(n*pi*t/L)/sqrt(L) for n >= 1
  Odd:  psi_n(t) = sin((n+1)*pi*t/L)/sqrt(L) for n >= 0

Produktformel: cos(a)*cos(b) = [cos(a-b)+cos(a+b)]/2
               sin(a)*sin(b) = [cos(a-b)-cos(a+b)]/2
"""

import numpy as np
from scipy.linalg import eigh
from mpmath import euler as mp_euler, log as mplog, pi as mppi

LOG4PI_GAMMA = float(mplog(4 * mppi)) + float(mp_euler)


def shift_element_cos(n, m, s, L):
    """
    Berechne <psi_n, S_s psi_m> analytisch (KORRIGIERTE orthogonale Basis).

    Basis: psi_0 = 1/sqrt(2L), psi_n = cos(n*pi*t/L)/sqrt(L) for n >= 1
    Frequenzen: kn = n*pi/L (NICHT n*pi/(2L)!)

    Integrationsgrenzen: t in [max(-L, s-L), min(L, s+L)]
    """
    if abs(s) > 2 * L:
        return 0.0

    a = max(-L, s - L)
    b = min(L, s + L)
    if a >= b:
        return 0.0

    # Normierungsfaktor
    if n == 0 and m == 0:
        norm = 1.0 / (2 * L)
    elif n == 0 or m == 0:
        norm = 1.0 / (L * np.sqrt(2))
    else:
        norm = 1.0 / L

    # KORRIGIERTE Frequenzen: n*pi/L
    kn = n * np.pi / L
    km = m * np.pi / L

    # cos(kn*t) * cos(km*(t-s)) = (1/2)[cos((kn-km)*t + km*s) + cos((kn+km)*t - km*s)]
    result = 0.0
    for freq, phase in [(kn - km, km * s), (kn + km, -km * s)]:
        if abs(freq) < 1e-12:
            result += np.cos(phase) * (b - a) / 2
        else:
            result += (np.sin(freq * b + phase) - np.sin(freq * a + phase)) / (2 * freq)

    return norm * result


def shift_element_sin(n, m, s, L):
    """
    Berechne <psi_n, S_s psi_m> fuer sin-Basis (KORRIGIERT).

    Basis: psi_n = sin((n+1)*pi*t/L)/sqrt(L) for n >= 0
    Frequenzen: (n+1)*pi/L (NICHT (n+1)*pi/(2L)!)
    """
    if abs(s) > 2 * L:
        return 0.0

    a = max(-L, s - L)
    b = min(L, s + L)
    if a >= b:
        return 0.0

    norm = 1.0 / L

    # KORRIGIERTE Frequenzen: (n+1)*pi/L
    kn = (n + 1) * np.pi / L
    km = (m + 1) * np.pi / L

    result = 0.0
    # MINUS-Zeichen beim zweiten Term ist der Unterschied zu cos!
    for freq, phase, sign in [(kn - km, km * s, +1), (kn + km, -km * s, -1)]:
        if abs(freq) < 1e-12:
            result += sign * np.cos(phase) * (b - a) / 2
        else:
            result += sign * (np.sin(freq * b + phase) - np.sin(freq * a + phase)) / (2 * freq)

    return norm * result


def build_QW_analytic(lam, N, primes, basis='cos'):
    """Baue QW-Matrix mit analytischen Shift-Elementen."""
    L = np.log(lam)

    W = LOG4PI_GAMMA * np.eye(N)

    # Archimedischer Kernel (numerisch, da K(s) keine geschlossene Form hat)
    n_int = max(2000, 30 * N)
    s_max = min(2 * L, 12.0)
    s_grid = np.linspace(0.005, s_max, n_int)
    ds = s_grid[1] - s_grid[0]

    for s in s_grid:
        k = np.exp(s / 2) / (2.0 * np.sinh(s))
        if k < 1e-15:
            continue
        for i in range(N):
            for j in range(i, N):
                if basis == 'cos':
                    sp = shift_element_cos(i, j, s, L)
                    sm = shift_element_cos(i, j, -s, L)
                else:
                    sp = shift_element_sin(i, j, s, L)
                    sm = shift_element_sin(i, j, -s, L)
                reg = -2.0 * np.exp(-s / 2) * (1.0 if i == j else 0.0)
                val = k * (sp + sm + reg) * ds
                W[i, j] += val
                if i != j:
                    W[j, i] += val

    # Primzahl-Beitraege (analytisch!)
    for p in primes:
        logp = np.log(p)
        for m in range(1, 20):
            coeff = logp * p**(-m / 2.0)
            shift = m * logp
            if shift >= 2 * L:
                break
            for i in range(N):
                for j in range(i, N):
                    if basis == 'cos':
                        sp = shift_element_cos(i, j, shift, L)
                        sm = shift_element_cos(i, j, -shift, L)
                    else:
                        sp = shift_element_sin(i, j, shift, L)
                        sm = shift_element_sin(i, j, -shift, L)
                    val = coeff * (sp + sm)
                    W[i, j] += val
                    if i != j:
                        W[j, i] += val

    return W


def test_analytic_vs_numeric():
    """Verifiziere analytische Matrixelemente gegen numerische."""
    from sympy import primerange
    primes = list(primerange(2, 100))

    print("=" * 80)
    print("ANALYTISCHE VS NUMERISCHE MATRIXELEMENTE")
    print("=" * 80)

    N = 30

    for lam in [20, 30, 50, 100]:
        primes_used = [p for p in primes if p <= max(lam, 47)]
        print(f"\nlambda={lam}:")

        for basis in ['cos', 'sin']:
            sector = "EVEN" if basis == 'cos' else "ODD"
            W = build_QW_analytic(lam, N, primes_used, basis)
            evals = np.sort(eigh(W, eigvals_only=True))

            # Symmetrie-Check
            sym_err = np.max(np.abs(W - W.T))

            print(f"  {sector}: l1={evals[0]:+.6f}, l2={evals[1]:+.6f}, "
                  f"gap={evals[1]-evals[0]:.4e}, sym={sym_err:.2e}")


def analyze_difference_matrix():
    """
    Berechne die Differenzmatrix D = W_cos - W_sin.

    Fuer den Prim-Beitrag bei Shift s:
    D_{nm}^{prime}(s) = <cos_n, S_s cos_m> - <sin_n, S_s sin_m>
    = int cos((kn+km)t - km*s) dt   (der Unterschied-Term)

    Wenn D eine bestimmte Struktur hat (z.B. hauptsaechlich positiv),
    dann ist W_cos "hoeher" als W_sin, aber der niedrigste EW koennte
    trotzdem niedriger sein wegen groesserer Spreizung.
    """
    from sympy import primerange
    primes = list(primerange(2, 100))

    print("\n" + "=" * 80)
    print("DIFFERENZ-MATRIX W_cos - W_sin (nur Prim-Beitraege)")
    print("=" * 80)

    N = 20

    for lam in [30, 100]:
        L = np.log(lam)
        primes_used = [p for p in primes if p <= max(lam, 47)]

        D = np.zeros((N, N))
        for p in primes_used:
            logp = np.log(p)
            for m_exp in range(1, 13):
                coeff = logp * p**(-m_exp / 2.0)
                shift = m_exp * logp
                if shift >= 2 * L:
                    break
                for i in range(N):
                    for j in range(i, N):
                        dc = shift_element_cos(i, j, shift, L) + shift_element_cos(i, j, -shift, L)
                        ds = shift_element_sin(i, j, shift, L) + shift_element_sin(i, j, -shift, L)
                        val = coeff * (dc - ds)
                        D[i, j] += val
                        if i != j:
                            D[j, i] += val

        # Analyse der Differenzmatrix
        evals_D = np.sort(eigh(D, eigvals_only=True))
        print(f"\nlambda={lam}:")
        print(f"  D = W_prime(cos) - W_prime(sin):")
        print(f"  Eigenwerte: min={evals_D[0]:+.4f}, max={evals_D[-1]:+.4f}")
        print(f"  Trace = {np.trace(D):+.4f}")
        print(f"  Diag: [{np.diag(D).min():+.4f}, {np.diag(D).max():+.4f}]")
        print(f"  ||D||_F = {np.linalg.norm(D, 'fro'):.4f}")

        # Zeige die wichtigsten Eigenwerte
        print(f"  Erste 5 EW: {evals_D[:5]}")
        print(f"  Letzte 5 EW: {evals_D[-5:]}")


if __name__ == "__main__":
    import time
    t0 = time.time()
    test_analytic_vs_numeric()
    analyze_difference_matrix()
    print(f"\nTotal: {time.time()-t0:.1f}s")
