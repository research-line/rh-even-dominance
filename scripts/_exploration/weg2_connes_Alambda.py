#!/usr/bin/env python3
"""
weg2_connes_Alambda.py
======================
Connes' Operator A_lambda mit den KORREKTEN Formeln aus arXiv:2602.04022v1.

Konstruktion:
  QW_lambda(f,f) = <A_lambda f | f>  auf L^2([lambda^{-1}, lambda], du/u)

  Die Quadratform: QW(phi) = W_R(psi) + sum_p W_p(psi)
  wobei psi(v) = integral phi(u) phi(uv) du/u  (Faltung auf R_+*)

  Lokale Beitraege:
    W_p(f) = (log p) sum_{m=1}^M p^{-m/2} (f(p^m) + f(p^{-m}))   (Gl. 9)
    W_R(f) = (log 4pi + gamma) f(1) + integral-Term                (Gl. 10)

  Weils Aequivalenz: RH <==> sum_v W_v(g * g*) <= 0 fuer alle g

Tests:
  1. Operator A_lambda fuer verschiedene lambda bauen
  2. Spektrum: Einfachheit des kleinsten EW?
  3. Eigenfunktion: Ist sie gerade?
  4. Nullstellen der FT der Eigenfunktion: Sind sie reell?
  5. Konvergenz: Approximieren die Nullstellen die Zeta-Nullstellen?
"""

import numpy as np
from mpmath import mp, im, zetazero, euler as mp_euler, log as mplog, pi as mppi
from scipy.integrate import quad
from scipy.linalg import eigh

mp.dps = 30

# ===========================================================================
# Connes' Explizitformel-Terme
# ===========================================================================

LOG4PI_GAMMA = float(mplog(4 * mppi)) + float(mp_euler)  # log(4pi) + gamma

def W_p_matrix(p, N_basis, lam, M_terms=10):
    """
    Primstellen-Beitrag W_p zur quadratischen Form.
    W_p(f) = (log p) sum_{m=1}^M p^{-m/2} (f(p^m) + f(p^{-m}))

    Auf die Faltung psi = phi * phi* angewandt:
    W_p(psi) = (log p) sum_m p^{-m/2} * (psi(p^m) + psi(p^{-m}))

    psi(v) = integral phi(u) phi(uv) du/u = <phi, L_v phi> (inneres Produkt)
    wobei L_v phi(u) = phi(u/v) die Translation auf R_+* ist.

    Also: W_p(psi) = (log p) sum_m p^{-m/2} * (<phi, L_{p^m} phi> + <phi, L_{p^{-m}} phi>)

    In Matrixform: W_p = (log p) sum_m p^{-m/2} * (S_{p^m} + S_{p^{-m}})
    wobei (S_v)_{ij} = <phi_i, L_v phi_j> = integral phi_i(u) phi_j(u/v) du/u
    """
    logp = np.log(p)

    # Diskretisierung: Basis auf [lambda^{-1}, lambda] mit du/u
    # Umparametrisierung: t = log(u), t in [-log(lambda), log(lambda)]
    L = np.log(lam)
    n_quad = 500
    t_grid = np.linspace(-L, L, n_quad)
    dt = t_grid[1] - t_grid[0]

    # Cosinus-Basis auf [-L, L] (gerade Funktionen)
    def basis(n, t):
        if n == 0:
            return 1.0 / np.sqrt(2 * L)
        return np.cos(n * np.pi * t / (2 * L)) / np.sqrt(L)

    # Basis-Werte auf Grid
    phi_grid = np.zeros((N_basis, n_quad))
    for n in range(N_basis):
        for i, t in enumerate(t_grid):
            phi_grid[n, i] = basis(n, t)

    W = np.zeros((N_basis, N_basis))

    for m in range(1, M_terms + 1):
        coeff = logp * p**(-m / 2.0)
        shift = m * logp  # log(p^m)

        for sign in [1, -1]:
            s = sign * shift
            # <phi_i, L_{p^{sign*m}} phi_j> = integral phi_i(t) phi_j(t - s) dt
            # phi_j(t - s) ist phi_j verschoben um s

            # Fuer den verschobenen Basis-Vektor
            phi_shifted = np.zeros((N_basis, n_quad))
            for n in range(N_basis):
                for i, t in enumerate(t_grid):
                    t_shifted = t - s
                    if abs(t_shifted) <= L:
                        phi_shifted[n, i] = basis(n, t_shifted)
                    # sonst 0 (ausserhalb Support)

            # Overlap-Matrix: S_{ij} = integral phi_i(t) phi_j(t-s) dt
            S = phi_grid @ phi_shifted.T * dt

            W += coeff * S

    return W

def W_arch_matrix(N_basis, lam):
    """
    Archimedischer Beitrag W_R zur quadratischen Form.
    W_R(f) = (log 4pi + gamma) f(1) + integral_1^inf (f(x)+f(x^{-1})-2x^{-1/2}f(1)) * x^{1/2}/(x-x^{-1}) d*x

    Auf psi = phi * phi* angewandt, wird dies zur Bilinearform.
    """
    L = np.log(lam)
    n_quad = 500
    t_grid = np.linspace(-L, L, n_quad)
    dt = t_grid[1] - t_grid[0]

    def basis(n, t):
        if n == 0:
            return 1.0 / np.sqrt(2 * L)
        return np.cos(n * np.pi * t / (2 * L)) / np.sqrt(L)

    # Basis-Werte
    phi_grid = np.zeros((N_basis, n_quad))
    for n in range(N_basis):
        for i, t in enumerate(t_grid):
            phi_grid[n, i] = basis(n, t)

    # Term 1: (log 4pi + gamma) * psi(1)
    # psi(1) = integral phi(u)^2 du/u = integral phi(t)^2 dt = <phi, phi>
    # Also: Term 1 = (log 4pi + gamma) * sum_ij c_i c_j delta_{ij} = (log4pi+gamma) * ||phi||^2
    # In Matrixform: Term1_{ij} = (log 4pi + gamma) * delta_{ij} ... NEIN
    # psi(v) = integral phi(u) phi(uv) du/u, also psi(1) = ||phi||^2_{L^2(du/u)}
    # Term 1 Beitrag zur QF: (log4pi + gamma) * ||phi||^2
    # In Matrixform: (log4pi + gamma) * I (Identitaet, weil Basis orthonormal bzgl du/u ~ dt)

    W = LOG4PI_GAMMA * np.eye(N_basis)

    # Term 2: Integral ueber den Kern
    # Das Integral wird zur Bilinearform ueber phi
    # psi(v) = integral phi(u) phi(uv) du/u
    # f(x) = psi(x) in Connes' Notation, also x = v = e^s
    # Integral = integral_1^inf (...) x^{1/2}/(x - x^{-1}) dx/x
    # = integral_0^inf (...) e^{s/2} / (e^s - e^{-s}) ds  (s = log x)
    # = integral_0^inf (...) 1/(2*sinh(s)) ds

    # Diskretisierung des Integrals
    n_int = 300
    s_grid = np.linspace(0.01, 6.0, n_int)  # s = log(x), x in [1, e^6]
    ds = s_grid[1] - s_grid[0]

    for idx_s, s in enumerate(s_grid):
        # Kernel-Faktor: 1 / (2*sinh(s))
        kernel = 1.0 / (2.0 * np.sinh(s))

        # psi(e^s) + psi(e^{-s}) - 2*e^{-s/2}*psi(1)
        # psi(e^s) = <phi(t), phi(t-s)>
        # In Matrixform: Overlap S_s

        # Overlap fuer shift +s
        phi_plus = np.zeros((N_basis, n_quad))
        phi_minus = np.zeros((N_basis, n_quad))
        for n in range(N_basis):
            for i, t in enumerate(t_grid):
                t_plus = t - s
                t_minus = t + s
                if abs(t_plus) <= L:
                    phi_plus[n, i] = basis(n, t_plus)
                if abs(t_minus) <= L:
                    phi_minus[n, i] = basis(n, t_minus)

        S_plus = phi_grid @ phi_plus.T * dt   # <phi_i, L_{e^s} phi_j>
        S_minus = phi_grid @ phi_minus.T * dt  # <phi_i, L_{e^{-s}} phi_j>

        # psi(1) = <phi_i, phi_j> = delta_{ij}
        contrib = (S_plus + S_minus - 2.0 * np.exp(-s/2) * np.eye(N_basis)) * kernel * ds
        W += contrib

    return W

# ===========================================================================
# Vollstaendiger Operator A_lambda
# ===========================================================================

def build_A_lambda(lam, N_basis, primes, M_terms=10):
    """
    Baue A_lambda = W_arch - sum_p W_p  (Weil-Positivitaet: Negativ-Summe).

    Connes' Konvention: sum_v W_v(g*g*) <= 0 unter RH
    Also: Q_W = -sum_v W_v = W_arch_negiert + sum_p W_p ... NEIN

    Korrekt: Die explizite Formel gibt
    f-hat(i/2) + f-hat(-i/2) - sum_rho f-hat(rho) = sum_v W_v(f)

    Fuer f = g*g* (Faltung): sum_rho |g-hat(rho)|^2 = Polterme - sum_v W_v(g*g*)
    Da sum_rho >= 0, brauchen wir: Polterme >= sum_v W_v

    Die Quadratform auf die der Operator A_lambda wirkt ist:
    QW_lambda(phi) = sum_v W_v(phi * phi*)  (wobei phi Traeger in [lam^{-1}, lam])
    """
    print(f"    Baue W_arch (N={N_basis}, lambda={lam:.2f})...", end=" ", flush=True)
    W_arch = W_arch_matrix(N_basis, lam)
    print("OK")

    W_prime_total = np.zeros((N_basis, N_basis))
    for p in primes:
        W_prime_total += W_p_matrix(p, N_basis, lam, M_terms)

    # Die Weil-Quadratform: QW = W_R + sum_p W_p  (Vorzeichen wie in Connes)
    # RH <==> QW(g*g*) <= 0 fuer alle g mit g-hat(+-i/2) = 0
    # D.h. A_lambda hat Spektrum <= 0 (auf dem richtigen Unterraum)

    # Wir berechnen: QW = W_arch + W_prime
    QW = W_arch + W_prime_total

    return QW, W_arch, W_prime_total

# ===========================================================================
# Nullstellen der Fouriertransformierten
# ===========================================================================

def find_ft_zeros(evec, N_basis, lam, n_grid=5000):
    """Fouriertransformierte der Eigenfunktion und deren Nullstellen."""
    L = np.log(lam)

    def basis(n, t):
        if n == 0:
            return 1.0 / np.sqrt(2 * L)
        return np.cos(n * np.pi * t / (2 * L)) / np.sqrt(L)

    # Eigenfunktion auf feinem Grid
    t_fine = np.linspace(-L, L, n_grid)
    f_vals = np.zeros(n_grid)
    for n in range(N_basis):
        for i, t in enumerate(t_fine):
            f_vals[i] += evec[n] * basis(n, t)

    # Fouriertransformierte: F(xi) = integral f(t) e^{-i*xi*t} dt
    # Fuer reelles f mit geradem Support: F(xi) = 2 * integral_0^L f(t) cos(xi*t) dt
    xi_grid = np.linspace(0, 120, 3000)
    dt = t_fine[1] - t_fine[0]

    F_vals = np.zeros(len(xi_grid))
    for j, xi in enumerate(xi_grid):
        F_vals[j] = np.sum(f_vals * np.cos(xi * t_fine)) * dt

    # Nullstellen
    zeros = []
    for j in range(len(F_vals) - 1):
        if F_vals[j] * F_vals[j+1] < 0:
            xi0 = xi_grid[j] - F_vals[j] * (xi_grid[j+1] - xi_grid[j]) / (F_vals[j+1] - F_vals[j])
            zeros.append(xi0)

    return np.array(zeros), xi_grid, F_vals

# ===========================================================================
# HAUPTTEST
# ===========================================================================

if __name__ == "__main__":
    print("=" * 75)
    print("WEG 2: CONNES' OPERATOR A_lambda")
    print("  Korrekte Konstruktion nach arXiv:2602.04022v1")
    print("=" * 75)

    # Echte Zeta-Nullstellen als Referenz
    print("\n  Lade Zeta-Nullstellen...")
    gammas = []
    for k in range(1, 31):
        gammas.append(float(im(zetazero(k))))
    gammas = np.array(gammas)
    print(f"  gamma_1 = {gammas[0]:.6f}, ..., gamma_30 = {gammas[-1]:.6f}")

    # Primzahlen
    from sympy import primerange
    all_primes = list(primerange(2, 100))

    # Test fuer verschiedene lambda
    lambdas = [3, 5, 8, 13]
    N_BASIS = 20

    for lam in lambdas:
        print(f"\n{'='*75}")
        print(f"  lambda = {lam}, L = 2*log(lambda) = {2*np.log(lam):.4f}")
        print(f"  Support: [1/{lam}, {lam}] auf R_+*, [-{np.log(lam):.3f}, {np.log(lam):.3f}] auf R")
        print(f"  Primzahlen p^m <= lambda: ", end="")

        # Welche Primzahl-Potenzen fallen in den Support?
        active_primes = [p for p in all_primes if p <= lam**2]
        print(f"{len(active_primes)} Primzahlen <= lambda^2={lam**2}")

        primes_used = all_primes[:min(15, len(all_primes))]  # max 15 Primzahlen
        print(f"  Verwende {len(primes_used)} Primzahlen: {primes_used}")

        QW, W_arch, W_prime = build_A_lambda(lam, N_BASIS, primes_used, M_terms=8)

        # Spektrum
        evals, evecs = eigh(QW)
        print(f"\n  Spektrum von QW_lambda:")
        print(f"    Eigenwerte: {np.array2string(evals[:8], precision=6, separator=', ')}")
        print(f"    ... {np.array2string(evals[-3:], precision=6, separator=', ')}")
        print(f"    n_negativ = {np.sum(evals < -1e-10)}, n_positiv = {np.sum(evals > 1e-10)}")
        print(f"    lambda_min = {evals[0]:+.8e}")
        print(f"    lambda_2   = {evals[1]:+.8e}")
        print(f"    Luecke     = {evals[1] - evals[0]:.8e}")

        # Ist der kleinste EW einfach?
        gap_ratio = (evals[1] - evals[0]) / (evals[-1] - evals[0]) if evals[-1] != evals[0] else 0
        print(f"    Relative Luecke = {gap_ratio:.6f}")

        # Symmetrie der Eigenfunktion (gerade?)
        v0 = evecs[:, 0]
        # Gerade Basis -> alle Koeffizienten sollten "gerade" sein
        # Die Cosinus-Basis ist automatisch gerade!
        print(f"    EV_0 Koeffizienten: {np.array2string(v0[:6], precision=4, separator=', ')}")

        # Nullstellen der Fouriertransformierten
        zeros, xi_grid, F_vals = find_ft_zeros(v0, N_BASIS, lam)
        print(f"\n  Nullstellen der FT von EV_0:")
        print(f"    Anzahl: {len(zeros)}")
        if len(zeros) > 0:
            print(f"    Erste 5: {np.array2string(zeros[:5], precision=6, separator=', ')}")

            # Vergleich mit Zeta-Nullstellen
            for k in range(min(3, len(gammas))):
                if len(zeros) > 0:
                    dists = np.abs(zeros - gammas[k])
                    best = np.min(dists)
                    best_idx = np.argmin(dists)
                    print(f"    gamma_{k+1} = {gammas[k]:.6f}, "
                          f"naechste FT-Nullstelle = {zeros[best_idx]:.6f}, "
                          f"Delta = {best:.6f}")

    # FAZIT
    print(f"\n{'='*75}")
    print(f"FAZIT: CONNES' OPERATOR A_lambda")
    print(f"{'='*75}")
    print(f"""
  PRUEFE:
  1. Ist QW_lambda negativ semidefinit? (Weil: <= 0 unter RH)
  2. Ist der kleinste (negativste) EW einfach?
  3. Hat die zugehoerige Eigenfunktion eine FT mit reellen Nullstellen?
  4. Approximieren diese Nullstellen die Zeta-Nullstellen?

  BEACHTE: Die Implementierung ist noch APPROXIMATIV:
  - Endliche Quadratur (n=500 Punkte)
  - Endliche Primzahl-Summe (M_terms=8)
  - Endliche Basis (N=20)
  Die Ergebnisse muessen mit feinerer Diskretisierung validiert werden.
""")
