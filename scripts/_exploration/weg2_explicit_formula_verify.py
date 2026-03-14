#!/usr/bin/env python3
"""
weg2_explicit_formula_verify.py
===============================
Verifiziere die Weil-Explizitformel numerisch, dann baue Q_W korrekt.

Methode:
  1. Berechne Sigma_gamma h(gamma) direkt aus bekannten Nullstellen
  2. Berechne die RECHTE SEITE der Explizitformel (Primzahlen + arch.)
  3. Finde die fehlenden Terme durch Vergleich
  4. Baue Q_W mit der verifizierten Formel

Guinand-Weil Explizitformel (Standard-Form, siehe Iwaniec-Kowalski 5.53):

  Fuer h gerade, analytisch im Streifen |Im(s)| < 1/2 + epsilon:

  Sigma_gamma h(gamma) = h(i/2) + h(-i/2)
                        + (1/2pi) integral h(t) Phi(t) dt
                        - 2 * Sigma_{n=1}^inf Lambda(n)/sqrt(n) * g(log n)

  wobei:
    g(x) = (1/2pi) integral h(t) e^{ixt} dt   (Fourier-Transformierte von h)
    Phi(t) = Re[Gamma'/Gamma(1/4 + it/2)] + log(pi)

  Alternativer Ausdruck fuer den archimedischen Term:
    (1/2pi) integral h(t) Phi(t) dt
    = integral_0^inf [h(0)(e^{x/2}+e^{-x/2})/(e^x-1) - 2h(0)/x
                      - (e^{-x/2}g(x) + e^{x/2}g(-x))] dx  +  h(0)*(gamma_E + log(4pi))
"""

import numpy as np
from mpmath import (mp, mpf, mpc, im, re, zetazero, digamma, loggamma,
                    log, pi, euler, exp, cos, sin, quad, inf, fsum, mangoldt)
import sys

mp.dps = 30

# ===========================================================================
# 1. Testfunktionen
# ===========================================================================

def gaussian_h(t, alpha=0.1):
    """h(t) = exp(-alpha * t^2), gerade Testfunktion."""
    return float(exp(-alpha * mpf(t)**2))

def gaussian_g(x, alpha=0.1):
    """Fourier-Transformierte von h: g(x) = (1/2pi) int h(t) e^{ixt} dt
    = sqrt(1/(4*pi*alpha)) * exp(-x^2 / (4*alpha))"""
    return float(1.0 / (2 * mp.sqrt(mp.pi * alpha)) * exp(-mpf(x)**2 / (4 * alpha)))

def bump_h(t, T0=50):
    """h(t) = cos(pi*t/(2*T0))^2 fuer |t| < T0, sonst 0."""
    t = abs(t)
    if t >= T0:
        return 0.0
    return float(cos(mp.pi * t / (2 * T0))**2)

# ===========================================================================
# 2. Linke Seite: Summe ueber Nullstellen
# ===========================================================================

def left_side(h_func, K_zeros=200, **kwargs):
    """Sigma_gamma h(gamma_k) fuer die ersten K Nullstellen."""
    total = mpf(0)
    for k in range(1, K_zeros + 1):
        gamma_k = im(zetazero(k))
        # Beide Vorzeichen: gamma und -gamma (h ist gerade)
        total += 2 * h_func(float(gamma_k), **kwargs)
    return float(total)

# ===========================================================================
# 3. Rechte Seite: Explizitformel
# ===========================================================================

def right_side(h_func, g_func, N_primes=500, M_powers=20, **kwargs):
    """
    Rechte Seite der Guinand-Weil-Formel:

    R = h(i/2) + h(-i/2)
      + (1/2pi) int_{-inf}^{inf} h(t) Phi(t) dt
      - 2 * Sigma_{p^m} (log p / p^{m/2}) * g(m*log p)

    h(i/2) fuer Gauss: exp(-alpha * (i/2)^2) = exp(alpha/4)
    h(-i/2): dasselbe (h gerade)
    """
    # --- Pol-Beitrag: h(i/2) + h(-i/2) ---
    # Fuer Gauss: h(z) = exp(-alpha*z^2), analytische Fortsetzung
    alpha = kwargs.get('alpha', 0.1)
    # h(i/2) = exp(-alpha*(i/2)^2) = exp(-alpha*(-1/4)) = exp(alpha/4)
    h_half = float(exp(alpha / 4))  # h(i/2) = h(-i/2) fuer gerade h
    pole_term = 2 * h_half

    # --- Archimedischer Term ---
    # (1/2pi) int h(t) Phi(t) dt
    # Phi(t) = Re[psi(1/4 + it/2)] + log(pi)
    # wobei psi = Gamma'/Gamma (Digamma)

    def arch_integrand(t):
        t = mpf(t)
        if abs(t) < 1e-15:
            # Phi(0) = Re[psi(1/4)] + log(pi)
            phi = re(digamma(mpf(0.25))) + log(pi)
        else:
            phi = re(digamma(mpf(0.25) + mpc(0, t/2))) + log(pi)
        return h_func(float(t), **kwargs) * float(phi)

    # Numerische Integration (symmetrisch, h gerade)
    n_quad = 2000
    t_max = 200.0  # weit genug fuer Gauss mit alpha=0.1
    t_vals = np.linspace(0.001, t_max, n_quad)
    dt = t_vals[1] - t_vals[0]

    arch_integral = 0.0
    for t in t_vals:
        arch_integral += arch_integrand(t) * dt
    arch_integral *= 2  # Symmetrie: int_{-inf}^{inf} = 2 * int_0^{inf}
    arch_term = arch_integral / (2 * float(pi))

    # --- Primsummen-Term ---
    # -2 * Sigma_{n>=2} Lambda(n)/sqrt(n) * g(log n)
    # Lambda(n) = log p falls n = p^m, sonst 0
    # Also: -2 * Sigma_p Sigma_m log(p)/p^{m/2} * g(m*log p)

    from sympy import primerange
    primes = list(primerange(2, 10000))[:N_primes]

    prime_sum = 0.0
    for p in primes:
        logp = float(log(mpf(p)))
        for m in range(1, M_powers + 1):
            coeff = logp / p**(m / 2.0)
            g_val = g_func(m * logp, **kwargs)
            prime_sum += coeff * g_val
            # Konvergenz-Check
            if abs(coeff * g_val) < 1e-20:
                break
    prime_term = -2 * prime_sum

    return pole_term, arch_term, prime_term, pole_term + arch_term + prime_term

# ===========================================================================
# 4. Vergleich und Diagnose
# ===========================================================================

def verify_explicit_formula():
    """Verifiziere die Explizitformel numerisch."""
    print(f"{'='*70}")
    print(f"VERIFIKATION DER WEIL-EXPLIZITFORMEL")
    print(f"{'='*70}")

    for alpha in [0.01, 0.05, 0.1, 0.5]:
        print(f"\n  --- Gauss-Testfunktion, alpha = {alpha} ---")

        # Linke Seite
        K = 100
        L = left_side(gaussian_h, K_zeros=K, alpha=alpha)

        # Rechte Seite
        pole, arch, prime, R = right_side(gaussian_h, gaussian_g, alpha=alpha)

        diff = L - R
        rel_err = abs(diff) / abs(L) if abs(L) > 1e-15 else abs(diff)

        print(f"  Linke Seite (K={K}): {L:.10f}")
        print(f"  Rechte Seite:")
        print(f"    Pol-Term:   {pole:+.10f}")
        print(f"    Arch-Term:  {arch:+.10f}")
        print(f"    Prim-Term:  {prime:+.10f}")
        print(f"    SUMME:      {R:.10f}")
        print(f"  Differenz:    {diff:+.10e}")
        print(f"  Rel. Fehler:  {rel_err:.6e}")

        if rel_err > 0.01:
            print(f"  ** WARNUNG: Grosser Fehler! Formel oder Numerik fehlerhaft. **")

    # Konvergenz der linken Seite
    print(f"\n  --- Konvergenz der Nullstellen-Summe (alpha=0.1) ---")
    for K in [10, 20, 50, 100, 200]:
        L = left_side(gaussian_h, K_zeros=K, alpha=0.1)
        print(f"  K={K:3d}: L = {L:.10f}")

# ===========================================================================
# 5. Bilineare Form Q_W aus verifizierter Formel
# ===========================================================================

def build_QW_verified(N_basis, T, primes, K_zeros=100):
    """
    Baue Q_W mit der VERIFIZIERTEN Explizitformel.

    Q_{nm} = Sigma_gamma phi_n(gamma) * phi_m(gamma)
           = [Pol] + [Arch] + [Prim]

    Fuer phi_n(t) = sqrt(2/T) cos(n*pi*t/T):
      phi_n ist gerade
      phi_hat_n(x) = Fourier-Transformierte

    Q^{zeros}_{nm} = Sigma_gamma phi_n(gamma) * phi_m(gamma)

    Q^{formula}_{nm} benoetigt die bilineare Explizitformel.
    Bilineare Version: ersetze h(t) durch phi_n(t)*phi_m(t),
    ABER phi_n*phi_m ist NICHT die gleiche Testfunktion wie h!

    Stattdessen: Q_{nm} = Sigma_gamma phi_n(gamma)*phi_m(gamma)
    wobei wir die LINKE Seite direkt berechnen (aus Nullstellen).

    Fuer die RECHTE Seite brauchen wir:
    Die Explizitformel fuer h_{nm}(t) = phi_n(t)*phi_m(t).
    Dann: g_{nm}(x) = FT[h_{nm}](x) = (phi_n * phi_m)^hat(x)
    = convolution phi_hat_n * phi_hat_m (Faltung der FTs)
    """
    from mpmath import im as mp_im

    print(f"\n{'='*70}")
    print(f"Q_W AUS VERIFIZIERTER FORMEL (N={N_basis}, T={T})")
    print(f"{'='*70}")

    # Lade Nullstellen
    gammas = []
    for k in range(1, K_zeros + 1):
        gammas.append(float(mp_im(zetazero(k))))
    gammas = np.array(gammas)

    def phi(n, t):
        if n == 0:
            return 1.0 / np.sqrt(T)
        return np.sqrt(2.0 / T) * np.cos(n * np.pi * t / T)

    # Q aus Nullstellen (Gram-Matrix, Referenz)
    V = np.zeros((K_zeros, N_basis))
    for k in range(K_zeros):
        for n in range(N_basis):
            V[k, n] = phi(n, gammas[k])
    # Beide Vorzeichen: gamma und -gamma
    # phi_n ist gerade (cos), also phi_n(-gamma) = phi_n(gamma)
    # Sigma_{gamma>0} [phi_n(gamma)*phi_m(gamma) + phi_n(-gamma)*phi_m(-gamma)]
    # = 2 * Sigma_{gamma>0} phi_n(gamma)*phi_m(gamma)
    Q_zeros = 2 * V.T @ V

    evals_z = np.linalg.eigvalsh(Q_zeros)
    print(f"\n  Q_zeros: min EW = {evals_z[0]:.6e}, max EW = {evals_z[-1]:.6e}")
    print(f"  trace = {np.trace(Q_zeros):.6f}")
    print(f"  Kern (|EW| < 1e-4): {np.sum(np.abs(evals_z) < 1e-4)}")

    # Q aus Explizitformel: h_{nm}(t) = phi_n(t) * phi_m(t)
    # Fuer die Primseite brauchen wir g_{nm}(x) = FT[h_{nm}]
    # h_{nm}(t) = phi_n(t) * phi_m(t)
    # Fuer n,m >= 1: (2/T) cos(n*pi*t/T) cos(m*pi*t/T)
    #   = (1/T) [cos((n-m)*pi*t/T) + cos((n+m)*pi*t/T)]

    # FT von cos(omega*t) auf [-T, T]: 2*sin(omega*T - xi*T) / ... (sinc-artig)
    # Besser: numerisch berechnen

    # Pol-Term: h_{nm}(i/2) = phi_n(i/2) * phi_m(i/2)
    # phi_n(i/2) = sqrt(2/T) * cos(n*pi*i/(2T)) = sqrt(2/T) * cosh(n*pi/(2T))
    Q_pole = np.zeros((N_basis, N_basis))
    for n in range(N_basis):
        if n == 0:
            phi_n_half = 1.0 / np.sqrt(T) * np.cosh(0)  # = 1/sqrt(T)
        else:
            phi_n_half = np.sqrt(2.0 / T) * np.cosh(n * np.pi / (2 * T))
        for m in range(n, N_basis):
            if m == 0:
                phi_m_half = 1.0 / np.sqrt(T)
            else:
                phi_m_half = np.sqrt(2.0 / T) * np.cosh(m * np.pi / (2 * T))
            # Pol: 2 * phi_n(i/2) * phi_m(i/2)
            val = 2 * phi_n_half * phi_m_half
            Q_pole[n, m] = val
            if m != n:
                Q_pole[m, n] = val

    print(f"\n  Q_pole: trace = {np.trace(Q_pole):.6f}")

    # Archimedischer Term: (1/2pi) * 2 * int_0^T phi_n(t) phi_m(t) Phi(t) dt
    # (Faktor 2 wegen Symmetrie t <-> -t, phi gerade)
    log_pi = float(log(pi))
    n_quad = 1000
    t_quad = np.linspace(0.1, min(T - 0.1, 200), n_quad)
    dt_q = t_quad[1] - t_quad[0]

    Phi_vals = np.zeros(n_quad)
    for i, t in enumerate(t_quad):
        Phi_vals[i] = float(re(digamma(mpf(0.25) + mpc(0, t/2)))) + log_pi

    phi_grid = np.zeros((N_basis, n_quad))
    for n in range(N_basis):
        for i, t in enumerate(t_quad):
            phi_grid[n, i] = phi(n, t)

    Q_arch = np.zeros((N_basis, N_basis))
    for n in range(N_basis):
        for m in range(n, N_basis):
            # (1/2pi) * 2 * integral phi_n * phi_m * Phi dt
            integrand = phi_grid[n] * phi_grid[m] * Phi_vals
            val = 2 * np.sum(integrand) * dt_q / (2 * float(pi))
            Q_arch[n, m] = val
            if m != n:
                Q_arch[m, n] = val

    print(f"  Q_arch: trace = {np.trace(Q_arch):.6f}")

    # Prim-Term: -2 * Sigma_p Sigma_m log(p)/p^{m/2} * g_{nm}(m*log p)
    # g_{nm}(x) = FT[phi_n * phi_m](x) = (1/2pi) int phi_n(t) phi_m(t) e^{ixt} dt
    # Fuer gerade h: g(x) = (1/pi) int_0^T h(t) cos(xt) dt

    # Berechne g_{nm}(xi) numerisch
    def g_nm(n_idx, m_idx, xi):
        """g_{nm}(xi) = (1/pi) int_0^T phi_n(t) phi_m(t) cos(xi*t) dt"""
        integrand = phi_grid[n_idx] * phi_grid[m_idx] * np.cos(xi * t_quad)
        return np.sum(integrand) * dt_q / float(pi)

    Q_prime = np.zeros((N_basis, N_basis))
    from sympy import primerange as pr
    prime_list = list(pr(2, 5000))[:len(primes) if isinstance(primes, list) else primes]

    for p in prime_list:
        logp = np.log(p)
        for k in range(1, 21):
            coeff = float(log(mpf(p))) / p**(k / 2.0)
            if coeff < 1e-20:
                break
            xi = k * logp
            for n in range(N_basis):
                for m in range(n, N_basis):
                    g_val = g_nm(n, m, xi)
                    val = -2 * coeff * g_val
                    Q_prime[n, m] += val
                    if m != n:
                        Q_prime[m, n] += val

    print(f"  Q_prime: trace = {np.trace(Q_prime):.6f}")

    Q_formula = Q_pole + Q_arch + Q_prime
    evals_f = np.linalg.eigvalsh(Q_formula)
    print(f"\n  Q_formula = Q_pole + Q_arch + Q_prime:")
    print(f"    min EW = {evals_f[0]:.6e}, max EW = {evals_f[-1]:.6e}")
    print(f"    trace = {np.trace(Q_formula):.6f}")
    print(f"    n_neg = {np.sum(evals_f < -1e-10)}")

    # Vergleich
    diff = np.linalg.norm(Q_zeros - Q_formula, 'fro')
    print(f"\n  Vergleich:")
    print(f"    ||Q_zeros - Q_formula|| = {diff:.6e}")
    print(f"    ||Q_zeros|| = {np.linalg.norm(Q_zeros, 'fro'):.6e}")
    print(f"    Rel. Fehler: {diff / np.linalg.norm(Q_zeros, 'fro'):.6e}")

    return Q_zeros, Q_formula, Q_pole, Q_arch, Q_prime

# ===========================================================================
# HAUPTPROGRAMM
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("WEG 2: EXPLIZITFORMEL-VERIFIKATION")
    print("=" * 70)

    # 1. Verifikation fuer Gauss-Testfunktion
    verify_explicit_formula()

    # 2. Q_W mit verifizierter Formel
    print(f"\n\n{'='*70}")
    print(f"Q_W MIT VERIFIZIERTER FORMEL")
    print(f"{'='*70}")

    N_BASIS = 10
    T = 120.0
    primes = [2, 3, 5, 7, 11, 13]

    Q_z, Q_f, Q_p, Q_a, Q_pr = build_QW_verified(N_BASIS, T, primes, K_zeros=50)

    # Kern-Analyse
    evals, evecs = np.linalg.eigh(Q_z)
    print(f"\n  Kern-Analyse von Q_zeros:")
    for j in range(min(10, len(evals))):
        marker = " <-- KERN" if abs(evals[j]) < 0.01 else ""
        print(f"    EW_{j}: {evals[j]:+.6e}{marker}")
