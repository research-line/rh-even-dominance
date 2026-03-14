#!/usr/bin/env python3
"""
weg2_connes_QW.py
=================
Weg 2: Connes' Weil-Quadratische Form Q_W

Konstruktion von Q_W auf ZWEI Wegen:
  (A) "Nullstellen-Seite": Q_{nm} = Σ_k φ_n(γ_k) · φ_m(γ_k)   [Gram-Matrix, trivial PSD]
  (B) "Primzahl-Seite":    Q_{nm} = Q_arch + Q_pole - Q_primes  [Explizite Formel]

Kern-Einsichten (aus Connes 2025, via User-Analyse):
  1. Q_W >= 0 (PSD) mit grossem Kern (Dim 50-100)
  2. Nullstellen-Approximationen liegen im Kern
  3. Spektralluecke >= log 5 fuer Dirichlet-Charaktere
  4. Orthogonalitaet Kern/Range = No-Coordination in Operatorform
"""

import numpy as np
from mpmath import mp, mpf, im, zetazero, digamma, log, pi, euler
import sys

mp.dps = 25

# ===========================================================================
# 1. Zeta-Nullstellen laden
# ===========================================================================

def load_zeros(K):
    """Lade die ersten K Nullstellen (Imaginaerteile gamma_k)."""
    gammas = []
    for k in range(1, K + 1):
        g = float(im(zetazero(k)))
        gammas.append(g)
    return np.array(gammas)

# ===========================================================================
# 2. Basis-Funktionen: Kosinus auf [0, T]
# ===========================================================================

def cosine_basis(n, t, T):
    """φ_n(t) = sqrt(2/T) * cos(n*pi*t/T), n >= 1; φ_0(t) = 1/sqrt(T)."""
    if n == 0:
        return 1.0 / np.sqrt(T)
    return np.sqrt(2.0 / T) * np.cos(n * np.pi * t / T)

# ===========================================================================
# 3. Q_W von der Nullstellen-Seite (Gram-Matrix)
# ===========================================================================

def QW_from_zeros(gammas, N_basis, T):
    """
    Q_{nm} = Σ_k φ_n(γ_k) * φ_m(γ_k)
    Dies ist eine Gram-Matrix => automatisch PSD.
    """
    K = len(gammas)
    # Evaluationsmatrix: V_{kn} = φ_n(γ_k)
    V = np.zeros((K, N_basis))
    for k in range(K):
        for n in range(N_basis):
            V[k, n] = cosine_basis(n, gammas[k], T)
    # Q = V^T V
    Q = V.T @ V
    return Q, V

# ===========================================================================
# 4. Q_W von der Primzahl-Seite (Explizite Formel)
# ===========================================================================

def phi_hat_cos(n, xi, T):
    """
    Fourier-Cosinus-Transformierte der Basis-Funktion:
    ĥ_n(xi) = ∫_0^T φ_n(t) cos(xi*t) dt

    Fuer φ_n(t) = sqrt(2/T) cos(n*pi*t/T):
    ĥ_n(xi) = sqrt(2/T) * ∫_0^T cos(n*pi*t/T) cos(xi*t) dt
    = sqrt(2/T) * T/2 * [sinc((n*pi/T - xi)*T/(2*pi)) + sinc((n*pi/T + xi)*T/(2*pi))]
    ... besser: direkt berechnen
    """
    omega_n = n * np.pi / T
    if n == 0:
        # φ_0 = 1/sqrt(T)
        if abs(xi) < 1e-15:
            return T / np.sqrt(T)  # = sqrt(T)
        return np.sin(xi * T) / (xi * np.sqrt(T))

    # φ_n = sqrt(2/T) cos(omega_n * t)
    c = np.sqrt(2.0 / T)
    # ∫_0^T cos(omega_n * t) cos(xi * t) dt
    # = 0.5 * ∫_0^T [cos((omega_n-xi)t) + cos((omega_n+xi)t)] dt
    def sinc_term(w):
        if abs(w) < 1e-15:
            return T
        return np.sin(w * T) / w

    integral = 0.5 * (sinc_term(omega_n - xi) + sinc_term(omega_n + xi))
    return c * integral

def QW_from_primes(N_basis, T, primes, M_terms=5):
    """
    Q_W von der Primzahl-Seite der Weil-Explizitformel.

    Bilineare Version der Explizitformel:
    Σ_γ φ_n(γ) φ_m(γ) = Q_arch(n,m) + Q_pole(n,m) + Q_prime(n,m)

    Q_prime(n,m) = 2 Σ_p Σ_{k=1}^M (log p / p^{k/2}) * ĥ_n(k*log p) * ĥ_m(k*log p)

    Q_arch(n,m) = (1/pi) ∫_0^∞ φ_n(t) φ_m(t) * Φ(t) dt
    wobei Φ(t) = Re[ψ(1/4 + it/2)] + log(4π)/2 + γ/2

    Q_pole(n,m) beinhaltet Beitraege vom trivialen Nullstellen und Pol.
    """
    Q_prime = np.zeros((N_basis, N_basis))
    Q_arch = np.zeros((N_basis, N_basis))
    Q_pole = np.zeros((N_basis, N_basis))

    # --- Prime contribution ---
    for p in primes:
        logp = np.log(p)
        for k in range(1, M_terms + 1):
            coeff = 2.0 * logp / p**(k / 2.0)
            xi = k * logp
            for n in range(N_basis):
                h_n = phi_hat_cos(n, xi, T)
                for m in range(n, N_basis):
                    h_m = phi_hat_cos(m, xi, T)
                    val = coeff * h_n * h_m
                    Q_prime[n, m] += val
                    if m != n:
                        Q_prime[m, n] += val

    # --- Archimedean contribution ---
    # Numerische Integration: Φ(t) = Re[ψ(1/4 + it/2)] + log(4π)/2 + γ_E/2
    # ψ ist die Digamma-Funktion
    log4pi_half = float(log(4 * pi)) / 2
    euler_half = float(euler) / 2

    n_quad = 500
    t_quad = np.linspace(0.01, T, n_quad)
    dt = t_quad[1] - t_quad[0]

    Phi_vals = np.zeros(n_quad)
    for i, t in enumerate(t_quad):
        psi_val = float(digamma(0.25 + 1j * t / 2).real)
        Phi_vals[i] = psi_val + log4pi_half + euler_half

    # Q_arch(n,m) = (1/π) ∫_0^T φ_n(t) φ_m(t) Φ(t) dt
    phi_vals = np.zeros((N_basis, n_quad))
    for n in range(N_basis):
        for i, t in enumerate(t_quad):
            phi_vals[n, i] = cosine_basis(n, t, T)

    for n in range(N_basis):
        for m in range(n, N_basis):
            integrand = phi_vals[n] * phi_vals[m] * Phi_vals
            val = np.sum(integrand) * dt / np.pi
            Q_arch[n, m] = val
            if m != n:
                Q_arch[m, n] = val

    # --- Pole contribution ---
    # Beitrag vom Pol bei s=1: ergibt einen Term mit h(i/2)
    # In unserer Parameterisierung: ρ=1/2+iγ, γ=i/2 gibt s=0 (trivialer Wert)
    # Einfachste Naeherung: Q_pole ≈ 0 (klein gegenueber arch und prime)
    # TODO: Praezisere Implementierung

    return Q_arch, Q_prime, Q_pole, Q_arch - Q_prime + Q_pole

# ===========================================================================
# 5. Vergleich und Analyse
# ===========================================================================

def compare_both_sides(K_zeros, N_basis, T, primes):
    """Vergleiche Nullstellen-Seite mit Primzahl-Seite."""
    print(f"\n{'='*70}")
    print(f"VERGLEICH: Nullstellen-Seite vs. Primzahl-Seite")
    print(f"  K_zeros={K_zeros}, N_basis={N_basis}, T={T:.1f}")
    print(f"  Primzahlen: {primes}")
    print(f"{'='*70}")

    gammas = load_zeros(K_zeros)

    # Nullstellen-Seite
    Q_zeros, V = QW_from_zeros(gammas, N_basis, T)

    # Primzahl-Seite
    Q_arch, Q_prime, Q_pole, Q_formula = QW_from_primes(N_basis, T, primes)

    # Vergleich
    print(f"\n  Nullstellen-Seite:")
    evals_z = np.linalg.eigvalsh(Q_zeros)
    print(f"    Eigenwerte (min, max): {evals_z[0]:.6e}, {evals_z[-1]:.6e}")
    print(f"    Anzahl EW < 1e-10: {np.sum(evals_z < 1e-10)}")
    print(f"    Trace: {np.trace(Q_zeros):.6f}")
    print(f"    sum(Q): {np.sum(Q_zeros):.6f}")

    print(f"\n  Primzahl-Seite:")
    evals_f = np.linalg.eigvalsh(Q_formula)
    print(f"    Eigenwerte (min, max): {evals_f[0]:.6e}, {evals_f[-1]:.6e}")
    print(f"    Anzahl EW < 0: {np.sum(evals_f < 0)}")
    print(f"    Trace: {np.trace(Q_formula):.6f}")
    print(f"    sum(Q): {np.sum(Q_formula):.6f}")

    print(f"\n  Einzelbeitraege:")
    print(f"    Q_arch trace: {np.trace(Q_arch):.6f}")
    print(f"    Q_prime trace: {np.trace(Q_prime):.6f}")
    print(f"    Q_pole trace: {np.trace(Q_pole):.6f}")

    # Differenz
    diff = Q_zeros - Q_formula
    print(f"\n  Differenz (Nullst. - Formel):")
    print(f"    max |diff|: {np.max(np.abs(diff)):.6e}")
    print(f"    Frobenius: {np.linalg.norm(diff):.6e}")

    return Q_zeros, Q_formula, V, evals_z, evals_f

# ===========================================================================
# 6. Kern-Analyse (Connes' Kernstruktur)
# ===========================================================================

def kernel_analysis(Q, name, threshold=1e-6):
    """Analysiere den Kern einer PSD-Matrix."""
    print(f"\n{'='*70}")
    print(f"KERN-ANALYSE: {name}")
    print(f"{'='*70}")

    evals, evecs = np.linalg.eigh(Q)
    N = len(evals)

    n_kernel = np.sum(np.abs(evals) < threshold)
    n_pos = np.sum(evals > threshold)
    n_neg = np.sum(evals < -threshold)

    print(f"\n  Dimension: {N}")
    print(f"  Kern (|λ| < {threshold}): {n_kernel}")
    print(f"  Positiv (λ > {threshold}): {n_pos}")
    print(f"  Negativ (λ < -{threshold}): {n_neg}")

    print(f"\n  Eigenwerte (alle):")
    for j in range(N):
        marker = " <-- KERN" if abs(evals[j]) < threshold else ""
        print(f"    λ_{j:2d} = {evals[j]:+12.6e}{marker}")

    return evals, evecs, n_kernel

# ===========================================================================
# 7. Spektralluecke und No-Coordination
# ===========================================================================

def spectral_gap_test(gammas, V, Q, N_basis, T):
    """
    Teste die Spektralluecke: Dirichlet-Charaktere im positiven Raum?

    Statt echter Dirichlet-Charaktere: teste "quasi-periodische" Funktionen
    f(t) = cos(t * log p) die einem einzelnen Primfaktor entsprechen.
    """
    print(f"\n{'='*70}")
    print(f"SPEKTRALLUECKE UND NO-COORDINATION")
    print(f"{'='*70}")

    evals, evecs = np.linalg.eigh(Q)

    # Einige Testvektoren:
    primes_test = [2, 3, 5, 7, 11]

    print(f"\n  Rayleigh-Quotienten R(v) = v^T Q v / v^T v:")
    print(f"  {'Vektor':>20} | {'R(v)':>12} | {'Anteil Kern':>12} | {'Anteil Pos':>12}")
    print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    threshold = 1e-6
    kern_mask = np.abs(evals) < threshold
    pos_mask = evals > threshold

    for p in primes_test:
        # v(t) = cos(t * log p), projiziert auf Basis
        logp = np.log(p)
        v_coords = np.zeros(N_basis)
        for n in range(N_basis):
            # <φ_n, cos(t*log p)> = ∫_0^T φ_n(t) cos(t*log p) dt
            # = phi_hat_cos(n, logp, T)  (fast dasselbe)
            v_coords[n] = phi_hat_cos(n, logp, T)

        if np.linalg.norm(v_coords) < 1e-15:
            continue
        v_n = v_coords / np.linalg.norm(v_coords)

        R = v_n @ Q @ v_n

        # Projektion auf Eigenraeume
        proj = evecs.T @ v_n
        anteil_kern = np.sum(proj[kern_mask]**2) if np.any(kern_mask) else 0
        anteil_pos = np.sum(proj[pos_mask]**2) if np.any(pos_mask) else 0

        print(f"  {'cos(t*log '+str(p)+')':>20} | {R:+12.6e} | {anteil_kern:12.6f} | {anteil_pos:12.6f}")

    # Konstante Funktion
    v_const = np.zeros(N_basis)
    v_const[0] = 1.0  # φ_0 ist die (normierte) Konstante
    R_const = v_const @ Q @ v_const
    proj_const = evecs.T @ v_const
    ak = np.sum(proj_const[kern_mask]**2) if np.any(kern_mask) else 0
    ap = np.sum(proj_const[pos_mask]**2) if np.any(pos_mask) else 0
    print(f"  {'Konstante 1':>20} | {R_const:+12.6e} | {ak:12.6f} | {ap:12.6f}")

    # Zufaelliger Vektor
    rng = np.random.RandomState(42)
    v_rand = rng.randn(N_basis)
    v_rand /= np.linalg.norm(v_rand)
    R_rand = v_rand @ Q @ v_rand
    proj_rand = evecs.T @ v_rand
    ak_r = np.sum(proj_rand[kern_mask]**2) if np.any(kern_mask) else 0
    ap_r = np.sum(proj_rand[pos_mask]**2) if np.any(pos_mask) else 0
    print(f"  {'Zufaellig':>20} | {R_rand:+12.6e} | {ak_r:12.6f} | {ap_r:12.6f}")


# ===========================================================================
# 8. Nullstellen-Approximation aus dem Kern
# ===========================================================================

def zero_approximation(gammas, V, Q, N_basis, T):
    """
    Connes' Methode: Nullstellen als Extremstellen von Q_W.
    Die Eigenvektoren des Kerns liefern Funktionen, deren Nullstellen
    die Zeta-Nullstellen approximieren.
    """
    print(f"\n{'='*70}")
    print(f"NULLSTELLEN-APPROXIMATION AUS DEM KERN")
    print(f"{'='*70}")

    evals, evecs = np.linalg.eigh(Q)
    threshold = 1e-6

    # Kern-Eigenvektoren
    kern_idx = [j for j in range(len(evals)) if abs(evals[j]) < threshold]
    print(f"\n  Kern-Dimension: {len(kern_idx)}")

    if len(kern_idx) == 0:
        print("  Kein Kern vorhanden (alle Eigenwerte != 0).")
        print("  Das ist erwartet wenn N_basis > K_zeros (ueberdeterminiert).")
        return

    # Fuer jeden Kern-EV: Rekonstruiere die Funktion und finde ihre Nullstellen
    t_grid = np.linspace(0.1, T - 0.1, 5000)

    for idx in kern_idx[:3]:
        v = evecs[:, idx]
        # Rekonstruiere f(t) = Σ v_n φ_n(t)
        f_vals = np.zeros(len(t_grid))
        for n in range(N_basis):
            for i, t in enumerate(t_grid):
                f_vals[i] += v[n] * cosine_basis(n, t, T)

        # Nullstellen finden (Vorzeichenwechsel)
        zeros_found = []
        for i in range(len(f_vals) - 1):
            if f_vals[i] * f_vals[i+1] < 0:
                # Lineare Interpolation
                t0 = t_grid[i] - f_vals[i] * (t_grid[i+1] - t_grid[i]) / (f_vals[i+1] - f_vals[i])
                zeros_found.append(t0)

        print(f"\n  Kern-EV {idx} (λ={evals[idx]:+.2e}):")
        print(f"    {len(zeros_found)} Nullstellen gefunden")
        if zeros_found:
            # Vergleiche mit echten Nullstellen
            n_match = 0
            for z in zeros_found[:10]:
                dists = np.abs(gammas - z)
                best_match = np.argmin(dists)
                matched = dists[best_match] < 0.5
                if matched:
                    n_match += 1
                    print(f"    t={z:.4f} -> γ_{best_match+1}={gammas[best_match]:.4f} "
                          f"(Δ={dists[best_match]:.4f})")
                else:
                    print(f"    t={z:.4f} -> keine Uebereinstimmung (min Δ={dists[best_match]:.2f})")

# ===========================================================================
# HAUPTPROGRAMM
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("WEG 2: CONNES' WEIL-QUADRATISCHE FORM Q_W")
    print("=" * 70)

    # Parameter
    K_zeros = 30      # Anzahl Zeta-Nullstellen
    N_basis = 20      # Basis-Dimension
    T = 120.0         # Intervall [0, T]
    primes = [2, 3, 5, 7, 11, 13]

    print(f"\n  Parameter: K={K_zeros} Nullstellen, N={N_basis} Basis, T={T}, S={primes}")
    print(f"  Lade Nullstellen...")
    gammas = load_zeros(K_zeros)
    print(f"  Nullstellen geladen: γ_1={gammas[0]:.4f}, ..., γ_{K_zeros}={gammas[-1]:.4f}")

    # 1. Vergleich beide Seiten
    Q_zeros, Q_formula, V, ev_z, ev_f = compare_both_sides(K_zeros, N_basis, T, primes)

    # 2. Kern-Analyse (Nullstellen-Seite)
    evals_z, evecs_z, nk_z = kernel_analysis(Q_zeros, "Nullstellen-Seite Q_zeros")

    # 3. Kern-Analyse (Primzahl-Seite)
    evals_f, evecs_f, nk_f = kernel_analysis(Q_formula, "Primzahl-Seite Q_formula")

    # 4. Spektralluecke
    spectral_gap_test(gammas, V, Q_zeros, N_basis, T)

    # 5. Nullstellen-Approximation
    zero_approximation(gammas, V, Q_zeros, N_basis, T)

    # FAZIT
    print(f"\n{'='*70}")
    print(f"FAZIT")
    print(f"{'='*70}")
    print(f"""
  STRUKTUR DER WEIL-QUADRATISCHEN FORM:

  1. Q_zeros (aus Nullstellen): Gram-Matrix, automatisch PSD.
     Kern-Dimension = max(0, N_basis - K_zeros) = max(0, {N_basis}-{K_zeros}).

  2. Q_formula (aus Primzahlen): Q_arch - Q_prime.
     Stimmt Q_formula mit Q_zeros ueberein?
     Falls ja: Die Explizite Formel liefert eine PSD-Zerlegung
     OHNE Kenntnis der Nullstellen.

  3. CONNES-EINSICHT: Die Positivitaet Q_W >= 0 kommt aus der
     Analytizitaet von xi + Funktionalgleichung.
     Die Nullstellen liegen im Kern (nicht im positiven Raum).
     Die Spektralluecke im positiven Raum (>= log 5) kommt
     aus der arithmetischen Unabhaengigkeit der Primzahlen.

  4. UEBERSETZUNGSBRUECKE:
     xi analytisch + FE  =>  Nullstellen im Kern von Q_W
     Euler-Produkt        =>  Spektralluecke im Range
     No-Coordination     =>  Orthogonalitaet Kern/Range
     Zusammen:            =>  Q_W >= 0, und Nullstellen auf krit. Linie
""")
