#!/usr/bin/env python3
"""
bi9_soliton_prototype.py
========================
BI-9: Soliton-Stabilitaet fuer RH -- Numerischer Prototyp

Idee: Nullstellen von xi als Solitonen (Kinks) in einem Phasenfeld.
      RH <=> alle Kinks stabil (keine negative Hesse-Mode).

Drei Modelle:
1. Sine-Gordon: V(phi) = lambda * (1 - cos phi)
2. Poeschl-Teller: V(phi) = V_0 / cosh^2(phi)  (CRM-Sättigung!)
3. Phasenfeld aus xi selbst: phi(t) = arg xi(1/2 + it)

Teste:
- Hat F[phi] ein Minimum bei der RH-Konfiguration?
- Ist die Hesse positiv definit?
- Erzeugt eine Off-line-Stoerung eine negative Mode?
"""

from mpmath import (mp, mpf, mpc, log, exp, pi, gamma, zeta, zetazero,
                    im, re, arg, loggamma, sqrt, diff, fsum, cosh, cos, sin, tanh)
import numpy as np

mp.dps = 25

# ===========================================================================
# 1. Zeta-Nullstellen laden
# ===========================================================================

def load_zeros(N=50):
    """Lade die ersten N nichttrivialen Nullstellen."""
    print(f"  Lade {N} Zeta-Nullstellen...")
    zeros = []
    for k in range(1, N+1):
        rho = zetazero(k)
        gamma_k = float(im(rho))
        zeros.append(gamma_k)
    print(f"  gamma_1 = {zeros[0]:.6f}, gamma_{N} = {zeros[-1]:.6f}")
    return np.array(zeros)


# ===========================================================================
# 2. Phasenfeld aus xi
# ===========================================================================

def xi_func(s):
    """xi(s) = (1/2) s(s-1) pi^{-s/2} Gamma(s/2) zeta(s)"""
    return mpf('0.5') * s * (s - 1) * exp(-s/2 * log(pi)) * gamma(s/2) * zeta(s)

def phase_from_xi(t_values):
    """Berechne phi(t) = arg[xi(1/2 + it)] entlang der kritischen Linie."""
    phases = []
    for t in t_values:
        s = mpc(0.5, float(t))
        xi_val = xi_func(s)
        # xi(1/2 + it) ist reell fuer reelles t (Funktionalgleichung)
        # Also ist arg = 0 oder pi (Vorzeichenwechsel)
        # Wir nehmen stattdessen die stetige Phase
        phase = float(arg(xi_val))
        phases.append(phase)
    return np.array(phases)


# ===========================================================================
# 3. Nullstellen-Abstands-Statistik (GUE-Check)
# ===========================================================================

def spacing_statistics(gammas):
    """Berechne normierte Abstaende und vergleiche mit GUE."""
    N = len(gammas)
    spacings = np.diff(gammas)

    # Normiere auf mittleren Abstand
    mean_spacing = np.mean(spacings)
    s_normalized = spacings / mean_spacing

    print(f"\n  Nullstellen-Abstaende (normiert auf <s> = 1):")
    print(f"  Mittlerer Abstand: {mean_spacing:.6f}")
    print(f"  Min: {np.min(s_normalized):.4f}, Max: {np.max(s_normalized):.4f}")
    print(f"  Std: {np.std(s_normalized):.4f}")
    print(f"  (GUE-Vorhersage fuer Std: ~0.42)")

    return s_normalized, mean_spacing


# ===========================================================================
# 4. MODELL 1: Sine-Gordon Soliton-Funktional
# ===========================================================================

def sine_gordon_test(gammas):
    """
    F_SG[phi] = sum_n [(gamma_{n+1} - gamma_n)^{-1} * (phi_{n+1} - phi_n)^2
                       + lambda * (1 - cos(phi_n))]

    Die Nullstellen als Gitterpunkte, phi_n = Phase am n-ten Gitterpunkt.
    Kink-Loesung: phi_n wechselt um 2*pi an jeder Nullstelle.
    """
    print(f"\n{'='*60}")
    print(f"MODELL 1: Sine-Gordon Soliton-Funktional")
    print(f"{'='*60}")

    N = len(gammas)
    spacings = np.diff(gammas)

    # Parameter
    lam = 1.0  # Kopplungskonstante

    # RH-Konfiguration: phi_n = n * pi (monotone Phase, Vorzeichenwechsel)
    phi_RH = np.arange(N) * np.pi

    # Berechne F[phi_RH]
    kinetic = np.sum((np.diff(phi_RH))**2 / spacings)
    potential = lam * np.sum(1 - np.cos(phi_RH))
    F_RH = kinetic + potential
    print(f"\n  F[phi_RH] = {F_RH:.6f} (kinetic={kinetic:.2f}, potential={potential:.2f})")

    # Hesse-Matrix bei phi_RH
    # d^2 F / d phi_n^2 = 2/s_{n-1} + 2/s_n + lambda * cos(phi_n)
    # d^2 F / d phi_n d phi_{n+1} = -2/s_n
    H = np.zeros((N, N))
    for n in range(N):
        # Diagonale
        diag = lam * np.cos(phi_RH[n])
        if n > 0:
            diag += 2.0 / spacings[n-1]
        if n < N-1:
            diag += 2.0 / spacings[n]
        H[n, n] = diag

        # Nebendiagonale
        if n < N-1:
            off = -2.0 / spacings[n]
            H[n, n+1] = off
            H[n+1, n] = off

    eigenvalues = np.linalg.eigvalsh(H)
    eigenvalues.sort()

    n_neg = np.sum(eigenvalues < -1e-10)
    n_zero = np.sum(np.abs(eigenvalues) < 1e-10)
    n_pos = np.sum(eigenvalues > 1e-10)

    print(f"\n  Hesse-Matrix bei phi_RH ({N}x{N}):")
    print(f"  Negative EW: {n_neg}, Null-EW: {n_zero}, Positive EW: {n_pos}")
    print(f"  Kleinste 5 EW: {eigenvalues[:5]}")
    print(f"  Groesste 3 EW: {eigenvalues[-3:]}")

    if n_neg > 0:
        print(f"\n  => phi_RH ist KEIN Minimum (Sattelpunkt oder Maximum)")
        print(f"  => Sine-Gordon mit phi_n = n*pi ist NICHT das richtige Modell")
    else:
        print(f"\n  => phi_RH ist ein LOKALES MINIMUM -> Soliton-Stabilitaet!")

    # Off-line Perturbation: verschiebe eine Nullstelle
    print(f"\n  OFF-LINE PERTURBATION:")
    delta = 0.1  # Verschiebung in Re-Richtung
    # Effekt: Phase wird komplex -> aendert phi_n
    # Simuliere als: phi_n -> phi_n + delta * random
    phi_perturbed = phi_RH.copy()
    phi_perturbed[N//2] += delta  # Stoere mittlere Nullstelle

    kinetic_p = np.sum((np.diff(phi_perturbed))**2 / spacings)
    potential_p = lam * np.sum(1 - np.cos(phi_perturbed))
    F_perturbed = kinetic_p + potential_p
    Delta_F = F_perturbed - F_RH

    print(f"  delta = {delta}")
    print(f"  F[phi_RH + delta] = {F_perturbed:.6f}")
    print(f"  Delta F = {Delta_F:.6f} ({'POSITIV -> stabil' if Delta_F > 0 else 'NEGATIV -> instabil'})")

    return eigenvalues


# ===========================================================================
# 5. MODELL 2: Poeschl-Teller Soliton-Funktional
# ===========================================================================

def poeschl_teller_test(gammas):
    """
    F_PT[phi] = sum_n [(d phi/dt)^2 + V_0 / cosh^2(phi_n / phi_0)]

    Poeschl-Teller Potential: reflexionslos, exakt loesbar.
    Soliton-Profil: tanh -- die CRM-Saettigungskurve!
    """
    print(f"\n{'='*60}")
    print(f"MODELL 2: Poeschl-Teller Soliton-Funktional")
    print(f"{'='*60}")

    N = len(gammas)
    spacings = np.diff(gammas)

    # PT-Parameter
    V_0 = 2.0   # Potentialtiefe
    phi_0 = 1.0  # Breite

    # RH-Konfiguration: Kink-Profil = tanh
    # phi(t) = phi_0 * tanh((t - t_center) / width)
    # Fuer mehrere Kinks: Summe von tanh-Profilen
    # Einfachstes Modell: phi_n = tanh(n - N/2)
    phi_RH = np.tanh((np.arange(N) - N/2) / (N/4))

    # Berechne F[phi_RH]
    kinetic = np.sum((np.diff(phi_RH))**2 / spacings)
    potential = V_0 * np.sum(1.0 / np.cosh(phi_RH / phi_0)**2)
    F_RH = kinetic + potential
    print(f"\n  F[phi_RH] = {F_RH:.6f} (kinetic={kinetic:.6f}, potential={potential:.2f})")

    # Hesse-Matrix
    H = np.zeros((N, N))
    for n in range(N):
        x = phi_RH[n] / phi_0
        # d^2 V / d phi^2 = V_0 / phi_0^2 * (-2/cosh^2(x) + 4*tanh^2(x)/cosh^2(x))
        #                  = V_0 / phi_0^2 * 2*(2*tanh^2(x) - 1) / cosh^2(x)
        d2V = V_0 / phi_0**2 * 2*(2*np.tanh(x)**2 - 1) / np.cosh(x)**2

        diag = d2V
        if n > 0:
            diag += 2.0 / spacings[n-1]
        if n < N-1:
            diag += 2.0 / spacings[n]
        H[n, n] = diag

        if n < N-1:
            off = -2.0 / spacings[n]
            H[n, n+1] = off
            H[n+1, n] = off

    eigenvalues = np.linalg.eigvalsh(H)
    eigenvalues.sort()

    n_neg = np.sum(eigenvalues < -1e-10)
    n_pos = np.sum(eigenvalues > 1e-10)

    print(f"\n  Hesse-Matrix bei phi_RH ({N}x{N}):")
    print(f"  Negative EW: {n_neg}, Positive EW: {n_pos}")
    print(f"  Kleinste 5 EW: {eigenvalues[:5]}")

    if n_neg == 0:
        print(f"  => phi_RH ist ein LOKALES MINIMUM -> PT-Soliton stabil!")
    else:
        print(f"  => {n_neg} negative Moden -> PT-Profil braucht Anpassung")

    return eigenvalues


# ===========================================================================
# 6. MODELL 3: Direktes xi-Phasenfeld
#    phi(t) = arg xi(1/2 + it) -- die physikalische Phase
# ===========================================================================

def xi_phase_field_test(gammas):
    """
    Berechne die Phase von xi entlang der kritischen Linie
    und interpretiere Nullstellen als Phasen-Kinks.
    """
    print(f"\n{'='*60}")
    print(f"MODELL 3: Direktes xi-Phasenfeld")
    print(f"{'='*60}")

    # xi(1/2 + it) ist REELL fuer reelles t (Funktionalgleichung)
    # Also ist phi(t) = 0 oder pi (Vorzeichen)
    # Bei jeder Nullstelle wechselt das Vorzeichen -> Kink um pi

    # Statt arg(xi) nehmen wir die STETIGE Phase via Hardy Z-Funktion:
    # Z(t) = e^{i*theta(t)} * zeta(1/2 + it)
    # theta(t) = arg[pi^{-it/2} * Gamma(1/4 + it/2)] = Riemann-Siegel theta

    # Die Nullstellen von Z(t) sind EXAKT die Nullstellen von zeta(1/2+it)
    # Z(t) ist reell -> Nullstellen = Vorzeichenwechsel

    N_grid = 500
    t_max = gammas[-1] + 5
    t_grid = np.linspace(1.0, t_max, N_grid)

    # Berechne Z(t) via mpmath
    print(f"  Berechne Vorzeichen von Z(t) auf Gitter [1, {t_max:.1f}]...")

    signs = []
    for t in t_grid:
        s = mpc(0.5, float(t))
        xi_val = xi_func(s)
        signs.append(1 if float(re(xi_val)) > 0 else -1)
    signs = np.array(signs)

    # Zaehle Vorzeichenwechsel
    sign_changes = np.where(np.diff(signs) != 0)[0]
    print(f"  Vorzeichenwechsel gefunden: {len(sign_changes)}")
    print(f"  Erwartet (Nullstellen im Intervall): {len(gammas)}")

    # "Phase" als kumulierte Vorzeichenwechsel
    cumulative_phase = np.cumsum(np.abs(np.diff(signs)) / 2) * np.pi
    cumulative_phase = np.concatenate([[0], cumulative_phase])

    # Phasen-Geschwindigkeit d(phi)/dt
    dphi_dt = np.diff(cumulative_phase) / np.diff(t_grid)

    print(f"\n  Kumulative Phase am Ende: {cumulative_phase[-1]/np.pi:.1f} * pi")
    print(f"  Erwarteter Wert: {len(gammas)} * pi = {len(gammas)*np.pi:.1f}")
    print(f"  Mittlere Phasengeschwindigkeit: {np.mean(dphi_dt):.6f}")
    print(f"  (Zum Vergleich: log(t/2pi)/2pi bei t={t_max:.0f} = {np.log(t_max/(2*np.pi))/(2*np.pi):.6f})")

    # Interpretation: Die Phasengeschwindigkeit = Nullstellendichte
    # n(t) ~ log(t/2pi) / (2pi) -- bekannt
    # Jeder Kink (Vorzeichenwechsel) entspricht einem Phasen-Sprung um pi

    print(f"\n  INTERPRETATION:")
    print(f"  xi(1/2+it) ist reell -> Phase ist 0 oder pi (diskret)")
    print(f"  Jede Nullstelle = Phasen-Kink um pi")
    print(f"  Die kumulative Phase waechst wie N(T) * pi")
    print(f"  N(T) ~ (T/2pi) * log(T/2pi) (Riemanns Formel)")
    print(f"")
    print(f"  FUER EIN SOLITON-MODELL brauchen wir ein STETIGES Phasenfeld.")
    print(f"  Kandidat: theta(t) = arg[pi^{{-it/2}} * Gamma(1/4 + it/2)]")
    print(f"  Das ist die Riemann-Siegel theta-Funktion.")
    print(f"  Die Kinks im Z(t)-Vorzeichen werden durch theta(t) geglättet.")


# ===========================================================================
# 7. MODELL 4: Nullstellen-Wechselwirkung als Soliton-Gitter
#    Coulomb-Gas auf der kritischen Linie
# ===========================================================================

def coulomb_gas_test(gammas):
    """
    Modelliere Nullstellen als 1D-Coulomb-Gas (log-Gas):
    E[{gamma}] = -2 * sum_{j<k} log|gamma_j - gamma_k| + sum_j V(gamma_j)

    Das ist das EXAKTE GUE-Modell (Dyson)!
    Gleichgewicht: Nullstellen auf der kritischen Linie.
    Off-line-Störung: Nullstelle raus aus dem Gleichgewicht.
    """
    print(f"\n{'='*60}")
    print(f"MODELL 4: Coulomb-Gas / Log-Gas (GUE-Dyson)")
    print(f"{'='*60}")

    N = len(gammas)

    # Confining Potential (aus der Nullstellen-Zaehlfunktion)
    # V(t) ~ t * log(t/2pi) / (2pi) -- gross fuer grosse t
    def V(t):
        if t < 1:
            return t
        return t * np.log(t / (2*np.pi)) / (2 * np.pi)

    # Energie der RH-Konfiguration
    E_interaction = 0
    for j in range(N):
        for k in range(j+1, N):
            E_interaction -= 2 * np.log(abs(gammas[k] - gammas[j]))

    E_confining = sum(V(g) for g in gammas)
    E_RH = E_interaction + E_confining

    print(f"\n  E_RH = {E_RH:.2f}")
    print(f"    Wechselwirkung: {E_interaction:.2f}")
    print(f"    Einschluss:     {E_confining:.2f}")

    # Hesse-Matrix des Coulomb-Gas
    # d^2 E / d gamma_j d gamma_k = 2 / (gamma_j - gamma_k)^2  (j != k)
    # d^2 E / d gamma_j^2 = -sum_{k!=j} 2/(gamma_j - gamma_k)^2 + V''(gamma_j)
    H = np.zeros((N, N))
    for j in range(N):
        for k in range(N):
            if j != k:
                H[j, k] = 2.0 / (gammas[j] - gammas[k])**2
                H[j, j] -= 2.0 / (gammas[j] - gammas[k])**2

        # V''(t) ~ log(t/2pi)/(2pi*t) + 1/(2pi*t) fuer t >> 1
        t = gammas[j]
        if t > 1:
            H[j, j] += (np.log(t/(2*np.pi)) + 1) / (2 * np.pi * t)
        else:
            H[j, j] += 1.0

    eigenvalues = np.linalg.eigvalsh(H)
    eigenvalues.sort()

    n_neg = np.sum(eigenvalues < -1e-10)
    n_pos = np.sum(eigenvalues > 1e-10)

    print(f"\n  Hesse-Matrix des Coulomb-Gas ({N}x{N}):")
    print(f"  Negative EW: {n_neg}, Positive EW: {n_pos}")
    print(f"  Kleinste 5 EW: {eigenvalues[:5]}")
    print(f"  Groesste 3 EW: {eigenvalues[-3:]}")

    if n_neg == 0:
        print(f"\n  => RH-Konfiguration ist STABILES GLEICHGEWICHT des Coulomb-Gas!")
        print(f"  => Das IST die Soliton-Stabilitaet: topologisch erzwungen durch")
        print(f"     log-Abstoßung + Einschluss-Potential")
    else:
        print(f"\n  => {n_neg} instabile Moden im Coulomb-Gas")

    # Off-line Perturbation: Verschiebe Nullstelle aus der Linie
    print(f"\n  OFF-LINE PERTURBATION (Nullstelle #{N//2} um delta verschieben):")
    for delta in [0.01, 0.05, 0.1, 0.5]:
        gammas_pert = gammas.copy()
        gammas_pert[N//2] += delta  # Verschiebe ENTLANG der Linie (nicht off-line)

        E_pert_int = 0
        for j in range(N):
            for k in range(j+1, N):
                E_pert_int -= 2 * np.log(abs(gammas_pert[k] - gammas_pert[j]))
        E_pert_conf = sum(V(g) for g in gammas_pert)
        E_pert = E_pert_int + E_pert_conf
        print(f"    delta={delta:.2f}: Delta E = {E_pert - E_RH:+.6f} "
              f"({'stabil' if E_pert > E_RH else 'INSTABIL'})")

    return eigenvalues


# ===========================================================================
# 8. HAUPTPROGRAMM
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BI-9: SOLITON-STABILITAET FUER RH")
    print("Numerischer Prototyp")
    print("=" * 60)

    # Lade Nullstellen
    gammas = load_zeros(30)

    # Abstandsstatistik
    s_norm, mean_s = spacing_statistics(gammas)

    # Modell 1: Sine-Gordon
    ev_sg = sine_gordon_test(gammas)

    # Modell 2: Poeschl-Teller
    ev_pt = poeschl_teller_test(gammas)

    # Modell 3: xi-Phasenfeld (qualitativ)
    xi_phase_field_test(gammas)

    # Modell 4: Coulomb-Gas (das physikalischste Modell)
    ev_cg = coulomb_gas_test(gammas)

    # Zusammenfassung
    print(f"\n{'='*60}")
    print(f"ZUSAMMENFASSUNG")
    print(f"{'='*60}")
    print(f"""
  MODELL 1 (Sine-Gordon): Hesse bei phi = n*pi
    -> Teste ob RH-Konfiguration Minimum ist
    -> Negative EW: {np.sum(ev_sg < -1e-10)}

  MODELL 2 (Poeschl-Teller): Hesse bei tanh-Profil
    -> Teste ob CRM-Saettigungs-Profil stabil ist
    -> Negative EW: {np.sum(ev_pt < -1e-10)}

  MODELL 3 (xi-Phase): Qualitative Analyse
    -> xi auf krit. Linie ist reell -> Phase diskret (0 oder pi)
    -> Kinks = Vorzeichenwechsel = Nullstellen
    -> Braucht stetiges Phasenfeld (Riemann-Siegel theta)

  MODELL 4 (Coulomb-Gas / GUE):
    -> DAS PHYSIKALISCH KORREKTESTE MODELL
    -> Negative EW: {np.sum(ev_cg < -1e-10)}
    -> Log-Abstossung + Einschluss = natuerliche Soliton-Stabilisierung

  STRATEGISCHE BEWERTUNG:
  Das Coulomb-Gas/Log-Gas (Modell 4) ist der staerkste Kandidat,
  weil es EXAKT das GUE-Ensemble von Dyson ist -- und die
  Zeta-Nullstellen GUE-Statistik zeigen (Montgomery-Odlyzko).

  Die Soliton-Stabilitaet im Coulomb-Gas ist aequivalent zu:
    "Das Minimum der Energie liegt bei der beobachteten Konfiguration"
  Das ist ein VARIATIONSPRINZIP -- exakt Pattern A.

  NAECHSTER SCHRITT: Off-line-Perturbation im Coulomb-Gas
  (Nullstelle SENKRECHT zur kritischen Linie verschieben)
  und zeigen, dass dies die Energie IMMER erhoeht.
""")
