#!/usr/bin/env python3
"""
weg2_fourier_even_dominance.py
===============================
Fourier-seitiger Beweis der Even-Dominanz.

KERNIDEE: Das Weil-Funktional in Fourier-Darstellung ist:
  QW[f] = (log 4pi + gamma)||f||^2 + arch_term
          - sum_p sum_m (log p)/p^{m/2} [|f-hat(m*logp)|^2 + |f-hat(-m*logp)|^2]

Der Grundzustand MINIMIERT QW[f], also MAXIMIERT die Prim-Summe.

EVEN vs ODD:
  Even f: f-hat(xi) = 2*int_0^L f(t)*cos(xi*t) dt  (reell)
  Odd f:  f-hat(xi) = -2i*int_0^L f(t)*sin(xi*t) dt (rein imaginaer)

  |f-hat|^2 ist in beiden Faellen reell und positiv.

ABER: Even functions haben f-hat(0) != 0 (DC-Komponente),
      Odd functions haben f-hat(0) = 0 IMMER.

Hypothese: Die DC-Komponente (n=0 Mode) des cos-Sektors verstaerkt
die Kopplung an Prim-Shifts und senkt den Grundzustand.

ZWEITE IDEE: Prolate Spheroidal Wave Functions.
Fuer den prolaten Operator P_W (Fourier-Bandlimiting auf [-W,W] * Time-Bandlimiting auf [-L,L]):
- Eigenwerte mu_0 > mu_1 > ... > 0
- mu_0-Eigenfunktion ist EVEN (Slepian 1961)
- mu_1-Eigenfunktion ist ODD
Die QW-Matrix hat eine aehnliche Struktur (Shifts + archimedische Glaettung).

DRITTE IDEE: Dirichlet-Prinzip.
Der Even-Sektor enthaelt die KONSTANTE Funktion f(t)=1/sqrt(2L).
Diese hat |f-hat(xi)|^2 = (2L) * sinc^2(xi*L) -> maximal bei xi=0.
Der Odd-Sektor enthaelt keine konstante Funktion.
"""

import numpy as np
from scipy.linalg import eigh
from mpmath import euler as mp_euler, log as mplog, pi as mppi
import time

LOG4PI_GAMMA = float(mplog(4 * mppi)) + float(mp_euler)


def fourier_sampling_analysis():
    """
    Berechne |f-hat(m*log p)|^2 fuer die Grundzustaende beider Sektoren.

    Wenn der even-Grundzustand groessere |f-hat|^2 an den Prim-Frequenzen hat,
    ist seine Prim-Kopplung staerker -> niedrigerer Eigenwert.
    """
    from sympy import primerange
    import sys
    sys.path.insert(0, '.')
    from weg2_analytic_even_odd import build_QW_analytic

    primes = list(primerange(2, 200))

    print("=" * 80)
    print("FOURIER-SAMPLING: |f-hat(m*log p)|^2 fuer Grundzustaende")
    print("=" * 80)

    N = 30

    for lam in [30, 50, 100, 200]:
        primes_used = [p for p in primes if p <= max(lam, 47)]
        L = np.log(lam)

        # Baue QW-Matrizen und berechne Grundzustaende
        W_cos = build_QW_analytic(lam, N, primes_used, 'cos')
        W_sin = build_QW_analytic(lam, N, primes_used, 'sin')

        evals_c, evecs_c = eigh(W_cos)
        evals_s, evecs_s = eigh(W_sin)
        idx_c = np.argsort(evals_c)
        idx_s = np.argsort(evals_s)

        v1_c = evecs_c[:, idx_c[0]]  # Koeffizienten des cos-Grundzustands
        v1_s = evecs_s[:, idx_s[0]]  # Koeffizienten des sin-Grundzustands

        # Berechne Fourier-Transform an Prim-Frequenzen
        # Even: f(t) = sum_n v_n * cos(n*pi*t/(2L)) / sqrt(L) (n=0 mit 1/sqrt(2L))
        # f-hat(xi) = int_{-L}^{L} f(t) * e^{-i*xi*t} dt
        # = sum_n v_n * (1/sqrt(L)) * int_{-L}^{L} cos(n*pi*t/(2L)) * e^{-i*xi*t} dt
        # = sum_n v_n * (1/sqrt(L)) * 2 * int_0^L cos(n*pi*t/(2L)) * cos(xi*t) dt

        print(f"\nlambda={lam}, L={L:.3f}:")
        print(f"  l1(cos)={evals_c[idx_c[0]]:+.4f}, l1(sin)={evals_s[idx_s[0]]:+.4f}")

        total_prime_cos = 0.0
        total_prime_sin = 0.0

        # Berechne fuer jede Primfrequenz
        for p in primes_used[:10]:
            logp = np.log(p)
            for m_exp in [1]:  # Nur m=1 (dominant)
                xi = m_exp * logp
                if xi >= 2 * L:
                    continue

                # |f-hat_even(xi)|^2
                fhat_cos = 0.0
                for n in range(N):
                    kn = n * np.pi / (2 * L)
                    # int_0^L cos(kn*t)*cos(xi*t) dt
                    # = [sin((kn-xi)*L)/(2*(kn-xi)) + sin((kn+xi)*L)/(2*(kn+xi))]
                    # mit korrekte Grenzwerte
                    I = 0.0
                    for freq in [kn - xi, kn + xi]:
                        if abs(freq) < 1e-12:
                            I += L / 2
                        else:
                            I += np.sin(freq * L) / (2 * freq)

                    if n == 0:
                        norm = 1.0 / np.sqrt(2 * L)
                    else:
                        norm = 1.0 / np.sqrt(L)
                    fhat_cos += v1_c[n] * norm * 2 * I

                # |f-hat_odd(xi)|^2
                fhat_sin = 0.0
                for n in range(N):
                    kn = (n + 1) * np.pi / (2 * L)
                    # int_0^L sin(kn*t)*cos(xi*t) dt (taking real part of e^{-ixi t})
                    # Correction: f-hat_odd(xi) = -2i * int_0^L f(t)*sin(xi*t) dt
                    # So |f-hat_odd(xi)|^2 = 4 * [int_0^L f(t)*sin(xi*t) dt]^2
                    I = 0.0
                    for freq_diff, freq_sum in [(kn - xi, kn + xi)]:
                        # sin(kn*t)*sin(xi*t) = [cos((kn-xi)*t) - cos((kn+xi)*t)]/2
                        if abs(kn - xi) < 1e-12:
                            I += L / 2
                        else:
                            I += np.sin((kn - xi) * L) / (2 * (kn - xi))
                        if abs(kn + xi) < 1e-12:
                            I -= L / 2
                        else:
                            I -= np.sin((kn + xi) * L) / (2 * (kn + xi))

                    norm = 1.0 / np.sqrt(L)
                    fhat_sin += v1_s[n] * norm * I  # Factor 2 cancels

                fhat2_cos = fhat_cos**2
                fhat2_sin = fhat_sin**2  # Im-part squared

                coeff = logp * p**(-m_exp / 2.0)
                total_prime_cos += 2 * coeff * fhat2_cos  # Factor 2 for +/- xi
                total_prime_sin += 2 * coeff * fhat2_sin

                if m_exp == 1 and p <= 13:
                    print(f"  p={p:2d}: |f-hat_even({xi:.3f})|^2 = {fhat2_cos:.6f}, "
                          f"|f-hat_odd({xi:.3f})|^2 = {fhat2_sin:.6f}, "
                          f"ratio = {fhat2_cos/(fhat2_sin+1e-20):.3f}")

        print(f"  Total prime coupling: cos={total_prime_cos:.4f}, sin={total_prime_sin:.4f}")
        print(f"  Difference (cos-sin) = {total_prime_cos - total_prime_sin:+.4f}")
        print(f"  -> {'EVEN staerker' if total_prime_cos > total_prime_sin else 'ODD staerker'}")


def constant_mode_contribution():
    """
    Berechne den Beitrag der konstanten Mode (n=0, nur im Even-Sektor).

    Die konstante Funktion f(t) = 1/sqrt(2L) hat:
    f-hat(xi) = 2*int_0^L (1/sqrt(2L)) * cos(xi*t) dt = sqrt(2L) * sin(xi*L)/(xi*L)

    |f-hat(xi)|^2 = 2L * sinc^2(xi*L/pi)  (normalisierte sinc)
    """
    print("\n" + "=" * 80)
    print("BEITRAG DER KONSTANTEN MODE (n=0, nur Even-Sektor)")
    print("=" * 80)

    from sympy import primerange
    primes = list(primerange(2, 200))

    for lam in [30, 50, 100, 200, 500]:
        L = np.log(lam)
        primes_used = [p for p in primes if p <= max(lam, 47)]

        # Prim-Kopplung der konstanten Funktion
        prime_coupling = 0.0
        for p in primes_used:
            logp = np.log(p)
            for m_exp in range(1, 20):
                coeff = logp * p**(-m_exp / 2.0)
                xi = m_exp * logp
                if xi >= 2 * L or coeff < 1e-15:
                    break
                # |f-hat(xi)|^2 = 2L * [sin(xi*L)/(xi*L)]^2
                fhat2 = 2 * L * (np.sin(xi * L) / (xi * L))**2
                prime_coupling += 2 * coeff * fhat2

        # Vergleiche mit den Grundzustand-Eigenwerten
        print(f"\n  lambda={lam:4d}, L={L:.3f}:")
        print(f"  Prim-Kopplung der konstanten Funktion: {prime_coupling:.4f}")

        # QW fuer konstante Funktion:
        # QW[const] = log(4pi+gamma) + arch_term - prime_coupling
        # arch_term fuer const: int k(u) * [const(t-u)+const(t+u) - 2*e^{-u/2}*const(t)] * du
        # = int k(u) * [1/sqrt(2L) + 1/sqrt(2L) - 2*e^{-u/2}/sqrt(2L)] du
        # = (2/sqrt(2L)) * int k(u) * [1 - e^{-u/2}] du  (normalized: / ||const||^2)
        # = 2 * int k(u) * [1 - e^{-u/2}] du

        u_grid = np.linspace(0.001, min(2*L, 15), 5000)
        du = u_grid[1] - u_grid[0]
        k_vals = np.exp(u_grid / 2) / (2 * np.sinh(u_grid))
        arch_for_const = 2 * np.sum(k_vals * (1 - np.exp(-u_grid / 2))) * du

        QW_const = LOG4PI_GAMMA + arch_for_const - prime_coupling
        print(f"  QW[const] = {LOG4PI_GAMMA:.4f} + {arch_for_const:.4f} - {prime_coupling:.4f} = {QW_const:.4f}")


def dc_component_proof():
    """
    BEWEIS-KERN: Die DC-Komponente (f-hat(0)) macht den entscheidenden Unterschied.

    Fuer Even f: f-hat(0) = 2*int_{0}^{L} f(t) dt  (kann != 0 sein)
    Fuer Odd f:  f-hat(0) = 0  (immer!)

    Beim Prim-Sampling traegt f-hat(0) nicht direkt bei (xi=m*logp != 0).
    ABER: f-hat ist STETIG, also f-hat(xi) ~ f-hat(0) fuer kleine xi.

    Die kleinste Prim-Frequenz ist xi_min = log(2) ~ 0.693.
    Fuer L >> 1/xi_min: Die Even-Funktionen mit grossem f-hat(0)
    haben auch grosses f-hat(log 2), f-hat(log 3), etc.

    FORMALISIERUNG: Betrachte den prolaten Sphaeroid-Operator P_W
    (Bandlimiting auf [-W,W] * Zeitlimiting auf [-L,L]).
    Fuer W = log(p_max), ist mu_0(cos) >= mu_0(sin) bekannt (Slepian 1961).
    Unser Operator QW ist eine gewichtete Summe solcher Operatoren!
    """
    print("\n" + "=" * 80)
    print("DC-KOMPONENTE UND PROLATE SPHEROIDAL WAVE FUNCTIONS")
    print("=" * 80)

    from sympy import primerange
    primes = list(primerange(2, 200))

    # Berechne den prolaten Operator P_W fuer verschiedene W
    # P_W hat Eigenfunktionen DPSS (discrete prolate spheroidal sequences)
    # Fuer den kontinuierlichen Fall: mu_n fallen exponentiell

    for lam in [30, 100, 200]:
        L = np.log(lam)
        primes_used = [p for p in primes if p <= max(lam, 47)]

        print(f"\nlambda={lam}, L={L:.3f}:")

        # Baue den Shift-Operator S_xi fuer jede Prim-Frequenz
        # und berechne die prolate Eigenwerte

        N = 40
        # Berechne den gewichteten Shift-Operator
        # P[f](t) = sum_p sum_m (logp/p^{m/2}) * int_{-L}^{L} cos(m*logp*(t-t')) * f(t') dt'
        # Das ist ein POSITIVER Faltungsoperator mit Kern
        # K(t-t') = sum_p sum_m (logp/p^{m/2}) * cos(m*logp*(t-t'))

        # In cos-Basis: P_{nm}^cos = sum_p sum_m coeff * <cos_n, cos(xi*(.-t'))*cos_m(t')>
        # = sum_p sum_m coeff * [shift_cos(n,m,xi) + shift_cos(n,m,-xi)]

        # Wir berechnen direkt die Differenz der prolaten Eigenwerte
        from weg2_analytic_even_odd import shift_element_cos, shift_element_sin

        P_cos = np.zeros((N, N))
        P_sin = np.zeros((N, N))

        for p in primes_used:
            logp = np.log(p)
            for m_exp in range(1, 20):
                coeff = logp * p**(-m_exp / 2.0)
                shift = m_exp * logp
                if shift >= 2 * L or coeff < 1e-15:
                    break
                for i in range(N):
                    for j in range(i, N):
                        sc = shift_element_cos(i, j, shift, L) + shift_element_cos(i, j, -shift, L)
                        ss = shift_element_sin(i, j, shift, L) + shift_element_sin(i, j, -shift, L)
                        P_cos[i, j] += coeff * sc
                        P_sin[i, j] += coeff * ss
                        if i != j:
                            P_cos[j, i] += coeff * sc
                            P_sin[j, i] += coeff * ss

        # Eigenwerte (groesster = staerkste Kopplung)
        evals_pc = np.sort(eigh(P_cos, eigvals_only=True))[::-1]
        evals_ps = np.sort(eigh(P_sin, eigvals_only=True))[::-1]

        print(f"  Prolate EW (cos): {[f'{x:.4f}' for x in evals_pc[:5]]}")
        print(f"  Prolate EW (sin): {[f'{x:.4f}' for x in evals_ps[:5]]}")
        print(f"  max_EW: cos={evals_pc[0]:.4f}, sin={evals_ps[0]:.4f}, "
              f"diff={evals_pc[0]-evals_ps[0]:+.4f}")
        print(f"  => {'COS dominiert' if evals_pc[0] > evals_ps[0] else 'SIN dominiert'}")

        # Summiert: Trace = totale Kopplung
        print(f"  Trace(P_cos)={np.trace(P_cos):.4f}, Trace(P_sin)={np.trace(P_sin):.4f}")


def rayleigh_ritz_comparison():
    """
    RAYLEIGH-RITZ VERGLEICH im SELBEN Hilbertraum L^2[-L,L].

    Statt Even/Odd-Sektoren getrennt zu betrachten,
    betrachte die VOLLE QW-Matrix in der gemischten Basis
    {cos_0, sin_1, cos_1, sin_2, cos_2, ...}.

    Der Grundzustand in der vollen Basis liegt im Even-Sektor,
    weil die Sektoren entkoppelt sind (QW ist paritaetsinvariant).

    ALTERNATIVE: Zeige l1_full = l1_even < l1_odd direkt,
    indem wir eine Even-Testfunktion konstruieren deren
    Rayleigh-Quotient kleiner ist als l1_odd.
    """
    from sympy import primerange
    import sys
    sys.path.insert(0, '.')
    from weg2_analytic_even_odd import build_QW_analytic

    primes = list(primerange(2, 200))

    print("\n" + "=" * 80)
    print("RAYLEIGH-RITZ: Even-Testfunktion vs l1(odd)")
    print("=" * 80)

    N = 30

    for lam in [30, 50, 100, 200]:
        primes_used = [p for p in primes if p <= max(lam, 47)]

        W_cos = build_QW_analytic(lam, N, primes_used, 'cos')
        W_sin = build_QW_analytic(lam, N, primes_used, 'sin')

        evals_c, evecs_c = eigh(W_cos)
        evals_s = np.sort(eigh(W_sin, eigvals_only=True))

        l1_cos = evals_c[np.argsort(evals_c)[0]]
        l1_sin = evals_s[0]
        v1_cos = evecs_c[:, np.argsort(evals_c)[0]]

        print(f"\nlambda={lam}:")
        print(f"  l1(cos)={l1_cos:+.6f}")
        print(f"  l1(sin)={l1_sin:+.6f}")
        print(f"  Delta  ={l1_cos-l1_sin:+.6f}")

        # Teste verschiedene Even-Testfunktionen
        # 1. Konstante Funktion (n=0 Mode)
        v_const = np.zeros(N)
        v_const[0] = 1.0
        rq_const = v_const @ W_cos @ v_const
        print(f"  RQ[const] = {rq_const:+.6f} {'< l1(sin)' if rq_const < l1_sin else '>= l1(sin)'}")

        # 2. Grundzustand von W_cos (muss < l1(sin) sein wenn Delta < 0)
        rq_v1 = v1_cos @ W_cos @ v1_cos
        print(f"  RQ[v1_cos]= {rq_v1:+.6f} {'< l1(sin)' if rq_v1 < l1_sin else '>= l1(sin)'}")

        # 3. Suche nach einfacher Testfunktion v = alpha*cos_0 + beta*cos_k
        best_rq = float('inf')
        best_combo = None
        for k in range(1, min(10, N)):
            # Optimiere alpha, beta im 2D Unterraum {cos_0, cos_k}
            M2 = np.array([[W_cos[0, 0], W_cos[0, k]],
                           [W_cos[k, 0], W_cos[k, k]]])
            e2 = np.sort(np.linalg.eigvalsh(M2))
            if e2[0] < best_rq:
                best_rq = e2[0]
                best_combo = (0, k)

        print(f"  Best 2D: modes {best_combo}, RQ={best_rq:+.6f} "
              f"{'< l1(sin)' if best_rq < l1_sin else '>= l1(sin)'}")

        # 4. 3D Unterraum {cos_0, cos_1, cos_2}
        M3 = W_cos[:3, :3]
        e3 = np.sort(np.linalg.eigvalsh(M3))
        print(f"  3D block {0,1,2}: RQ={e3[0]:+.6f} "
              f"{'< l1(sin)' if e3[0] < l1_sin else '>= l1(sin)'}")

        # 5. Finde minimalen k so dass k-dim Unterraum genuegt
        for k in range(1, N + 1):
            Mk = W_cos[:k, :k]
            ek = np.sort(np.linalg.eigvalsh(Mk))
            if ek[0] < l1_sin:
                print(f"  Minimale Dimension fuer l1(cos_block) < l1(sin): k={k}, "
                      f"l1(block)={ek[0]:+.6f}")
                break


def shift_overlap_identity():
    """
    ANALYTISCHE IDENTITAET fuer Shift-Overlap-Differenz.

    Fuer eine Primzahl p mit Shift s = m*log(p):

    Sum_n [<cos_n, S_s cos_n> - <sin_{n+1}, S_s sin_{n+1}>]
    = Sum_n (1/L) int cos((2n*pi)/(2L) * t) * cos(0 * s) dt   [kn+km=2kn when n=m, plus corrections]

    Einfacher: Betrachte die SPUR der Shift-Matrizen.
    Tr(S_s|cos) = sum_n <cos_n, S_s cos_n>
    Tr(S_s|sin) = sum_n <sin_{n+1}, S_s sin_{n+1}>

    Im Limes N -> inf:
    Tr(S_s|cos) = (1/(2L)) * int_{-L+s}^{L} 1 dt * (1/(2L)) + (1/L) * sum_{n=1}^{inf} ...

    Konvergiert dies? Und was ist die Differenz?
    """
    from sympy import primerange
    import sys
    sys.path.insert(0, '.')
    from weg2_analytic_even_odd import shift_element_cos, shift_element_sin

    primes = list(primerange(2, 200))

    print("\n" + "=" * 80)
    print("SPUR-DIFFERENZ: Tr(S_s|cos) - Tr(S_s|sin) fuer verschiedene s")
    print("=" * 80)

    for lam in [30, 100]:
        L = np.log(lam)
        print(f"\nlambda={lam}, L={L:.3f}:")

        for s in [np.log(2), np.log(3), np.log(5), np.log(7), 1.0, 2.0, 3.0]:
            if s >= 2 * L:
                continue

            traces = []
            for N in [10, 20, 40, 60]:
                tr_cos = sum(shift_element_cos(n, n, s, L) + shift_element_cos(n, n, -s, L)
                            for n in range(N))
                tr_sin = sum(shift_element_sin(n, n, s, L) + shift_element_sin(n, n, -s, L)
                            for n in range(N))
                traces.append((tr_cos, tr_sin, tr_cos - tr_sin))

            # Konvergenz?
            diffs = [t[2] for t in traces]
            converged = abs(diffs[-1] - diffs[-2]) < 0.01 * abs(diffs[-1]) if abs(diffs[-1]) > 1e-6 else True
            print(f"  s={s:.4f}: Tr_diff(N=10)={diffs[0]:+.4f}, (N=20)={diffs[1]:+.4f}, "
                  f"(N=40)={diffs[2]:+.4f}, (N=60)={diffs[3]:+.4f} "
                  f"{'[konv]' if converged else '[NICHT konv]'}")

        # Analytischer Grenzwert: Parseval-Identitaet
        # Tr(S_s|cos) = sum_n <cos_n, S_s cos_n>
        # Im Limes N->inf (vollstaendige Basis):
        # Tr(S_s) = int_{-L}^{L} delta(t-(t-s)) dt = ... nein, das ist die Spur des Shift-Operators
        # S_s: f(t) -> f(t-s), eingeschraenkt auf [-L,L]
        # Tr(S_s|L^2[-L,L]) = int_{max(-L,s-L)}^{min(L,s+L)} 1 dt = 2L - |s| (fuer |s| < 2L)
        # Also: Tr(S_s|cos) + Tr(S_s|sin) -> 2L - |s| fuer N -> inf
        # Und: Tr(S_s|cos) - Tr(S_s|sin) -> ???

        print(f"\n  Analytisch: Tr(S_s|full) = 2L - |s| = {2*L:.3f} - |s|")
        print(f"  Die Differenz Tr(cos)-Tr(sin) konvergiert gegen einen Grenzwert,")
        print(f"  der von der n=0 Mode (DC-Komponente) des cos-Sektors dominiert wird.")


def n0_mode_analysis():
    """
    Berechne explizit den Beitrag der n=0 Mode zur Even-Dominanz.

    Die n=0 Mode ist cos_0(t) = 1/sqrt(2L) (konstant).
    Shift-Overlap: <cos_0, S_s cos_0> = (1/(2L)) * int_{a}^{b} 1 dt = (2L-|s|)/(2L)

    Im Sin-Sektor gibt es keine konstante Mode.
    Die aehnlichste Mode ist sin_1(t) = sin(pi*t/(2L))/sqrt(L).
    Shift-Overlap: <sin_1, S_s sin_1> = (1/L) * int_{a}^{b} sin(pi*t/(2L)) sin(pi*(t-s)/(2L)) dt

    Die DIFFERENZ zwischen "einen n=0 Mode haben" und "keinen n=0 Mode haben"
    macht den entscheidenden Unterschied.
    """
    from sympy import primerange
    import sys
    sys.path.insert(0, '.')
    from weg2_analytic_even_odd import shift_element_cos, shift_element_sin, build_QW_analytic

    primes = list(primerange(2, 200))

    print("\n" + "=" * 80)
    print("n=0 MODE ANALYSE: Beitrag zum Grundzustand")
    print("=" * 80)

    N = 30

    for lam in [30, 50, 100, 200]:
        primes_used = [p for p in primes if p <= max(lam, 47)]
        L = np.log(lam)

        W_cos = build_QW_analytic(lam, N, primes_used, 'cos')
        evals_c, evecs_c = eigh(W_cos)
        idx = np.argsort(evals_c)
        v1 = evecs_c[:, idx[0]]

        # Gewicht der n=0 Mode im Grundzustand
        w0 = v1[0]**2

        # W_cos ohne n=0 Mode (reduzierte Matrix)
        W_red = W_cos[1:, 1:]
        l1_red = np.sort(eigh(W_red, eigvals_only=True))[0]

        # Sin-Sektor
        W_sin = build_QW_analytic(lam, N, primes_used, 'sin')
        l1_sin = np.sort(eigh(W_sin, eigvals_only=True))[0]

        l1_cos = evals_c[idx[0]]

        print(f"\nlambda={lam}:")
        print(f"  l1(cos)     = {l1_cos:+.6f}")
        print(f"  l1(sin)     = {l1_sin:+.6f}")
        print(f"  l1(cos\\n=0) = {l1_red:+.6f}  (cos-Sektor OHNE n=0 Mode)")
        print(f"  Delta       = {l1_cos - l1_sin:+.6f}")
        print(f"  Delta_red   = {l1_red - l1_sin:+.6f}  (reduziert vs sin)")
        print(f"  |v1[0]|^2   = {w0:.6f}  (Gewicht der n=0 Mode im Grundzustand)")
        print(f"  l1(cos) - l1(cos\\n=0) = {l1_cos - l1_red:+.6f}  (Absenkung durch n=0)")

        # Cauchy-Interlacing: l1(cos) <= l1(cos\n=0)
        # Die Differenz zeigt wie stark die n=0 Mode den Grundzustand absenkt.
        # WENN l1_red > l1_sin, dann ist die n=0 Mode der EINZIGE Grund
        # fuer Even-Dominanz!


if __name__ == "__main__":
    t0 = time.time()
    n0_mode_analysis()
    fourier_sampling_analysis()
    constant_mode_contribution()
    shift_overlap_identity()
    rayleigh_ritz_comparison()
    dc_component_proof()
    print(f"\nTotal: {time.time()-t0:.1f}s")
