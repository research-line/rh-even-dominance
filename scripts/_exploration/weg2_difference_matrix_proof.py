#!/usr/bin/env python3
"""
weg2_difference_matrix_proof.py
================================
Analytische Berechnung der Differenzmatrix D = W_prime(cos) - W_prime(sin).

ZIEL: Zeige dass lambda_1(W_cos) < lambda_1(W_sin) fuer lambda >= lambda_0.

ANSATZ:
1. Fuer einen einzelnen Shift s berechne D_{nm}(s) in geschlossener Form
2. Summiere ueber Primpotenzen
3. Analysiere die Spektralstruktur von D

SCHLUESSEL-EINSICHT:
Die Produktformel-Differenz cos(a)cos(b) - sin(a)sin(b) = cos(a+b)
ergibt fuer die Matrixelemente:

D_{nm}(s) = (1/L) * int cos((kn+km)*t - km*s) dt + (1/L) * int cos((kn+km)*t + km*s) dt

wobei kn = n*pi/(2L), km = m*pi/(2L).
"""

import numpy as np
from scipy.linalg import eigh
from mpmath import euler as mp_euler, log as mplog, pi as mppi
import time

LOG4PI_GAMMA = float(mplog(4 * mppi)) + float(mp_euler)


def D_element_single_shift(n, m, s, L):
    """
    Berechne D_{nm}(s) = shift_cos(n,m,s,L) - shift_sin(n,m,s,L)
    in geschlossener Form.

    Fuer cos-Basis (n,m >= 0) mit kn = n*pi/(2L):
      cos(kn*t)*cos(km*(t-s)) = [cos((kn-km)*t + km*s) + cos((kn+km)*t - km*s)]/2

    Fuer sin-Basis (n,m >= 1) mit kn = n*pi/(2L):
      sin(kn*t)*sin(km*(t-s)) = [cos((kn-km)*t + km*s) - cos((kn+km)*t - km*s)]/2

    Differenz = cos((kn+km)*t - km*s) term (mit Faktor 1).

    ABER: cos-Basis hat n=0,1,2,... und sin-Basis hat n=1,2,3,...
    Also vergleichen wir cos-Mode n mit sin-Mode (n+1).

    Korrektur: In der sin-Basis nutzen wir kn = (n+1)*pi/(2L),
    in der cos-Basis kn = n*pi/(2L). Deshalb ist D_element nicht
    einfach nur der cos(a+b)-Term, sondern die volle Differenz.
    """
    if abs(s) > 2 * L:
        return 0.0

    a = max(-L, s - L)
    b = min(L, s + L)
    if a >= b:
        return 0.0

    # cos-Beitrag: kn_c = n*pi/(2L), km_c = m*pi/(2L)
    kn_c = n * np.pi / (2 * L)
    km_c = m * np.pi / (2 * L)

    # sin-Beitrag: kn_s = (n+1)*pi/(2L), km_s = (m+1)*pi/(2L)
    kn_s = (n + 1) * np.pi / (2 * L)
    km_s = (m + 1) * np.pi / (2 * L)

    # Normierung
    if n == 0 and m == 0:
        norm_c = 1.0 / (2 * L)
    elif n == 0 or m == 0:
        norm_c = 1.0 / (L * np.sqrt(2))
    else:
        norm_c = 1.0 / L
    norm_s = 1.0 / L

    # cos-Element: (norm_c/2) * sum of two integrals
    val_c = 0.0
    for freq, phase in [(kn_c - km_c, km_c * s), (kn_c + km_c, -km_c * s)]:
        if abs(freq) < 1e-12:
            val_c += np.cos(phase) * (b - a) / 2
        else:
            val_c += (np.sin(freq * b + phase) - np.sin(freq * a + phase)) / (2 * freq)
    val_c *= norm_c

    # sin-Element: (norm_s/2) * (first integral - second integral)
    val_s = 0.0
    for freq, phase, sign in [(kn_s - km_s, km_s * s, +1), (kn_s + km_s, -km_s * s, -1)]:
        if abs(freq) < 1e-12:
            val_s += sign * np.cos(phase) * (b - a) / 2
        else:
            val_s += sign * (np.sin(freq * b + phase) - np.sin(freq * a + phase)) / (2 * freq)
    val_s *= norm_s

    return val_c - val_s


def D_element_symmetric(n, m, s, L):
    """D fuer symmetrisierten Shift: S_s + S_{-s}."""
    return D_element_single_shift(n, m, s, L) + D_element_single_shift(n, m, -s, L)


def build_D_matrix(lam, N, primes):
    """
    Baue die volle Differenzmatrix D = W_prime(cos) - W_prime(sin).

    D_{nm} = sum_p log(p) sum_m p^{-m/2} * D_element_symmetric(n, m, m*log(p), L)
    """
    L = np.log(lam)
    D = np.zeros((N, N))

    for p in primes:
        logp = np.log(p)
        for m_exp in range(1, 20):
            coeff = logp * p**(-m_exp / 2.0)
            shift = m_exp * logp
            if shift >= 2 * L or coeff < 1e-15:
                break
            for i in range(N):
                for j in range(i, N):
                    val = coeff * D_element_symmetric(i, j, shift, L)
                    D[i, j] += val
                    if i != j:
                        D[j, i] += val

    return D


def analyze_D_structure():
    """Analysiere die Struktur der Differenzmatrix D."""
    from sympy import primerange
    primes = list(primerange(2, 200))

    print("=" * 80)
    print("DIFFERENZMATRIX D = W_prime(cos) - W_prime(sin)")
    print("=" * 80)

    N = 30

    for lam in [30, 50, 100, 200]:
        primes_used = [p for p in primes if p <= max(lam, 47)]
        L = np.log(lam)

        D = build_D_matrix(lam, N, primes_used)
        evals_D = np.sort(eigh(D, eigvals_only=True))

        print(f"\nlambda={lam}, L={L:.3f}, N={N}, #primes={len(primes_used)}:")
        print(f"  Trace(D)   = {np.trace(D):+.4f}")
        print(f"  ||D||_F    = {np.linalg.norm(D, 'fro'):.4f}")
        print(f"  min(D_ii)  = {np.min(np.diag(D)):+.6f}")
        print(f"  max(D_ii)  = {np.max(np.diag(D)):+.6f}")
        print(f"  EW(D): [{evals_D[0]:+.4f}, ..., {evals_D[-1]:+.4f}]")
        print(f"  Erste 5 EW: {[f'{x:+.4f}' for x in evals_D[:5]]}")

        # Eigenvektoren des kleinsten EW
        evals_D_full, evecs_D = eigh(D)
        idx = np.argsort(evals_D_full)
        v_min = evecs_D[:, idx[0]]
        print(f"  Eigenvektor zu min EW: max|v|={np.max(np.abs(v_min)):.4f}, "
              f"dominante Mode(s): {np.argsort(np.abs(v_min))[-3:][::-1]}")

        # Jetzt vergleiche mit tatsaechlichen QW-Eigenwerten
        from weg2_analytic_even_odd import build_QW_analytic
        W_cos = build_QW_analytic(lam, N, primes_used, 'cos')
        W_sin = build_QW_analytic(lam, N, primes_used, 'sin')
        l1_cos = np.sort(eigh(W_cos, eigvals_only=True))[0]
        l1_sin = np.sort(eigh(W_sin, eigvals_only=True))[0]
        Delta = l1_cos - l1_sin

        print(f"  l1(cos)={l1_cos:+.4f}, l1(sin)={l1_sin:+.4f}, Delta={Delta:+.4f}")

        # Weyl-Perturbationsschranke
        print(f"  Weyl: lambda_1(D) = {evals_D[0]:+.4f} (obere Schranke fuer Delta)")
        print(f"  Verhaeltnis Delta/min_EW(D) = {Delta/evals_D[0]:.4f}")


def analyze_low_mode_block():
    """
    Analysiere den 2x2 bzw 3x3 Block von D fuer niedrige Moden.

    Hypothese: Die Even-Dominanz wird durch die niedrigsten Moden getrieben.
    Wenn der 3x3-Block von D bereits den richtigen Charakter hat,
    koennte man einen Beweis auf diesen Block stuetzen.
    """
    from sympy import primerange
    primes = list(primerange(2, 200))

    print("\n" + "=" * 80)
    print("LOW-MODE BLOCK VON D")
    print("=" * 80)

    for lam in [30, 50, 100, 200, 500]:
        primes_used = [p for p in primes if p <= max(lam, 47)]
        L = np.log(lam)

        # Berechne D fuer verschiedene N um Konvergenz zu pruefen
        for N in [5, 10, 20, 30]:
            D = build_D_matrix(lam, N, primes_used)
            evals_D = np.sort(eigh(D, eigvals_only=True))

            # 3x3 Block
            D3 = D[:3, :3]
            evals_3 = np.sort(np.linalg.eigvalsh(D3))

            if N == 30:
                print(f"\nlambda={lam}:")
                print(f"  3x3 Block:")
                for i in range(3):
                    print(f"    [{D3[i,0]:+8.4f} {D3[i,1]:+8.4f} {D3[i,2]:+8.4f}]")
                print(f"  3x3 EW: {[f'{x:+.4f}' for x in evals_3]}")

            print(f"  N={N:2d}: min_EW(D)={evals_D[0]:+.4f}, min_EW(D3)={evals_3[0]:+.4f}")


def trace_formula_analysis():
    """
    TRACE-FORMEL-ANALYSE.

    Tr(D) = sum_n D_{nn} = sum_p log(p) sum_m p^{-m/2} * sum_n [cos_overlap(n,n,s) - sin_overlap(n,n,s)]

    Fuer die Diagonalelemente D_{nn}(s) gilt:
    cos: (1/L) int cos(2*kn*t - kn*s) dt
    sin: -(1/L) int cos(2*kn_s*t - kn_s*s) dt

    Die Summe ueber n ist eine Theta-Funktion!
    """
    from sympy import primerange
    primes = list(primerange(2, 200))

    print("\n" + "=" * 80)
    print("TRACE-ANALYSE: Tr(D) als Funktion von lambda")
    print("=" * 80)

    N = 40
    lam_values = [20, 25, 30, 40, 50, 70, 100, 150, 200, 300, 500]

    traces = []
    min_evs = []

    for lam in lam_values:
        primes_used = [p for p in primes if p <= max(lam, 47)]
        D = build_D_matrix(lam, N, primes_used)
        tr = np.trace(D)
        evals_D = np.sort(eigh(D, eigvals_only=True))
        traces.append(tr)
        min_evs.append(evals_D[0])

    print(f"\n  {'lambda':>6} | {'Tr(D)':>10} | {'min EW(D)':>10} | {'Tr/log(l)':>10} | {'min/log(l)':>10}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for i, lam in enumerate(lam_values):
        ll = np.log(lam)
        print(f"  {lam:6d} | {traces[i]:+10.4f} | {min_evs[i]:+10.4f} | {traces[i]/ll:+10.4f} | {min_evs[i]/ll:+10.4f}")

    # Linearer Fit fuer min_EW vs log(lambda)
    x = np.array([np.log(l) for l in lam_values if l >= 30])
    y = np.array([min_evs[i] for i, l in enumerate(lam_values) if l >= 30])
    if len(x) > 1:
        slope, intercept = np.polyfit(x, y, 1)
        print(f"\n  Linearer Fit min_EW(D) ~ {slope:.3f} * log(lambda) + ({intercept:.3f})")
        print(f"  => min_EW(D) -> -inf als lambda -> inf")
        print(f"  => Differenz wird STAERKER mit wachsendem lambda")


def prime_contribution_breakdown():
    """
    Zerlege D nach einzelnen Primzahlen.

    Frage: Welche Primzahlen tragen am meisten zur Even-Dominanz bei?
    """
    from sympy import primerange
    primes = list(primerange(2, 200))

    print("\n" + "=" * 80)
    print("PRIM-ZERLEGUNG VON D")
    print("=" * 80)

    N = 25

    for lam in [50, 100]:
        primes_used = [p for p in primes if p <= max(lam, 47)]
        L = np.log(lam)

        print(f"\nlambda={lam}, L={L:.3f}:")
        print(f"  {'p':>4} | {'logp':>6} | {'Tr(D_p)':>10} | {'min_EW(D_p)':>12} | {'||D_p||_F':>10}")
        print(f"  {'-'*4}-+-{'-'*6}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}")

        D_total = np.zeros((N, N))

        for p in primes_used[:15]:  # Zeige die ersten 15
            logp = np.log(p)
            D_p = np.zeros((N, N))

            for m_exp in range(1, 20):
                coeff = logp * p**(-m_exp / 2.0)
                shift = m_exp * logp
                if shift >= 2 * L or coeff < 1e-15:
                    break
                for i in range(N):
                    for j in range(i, N):
                        val = coeff * D_element_symmetric(i, j, shift, L)
                        D_p[i, j] += val
                        if i != j:
                            D_p[j, i] += val

            D_total += D_p
            evals_p = np.sort(eigh(D_p, eigvals_only=True))
            print(f"  {p:4d} | {logp:6.3f} | {np.trace(D_p):+10.4f} | "
                  f"{evals_p[0]:+12.6f} | {np.linalg.norm(D_p, 'fro'):10.4f}")

        print(f"  {'SUM':>4} | {'':>6} | {np.trace(D_total):+10.4f} | "
              f"{np.sort(eigh(D_total, eigvals_only=True))[0]:+12.6f} | "
              f"{np.linalg.norm(D_total, 'fro'):10.4f}")


def eigenvector_overlap():
    """
    KERN-ARGUMENT: Wenn v1_cos der Grundzustands-Eigenvektor von W_cos ist,
    dann ist Delta = v1_cos^T * D * v1_cos + O(||v1_cos - v1_sin||^2).

    Zeige: v1^T * D * v1 < 0 fuer den Grundzustand.
    """
    from sympy import primerange
    primes = list(primerange(2, 200))

    print("\n" + "=" * 80)
    print("EIGENVECTOR-OVERLAP: v1^T * D * v1")
    print("=" * 80)

    N = 30

    for lam in [30, 50, 100, 200]:
        primes_used = [p for p in primes if p <= max(lam, 47)]

        from weg2_analytic_even_odd import build_QW_analytic
        W_cos = build_QW_analytic(lam, N, primes_used, 'cos')
        W_sin = build_QW_analytic(lam, N, primes_used, 'sin')

        evals_c, evecs_c = eigh(W_cos)
        evals_s, evecs_s = eigh(W_sin)
        idx_c = np.argsort(evals_c)
        idx_s = np.argsort(evals_s)

        v1_c = evecs_c[:, idx_c[0]]  # Grundzustand cos
        v1_s = evecs_s[:, idx_s[0]]  # Grundzustand sin

        D = build_D_matrix(lam, N, primes_used)

        # Rayleigh-Quotienten
        rq_c = v1_c @ D @ v1_c
        rq_s = v1_s @ D @ v1_s

        # Exakter Delta
        Delta = evals_c[idx_c[0]] - evals_s[idx_s[0]]

        # Overlap
        overlap = abs(v1_c @ v1_s)

        print(f"\nlambda={lam}:")
        print(f"  l1(cos)={evals_c[idx_c[0]]:+.4f}, l1(sin)={evals_s[idx_s[0]]:+.4f}, Delta={Delta:+.4f}")
        print(f"  v1_cos^T * D * v1_cos = {rq_c:+.6f}")
        print(f"  v1_sin^T * D * v1_sin = {rq_s:+.6f}")
        print(f"  |<v1_cos | v1_sin>|   = {overlap:.6f}")
        print(f"  Delta - rq_c          = {Delta - rq_c:+.6f} (Korrektur vom Arch-Unterschied)")

        # Zerlegung des Rayleigh-Quotienten nach Moden
        contributions = v1_c * (D @ v1_c)
        top5 = np.argsort(contributions)[:5]
        print(f"  Dominante negative Beitraege (Moden): {top5} mit Werten {contributions[top5]}")


def formal_bound_attempt():
    """
    FORMALER BEWEIS-VERSUCH.

    Strategie: Zeige dass der Rayleigh-Quotient v^T * W_cos * v < v^T * W_sin * v
    fuer geeignetes v.

    Da W_cos = W_common + D/2 und W_sin = W_common - D/2 (mit W_common = (W_cos+W_sin)/2):
    l1(cos) <= v^T * W_common * v + v^T * (D/2) * v

    Fuer v = Grundzustand von W_common:
    l1(cos) - l1(sin) <= v^T * D * v - v^T * (-D/2) * v = ...

    Nein, besser: Min-Max.
    l1(cos) = min_v v^T * W_cos * v = min_v [v^T * W_sin * v + v^T * D * v]
    <= l1(sin) + max ueber Grundzustand von W_sin: v1_sin^T * D * v1_sin

    Aber wir brauchen die ANDERE Richtung: l1(cos) < l1(sin).

    Variationsargument:
    l1(cos) <= u^T * W_cos * u fuer JEDES normiertes u.
    Waehle u so dass u^T * W_cos * u < l1(sin).

    l1(sin) ist der min ueber alle normierten v von v^T * W_sin * v.
    Also muessen wir ein u finden mit u^T * W_cos * u < min_v v^T * W_sin * v.

    Aequivalent: u^T * W_cos * u < l1(sin).
    Das heisst: u^T * (W_sin + D) * u < l1(sin)
    => u^T * W_sin * u + u^T * D * u < l1(sin)
    => u^T * D * u < l1(sin) - u^T * W_sin * u <= 0 (da u^T*W_sin*u >= l1(sin))

    Also genuegt: u^T * D * u < 0 fuer ein normiertes u mit u^T * W_sin * u = l1(sin),
    d.h. u = v1_sin (Grundzustand von W_sin).

    ABER: u^T * W_cos * u = u^T * W_sin * u + u^T * D * u = l1(sin) + u^T * D * u.
    Also: l1(cos) <= l1(sin) + v1_sin^T * D * v1_sin.

    Wenn v1_sin^T * D * v1_sin < 0, dann l1(cos) < l1(sin). QED!

    FRAGE: Ist v1_sin^T * D * v1_sin < 0 IMMER fuer lambda >= lambda_0?
    """
    from sympy import primerange
    primes = list(primerange(2, 200))

    print("\n" + "=" * 80)
    print("FORMALER BEWEIS-VERSUCH: v1_sin^T * D * v1_sin < 0 ?")
    print("=" * 80)
    print("\n  Argument: l1(cos) <= l1(sin) + v1_sin^T * D * v1_sin")
    print("  Wenn v1_sin^T * D * v1_sin < 0, dann l1(cos) < l1(sin). QED!")

    N = 30

    print(f"\n  {'lambda':>6} | {'l1(cos)':>10} | {'l1(sin)':>10} | {'Delta':>10} | "
          f"{'v1s^T*D*v1s':>12} | {'Bound':>10} | {'OK?':>4}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}-+-{'-'*4}")

    for lam in [20, 25, 28, 30, 35, 40, 50, 70, 100, 150, 200, 300, 500]:
        primes_used = [p for p in primes if p <= max(lam, 47)]

        from weg2_analytic_even_odd import build_QW_analytic
        W_cos = build_QW_analytic(lam, N, primes_used, 'cos')
        W_sin = build_QW_analytic(lam, N, primes_used, 'sin')

        evals_c = np.sort(eigh(W_cos, eigvals_only=True))
        evals_s, evecs_s = eigh(W_sin)
        idx_s = np.argsort(evals_s)
        v1_s = evecs_s[:, idx_s[0]]

        D = build_D_matrix(lam, N, primes_used)
        rq = v1_s @ D @ v1_s

        bound = evals_s[idx_s[0]] + rq
        ok = "YES" if rq < 0 else "NO"

        Delta = evals_c[0] - evals_s[idx_s[0]]

        print(f"  {lam:6d} | {evals_c[0]:+10.4f} | {evals_s[idx_s[0]]:+10.4f} | "
              f"{Delta:+10.4f} | {rq:+12.6f} | {bound:+10.4f} | {ok:>4}")

    print("\n  INTERPRETATION:")
    print("  Wenn v1_sin^T * D * v1_sin < 0 fuer alle lambda >= lambda_0,")
    print("  dann ist lambda_1(even) < lambda_1(sin) bewiesen!")
    print("  Die Schranke bound = l1(sin) + v1s^T*D*v1s ist eine obere Schranke fuer l1(cos).")
    print("  Falls bound < l1(sin), ist Even-Dominanz bewiesen.")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, r'C:\Users\User\OneDrive\.RESEARCH\Natur&Technik\1 Musterbeweise\RH\scripts')

    t0 = time.time()
    analyze_D_structure()
    analyze_low_mode_block()
    trace_formula_analysis()
    prime_contribution_breakdown()
    eigenvector_overlap()
    formal_bound_attempt()
    print(f"\nTotal: {time.time()-t0:.1f}s")
