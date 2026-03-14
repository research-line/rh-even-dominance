#!/usr/bin/env python3
"""
weg2_gap_asymptotik.py
======================
Analyse der Abnahme-Rate von lambda_min(Q_W^S) bei S -> inf.

Zentrale Frage: Konvergiert lambda_min -> 0 oder -> c > 0?

Strategie:
  1. Feinere Abtastung von lambda_min vs. |S|
  2. Fit an verschiedene Modelle: c/log(S), c/S^alpha, c*exp(-S), c - a*log(S)
  3. Extrapolation auf S -> inf
  4. ANALYTISCHE Abschaetzung der Primzahl-Teilsumme
"""

import numpy as np
from mpmath import mp, im, zetazero, digamma, log, pi, euler
from sympy import primerange
from scipy.optimize import curve_fit

mp.dps = 25

# ===========================================================================
# Q_W Konstruktion (aus weg2_connes_QW.py, kompakt)
# ===========================================================================

def cosine_basis(n, t, T):
    if n == 0:
        return 1.0 / np.sqrt(T)
    return np.sqrt(2.0 / T) * np.cos(n * np.pi * t / T)

def phi_hat_cos(n, xi, T):
    omega_n = n * np.pi / T
    if n == 0:
        if abs(xi) < 1e-15:
            return np.sqrt(T)
        return np.sin(xi * T) / (xi * np.sqrt(T))
    c = np.sqrt(2.0 / T)
    def sinc_term(w):
        if abs(w) < 1e-15:
            return T
        return np.sin(w * T) / w
    return c * 0.5 * (sinc_term(omega_n - xi) + sinc_term(omega_n + xi))

def build_Q_arch(N_basis, T, n_quad=500):
    """Archimedischer Beitrag (FEST, unabhaengig von S)."""
    Q_arch = np.zeros((N_basis, N_basis))
    log4pi_half = float(log(4 * pi)) / 2
    euler_half = float(euler) / 2

    t_quad = np.linspace(0.1, T - 0.1, n_quad)
    dt = t_quad[1] - t_quad[0]

    Phi_vals = np.zeros(n_quad)
    for i, t in enumerate(t_quad):
        Phi_vals[i] = float(digamma(0.25 + 1j * t / 2).real) + log4pi_half + euler_half

    phi_grid = np.zeros((N_basis, n_quad))
    for n in range(N_basis):
        for i, t in enumerate(t_quad):
            phi_grid[n, i] = cosine_basis(n, t, T)

    for n in range(N_basis):
        for m in range(n, N_basis):
            val = np.sum(phi_grid[n] * phi_grid[m] * Phi_vals) * dt / np.pi
            Q_arch[n, m] = val
            if m != n:
                Q_arch[m, n] = val
    return Q_arch

def build_Q_prime(N_basis, T, primes, M_terms=5):
    """Primzahl-Beitrag (waechst mit S)."""
    Q_prime = np.zeros((N_basis, N_basis))
    for p in primes:
        logp = np.log(p)
        for k in range(1, M_terms + 1):
            coeff = 2.0 * logp / p**(k / 2.0)
            xi = k * logp
            h_vals = np.array([phi_hat_cos(n, xi, T) for n in range(N_basis)])
            Q_prime += coeff * np.outer(h_vals, h_vals)
    return Q_prime

# ===========================================================================
# Feine Abtastung
# ===========================================================================

def fine_sampling(N_basis, T):
    """Lambda_min fuer viele S-Werte."""
    print(f"\n{'='*70}")
    print(f"FEINE ABTASTUNG: lambda_min vs. |S|")
    print(f"  N_basis={N_basis}, T={T}")
    print(f"{'='*70}")

    # Archimedischer Term (einmal berechnen)
    print("  Berechne Q_arch (einmalig)...")
    Q_arch = build_Q_arch(N_basis, T)
    print(f"  Q_arch fertig. trace = {np.trace(Q_arch):.4f}")

    all_primes = list(primerange(2, 5000))

    # S-Werte: 1, 2, 3, ..., 10, 15, 20, ..., 50, 75, 100, ..., 500, 669
    S_values = list(range(1, 11)) + list(range(15, 55, 5)) + \
               [75, 100, 150, 200, 300, 500, 669]
    S_values = [s for s in S_values if s <= len(all_primes)]

    results = []
    Q_prime_cumul = np.zeros((N_basis, N_basis))
    prime_idx = 0

    print(f"\n  {'|S|':>5} | {'p_max':>6} | {'lambda_min':>12} | {'trace(Q_p)':>12} | "
          f"{'trace(Q_a-Qp)':>14} | {'min EW ratio':>12}")
    print(f"  {'-'*5}-+-{'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*14}-+-{'-'*12}")

    prev_lam_min = None

    for S_size in S_values:
        # Inkrementell: nur neue Primzahlen hinzufuegen
        while prime_idx < S_size and prime_idx < len(all_primes):
            p = all_primes[prime_idx]
            logp = np.log(p)
            for k in range(1, 6):
                coeff = 2.0 * logp / p**(k / 2.0)
                xi = k * logp
                h_vals = np.array([phi_hat_cos(n, xi, T) for n in range(N_basis)])
                Q_prime_cumul += coeff * np.outer(h_vals, h_vals)
            prime_idx += 1

        Q_total = Q_arch - Q_prime_cumul
        evals = np.linalg.eigvalsh(Q_total)
        lam_min = evals[0]

        trace_prime = np.trace(Q_prime_cumul)
        trace_total = np.trace(Q_total)

        ratio = lam_min / prev_lam_min if prev_lam_min and prev_lam_min != 0 else 1.0
        prev_lam_min = lam_min

        results.append((S_size, all_primes[S_size - 1], lam_min, trace_prime, trace_total))

        if S_size <= 10 or S_size % 25 == 0 or S_size in [15, 20, 50, 100, 200, 500, 669]:
            print(f"  {S_size:5d} | {all_primes[S_size-1]:6d} | {lam_min:+12.6e} | "
                  f"{trace_prime:12.6e} | {trace_total:+14.6e} | {ratio:12.6f}")

    return results, Q_arch

# ===========================================================================
# Fit-Modelle
# ===========================================================================

def fit_models(results):
    """Fitte verschiedene Modelle an lambda_min(S)."""
    print(f"\n{'='*70}")
    print(f"FIT-MODELLE fuer lambda_min(|S|)")
    print(f"{'='*70}")

    S_vals = np.array([r[0] for r in results], dtype=float)
    lam_vals = np.array([r[2] for r in results])

    # Nur Werte mit S >= 3 fuer stabilen Fit
    mask = S_vals >= 3
    S = S_vals[mask]
    lam = lam_vals[mask]

    # Modell 1: lambda = a - b * log(S)
    def model_log(x, a, b):
        return a - b * np.log(x)

    # Modell 2: lambda = a - b / sqrt(S)  (aehnlich PNT-Korrektur)
    def model_sqrt(x, a, b):
        return a - b / np.sqrt(x)

    # Modell 3: lambda = a * exp(-b * S)
    def model_exp(x, a, b):
        return a * np.exp(-b * x)

    # Modell 4: lambda = a - b * S^alpha
    def model_power(x, a, b, alpha):
        return a - b * x**alpha

    # Modell 5: lambda = c + a / log(S)
    def model_invlog(x, c, a):
        return c + a / np.log(x)

    # Modell 6: lambda = a - b * log(log(S))
    def model_loglog(x, a, b):
        return a - b * np.log(np.log(x + 1))

    models = [
        ("a - b*log(S)", model_log, (0.4, 0.01), 2),
        ("a - b/sqrt(S)", model_sqrt, (0.35, 0.1), 2),
        ("c + a/log(S)", model_invlog, (0.3, 0.1), 2),
        ("a - b*log(log(S))", model_loglog, (0.5, 0.05), 2),
    ]

    print(f"\n  {'Modell':>25} | {'Parameter':>30} | {'Residual':>12} | {'S->inf':>12}")
    print(f"  {'-'*25}-+-{'-'*30}-+-{'-'*12}-+-{'-'*12}")

    best_residual = float('inf')
    best_model = None

    for name, func, p0, n_params in models:
        try:
            popt, _ = curve_fit(func, S, lam, p0=p0, maxfev=10000)
            pred = func(S, *popt)
            residual = np.sqrt(np.mean((pred - lam)**2))

            # Extrapolation S -> inf
            if 'log(S)' in name and 'log(log' not in name and '/' not in name:
                lam_inf = -float('inf')  # a - b*log(inf) -> -inf
                lam_inf_str = "-inf"
            elif '/sqrt' in name:
                lam_inf = popt[0]
                lam_inf_str = f"{lam_inf:.6f}"
            elif '/log' in name:
                lam_inf = popt[0]
                lam_inf_str = f"{lam_inf:.6f}"
            elif 'loglog' in name:
                lam_inf = -float('inf')
                lam_inf_str = "-inf"
            else:
                lam_inf = 0
                lam_inf_str = "???"

            param_str = ", ".join(f"{p:.6f}" for p in popt)
            print(f"  {name:>25} | {param_str:>30} | {residual:12.6e} | {lam_inf_str:>12}")

            if residual < best_residual:
                best_residual = residual
                best_model = (name, popt, func, lam_inf_str)
        except Exception as e:
            print(f"  {name:>25} | {'FIT FAILED':>30} | {'---':>12} | {'---':>12}")

    if best_model:
        print(f"\n  BESTES MODELL: {best_model[0]}")
        print(f"  Parameter: {best_model[1]}")
        print(f"  lambda_min(S->inf) -> {best_model[3]}")

    return best_model

# ===========================================================================
# Analytische Abschaetzung der Primsumme
# ===========================================================================

def analytical_prime_sum():
    """
    Berechne die VOLLE Primsumme (S -> inf) analytisch.

    Sigma_{p prim} log(p) / p^{1/2} divergiert!
    (weil Sigma log(p) / p^{1/2} ~ integral log(x)/x^{1/2} dx -> inf)

    Sigma_{p} log(p) / p^{k/2} konvergiert fuer k >= 3:
    Fuer k=1: -zeta'(1/2)/zeta(1/2) (divergiert, Re(s)=1/2 ist kritisch!)
    Fuer k=2: -zeta'(1)/zeta(1) (divergiert wegen Pol bei s=1)
    Fuer k=3: endlich

    Also: Der Term k=1 in der Primsumme DIVERGIERT!
    Das bedeutet: trace(Q_prime) -> inf fuer S -> inf.
    """
    print(f"\n{'='*70}")
    print(f"ANALYTISCHE ABSCHAETZUNG DER PRIMSUMME")
    print(f"{'='*70}")

    primes = list(primerange(2, 100000))

    # Partielle Summen fuer verschiedene k
    for k in [1, 2, 3, 4, 5]:
        partial_sums = []
        running = 0.0
        checkpoints = [10, 50, 100, 500, 1000, 5000, len(primes)]
        cp_idx = 0
        for i, p in enumerate(primes):
            running += np.log(p) / p**(k / 2.0)
            if cp_idx < len(checkpoints) and (i + 1) == checkpoints[cp_idx]:
                partial_sums.append((i + 1, p, running))
                cp_idx += 1

        print(f"\n  k={k}: Sigma log(p) / p^(k/2)")
        for n, p_max, s in partial_sums:
            print(f"    N={n:5d}, p_max={p_max:6d}: S = {s:.6f}")

        # Konvergenz?
        if len(partial_sums) >= 2:
            ratio = partial_sums[-1][2] / partial_sums[-2][2]
            if ratio > 1.05:
                print(f"    => DIVERGIERT (Verhaeltnis letzte/vorletzte: {ratio:.4f})")
            else:
                print(f"    => Konvergiert (Verhaeltnis: {ratio:.4f})")

    # Zusammenfassung
    print(f"\n  ZUSAMMENFASSUNG:")
    print(f"  k=1: Sigma log(p)/sqrt(p) DIVERGIERT (~2*sqrt(N) nach PNT)")
    print(f"  k=2: Sigma log(p)/p DIVERGIERT (log-artig, ~log(log(N)))")
    print(f"  k=3: Sigma log(p)/p^(3/2) KONVERGIERT")
    print(f"  k>=3: Alle konvergieren")
    print(f"")
    print(f"  KONSEQUENZ: trace(Q_prime) -> inf wegen k=1 und k=2 Termen!")
    print(f"  Aber trace(Q_arch) ist FEST.")
    print(f"  Also: trace(Q_W^S) = trace(Q_arch) - trace(Q_prime^S) -> -inf")
    print(f"")
    print(f"  ABER: trace(Q) < 0 bedeutet NICHT lambda_min < 0!")
    print(f"  Die Spur ist die SUMME aller Eigenwerte.")
    print(f"  Es koennte sein, dass ein EW -> -inf geht, waehrend")
    print(f"  lambda_min stabil bleibt (oder umgekehrt).")
    print(f"")
    print(f"  ENTSCHEIDEND: Nicht trace, sondern lambda_min(Q_W^S).")
    print(f"  Die divergierende Primsumme MUSS irgendwann Q_W^S")
    print(f"  indefinit machen -- ausser die Operatorstruktur verhindert es.")

# ===========================================================================
# Trace-Divergenz vs. lambda_min
# ===========================================================================

def trace_vs_lambdamin(results, Q_arch, N_basis, T):
    """Vergleiche trace-Divergenz mit lambda_min-Verhalten."""
    print(f"\n{'='*70}")
    print(f"TRACE vs. LAMBDA_MIN")
    print(f"{'='*70}")

    trace_arch = np.trace(Q_arch)

    print(f"\n  trace(Q_arch) = {trace_arch:.4f} (fest)")
    print(f"\n  {'|S|':>5} | {'trace(Q_p)':>12} | {'trace(Q_W)':>12} | {'lambda_min':>12} | {'lambda_max':>12}")
    print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    all_primes = list(primerange(2, 5000))
    Q_prime = np.zeros((N_basis, N_basis))

    for S_size, p_max, lam_min, trace_prime, trace_total in results:
        if S_size in [1, 5, 10, 25, 50, 100, 200, 500, 669]:
            # Recompute max EV
            # We have the data from results, but not max EV. Use trace info.
            print(f"  {S_size:5d} | {trace_prime:12.4f} | {trace_total:+12.4f} | "
                  f"{lam_min:+12.6e} | {'---':>12}")

# ===========================================================================
# HAUPTPROGRAMM
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("WEG 2: ASYMPTOTIK DER SPEKTRALLUECKE")
    print("=" * 70)

    N_BASIS = 12  # etwas kleiner fuer Geschwindigkeit
    T = 120.0

    # 1. Feine Abtastung
    results, Q_arch = fine_sampling(N_BASIS, T)

    # 2. Fit-Modelle
    best = fit_models(results)

    # 3. Analytische Primsumme
    analytical_prime_sum()

    # 4. Trace vs lambda_min
    trace_vs_lambdamin(results, Q_arch, N_BASIS, T)

    # FAZIT
    print(f"\n{'='*70}")
    print(f"FAZIT: ASYMPTOTIK")
    print(f"{'='*70}")
    print(f"""
  DREI ERKENNTNISSE:

  1. Die Primsumme (k=1 Term: Sigma log(p)/sqrt(p)) DIVERGIERT.
     trace(Q_prime) -> inf, also trace(Q_W^S) -> -inf.
     Das bedeutet: Q_W^S kann nicht fuer alle S PSD bleiben,
     WENN die negativen Eigenwerte sich gleichmaessig verteilen.

  2. ABER: Die Eigenwertstruktur ist NICHT gleichmaessig.
     lambda_min koennte gegen 0 konvergieren, waehrend einige
     Eigenwerte stark negativ werden. Das ist genau die Kern-Entstehung.

  3. Die FIT-ERGEBNISSE zeigen die Rate der Abnahme.
     Wenn lambda_min ~ c + a/log(S), dann lambda_min -> c > 0 (PSD bleibt).
     Wenn lambda_min ~ a - b*log(S), dann lambda_min -> -inf (PSD bricht).

  STRATEGISCHE KONSEQUENZ:
  - Falls lambda_min -> 0: Connes' Kern entsteht. Ideal fuer BI-11.
  - Falls lambda_min -> -inf: Q_W^S wird irgendwann indefinit.
    Dann ist die "richtige" Q_W die mit ALLEN Primen, und
    die PSD-Eigenschaft gilt nur unter RH (nicht als Beweiswerkzeug).
  - Falls lambda_min -> c > 0: Kein Kern, Q_W bleibt strikt positiv.
    Dann approximieren die Kern-EVs von Q_zeros die Zeta-Nullstellen
    auf eine Weise, die Q_formula NICHT reproduzieren kann.

  In jedem Fall: Die Divergenz der Primsumme (k=1 Term) ist das
  zentrale analytische Faktum. Es zwingt trace(Q_W) -> -inf,
  und die Eigenwertverteilung entscheidet alles.
""")
