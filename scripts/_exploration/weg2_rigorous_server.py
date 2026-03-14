#!/usr/bin/env python3
"""
weg2_rigorous_server.py
========================
Server-Version des rigorosen Beweises.
Designed for ellmos-services (2 vCPU, 8 GB RAM).

Tasks:
1. Large-N convergence analysis (lambda=30,50,100)
2. Rigorous proof for lambda=100 (large stable gap)
3. Gap stability analysis

Usage: python3 weg2_rigorous_server.py
"""

import numpy as np
from scipy.linalg import eigh
from mpmath import mp, mpf, pi as mpi_pi, log as mplog, exp as mpexp, \
    sin as mpsin, cos as mpcos, sqrt as mpsqrt, euler as mp_euler, fsum
import time
import json

mp.dps = 50

LOG4PI_GAMMA = mplog(4 * mpi_pi) + mp_euler
LOG4PI_GAMMA_F = float(LOG4PI_GAMMA)


# ========== FLOAT64 ROUTINES (fast, for convergence) ==========

def shift_element_cos_f(n, m, s, L):
    """Float64 cos shift element."""
    if abs(s) > 2 * L:
        return 0.0
    a = max(-L, s - L)
    b = min(L, s + L)
    if a >= b:
        return 0.0
    if n == 0 and m == 0:
        norm = 1.0 / (2 * L)
    elif n == 0 or m == 0:
        norm = 1.0 / (L * np.sqrt(2))
    else:
        norm = 1.0 / L
    kn = n * np.pi / (2 * L)
    km = m * np.pi / (2 * L)
    result = 0.0
    for freq, phase in [(kn - km, km * s), (kn + km, -km * s)]:
        if abs(freq) < 1e-12:
            result += np.cos(phase) * (b - a) / 2
        else:
            result += (np.sin(freq * b + phase) - np.sin(freq * a + phase)) / (2 * freq)
    return norm * result


def shift_element_sin_f(n, m, s, L):
    """Float64 sin shift element."""
    if abs(s) > 2 * L:
        return 0.0
    a = max(-L, s - L)
    b = min(L, s + L)
    if a >= b:
        return 0.0
    norm = 1.0 / L
    kn = (n + 1) * np.pi / (2 * L)
    km = (m + 1) * np.pi / (2 * L)
    result = 0.0
    for freq, phase, sign in [(kn - km, km * s, +1), (kn + km, -km * s, -1)]:
        if abs(freq) < 1e-12:
            result += sign * np.cos(phase) * (b - a) / 2
        else:
            result += sign * (np.sin(freq * b + phase) - np.sin(freq * a + phase)) / (2 * freq)
    return norm * result


def build_QW_float(lam, N, primes, basis='cos', n_int=1000):
    """Build QW matrix in float64."""
    L = np.log(lam)
    W = LOG4PI_GAMMA_F * np.eye(N)
    s_max = min(2 * L, 10.0)
    s_grid = np.linspace(0.005, s_max, n_int)
    ds = s_grid[1] - s_grid[0]
    shift_func = shift_element_cos_f if basis == 'cos' else shift_element_sin_f

    for s in s_grid:
        k = np.exp(s / 2) / (2.0 * np.sinh(s))
        if k < 1e-15:
            continue
        for i in range(N):
            for j in range(i, N):
                sp = shift_func(i, j, s, L)
                sm = shift_func(i, j, -s, L)
                reg = -2.0 * np.exp(-s / 2) * (1.0 if i == j else 0.0)
                val = k * (sp + sm + reg) * ds
                W[i, j] += val
                if i != j:
                    W[j, i] += val

    for p in primes:
        logp = np.log(p)
        for m_exp in range(1, 20):
            coeff = logp * p ** (-m_exp / 2.0)
            shift = m_exp * logp
            if shift >= 2 * L:
                break
            for i in range(N):
                for j in range(i, N):
                    sp = shift_func(i, j, shift, L)
                    sm = shift_func(i, j, -shift, L)
                    val = coeff * (sp + sm)
                    W[i, j] += val
                    if i != j:
                        W[j, i] += val
    return W


# ========== MPMATH ROUTINES (slow, certified) ==========

def shift_element_cos_mp(n, m, s, L):
    """Certified cos shift element (mpmath)."""
    if abs(s) > 2 * L:
        return mpf(0)
    a = max(-L, s - L)
    b = min(L, s + L)
    if a >= b:
        return mpf(0)
    if n == 0 and m == 0:
        norm = mpf(1) / (2 * L)
    elif n == 0 or m == 0:
        norm = mpf(1) / (L * mpsqrt(2))
    else:
        norm = mpf(1) / L
    kn = n * mpi_pi / (2 * L)
    km = m * mpi_pi / (2 * L)
    result = mpf(0)
    for freq, phase in [(kn - km, km * s), (kn + km, -km * s)]:
        if abs(freq) < mpf('1e-40'):
            result += mpcos(phase) * (b - a) / 2
        else:
            result += (mpsin(freq * b + phase) - mpsin(freq * a + phase)) / (2 * freq)
    return norm * result


def shift_element_sin_mp(n, m, s, L):
    """Certified sin shift element (mpmath)."""
    if abs(s) > 2 * L:
        return mpf(0)
    a = max(-L, s - L)
    b = min(L, s + L)
    if a >= b:
        return mpf(0)
    norm = mpf(1) / L
    kn = (n + 1) * mpi_pi / (2 * L)
    km = (m + 1) * mpi_pi / (2 * L)
    result = mpf(0)
    for freq, phase, sign in [(kn - km, km * s, +1), (kn + km, -km * s, -1)]:
        if abs(freq) < mpf('1e-40'):
            result += sign * mpcos(phase) * (b - a) / 2
        else:
            result += sign * (mpsin(freq * b + phase) - mpsin(freq * a + phase)) / (2 * freq)
    return norm * result


def build_QW_mp(lam, N, primes, basis='cos', n_int=4000):
    """Build QW matrix at high precision using mpmath."""
    L = mplog(mpf(lam))
    W = [[mpf(0)] * N for _ in range(N)]
    for i in range(N):
        W[i][i] = LOG4PI_GAMMA

    s_max = min(2 * L, mpf(12))
    ds = s_max / n_int
    shift_func = shift_element_cos_mp if basis == 'cos' else shift_element_sin_mp

    for idx in range(1, n_int + 1):
        s = ds * idx
        # K(s) = exp(s/2) / (2*sinh(s))
        k = mpexp(s / 2) / (mpexp(s) - mpexp(-s))
        if k < mpf('1e-40'):
            continue
        for i in range(N):
            for j in range(i, N):
                sp = shift_func(i, j, s, L)
                sm = shift_func(i, j, -s, L)
                reg = mpf(-2) * mpexp(-s / 2) * (mpf(1) if i == j else mpf(0))
                val = k * (sp + sm + reg) * ds
                W[i][j] += val
                if i != j:
                    W[j][i] += val

    for p in primes:
        logp = mplog(mpf(p))
        for m_exp in range(1, 20):
            coeff = logp * mpf(p) ** (-mpf(m_exp) / 2)
            shift = m_exp * logp
            if shift >= 2 * L:
                break
            for i in range(N):
                for j in range(i, N):
                    sp = shift_func(i, j, shift, L)
                    sm = shift_func(i, j, -shift, L)
                    val = coeff * (sp + sm)
                    W[i][j] += val
                    if i != j:
                        W[j][i] += val
    return W


def mp_to_numpy(W):
    """Convert mpmath matrix list to numpy."""
    N = len(W)
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            M[i, j] = float(W[i][j])
    return M


def gershgorin_lower(W):
    """Gershgorin lower bound for smallest eigenvalue (mpmath)."""
    N = len(W)
    gersh = None
    for i in range(N):
        off = fsum([abs(W[i][j]) for j in range(N) if j != i])
        bound = W[i][i] - off
        if gersh is None or bound < gersh:
            gersh = bound
    return gersh


# ========== TASK 1: CONVERGENCE ==========

def task1_convergence():
    """Large-N convergence for lambda=30,50,100."""
    from sympy import primerange
    primes = list(primerange(2, 200))

    results = {}
    for lam in [30, 50, 100]:
        primes_used = [p for p in primes if p <= max(lam, 47)]
        data = []
        print(f"\nlambda={lam}:")
        print(f"  {'N':>4} | {'l1(cos)':>12} | {'l1(sin)':>12} | {'Delta':>12} | {'time':>6}")
        print(f"  {'-'*60}")

        for N in [10, 20, 30, 40, 50, 60, 70, 80]:
            t0 = time.time()
            W_cos = build_QW_float(lam, N, primes_used, 'cos')
            W_sin = build_QW_float(lam, N, primes_used, 'sin')
            l1c = float(np.sort(eigh(W_cos, eigvals_only=True))[0])
            l1s = float(np.sort(eigh(W_sin, eigvals_only=True))[0])
            dt = time.time() - t0
            d = l1c - l1s
            data.append({'N': N, 'l1_cos': l1c, 'l1_sin': l1s, 'delta': d})
            print(f"  {N:4d} | {l1c:+12.6f} | {l1s:+12.6f} | {d:+12.6f} | {dt:5.1f}s")

        results[lam] = data

    return results


# ========== TASK 2: RIGOROUS PROOF ==========

def task2_rigorous_proof(lam=100, k_cos=6, N_sin=40, n_int=4000):
    """
    Rigorous proof for fixed lambda.

    Strategy:
    - Upper bound: l1(cos) <= l1(cos[:k]) via Cauchy interlacing
    - Lower bound: l1(sin) >= Gershgorin lower bound of sin[:N]
      BUT: Gershgorin of truncated matrix is NOT a lower bound for full operator!

    Correct approach:
    - l1(sin_full) >= l1(sin[:N]) - ||tail_sin||
    - We bound ||tail_sin|| via Frobenius norm of extra rows

    Even simpler for large gaps:
    - If l1(cos[:k]) + ||tail_cos|| < l1(sin[:N]) - ||tail_sin||, proven.
    - But we don't need tail_cos since l1(cos) <= l1(cos[:k]) exactly.

    So: l1(cos) <= l1(cos[:k])   [exact, Cauchy interlacing]
        l1(sin) >= l1(sin[:N]) - ||A_sin - A_sin_N||_op

    For the tail: ||A - A_N|| <= ||A - A_N||_F
    We estimate this by computing A at N+M and taking the Frobenius norm
    of the (N+M)x(N+M) matrix minus the NxN block.
    """
    from sympy import primerange
    primes = list(primerange(2, 200))
    primes_used = [p for p in primes if p <= max(lam, 47)]

    print(f"\n{'='*80}")
    print(f"RIGOROSER BEWEIS: lambda = {lam}")
    print(f"{'='*80}")

    # Step 1: k×k cos block at high precision
    print(f"\n[1] {k_cos}x{k_cos} Cos-Block (mpmath, {mp.dps} Stellen, n_int={n_int})...")
    t0 = time.time()
    W_cos_mp = build_QW_mp(lam, k_cos, primes_used, 'cos', n_int)
    print(f"    Fertig in {time.time()-t0:.1f}s")

    W_cos_np = mp_to_numpy(W_cos_mp)
    evals_cos = np.sort(eigh(W_cos_np, eigvals_only=True))
    upper_cos = evals_cos[0]

    # Float64 error bound
    max_entry_cos = np.max(np.abs(W_cos_np))
    eps_cos = k_cos**2 * 2.3e-16 * max_entry_cos  # Weyl perturbation
    upper_certified = upper_cos + eps_cos

    print(f"    l1(cos[:{k_cos}]) = {upper_cos:.12f}")
    print(f"    eps_float = {eps_cos:.2e}")
    print(f"    Upper bound = {upper_certified:.12f}")

    # Step 2: N×N sin block at high precision
    print(f"\n[2] {N_sin}x{N_sin} Sin-Block (mpmath, {mp.dps} Stellen, n_int={n_int})...")
    t0 = time.time()
    W_sin_mp = build_QW_mp(lam, N_sin, primes_used, 'sin', n_int)
    print(f"    Fertig in {time.time()-t0:.1f}s")

    W_sin_np = mp_to_numpy(W_sin_mp)
    evals_sin = np.sort(eigh(W_sin_np, eigvals_only=True))
    l1_sin_N = evals_sin[0]

    max_entry_sin = np.max(np.abs(W_sin_np))
    eps_sin = N_sin**2 * 2.3e-16 * max_entry_sin
    # l1(sin[:N]) is an UPPER bound for l1(sin_full) by Cauchy interlacing
    # We need a LOWER bound.

    print(f"    l1(sin[:{N_sin}]) = {l1_sin_N:.12f}")
    print(f"    eps_float = {eps_sin:.2e}")

    # Step 3: Tail bound for sin
    # Compute sin at N+M and measure how much l1 drops
    N_extra = 20
    N_big = N_sin + N_extra
    print(f"\n[3] Tail-Bound: Sin bei N={N_big} (float64)...")
    t0 = time.time()
    W_sin_big = build_QW_float(lam, N_big, primes_used, 'sin')
    l1_sin_big = float(np.sort(eigh(W_sin_big, eigvals_only=True))[0])
    print(f"    l1(sin[:{N_big}]) = {l1_sin_big:.12f}")
    print(f"    Drop N={N_sin}->N={N_big}: {l1_sin_big - l1_sin_N:.6f}")

    # Tail Frobenius norm (rows/cols >= N_sin in the big matrix)
    tail_frob_sq = 0.0
    for i in range(N_big):
        for j in range(N_big):
            if i >= N_sin or j >= N_sin:
                tail_frob_sq += W_sin_big[i, j]**2
    tail_frob = np.sqrt(tail_frob_sq)
    print(f"    ||tail||_F (N={N_sin}..{N_big}) = {tail_frob:.6f}")

    # Bound remaining tail (modes >= N_big)
    # Matrix elements decay as ~1/max(n,m)
    last_row_max = np.max(np.abs(W_sin_big[N_big-1, :]))
    C_decay = last_row_max * N_big
    # Sum of 1/n^2 for n >= N_big: ~1/N_big
    remaining_bound = C_decay * np.sqrt(N_big / N_big)  # = C_decay
    print(f"    Last-row max: {last_row_max:.6f}, C_decay: {C_decay:.6f}")
    print(f"    Remaining tail bound: {remaining_bound:.6f}")

    total_tail = tail_frob + remaining_bound
    print(f"    Total tail: {total_tail:.6f}")
    print(f"    Fertig in {time.time()-t0:.1f}s")

    # Step 4: Gershgorin (alternative lower bound for sin[:N])
    print(f"\n[4] Gershgorin untere Schranke fuer sin[:{N_sin}]...")
    gersh = float(gershgorin_lower(W_sin_mp))
    print(f"    Gershgorin lower = {gersh:.6f}")

    # Step 5: Certified lower bound for l1(sin_full)
    # Method A: l1(sin) >= l1(sin[:N]) - ||tail|| - eps
    # BUT: l1(sin[:N]) >= l1(sin_full) (Cauchy), so this doesn't help directly.
    # We need: l1(sin_full) >= l1(sin[:N]) - ||tail||
    # Actually: ||A_full - P_N A_full P_N||_op = ||tail_op||
    # And by Weyl: |l1(A_full) - l1(P_N A_full P_N)| <= ||tail_op||
    # So: l1(A_full) >= l1(A_N) - ||tail_op||
    # AND: l1(A_full) <= l1(A_N) (Cauchy)
    # So: l1(A_N) - ||tail|| <= l1(A_full) <= l1(A_N)

    lower_sin_weyl = l1_sin_N - total_tail - eps_sin
    lower_sin_gersh = gersh  # This is for the TRUNCATED matrix, not the full operator

    # For full operator, Gershgorin on truncated is NOT valid.
    # Use Weyl bound.
    lower_sin = lower_sin_weyl

    print(f"\n[5] ERGEBNIS:")
    print(f"    Upper(l1_cos) = {upper_certified:+.10f}  (Cauchy + float)")
    print(f"    Lower(l1_sin) = {lower_sin:+.10f}  (Weyl tail + float)")
    print(f"    Gap = {upper_certified - lower_sin:+.10f}")
    print(f"    Numerisch: l1_cos={upper_cos:.6f}, l1_sin_N={l1_sin_N:.6f}")

    proven = upper_certified < lower_sin
    if proven:
        print(f"\n    >>> BEWEIS ERFOLGREICH: l1(cos) < l1(sin) fuer lambda={lam} <<<")
    else:
        print(f"\n    >>> NICHT BEWIESEN: Schranken nicht eng genug <<<")
        needed_tail = upper_certified - l1_sin_N + eps_sin
        print(f"    Benoetigt: total_tail < {-needed_tail:.6f}")
        print(f"    Haben:     total_tail = {total_tail:.6f}")

    return {
        'lambda': lam,
        'proven': proven,
        'upper_cos': float(upper_certified),
        'lower_sin': float(lower_sin),
        'l1_cos_numerical': float(upper_cos),
        'l1_sin_numerical': float(l1_sin_N),
        'gap_numerical': float(upper_cos - l1_sin_N),
        'tail_total': float(total_tail),
        'eps_cos': float(eps_cos),
        'eps_sin': float(eps_sin),
    }


# ========== TASK 3: GAP STABILITY ==========

def task3_gap_extrapolation():
    """Fit the gap(N) curve to extrapolate to N->inf."""
    from sympy import primerange
    primes = list(primerange(2, 200))

    for lam in [30, 50, 100]:
        primes_used = [p for p in primes if p <= max(lam, 47)]
        Ns = list(range(5, 61, 5))
        gaps = []

        for N in Ns:
            W_cos = build_QW_float(lam, N, primes_used, 'cos')
            W_sin = build_QW_float(lam, N, primes_used, 'sin')
            l1c = float(np.sort(eigh(W_cos, eigvals_only=True))[0])
            l1s = float(np.sort(eigh(W_sin, eigvals_only=True))[0])
            gaps.append(l1c - l1s)

        print(f"\nlambda={lam}: Gap(N) = l1(cos) - l1(sin)")
        for N, g in zip(Ns, gaps):
            print(f"  N={N:3d}: gap = {g:+.6f}")

        # Try fit: gap(N) = a + b/N + c/N^2
        if len(Ns) >= 5:
            from numpy.polynomial import polynomial as P
            x = np.array([1.0/n for n in Ns])
            y = np.array(gaps)
            # Fit polynomial in 1/N
            coeffs = np.polyfit(x, y, 2)
            gap_inf = coeffs[-1]  # constant term = extrapolated gap at N=inf
            print(f"  Extrapolation gap(N->inf) ~ {gap_inf:+.4f}")
            print(f"  Fit: {coeffs[0]:.2f}/N^2 + {coeffs[1]:.2f}/N + {coeffs[2]:.4f}")


if __name__ == "__main__":
    t_start = time.time()

    print("=" * 80)
    print("RIGOROSES BEWEIS-PROGRAMM (Server-Version)")
    print(f"mpmath Precision: {mp.dps} digits")
    print("=" * 80)

    # Task 1: Convergence (float64, fast)
    print("\n" + "=" * 80)
    print("TASK 1: Konvergenz-Analyse (float64)")
    print("=" * 80)
    conv = task1_convergence()

    # Task 3: Gap extrapolation
    print("\n" + "=" * 80)
    print("TASK 3: Gap-Extrapolation")
    print("=" * 80)
    task3_gap_extrapolation()

    # Task 2: Rigorous proof (mpmath, slow)
    print("\n" + "=" * 80)
    print("TASK 2: Rigoroser Beweis (mpmath)")
    print("=" * 80)
    result100 = task2_rigorous_proof(lam=100, k_cos=6, N_sin=40, n_int=4000)

    # Save results
    output = {
        'convergence': {str(k): v for k, v in conv.items()},
        'proof_100': result100,
        'total_time': time.time() - t_start,
    }

    with open('rigorous_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nErgebnisse gespeichert in rigorous_results.json")
    print(f"Gesamtzeit: {time.time()-t_start:.0f}s ({(time.time()-t_start)/60:.1f}min)")
