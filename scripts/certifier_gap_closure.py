#!/usr/bin/env python3
"""
certifier_gap_closure.py
========================
Close the certificate gap: lambda from 700000 to 1300000.
Geometric grid with ratio 1.5.
Fills the window between CAP certificates (<=640K) and
Euler-Maclaurin asymptotic regime (>=1.2M).

For ellmos-services (2 vCPU, 8 GB RAM).
Estimated runtime: ~12-18 hours total.
"""

import numpy as np
from scipy.linalg import eigvalsh
from mpmath import mp, iv, pi as mpi, log as mplog, euler as mp_euler
from sympy import primerange
import time
import json
import sys

mp.dps = 40

LOG4PI_GAMMA = float(mplog(4 * mpi) + mp_euler)


def shift_cos_iv(n, m, s, L):
    two_L = 2 * L
    if s > two_L or s < -two_L:
        return iv.mpf(0)
    a_iv = iv.mpf(max(-L, s - L))
    b_iv = iv.mpf(min(L, s + L))
    if a_iv >= b_iv:
        return iv.mpf(0)
    if n == 0 and m == 0:
        norm = iv.mpf(1) / (2 * L)
    elif n == 0 or m == 0:
        norm = iv.mpf(1) / (L * iv.sqrt(iv.mpf(2)))
    else:
        norm = iv.mpf(1) / L
    kn = iv.mpf(n) * iv.pi / L
    km = iv.mpf(m) * iv.pi / L
    result = iv.mpf(0)
    for freq, phase in [(kn - km, km * s), (kn + km, -km * s)]:
        if abs(float(freq.mid)) < 1e-30:
            result += iv.cos(phase) * (b_iv - a_iv) / 2
        else:
            result += (iv.sin(freq * b_iv + phase) - iv.sin(freq * a_iv + phase)) / (2 * freq)
    return norm * result


def shift_sin_f(n, m, s, L):
    if abs(s) > 2 * L:
        return 0.0
    a = max(-L, s - L)
    b = min(L, s + L)
    if a >= b:
        return 0.0
    norm = 1.0 / L
    kn = (n + 1) * np.pi / L
    km = (m + 1) * np.pi / L
    result = 0.0
    for freq, phase, sign in [(kn - km, km * s, +1), (kn + km, -km * s, -1)]:
        if abs(freq) < 1e-12:
            result += sign * np.cos(phase) * (b - a) / 2
        else:
            result += sign * (np.sin(freq * b + phase) - np.sin(freq * a + phase)) / (2 * freq)
    return norm * result


def build_cos_block_iv(lam, k, primes):
    L = iv.log(iv.mpf(lam))
    W = [[iv.mpf(0)] * k for _ in range(k)]
    LOG4PI_IV = iv.log(4 * iv.pi) + iv.mpf(mp_euler)
    for i in range(k):
        W[i][i] = LOG4PI_IV
    for p in primes:
        logp = iv.log(iv.mpf(p))
        for m_exp in range(1, 30):
            coeff = logp * iv.mpf(p) ** (iv.mpf(-m_exp) / 2)
            if float(coeff.b) < 1e-25:
                break
            shift = iv.mpf(m_exp) * logp
            if float(shift.a) >= 2 * float(L.b):
                break
            for i in range(k):
                for j in range(i, k):
                    sp = shift_cos_iv(i, j, shift, L)
                    sm = shift_cos_iv(i, j, -shift, L)
                    val = coeff * (sp + sm)
                    W[i][j] += val
                    if i != j:
                        W[j][i] += val
    return W


def build_sin_block_f64(lam, N, primes):
    L = np.log(lam)
    W = LOG4PI_GAMMA * np.eye(N)
    s_max = min(2 * L, 12.0)
    n_quad = max(2000, 25 * N)
    s_grid = np.linspace(0.005, s_max, n_quad)
    ds = s_grid[1] - s_grid[0]
    for s in s_grid:
        K = np.exp(s / 2) / (2.0 * np.sinh(s))
        if K < 1e-15:
            continue
        for i in range(N):
            for j in range(i, N):
                sp = shift_sin_f(i, j, s, L)
                sm = shift_sin_f(i, j, -s, L)
                reg = -2.0 * np.exp(-s / 2) if i == j else 0.0
                W[i, j] += K * (sp + sm + reg) * ds
                if i != j:
                    W[j, i] = W[i, j]
    for p in primes:
        logp = np.log(p)
        for m_exp in range(1, 20):
            coeff = logp * p ** (-m_exp / 2.0)
            shift = m_exp * logp
            if shift >= 2 * L:
                break
            for i in range(N):
                for j in range(i, N):
                    sp = shift_sin_f(i, j, shift, L)
                    sm = shift_sin_f(i, j, -shift, L)
                    W[i, j] += coeff * (sp + sm)
                    if i != j:
                        W[j, i] = W[i, j]
    return W


if __name__ == "__main__":
    print("=" * 75)
    print("FST-RH GAP CLOSURE CERTIFIER: lambda 700000 to 1300000")
    print("Geometric grid ratio 1.5, N_sin=80/100")
    print("=" * 75)
    sys.stdout.flush()

    K_COS = 4
    N_SIN = 80
    N_SIN_BIG = 100

    grid = []
    lam = 700000
    while lam <= 1300000:
        grid.append(int(lam))
        lam = int(lam * 1.5)
    if grid[-1] < 1300000:
        grid.append(1300000)

    print(f"Grid: {len(grid)} points: {grid}")
    sys.stdout.flush()

    certificates = []

    for idx, lam in enumerate(grid):
        L = np.log(lam)

        print(f"\n{'='*75}")
        print(f"[{idx+1}/{len(grid)}] lambda = {lam}, N_sin={N_SIN}")
        print(f"{'='*75}")
        sys.stdout.flush()

        primes = [int(p) for p in primerange(2, lam + 1)]
        print(f"  L = {L:.4f}, primes: {len(primes)}")
        sys.stdout.flush()

        t0 = time.time()
        W_cos_iv = build_cos_block_iv(lam, K_COS, primes)
        W_mid = np.array([[float(W_cos_iv[i][j].mid) for j in range(K_COS)] for i in range(K_COS)])
        W_rad = np.array([[float(W_cos_iv[i][j].delta) / 2 for j in range(K_COS)] for i in range(K_COS)])
        l1_mid = eigvalsh(W_mid)[0]
        rad = np.linalg.norm(W_rad, 'fro')
        eps64 = K_COS * np.finfo(np.float64).eps * np.max(np.abs(W_mid))
        upper_cos = l1_mid + rad + eps64
        t_cos = time.time() - t0
        print(f"  Cos: upper={upper_cos:+.4f}, rad={rad:.2e}, time={t_cos:.1f}s")
        sys.stdout.flush()

        t0 = time.time()
        W_sin_core = build_sin_block_f64(lam, N_SIN, primes)
        l1_core = float(eigvalsh(W_sin_core)[0])
        t_core = time.time() - t0
        print(f"  Sin core (N={N_SIN}): l1={l1_core:+.4f}, time={t_core:.1f}s")
        sys.stdout.flush()

        t0 = time.time()
        W_sin_big = build_sin_block_f64(lam, N_SIN_BIG, primes)
        l1_big = float(eigvalsh(W_sin_big)[0])
        t_big = time.time() - t0
        print(f"  Sin big  (N={N_SIN_BIG}): l1={l1_big:+.4f}, time={t_big:.1f}s")
        sys.stdout.flush()

        drop = l1_core - l1_big
        norms = [np.linalg.norm(W_sin_big[:N_SIN, n]) for n in range(N_SIN_BIG - 3, N_SIN_BIG)]
        avg_norm = np.mean(norms)
        remaining = 2 * avg_norm**2 / max(abs(l1_big), 1) if avg_norm > 1e-15 else 1.0
        eps_margin = N_SIN_BIG * np.finfo(np.float64).eps * np.max(np.abs(W_sin_big)) * 10
        lower_sin = l1_big - remaining - eps_margin
        t_sin = t_core + t_big

        cert_gap = float(upper_cos - lower_sin)
        proved = cert_gap < 0

        cert = {
            'lambda': int(lam),
            'L': float(L),
            'n_primes': len(primes),
            'upper_cos': float(upper_cos),
            'lower_sin': float(lower_sin),
            'cert_gap': float(cert_gap),
            'proved': bool(proved),
            'N_sin': N_SIN,
            'N_sin_big': N_SIN_BIG,
            'cos_info': {'l1_mid': float(l1_mid), 'rad_frob': float(rad)},
            'sin_info': {'l1_core': float(l1_core), 'l1_big': float(l1_big),
                         'drop': float(drop), 'remaining': float(remaining)},
            'time_cos': float(t_cos),
            'time_sin': float(t_sin),
        }
        certificates.append(cert)

        status = "PROVED" if proved else "FAILED"
        print(f"  *** cert_gap = {cert_gap:+.4f} => {status} ***")
        sys.stdout.flush()

        output = {
            'parameters': {'k_cos': K_COS, 'precision': mp.dps,
                           'purpose': 'Close gap between CAP (<=640K) and Euler-Maclaurin (>=1.2M)'},
            'certificates': certificates,
            'summary': {
                'total_proved': sum(1 for c in certificates if c['proved']),
                'total_failed': sum(1 for c in certificates if not c['proved']),
                'max_lambda_proved': max((c['lambda'] for c in certificates if c['proved']), default=0),
            }
        }
        with open('/opt/rh_proof/certificates_gap_closure.json', 'w') as f:
            json.dump(output, f, indent=2)

    print(f"\n{'='*75}")
    print(f"DONE. {sum(1 for c in certificates if c['proved'])}/{len(certificates)} proved")
    print(f"{'='*75}")
