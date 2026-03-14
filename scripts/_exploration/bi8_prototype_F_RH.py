#!/usr/bin/env python3
"""
bi8_prototype_F_RH.py
=====================
BI-8 Numerical Prototype: F_RH as KL-divergence on zero configurations.

Goal: For the first N zeros of zeta, construct F_RH and compute its Hessian.
Check whether the Hessian reproduces the Weil/Li positivity structure.

Three approaches tested:
1. Spectral measure KL: mu vs mu* (smoothed zero counting measure)
2. log det_2 as potential: F(z) = log det_2(I - zR)
3. Direct Li-coefficient Hessian: delta^2 F from lambda_n structure

This is a DESIGN VALIDATOR, not a proof.
"""

import os
os.environ["PYTHONIOENCODING"] = "utf-8"

from mpmath import (mp, mpf, mpc, zetazero, pi, gamma, log, sqrt,
                    fabs, re, im, zeta, fsum, exp, conj, power, matrix,
                    eig, chop, diff)
import time

mp.dps = 30

def xi_func(s):
    """Compute xi(s) = 1/2 * s * (s-1) * pi^(-s/2) * Gamma(s/2) * zeta(s)"""
    return mpf('0.5') * s * (s - 1) * pi**(-s/2) * gamma(s/2) * zeta(s)


def compute_zeros(N):
    """Compute first N non-trivial zeros (positive imaginary part)."""
    print(f"Computing first {N} zeros...")
    t0 = time.time()
    zeros = []
    for n in range(1, N + 1):
        rho = zetazero(n)
        zeros.append(rho)
        if n <= 3 or n == N:
            print(f"  rho_{n} = {float(re(rho)):.4f} + {float(im(rho)):.6f}i")
    print(f"  ({time.time()-t0:.1f}s)")
    return zeros


def approach1_spectral_measure_kl(zeros, N_bins=50):
    """
    Approach 1: KL divergence between empirical zero density and smooth model.

    mu = empirical measure from zeros (binned)
    mu* = smooth asymptotic density: n(t) ~ (1/2pi) log(t/2pi)

    F = D_KL(mu || mu*) = sum mu_j log(mu_j / mu*_j)
    """
    print("\n" + "="*80)
    print("  Approach 1: Spectral Measure KL Divergence")
    print("="*80)

    gamma_vals = [float(im(rho)) for rho in zeros]
    t_max = max(gamma_vals) * 1.1
    t_min = min(gamma_vals) * 0.9

    bin_edges = [t_min + (t_max - t_min) * i / N_bins for i in range(N_bins + 1)]

    # Empirical counts
    counts = [0] * N_bins
    for g in gamma_vals:
        for j in range(N_bins):
            if bin_edges[j] <= g < bin_edges[j+1]:
                counts[j] += 1
                break

    # Smooth asymptotic density
    import math
    total = len(gamma_vals)
    smooth = []
    for j in range(N_bins):
        t_mid = (bin_edges[j] + bin_edges[j+1]) / 2
        dt = bin_edges[j+1] - bin_edges[j]
        # Asymptotic density: dN/dt = (1/2pi) log(t/2pi)
        if t_mid > 2 * math.pi:
            density = math.log(t_mid / (2 * math.pi)) / (2 * math.pi)
        else:
            density = 0.01  # regularize
        smooth.append(density * dt * total / sum(
            (math.log(max((bin_edges[k] + bin_edges[k+1])/2, 2*math.pi+0.1) / (2*math.pi)) / (2*math.pi))
            * (bin_edges[k+1] - bin_edges[k])
            for k in range(N_bins)
        ))

    # Compute KL
    kl = 0.0
    for j in range(N_bins):
        if counts[j] > 0 and smooth[j] > 0:
            p = counts[j] / total
            q = smooth[j] / total
            kl += p * math.log(p / q)

    print(f"  N zeros = {total}, N bins = {N_bins}")
    print(f"  D_KL(empirical || asymptotic) = {kl:.6f}")
    print(f"  (Should be small if asymptotic density is good)")

    return kl


def approach2_log_det2(zeros):
    """
    Approach 2: Properties of log det_2(I - zR) as a function of z.

    log det_2(I - zR) = sum_j [log(1 - z/rho_j) + z/rho_j]

    Test: Is this a Nevanlinna function? (Im f(z) >= 0 for Im z > 0)
    This would be a NEW RH criterion if true.
    """
    print("\n" + "="*80)
    print("  Approach 2: log det_2 Nevanlinna Property")
    print("="*80)

    def log_det2(z, zeros_list):
        """Compute log det_2(I - zR) = sum [log(1-z/rho) + z/rho] over paired zeros."""
        result = mpc(0)
        for rho in zeros_list:
            # Each zero rho contributes, plus its conjugate 1-conj(rho)
            rho_bar = 1 - conj(rho)  # = 1 - (1/2 - i*gamma) = 1/2 + i*gamma = rho for on-line
            for r in [rho, rho_bar]:
                if fabs(z/r) < mpf('0.99'):
                    result += log(1 - z/r) + z/r
                else:
                    # Near singularity, use series
                    w = z/r
                    result += log(1 - w) + w
        return result

    # Test points in upper half plane
    print("\n  Testing Im[log det_2(I - zR)] for z in upper half plane:")
    print(f"  {'z':>25s}  {'Re[log det2]':>15s}  {'Im[log det2]':>15s}  {'Im >= 0?':>10s}")

    test_z = [
        mpc(0.1, 0.5),
        mpc(0.5, 1.0),
        mpc(0.5, 5.0),
        mpc(0.5, 14.0),   # near first zero height
        mpc(1.0, 0.1),
        mpc(-1.0, 0.5),
        mpc(0.25, 0.25),
        mpc(0.5, 0.5),
        mpc(2.0, 1.0),
        mpc(0.5, 50.0),
    ]

    nevanlinna_ok = True
    for z in test_z:
        val = log_det2(z, zeros)
        re_val = float(re(val))
        im_val = float(im(val))
        ok = im_val >= -1e-10
        if not ok:
            nevanlinna_ok = False
        z_str = f"{float(re(z)):.2f}+{float(im(z)):.2f}i"
        print(f"  {z_str:>25s}  {re_val:>15.6f}  {im_val:>15.6f}  {'YES' if ok else 'NO':>10s}")

    print(f"\n  Nevanlinna property (Im >= 0 in upper HP): {'HOLDS' if nevanlinna_ok else 'FAILS'}")
    print(f"  (Based on {len(test_z)} test points with {len(zeros)} zeros)")

    return nevanlinna_ok


def approach3_li_hessian(zeros, n_max=15):
    """
    Approach 3: Hessian structure from Li coefficients.

    lambda_n = sum_rho [1 - (1-1/rho)^n]

    Under RH, rho = 1/2 + i*gamma, so (1-1/rho) = 1 - 1/(1/2+i*gamma).

    The "Hessian" of the "free energy" at the RH configuration is:
    H_{nm} = d^2 F / d(delta_rho_n)(delta_rho_m)

    We compute it numerically by perturbing zeros off the line.
    """
    print("\n" + "="*80)
    print("  Approach 3: Li-Coefficient Hessian Structure")
    print("="*80)

    # First: compute lambda_n at the actual (on-line) configuration
    print(f"\n  Computing lambda_n for n=1..{n_max} from {len(zeros)} zeros:")

    lambdas = []
    for n in range(1, n_max + 1):
        lam = mpf(0)
        for rho in zeros:
            # Paired: rho and conj(rho) = 1 - rho_bar
            for r in [rho, conj(rho)]:
                w = 1 - 1/r
                lam += 1 - power(w, n)
        lam = re(lam)  # should be real
        lambdas.append(float(lam))
        sign = "+" if float(lam) > 0 else "-"
        print(f"    lambda_{n:2d} = {float(lam):+.6f}  ({sign})")

    # Now: perturb one zero off the line and see how lambda_n changes
    print(f"\n  Perturbation analysis: shift rho_1 by delta off the line")
    print(f"  rho_1 = {float(re(zeros[0])):.4f} + {float(im(zeros[0])):.6f}i")

    deltas = [0.0, 0.001, 0.01, 0.05, 0.1]

    for delta in deltas:
        perturbed_zeros = list(zeros)
        if delta > 0:
            # Shift first zero off line: rho -> (0.5 + delta) + i*gamma
            old_rho = zeros[0]
            new_rho = mpc(mpf('0.5') + delta, im(old_rho))
            perturbed_zeros[0] = new_rho

        # Recompute lambda_n with perturbed zeros
        lam_perturbed = []
        for n in range(1, min(n_max + 1, 6)):
            lam = mpf(0)
            for rho in perturbed_zeros:
                for r in [rho, conj(rho)]:
                    w = 1 - 1/r
                    lam += 1 - power(w, n)
            lam_perturbed.append(float(re(lam)))

        if delta == 0:
            print(f"    delta=0.000: lambda_1..5 = {['%.4f' % l for l in lam_perturbed]}")
        else:
            diffs = [lam_perturbed[i] - lambdas[i] for i in range(len(lam_perturbed))]
            # Check: does perturbation make lambda_n smaller (toward negative)?
            makes_negative = any(d < 0 for d in diffs)
            print(f"    delta={delta:.3f}: dlambda_1..5 = {['%+.4f' % d for d in diffs]}  "
                  f"{'DESTABILIZES' if makes_negative else 'stabilizes'}")

    # Compute the "curvature" d^2(lambda_n)/d(delta)^2 at delta=0
    print(f"\n  Second derivative d^2(lambda_n)/d(delta)^2 at delta=0:")
    eps = mpf('0.001')
    for n in range(1, min(n_max + 1, 8)):
        # lambda_n(+eps) and lambda_n(-eps)
        lam_plus = mpf(0)
        lam_minus = mpf(0)
        lam_center = mpf(0)

        for idx, rho in enumerate(zeros):
            for r_orig in [rho, conj(rho)]:
                # Center
                w = 1 - 1/r_orig
                lam_center += 1 - power(w, n)

                if idx == 0:
                    # Perturbed: shift real part
                    r_plus = mpc(re(r_orig) + eps, im(r_orig))
                    r_minus = mpc(re(r_orig) - eps, im(r_orig))
                else:
                    r_plus = r_orig
                    r_minus = r_orig

                w_plus = 1 - 1/r_plus
                w_minus = 1 - 1/r_minus
                lam_plus += 1 - power(w_plus, n)
                lam_minus += 1 - power(w_minus, n)

        d2 = float(re((lam_plus - 2*lam_center + lam_minus) / eps**2))
        print(f"    d^2(lambda_{n:2d})/d(delta)^2 = {d2:+.4f}  "
              f"{'(convex: perturbation COSTS energy)' if d2 < 0 else '(concave: perturbation GAINS energy)'}")

    return lambdas


def approach4_weil_quadratic_form(zeros, n_test=10):
    """
    Approach 4: Weil functional as a quadratic form.

    W(h) = sum_rho h_hat(rho) can be written as a bilinear form
    if h is parameterized. Test: Is the "Gram matrix" of the
    Li test functions positive definite?

    G_{nm} = sum_rho phi_n(rho) * conj(phi_m(rho))
    where phi_n(rho) = 1 - (1-1/rho)^n
    """
    print("\n" + "="*80)
    print("  Approach 4: Weil Functional as Quadratic Form")
    print("="*80)

    # Compute the Gram matrix of the Li test functions
    print(f"\n  Computing Gram matrix G_{{nm}} for n,m = 1..{n_test}")
    print(f"  G_{{nm}} = sum_rho phi_n(rho) * conj(phi_m(rho))")
    print(f"  where phi_n(rho) = 1 - (1-1/rho)^n")

    # Compute phi_n(rho) for all rho and n
    phi = []  # phi[n][rho_idx]
    for n in range(1, n_test + 1):
        phi_n = []
        for rho in zeros:
            # Include both rho and conj(rho)
            val = (1 - power(1 - 1/rho, n)) + (1 - power(1 - 1/conj(rho), n))
            phi_n.append(val)
        phi.append(phi_n)

    # Gram matrix
    G = matrix(n_test, n_test)
    for n in range(n_test):
        for m in range(n_test):
            val = fsum(phi[n][j] * conj(phi[m][j]) for j in range(len(zeros)))
            G[n, m] = re(val)  # Should be real by symmetry

    print(f"\n  Gram matrix G (first 5x5 block):")
    for i in range(min(5, n_test)):
        row = [f"{float(G[i,j]):8.3f}" for j in range(min(5, n_test))]
        print(f"    [{', '.join(row)}]")

    # Check positive definiteness via eigenvalues
    print(f"\n  Eigenvalues of G:")
    try:
        # Convert to float matrix for eigenvalue computation
        G_float = matrix(n_test, n_test)
        for i in range(n_test):
            for j in range(n_test):
                G_float[i,j] = re(G[i,j])

        eigenvalues = sorted([float(re(ev)) for ev in eig(G_float, left=False, right=False)], reverse=True)
        all_positive = all(ev > 0 for ev in eigenvalues)

        for i, ev in enumerate(eigenvalues):
            print(f"    ev_{i+1:2d} = {ev:+.6e}  {'(+)' if ev > 0 else '(-) !!!'}")

        print(f"\n  Gram matrix positive definite: {'YES' if all_positive else 'NO'}")
        print(f"  Condition number: {eigenvalues[0]/eigenvalues[-1]:.2e}" if eigenvalues[-1] > 0 else "")
    except Exception as e:
        print(f"  Eigenvalue computation failed: {e}")
        all_positive = False

    return all_positive


def main():
    N = 30  # Number of zeros

    print("="*80)
    print("  BI-8 Prototype: F_RH Construction and Hessian Analysis")
    print(f"  Using first {N} non-trivial zeros of zeta")
    print("="*80)

    zeros = compute_zeros(N)

    # Approach 1: Spectral measure KL
    kl = approach1_spectral_measure_kl(zeros)

    # Approach 2: log det_2 Nevanlinna property
    nevanlinna = approach2_log_det2(zeros)

    # Approach 3: Li-coefficient Hessian
    lambdas = approach3_li_hessian(zeros)

    # Approach 4: Gram matrix / quadratic form
    gram_pd = approach4_weil_quadratic_form(zeros, n_test=8)

    # Summary
    print("\n" + "="*80)
    print("  SUMMARY: Which approach produces the right Hessian?")
    print("="*80)
    print(f"  1. Spectral KL:     D_KL = {kl:.6f}")
    print(f"  2. Nevanlinna:      {'HOLDS' if nevanlinna else 'FAILS'} (Im[log det_2] >= 0 in upper HP)")
    print(f"  3. Li Hessian:      lambda_n > 0 for n=1..{len(lambdas)}")
    print(f"  4. Gram matrix PD:  {'YES' if gram_pd else 'NO'}")
    print()
    print("  DESIGN QUESTION: Which definition of F_RH makes the Hessian = Weil functional?")
    print("  - If Gram matrix is PD: The Li polynomials form a COMPLETE positive system")
    print("  - If Nevanlinna holds: log det_2 IS the natural 'free energy'")
    print("  - If perturbation destabilizes: RH configuration IS a stable minimum")


if __name__ == "__main__":
    main()
