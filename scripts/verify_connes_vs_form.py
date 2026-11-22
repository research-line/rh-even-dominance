"""
verify_connes_vs_form.py

Numerische Validierung der korrekten Connes-vS 2025 Operator-Form.

Setup (Connes-vS):
  Hilbertraum: L^2([0, L]), L = log lambda
  Basis: U_n(x) = exp(2 pi i n x/L) / sqrt(L), n in Z, supported on [0, L].
  Form: <U_m | U_n>_Q = integral_0^L [(U_m^* * U_n)(y) + (U_m^* * U_n)(-y)] dD(y)
  Weil distribution D auf [0, L]:
    D = (log 4pi + gamma) delta_0 + (archim kernel auf (0, L]) - sum_p sum_m (log p/p^(m/2)) delta_{m log p}

Diagonale (m = n):
  <U_n | U_n>_Q = integral_0^L 2 (1 - y/L) cos(2 pi n y/L) dD(y)

Off-Diagonale (m = -n, n != 0):
  <U_n | U_{-n}>_Q = -1/(pi n) * integral_0^L sin(2 pi n y/L) dD(y) =: -S_n

Reelle Cos/Sin-Basis innerhalb des U_n-Frameworks:
  cos_n := (U_n + U_{-n}) / sqrt(2), n >= 1
  sin_n := (U_n - U_{-n}) / (i sqrt(2)), n >= 1

Diagonalen in der Cos/Sin-Basis:
  <cos_n | cos_n>_Q = D_n - S_n
  <sin_n | sin_n>_Q = D_n + S_n

Wo:
  D_n = integral_0^L 2(1 - y/L) cos(2 pi n y/L) dD(y)
  S_n = (1/(pi n)) integral_0^L sin(2 pi n y/L) dD(y)

Verifikation: Cos und Sin Diagonalen unterscheiden sich um 2 S_n.

Erwartung: S_n != 0 fuer Weil-Distribution -> echte spektrale Asymmetrie.
"""

import math
import numpy as np
if not hasattr(np, "trapz"):  # NumPy >= 2.0 renamed trapz to trapezoid
    np.trapz = np.trapezoid
from sympy import primerange

GAMMA = 0.5772156649015329  # Euler-Mascheroni


def archim_kernel(y):
    """Archimedean kernel e^(y/2) / (2 sinh y) auf (0, infty)."""
    if y <= 0:
        return 0.0
    if y > 50:
        return 0.0  # exponentially small
    return math.exp(y / 2.0) / (2.0 * math.sinh(y))


def D_n_weil(n, lam, archim_nodes=2000):
    """
    D_n = integral_0^L 2 (1 - y/L) cos(2 pi n y/L) dD(y)

    Weil-Distribution D auf [0, L]:
      delta_0-term: 2 (log 4pi + gamma) * cos(0) = 2 (log 4pi + gamma)
      Hmm but delta at 0 — the weight at y=0 is 2(1-0)*cos(0) = 2.
      Aber: in unserem D ist die Konstante eigentlich ein delta_0,
      hier separat behandelt.

    Effektiv:
      D_n = 2(log 4pi + gamma)  (from delta_0, weight 2*1*1=2 at y=0,
                                  with mass log 4pi + gamma)
          + integral_0^L 2(1-y/L) cos(2 pi n y/L) k_arch(y) dy
            - 2 (1 - 0) integral_0^L cos(0) * e^(-y/2)/(2 sinh y)? hmm regularization
            wait, let's stick with the simpler form: integrand is k_arch(y) directly
          - sum_p sum_{m'} (log p/p^(m'/2)) * 2(1 - m' log p/L) cos(2 pi n m' log p/L)
            for m' log p < L (otherwise weight is 0 due to (1-y/L) being negative — actually,
            (1-y/L) = 0 at y=L and negative for y>L. So we stick with m' log p < L,
            i.e., p^m' < lambda.)
    """
    L = math.log(lam)

    # Constant term (delta at y=0)
    result = 2.0 * (math.log(4 * math.pi) + GAMMA)  # delta_0 contribution

    # Archimedean integral on (0, L]
    ys = np.linspace(0.001, L, archim_nodes)
    dy = ys[1] - ys[0]
    integrand = np.array([2.0 * (1 - y / L) * math.cos(2 * math.pi * n * y / L) * archim_kernel(y) for y in ys])
    # Also subtract the regularization term 2 e^(-y/2) which is to handle the singularity at 0
    # Actually in the EvenDom paper the integrand is [<f, (T_y + T_{-y})f> - 2 e^(-y/2) ||f||^2] * k_arch
    # For the diagonal n=n, we have <U_n, (T_y + T_{-y}) U_n> = 2(1-y/L)cos(2 pi n y/L)
    # Regularization: -2 e^(-y/2) for the delta_{nn} part
    regul = np.array([-2.0 * math.exp(-y / 2.0) * archim_kernel(y) for y in ys])
    archim_part = np.trapz(integrand + regul, ys)
    result += archim_part

    # Prime sum (with minus sign per Weil explicit formula; sign convention may need check)
    prime_part = 0.0
    pmax = int(lam) + 1
    for p in primerange(2, pmax + 1):
        logp = math.log(p)
        for m in range(1, 20):
            y = m * logp
            if y >= L:
                break
            weight = logp / (p ** (m / 2.0))
            kernel_val = 2.0 * (1 - y / L) * math.cos(2 * math.pi * n * y / L)
            prime_part -= weight * kernel_val  # MINUS sign per Weil formula
    result += prime_part
    return result


def S_n_weil(n, lam, archim_nodes=2000):
    """
    S_n = (1/(pi n)) * integral_0^L sin(2 pi n y/L) dD(y)

    Konstante: delta_0 gibt 0 (sin(0) = 0).
    Archim: integral of sin * k_arch.
    Primes: sum -log(p)/p^(m/2) sin(2 pi n m log p/L) / (pi n).
    """
    if n == 0:
        return 0.0
    L = math.log(lam)

    # Constant term: sin(0) = 0
    result = 0.0

    # Archimedean integral
    ys = np.linspace(0.001, L, archim_nodes)
    integrand = np.array([math.sin(2 * math.pi * n * y / L) * archim_kernel(y) for y in ys])
    # Note: regularization term -2 e^(-y/2) is for diagonal only, doesn't enter sin.
    archim_part = np.trapz(integrand, ys)
    result += archim_part

    # Prime sum
    prime_part = 0.0
    pmax = int(lam) + 1
    for p in primerange(2, pmax + 1):
        logp = math.log(p)
        for m in range(1, 20):
            y = m * logp
            if y >= L:
                break
            weight = logp / (p ** (m / 2.0))
            prime_part -= weight * math.sin(2 * math.pi * n * y / L)
    result += prime_part

    return result / (math.pi * n)


def main():
    print("=" * 80)
    print("Verify Connes-vS Operator Form: D_n and S_n, Cos/Sin Asymmetry")
    print("=" * 80)
    print()
    print("Hilbert space: L^2([0, L]), complex exponential basis U_n.")
    print("Cos diag = D_n - S_n, Sin diag = D_n + S_n.")
    print("Difference (Sin - Cos) = 2 S_n. NON-ZERO => spectral asymmetry.")
    print()

    lams = [100.0, 300.0, 1000.0]
    print(f"{'lambda':>8} {'n':>4} {'D_n':>14} {'S_n':>14} {'Cos diag':>14} {'Sin diag':>14} {'2 S_n':>14}")
    print("-" * 90)
    for lam in lams:
        for n in [1, 2, 3]:
            D = D_n_weil(n, lam)
            S = S_n_weil(n, lam)
            cos_diag = D - S
            sin_diag = D + S
            asym = 2 * S
            print(f"{lam:>8.0f} {n:>4} {D:>14.4f} {S:>14.4f} {cos_diag:>14.4f} {sin_diag:>14.4f} {asym:>14.4f}")
        print()

    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("If 2 S_n is non-zero, the cos- and sin-sector diagonals genuinely differ")
    print("in the correct Connes-vS form -> Even Dominance can be a non-trivial")
    print("question whose answer depends on the sign and magnitude of S_n.")
    print()
    print("If S_n > 0 for the minimum-eigenvalue mode, then Cos diag < Sin diag at")
    print("that mode, supporting Even Dominance asymptotically.")
    print()


if __name__ == "__main__":
    main()
