#!/usr/bin/env python3
"""
subleading_gap.py
=================
Degenerate perturbation theory for the even-odd gap.

M_L = M_inf - (4/L) * F' + O(1/L^2)

where M_inf has l1 = -1 (degenerate, multiplicity 2).
The gap arises from the PROJECTED perturbation F' on the l1-eigenspace.

If min_eigval(F'_cos_projected) < min_eigval(F'_sin_projected),
then even dominance holds at first order in 1/L.
"""
import numpy as np

def f_cos_profile(n, m, u_val, L_big=1000):
    d = u_val * L_big
    if abs(d) > 2*L_big: return 0.0
    a, b = max(-L_big, d-L_big), min(L_big, d+L_big)
    if a >= b: return 0.0
    if n==0 and m==0: norm = 1/(2*L_big)
    elif n==0 or m==0: norm = 1/(L_big*np.sqrt(2))
    else: norm = 1/L_big
    kn, km = n*np.pi/L_big, m*np.pi/L_big
    r = 0
    for freq, phase in [(kn-km, km*d), (kn+km, -km*d)]:
        if abs(freq) < 1e-12: r += np.cos(phase)*(b-a)/2
        else: r += (np.sin(freq*b+phase) - np.sin(freq*a+phase))/(2*freq)
    return norm * r

def f_sin_profile(n, m, u_val, L_big=1000):
    d = u_val * L_big
    if abs(d) > 2*L_big: return 0.0
    a, b = max(-L_big, d-L_big), min(L_big, d+L_big)
    if a >= b: return 0.0
    norm = 1/L_big
    kn, km = (n+1)*np.pi/L_big, (m+1)*np.pi/L_big
    r = 0
    for freq, phase, sign in [(kn-km, km*d, 1), (kn+km, -km*d, -1)]:
        if abs(freq) < 1e-12: r += sign*np.cos(phase)*(b-a)/2
        else: r += sign*(np.sin(freq*b+phase) - np.sin(freq*a+phase))/(2*freq)
    return norm * r


if __name__ == "__main__":
    h = 1e-6

    print("DEGENERATE PERTURBATION THEORY FOR THE GAP")
    print("=" * 60)
    print()
    print("M_inf_cos = diag(+1, -1, +1, -1)")
    print("  l1 = -1, eigenspace = {e_1, e_3}")
    print("M_inf_sin = diag(-1, +1, -1, +1)")
    print("  l1 = -1, eigenspace = {e_0, e_2}")
    print()

    # COS: perturbation on l1=-1 eigenspace {e_1, e_3}
    # F'[ii,jj] = f'_ij(u=1) for (i,j) in {(1,1),(1,3),(3,1),(3,3)}
    F_cos = np.zeros((2,2))
    for ii, i in enumerate([1, 3]):
        for jj, j in enumerate([1, 3]):
            f_at_1 = f_cos_profile(i, j, 1.0)
            f_at_1mh = f_cos_profile(i, j, 1.0 - h)
            F_cos[ii, jj] = (f_at_1 - f_at_1mh) / h

    # SIN: perturbation on l1=-1 eigenspace {e_0, e_2}
    F_sin = np.zeros((2,2))
    for ii, i in enumerate([0, 2]):
        for jj, j in enumerate([0, 2]):
            f_at_1 = f_sin_profile(i, j, 1.0)
            f_at_1mh = f_sin_profile(i, j, 1.0 - h)
            F_sin[ii, jj] = (f_at_1 - f_at_1mh) / h

    print("F'_cos (perturbation on {e1,e3}):")
    for i in range(2):
        print(f"  [{F_cos[i,0]:+.6f}  {F_cos[i,1]:+.6f}]")
    ev_cos = np.sort(np.linalg.eigvalsh(F_cos))
    print(f"  Eigenvalues: {ev_cos[0]:.6f}, {ev_cos[1]:.6f}")
    print()

    print("F'_sin (perturbation on {e0,e2}):")
    for i in range(2):
        print(f"  [{F_sin[i,0]:+.6f}  {F_sin[i,1]:+.6f}]")
    ev_sin = np.sort(np.linalg.eigvalsh(F_sin))
    print(f"  Eigenvalues: {ev_sin[0]:.6f}, {ev_sin[1]:.6f}")
    print()

    min_cos = ev_cos[0]
    min_sin = ev_sin[0]
    diff = min_cos - min_sin

    print(f"min_eigval(F'_cos) = {min_cos:.6f}")
    print(f"min_eigval(F'_sin) = {min_sin:.6f}")
    print(f"Difference = {diff:.6f}")
    print()

    print("Gap prediction: l1_cos - l1_sin ~ -(4/L) * (min_cos - min_sin)")
    for L in [5, 7, 9, 12, 15]:
        gap_pred = -4/L * diff
        print(f"  L={L:2d}: gap/sqrt(lam) ~ {gap_pred:.4f}")

    print()
    if diff < 0:
        print("*** RESULT: min_eigval(F'_cos) < min_eigval(F'_sin) ***")
        print("*** The even perturbation is MORE NEGATIVE ***")
        print("*** => Even dominance at first order in 1/L ***")
        print("*** => M1 follows from degenerate PT + remainder control ***")
    elif diff > 0:
        print("*** RESULT: min_eigval(F'_cos) > min_eigval(F'_sin) ***")
        print("*** The subleading correction DOES NOT favor even ***")
        print("*** => Gap must come from higher-order effects ***")
    else:
        print("*** RESULT: Equal at first order ***")

    # Also check: what are the actual endpoint derivatives analytically?
    print()
    print("ANALYTICAL CHECK:")
    print("f_cos(1,1; u) = cos(pi*u)*(1-u/2) - sin(pi*u)/(2*pi)")
    print("f'_cos(1,1; 1) = -pi*sin(pi)*(1/2) + cos(pi)*(-1/2) - cos(pi*1)/(2) = 0 + 1/2 - (-1/2) = 1")
    # Wait: let me compute properly
    # f(u) = cos(pi*u)*(1-u/2) - sin(pi*u)/(2*pi)
    # f'(u) = -pi*sin(pi*u)*(1-u/2) + cos(pi*u)*(-1/2) - cos(pi*u)/(2)
    #       = -pi*sin(pi*u)*(1-u/2) - cos(pi*u)
    # f'(1) = -pi*sin(pi)*0 - cos(pi) = -(-1) = 1
    print("f'_cos(1,1; 1) = 1  (analytisch)")

    # f_sin(0,0; u) = cos(pi*u)*(1-u/2) + sin(pi*u)/(2*pi)
    # f'(u) = -pi*sin(pi*u)*(1-u/2) + cos(pi*u)*(-1/2) + cos(pi*u)/(2)
    #       = -pi*sin(pi*u)*(1-u/2)
    # f'(1) = -pi*sin(pi)*0 = 0
    print("f'_sin(0,0; 1) = 0  (analytisch)")
    print()
    print("So F'_cos[0,0] = f'_cos(1,1;1) = 1")
    print("   F'_sin[0,0] = f'_sin(0,0;1) = 0")
    print("The DIAGONAL of the perturbation already differs!")
