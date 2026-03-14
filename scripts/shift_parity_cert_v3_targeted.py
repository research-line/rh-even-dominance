#!/usr/bin/env python3
"""
shift_parity_cert_v3_targeted.py
================================
Gezielte Zertifizierung der 169 Luecken aus v2.
Alle liegen um r ~ 1.8318 wo det ~ 0.

Strategie:
1. Feineres Gitter (100000 Subintervalle) im kritischen Bereich
2. Hoehere Praezision (60 Stellen)
3. Zusaetzlich: Direkte Eigenwert-Schranke via Gershgorin
"""

import mpmath
from mpmath import mpf, iv, nstr
import time

mpmath.mp.dps = 60  # 60 Stellen fuer hoehere Praezision

def D_entries_iv(r):
    """Alle 6 unabhaengigen Eintraege der 3x3-Matrix D(r) als Intervalle."""
    pi = iv.pi
    d00 = (2 - r) * (1 - iv.cos(pi * r)) - iv.sin(pi * r) / pi
    d11 = ((2 - r) * (iv.cos(pi * r) - iv.cos(2 * pi * r))
           - iv.sin(pi * r) / pi - iv.sin(2 * pi * r) / (2 * pi))
    d22 = ((2 - r) * (iv.cos(2 * pi * r) - iv.cos(3 * pi * r))
           - iv.sin(2 * pi * r) / (2 * pi) - iv.sin(3 * pi * r) / (3 * pi))
    d01 = ((3 * iv.sqrt(2) + 4) * iv.sin(pi * r) - 2 * iv.sin(2 * pi * r)) / (3 * pi)
    d02 = (-2 * iv.sqrt(2) * iv.sin(2 * pi * r) + iv.sin(3 * pi * r)
           - 3 * iv.sin(pi * r)) / (4 * pi)
    d12 = 2 * (19 * iv.sin(2 * pi * r) - 5 * iv.sin(pi * r)
               - 6 * iv.sin(3 * pi * r)) / (15 * pi)
    return d00, d11, d22, d01, d02, d12

def Tr_iv(r):
    """Trace von D_3."""
    return ((2 - r) * (1 - iv.cos(3 * iv.pi * r))
            - 2 * iv.sin(iv.pi * r) / iv.pi
            - iv.sin(2 * iv.pi * r) / iv.pi
            - iv.sin(3 * iv.pi * r) / (3 * iv.pi))

def det_iv(r):
    """Determinante von D_3."""
    d00, d11, d22, d01, d02, d12 = D_entries_iv(r)
    return (d00 * (d11 * d22 - d12**2)
            - d01 * (d01 * d22 - d12 * d02)
            + d02 * (d01 * d12 - d11 * d02))

def gershgorin_min_iv(r):
    """Untere Gershgorin-Schranke fuer kleinsten Eigenwert.

    Fuer symmetrische Matrix: lambda_min >= min_i (d_ii - sum_{j!=i} |d_ij|)
    """
    d00, d11, d22, d01, d02, d12 = D_entries_iv(r)

    # Gershgorin-Radien
    R0 = abs(d01) + abs(d02)
    R1 = abs(d01) + abs(d12)
    R2 = abs(d02) + abs(d12)

    # Untere Schranken fuer jeden Eigenwert
    lb0 = d00 - R0
    lb1 = d11 - R1
    lb2 = d22 - R2

    return lb0, lb1, lb2

def char_poly_bounds_iv(r):
    """Pruefe ob char. Polynom bei lambda=0 negatives Vorzeichen hat.
    p(0) = -det(D). Wenn det > 0, dann p(0) < 0, also mindestens eine neg. Nullstelle.
    Wenn det < 0, dann p(0) > 0 -- mehrdeutig.

    Besser: Pruefe p(-eps) fuer kleines eps > 0.
    p(lam) = -lam^3 + Tr*lam^2 - (sum 2x2 minors)*lam + det
    """
    d00, d11, d22, d01, d02, d12 = D_entries_iv(r)

    # 2x2 Hauptminoren
    m01 = d00 * d11 - d01**2
    m02 = d00 * d22 - d02**2
    m12 = d11 * d22 - d12**2
    sigma2 = m01 + m02 + m12  # Summe der 2x2 Minoren

    tr = d00 + d11 + d22
    det = d00 * m12 - d01 * (d01 * d22 - d12 * d02) + d02 * (d01 * d12 - d11 * d02)

    return tr, sigma2, det


if __name__ == "__main__":
    t0 = time.time()
    print("GEZIELTE ZERTIFIZIERUNG: Luecken-Region r ~ 1.83")
    print("=" * 70)

    # Kritischer Bereich aus v2: r in [1.70, 1.95]
    # Alle 169 Luecken lagen in [1.8317, 1.8318]
    # Erweitere leicht fuer Sicherheit
    regions = [
        (mpf('1.70'), mpf('1.85'), 5000),    # Breiter Bereich, mittelfeines Gitter
        (mpf('1.831'), mpf('1.833'), 50000),  # Enger Bereich, sehr feines Gitter
    ]

    total_certified = 0
    total_uncertified = 0
    uncertified_list = []

    for r_lo, r_hi, N in regions:
        print(f"\nRegion [{float(r_lo)}, {float(r_hi)}], {N} Intervalle")
        dr = (r_hi - r_lo) / N
        n_det = 0
        n_tr = 0
        n_gersh = 0
        n_unc = 0

        for i in range(N):
            ri_lo = r_lo + i * dr
            ri_hi = ri_lo + dr
            ri = iv.mpf([ri_lo, ri_hi])

            # Methode 1: det < 0
            d = det_iv(ri)
            if d.b < 0:
                n_det += 1
                continue

            # Methode 2: Tr < 0
            t = Tr_iv(ri)
            if t.b < 0:
                n_tr += 1
                continue

            # Methode 3: Gershgorin -- pruefe ob min Disk < 0
            lb0, lb1, lb2 = gershgorin_min_iv(ri)
            if lb0.b < 0 or lb1.b < 0 or lb2.b < 0:
                n_gersh += 1
                continue

            # Methode 4: Subintervalle (10x feiner)
            sub_ok = True
            n_sub = 20
            sub_dr = dr / n_sub
            for j in range(n_sub):
                rj_lo = ri_lo + j * sub_dr
                rj_hi = rj_lo + sub_dr
                rj = iv.mpf([rj_lo, rj_hi])

                d_s = det_iv(rj)
                if d_s.b < 0:
                    continue
                t_s = Tr_iv(rj)
                if t_s.b < 0:
                    continue
                lb0, lb1, lb2 = gershgorin_min_iv(rj)
                if lb0.b < 0 or lb1.b < 0 or lb2.b < 0:
                    continue

                # Noch feiner (20x)
                sub2_ok = True
                n_sub2 = 20
                sub2_dr = sub_dr / n_sub2
                for k in range(n_sub2):
                    rk_lo = rj_lo + k * sub2_dr
                    rk_hi = rk_lo + sub2_dr
                    rk = iv.mpf([rk_lo, rk_hi])

                    d_s2 = det_iv(rk)
                    if d_s2.b < 0:
                        continue
                    t_s2 = Tr_iv(rk)
                    if t_s2.b < 0:
                        continue
                    lb0, lb1, lb2 = gershgorin_min_iv(rk)
                    if lb0.b < 0 or lb1.b < 0 or lb2.b < 0:
                        continue

                    sub2_ok = False
                    uncertified_list.append((float(rk_lo), float(rk_hi),
                                            float(d_s2.a), float(d_s2.b),
                                            float(t_s2.a), float(t_s2.b)))

                if not sub2_ok:
                    sub_ok = False

            if sub_ok:
                n_gersh += 1  # zaehle als gemischt
            else:
                n_unc += 1

        print(f"  det < 0: {n_det}")
        print(f"  Tr < 0:  {n_tr}")
        print(f"  Gershgorin/Gemischt: {n_gersh}")
        print(f"  Nicht zertifiziert:  {n_unc}")
        total_certified += n_det + n_tr + n_gersh
        total_uncertified += n_unc

    dt = time.time() - t0

    print(f"\n{'='*70}")
    print(f"GESAMT: {total_certified} zertifiziert, {total_uncertified} offen")

    if total_uncertified == 0:
        print(f"\n*** LUECKEN GESCHLOSSEN! Vollstaendige Zertifizierung. ***")
    else:
        print(f"\n{total_uncertified} Regionen noch offen:")
        for reg in uncertified_list[:20]:
            print(f"  r in [{reg[0]:.12f}, {reg[1]:.12f}]: "
                  f"det in [{reg[2]:+.4e}, {reg[3]:+.4e}], "
                  f"Tr in [{reg[4]:+.4e}, {reg[5]:+.4e}]")

    # Diagnostik: Was passiert genau bei r = 1.8318?
    print(f"\n{'='*70}")
    print("DIAGNOSTIK bei r = 1.8318")
    mpmath.mp.dps = 60
    r_test = iv.mpf([mpf('1.83175'), mpf('1.83180')])
    d00, d11, d22, d01, d02, d12 = D_entries_iv(r_test)
    print(f"  D00 = [{float(d00.a):+.8e}, {float(d00.b):+.8e}]")
    print(f"  D11 = [{float(d11.a):+.8e}, {float(d11.b):+.8e}]")
    print(f"  D22 = [{float(d22.a):+.8e}, {float(d22.b):+.8e}]")
    print(f"  D01 = [{float(d01.a):+.8e}, {float(d01.b):+.8e}]")
    print(f"  D02 = [{float(d02.a):+.8e}, {float(d02.b):+.8e}]")
    print(f"  D12 = [{float(d12.a):+.8e}, {float(d12.b):+.8e}]")

    tr, sigma2, det = char_poly_bounds_iv(r_test)
    print(f"\n  Tr    = [{float(tr.a):+.8e}, {float(tr.b):+.8e}]")
    print(f"  Sigma2 = [{float(sigma2.a):+.8e}, {float(sigma2.b):+.8e}]")
    print(f"  det   = [{float(det.a):+.8e}, {float(det.b):+.8e}]")

    lb0, lb1, lb2 = gershgorin_min_iv(r_test)
    print(f"\n  Gershgorin untere Schranken:")
    print(f"    Disk 0: [{float(lb0.a):+.8e}, {float(lb0.b):+.8e}]")
    print(f"    Disk 1: [{float(lb1.a):+.8e}, {float(lb1.b):+.8e}]")
    print(f"    Disk 2: [{float(lb2.a):+.8e}, {float(lb2.b):+.8e}]")

    print(f"\nZeit: {dt:.1f}s")
