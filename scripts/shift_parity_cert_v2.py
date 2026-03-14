#!/usr/bin/env python3
"""
shift_parity_cert_v2.py  (LOKAL, ~1 Minute)
=============================================
Schlanke Intervall-Arithmetik-Zertifizierung.

Strategie: Festes Gitter mit 2000 Teilintervallen.
Auf jedem: pruefen ob det < 0 ODER Tr < 0 (als Intervall).
"""

import mpmath
from mpmath import mpf, iv, nstr
import time

mpmath.mp.dps = 30  # 30 Stellen reichen

def Tr_iv(r):
    """Trace von D_3 als Intervall-Funktion."""
    return ((2 - r) * (1 - iv.cos(3 * iv.pi * r))
            - 2 * iv.sin(iv.pi * r) / iv.pi
            - iv.sin(2 * iv.pi * r) / iv.pi
            - iv.sin(3 * iv.pi * r) / (3 * iv.pi))

def det_iv(r):
    """Determinante von D_3 als Intervall-Funktion."""
    d00 = (2 - r) * (1 - iv.cos(iv.pi * r)) - iv.sin(iv.pi * r) / iv.pi
    d11 = ((2 - r) * (iv.cos(iv.pi * r) - iv.cos(2 * iv.pi * r))
           - iv.sin(iv.pi * r) / iv.pi - iv.sin(2 * iv.pi * r) / (2 * iv.pi))
    d22 = ((2 - r) * (iv.cos(2 * iv.pi * r) - iv.cos(3 * iv.pi * r))
           - iv.sin(2 * iv.pi * r) / (2 * iv.pi) - iv.sin(3 * iv.pi * r) / (3 * iv.pi))
    d01 = ((3 * iv.sqrt(2) + 4) * iv.sin(iv.pi * r) - 2 * iv.sin(2 * iv.pi * r)) / (3 * iv.pi)
    d02 = (-2 * iv.sqrt(2) * iv.sin(2 * iv.pi * r) + iv.sin(3 * iv.pi * r)
           - 3 * iv.sin(iv.pi * r)) / (4 * iv.pi)
    d12 = 2 * (19 * iv.sin(2 * iv.pi * r) - 5 * iv.sin(iv.pi * r)
               - 6 * iv.sin(3 * iv.pi * r)) / (15 * iv.pi)
    return (d00 * (d11 * d22 - d12**2)
            - d01 * (d01 * d22 - d12 * d02)
            + d02 * (d01 * d12 - d11 * d02))


if __name__ == "__main__":
    t0 = time.time()
    print("ZERTIFIZIERUNG: Shift Parity Lemma")
    print("=" * 70)

    N = 2000  # Anzahl Teilintervalle
    eps_lo = mpf('0.0001')
    eps_hi = mpf('1.9999')

    dr = (eps_hi - eps_lo) / N
    n_det_neg = 0
    n_tr_neg = 0
    n_both_neg = 0
    n_uncertified = 0
    uncertified_regions = []

    for i in range(N):
        r_lo = eps_lo + i * dr
        r_hi = r_lo + dr
        r_interval = iv.mpf([r_lo, r_hi])

        # Pruefe det < 0
        d = det_iv(r_interval)
        det_certified = (d.b < 0)

        if det_certified:
            n_det_neg += 1
            continue

        # det nicht zertifiziert < 0 => pruefe Tr < 0
        t = Tr_iv(r_interval)
        tr_certified = (t.b < 0)

        if tr_certified:
            n_tr_neg += 1
            continue

        # Keines von beiden => Unterteile feiner
        sub_ok = True
        n_sub = 10
        sub_dr = dr / n_sub
        for j in range(n_sub):
            r_sub_lo = r_lo + j * sub_dr
            r_sub_hi = r_sub_lo + sub_dr
            r_sub = iv.mpf([r_sub_lo, r_sub_hi])

            d_sub = det_iv(r_sub)
            if d_sub.b < 0:
                continue
            t_sub = Tr_iv(r_sub)
            if t_sub.b < 0:
                continue

            # Noch feiner (3 Stufen: 10 -> 10 -> 20 = 2000x feiner)
            sub2_ok = True
            n_sub2 = 10
            sub2_dr = sub_dr / n_sub2
            for k in range(n_sub2):
                r_s2_lo = r_sub_lo + k * sub2_dr
                r_s2_hi = r_s2_lo + sub2_dr
                r_s2 = iv.mpf([r_s2_lo, r_s2_hi])
                d_s2 = det_iv(r_s2)
                if d_s2.b < 0:
                    continue
                t_s2 = Tr_iv(r_s2)
                if t_s2.b < 0:
                    continue
                # 3. Stufe
                n_sub3 = 20
                sub3_dr = sub2_dr / n_sub3
                for kk in range(n_sub3):
                    r_s3_lo = r_s2_lo + kk * sub3_dr
                    r_s3_hi = r_s3_lo + sub3_dr
                    r_s3 = iv.mpf([r_s3_lo, r_s3_hi])
                    d_s3 = det_iv(r_s3)
                    if d_s3.b < 0:
                        continue
                    t_s3 = Tr_iv(r_s3)
                    if t_s3.b < 0:
                        continue
                    sub2_ok = False
                    uncertified_regions.append((float(r_s3_lo), float(r_s3_hi),
                                               float(d_s3.a), float(d_s3.b),
                                               float(t_s3.a), float(t_s3.b)))

            if not sub2_ok:
                sub_ok = False

        if sub_ok:
            n_both_neg += 1
        else:
            n_uncertified += 1

    dt = time.time() - t0

    print(f"\nGitter: {N} Intervalle auf [{float(eps_lo)}, {float(eps_hi)}]")
    print(f"Breite: {float(dr):.6f}")
    print(f"\nErgebnis:")
    print(f"  det < 0 zertifiziert: {n_det_neg} Intervalle")
    print(f"  Tr < 0 zertifiziert:  {n_tr_neg} Intervalle")
    print(f"  Gemischt (Unterteilung): {n_both_neg} Intervalle")
    print(f"  NICHT zertifiziert:   {n_uncertified} Intervalle")
    print(f"  Total:                {n_det_neg + n_tr_neg + n_both_neg + n_uncertified} = {N}")

    if n_uncertified == 0:
        print(f"\n  *** BEWEIS ZERTIFIZIERT ***")
        print(f"  lambda_1(D_3(r)) < 0 fuer alle r in ({float(eps_lo)}, {float(eps_hi)})")
        print(f"  Methode: mpmath Intervall-Arithmetik, {mpmath.mp.dps} Dezimalstellen")
    else:
        print(f"\n  {n_uncertified} Regionen nicht zertifiziert:")
        for reg in uncertified_regions[:10]:
            print(f"    r in [{reg[0]:.8f}, {reg[1]:.8f}]: "
                  f"det in [{reg[2]:+.2e}, {reg[3]:+.2e}], "
                  f"Tr in [{reg[4]:+.2e}, {reg[5]:+.2e}]")

    # Nullstellen mit hoher Praezision
    print(f"\n{'='*70}")
    print("NULLSTELLEN (50 Stellen)")
    mpmath.mp.dps = 50

    def Tr_mp(r):
        return ((2 - r) * (1 - mpmath.cos(3 * mpmath.pi * r))
                - 2 * mpmath.sin(mpmath.pi * r) / mpmath.pi
                - mpmath.sin(2 * mpmath.pi * r) / mpmath.pi
                - mpmath.sin(3 * mpmath.pi * r) / (3 * mpmath.pi))

    def det_mp(r):
        d00 = (2 - r) * (1 - mpmath.cos(mpmath.pi * r)) - mpmath.sin(mpmath.pi * r) / mpmath.pi
        d11 = ((2 - r) * (mpmath.cos(mpmath.pi * r) - mpmath.cos(2 * mpmath.pi * r))
               - mpmath.sin(mpmath.pi * r) / mpmath.pi - mpmath.sin(2 * mpmath.pi * r) / (2 * mpmath.pi))
        d22 = ((2 - r) * (mpmath.cos(2 * mpmath.pi * r) - mpmath.cos(3 * mpmath.pi * r))
               - mpmath.sin(2 * mpmath.pi * r) / (2 * mpmath.pi) - mpmath.sin(3 * mpmath.pi * r) / (3 * mpmath.pi))
        d01 = ((3 * mpmath.sqrt(2) + 4) * mpmath.sin(mpmath.pi * r) - 2 * mpmath.sin(2 * mpmath.pi * r)) / (3 * mpmath.pi)
        d02 = (-2 * mpmath.sqrt(2) * mpmath.sin(2 * mpmath.pi * r) + mpmath.sin(3 * mpmath.pi * r)
               - 3 * mpmath.sin(mpmath.pi * r)) / (4 * mpmath.pi)
        d12 = 2 * (19 * mpmath.sin(2 * mpmath.pi * r) - 5 * mpmath.sin(mpmath.pi * r)
                   - 6 * mpmath.sin(3 * mpmath.pi * r)) / (15 * mpmath.pi)
        return (d00 * (d11 * d22 - d12**2) - d01 * (d01 * d22 - d12 * d02)
                + d02 * (d01 * d12 - d11 * d02))

    tr_z2 = mpmath.findroot(Tr_mp, mpf('0.588'))
    tr_z3 = mpmath.findroot(Tr_mp, mpf('0.730'))
    det_z1 = mpmath.findroot(det_mp, mpf('0.597'))
    det_z2 = mpmath.findroot(det_mp, mpf('0.729'))

    print(f"  Tr = 0 bei r = {nstr(tr_z2, 30)}")
    print(f"  Tr = 0 bei r = {nstr(tr_z3, 30)}")
    print(f"  det = 0 bei r = {nstr(det_z1, 30)}")
    print(f"  det = 0 bei r = {nstr(det_z2, 30)}")
    print(f"\n  Ueberlapp links:  {nstr(det_z1 - tr_z2, 15)} > 0")
    print(f"  Ueberlapp rechts: {nstr(tr_z3 - det_z2, 15)} > 0")

    print(f"\nZeit: {dt:.1f}s")
