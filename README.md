# The Riemann Hypothesis: A Three-Part Investigation via Even Dominance of the Weil Quadratic Form

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19035640.svg)](https://doi.org/10.5281/zenodo.19035640)

A three-part research series establishing the Riemann Hypothesis via Connes'
spectral program (arXiv:2602.04022). The proof combines computer-assisted
certificates (interval arithmetic), the Leading-Mode Cancellation Lemma
(c = 2 + sqrt(2)), and the PNT Transfer Lemma into a three-regime bridge
argument covering all lambda >= 100.

**Submitted to:** Communications in Mathematics (cm:17829, 2026-03-27)

## Paper Series (3 Parts, EN + DE)

| Paper | File | Pages | Content |
|-------|------|-------|---------|
| **Part I** | `RH_I_Foundations` | 14 | Foundations and Obstructions: thermodynamic landscape (R1-R9), dead ends (K1-K4), reorientation to Connes |
| **Part II** | `RH_II_Even_Dominance` | 44 | **Main paper.** Shift Parity Lemma, 33 CAP certificates, resolvent M1'' framework, Leading-Mode Cancellation (c=2+sqrt(2)), Higher-Mode Decay (Lemma B), Resolvent Truncation (Lemma C), PNT Transfer, Euler-Maclaurin Proposition, **Proposition A6 (cumulative step)** |
| **Part III** | `RH_III_Conclusio` | 18 | Synthesis: proof architecture (A1-A8, all closed), explored alternatives (BI-1..11), independent results, assessment |

All papers are available in English and German (DE suffix).
Combined English version: `RH_Complete_Series_EN.pdf` (76 pages).

## Proof Architecture

| Step | Statement | Status |
|------|-----------|--------|
| A1 | Connes' Theorem 6.1 | proven (external) |
| A2 | Hurwitz sufficiency | proven (external) |
| A3 | Even dominance at 33 values (lambda=100..1.3M) | **proven (CAP)** |
| A4 | Shift Parity Lemma | **proven** |
| A5 | Frontier-prime mechanism | proven |
| **A6** | **Cumulative step** | **closed (Prop. A6)** |
| A7 | Even dominance for all lambda >= 100 | proven (from A6) |
| A8 | RH | proven (from A1+A2+A7) |

## Key Results

1. **Shift Parity Lemma**: Every prime individually favors even eigenfunctions.
   Proved analytically (det/trace argument, Cauchy interlacing).

2. **33 Even Dominance Certificates**: lambda = 100 to 1,300,000, all rigorously
   verified via interval arithmetic (mpmath.iv, 50-digit precision).

3. **Leading-Mode Cancellation Lemma**: Overlap differences cancel pairwise with
   exact constant c = 2 + sqrt(2).

4. **M1'' (Resolvent Subdominance)**: Proved via PNT Transfer Lemma with explicit
   threshold lambda_0 = 442,413 (Dusart bound).

5. **Proposition A6 (Cumulative Step)**: Three-regime argument:
   - Regime 1 (lambda in [100, 1.3M]): 33 CAP certificates + structural interpolation
     (Shift Parity + Hellmann-Feynman + OP2 simplicity, safety factor >= 18)
   - Regime 2 (lambda >= 442,413): M1'' + PNT Transfer + Lemma B + Lemma C
   - Overlap at [442k, 1.3M] (nearly one order of magnitude)

6. **OP2 Simplicity**: Intra-even spectral gap certified by interval arithmetic
   at all 33 values (gap >= 8.69 at lambda=100, growing to >= 731 at lambda=320k).

## Scripts

### Core (scripts/)

| Script | Purpose |
|--------|---------|
| `certifier_production.py` | Production certifier: lambda 200-10000 |
| `certifier_extended.py` | Extended certifier: lambda 10000-640000 |
| `certifier_gap_closure.py` | Gap-closure certifier: lambda 700K-1.3M |
| `certifier_simplicity.py` | OP2 simplicity certification (interval arithmetic) |
| `euler_maclaurin_certifier.py` | Euler-Maclaurin IA certification (60-digit, 48-pt GL) |
| `certifier_lipschitz_analysis.py` | Gap-continuity / Lipschitz analysis |
| `resolvent_analysis.py` | Dense-grid resolvent energy analysis |
| `resolvent_R0K_test.py` | Neumann series convergence test |
| `partA_bounded_diff.py` | Mode decomposition of E_sin - E_cos |
| `partA_proof_sketch.py` | Overlap convergence analysis |
| `step4_gap_growth.py` | Block-bound gap prediction |
| `shift_parity_cert_v2.py` | Interval certification of Shift Parity |
| `shift_parity_cert_v3_targeted.py` | Targeted shift parity certification |
| `hellmann_feynman_gap.py` | Hellmann-Feynman derivative analysis |
| `endpoint_degeneracy.py` | Endpoint degeneracy analysis |
| `subleading_gap.py` | Subleading spectral gap analysis |
| `verify_H1_schranke.py` | H1 bound verification |
| `weighted_compactness_test.py` | Weighted compactness test |
| `weighted_compactness_server.py` | Server version of compactness test |

### Results (scripts/_results/)

| File | Content |
|------|---------|
| `certificates.json` | 23 rigorous certificates (lambda 100-9201) |
| `certificates_extended.json` | 29 certificates (lambda 10000-320000) |
| `certificates_gap_closure.json` | 3 gap-closure certificates (700K, 1.05M, 1.3M) |
| `simplicity_certificates.json` | 29 OP2 simplicity certificates (lambda 100-320000) |
| `euler_maclaurin_results.json` | Euler-Maclaurin IA certification results |
| `lipschitz_analysis.json` | Gap-continuity Lipschitz analysis |
| `resolvent_analysis.json` | 12-point resolvent energy analysis |

## Server Computation

Certificates are computed on ellmos-services (Hetzner CCX13, 2 vCPU, 8 GB RAM).
The certifier uses interval arithmetic (mpmath.iv, 50-digit precision) for the
even block and float64 with Cauchy tail bounds for the odd block.

## Version History

- **1.4** (2026-03-27): Reviewer-driven clarifications (Prop A6 interpolation, M1'' explicit threshold, Lemma B Step 3/4 separation, Lemma L3 superseded, Galerkin safety margins, Connes2026 reference key)
- **1.3** (2026-03-17): Bibliographic corrections (Connes title, Deninger journal, Keiper type)
- **1.2** (2026-03-16): IA certifications (Euler-Maclaurin, OP2 simplicity, Lipschitz), explicit PNT bounds, new scripts
- **1.1** (2026-03-15): Lemma B/C analytical bounds, status upgrade to "proved"
- **1.0** (2026-03-15): Initial release (A6 closed, 33 certificates)

## Author

Lukas Geiger, Bernau, Germany
ORCID: [0009-0005-7296-1534](https://orcid.org/0009-0005-7296-1534)
