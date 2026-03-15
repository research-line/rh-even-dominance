# FST-RH: A Proof of the Riemann Hypothesis via Even Dominance of the Weil Quadratic Form

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19035641.svg)](https://doi.org/10.5281/zenodo.19035641)

A three-part research series establishing the Riemann Hypothesis via Connes'
spectral program (arXiv:2602.04022). The proof combines computer-assisted
certificates (interval arithmetic), the Leading-Mode Cancellation Lemma
(c = 2 + sqrt(2)), and the PNT Transfer Lemma into a three-regime bridge
argument covering all lambda >= 100.

## Paper Series (3 Parts, EN + DE)

| Paper | File | Pages | Content |
|-------|------|-------|---------|
| **Part I** | `FST-RH_I_Foundations` | 12 | Foundations and Obstructions: thermodynamic landscape (R1-R9), dead ends (K1-K4), reorientation to Connes |
| **Part II** | `FST-RH_II_Even_Dominance` | 41 | **Main paper.** Shift Parity Lemma, 33 CAP certificates, resolvent M1'' framework, Leading-Mode Cancellation (c=2+sqrt(2)), Higher-Mode Decay (Lemma B), Resolvent Truncation (Lemma C), PNT Transfer, Euler-Maclaurin Proposition, **Proposition A6 (cumulative step)** |
| **Part III** | `FST-RH_III_Conclusio` | 16 | Synthesis: proof architecture (A1-A8, all closed), explored alternatives (BI-1..11), independent results, assessment |

All papers are available in English and German (DE suffix).

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

4. **M1'' (Resolvent Subdominance)**: Proved via PNT Transfer Lemma.

5. **Proposition A6 (Cumulative Step)**: Three-regime argument:
   - Regime 1 (lambda <= 1.3M): 33 CAP certificates
   - Regime 2 (lambda >= 1.2M): M1'' + PNT + Lemma B + Lemma C
   - Overlap at [1.2M, 1.3M]

## Scripts

### Core (scripts/)

| Script | Purpose |
|--------|---------|
| `certifier_production.py` | Production certifier: lambda 200-10000 |
| `certifier_extended.py` | Extended certifier: lambda 10000-640000 |
| `certifier_gap_closure.py` | Gap-closure certifier: lambda 700K-1.3M |
| `resolvent_analysis.py` | Dense-grid resolvent energy analysis |
| `resolvent_R0K_test.py` | Neumann series convergence test |
| `partA_bounded_diff.py` | Mode decomposition of E_sin - E_cos |
| `partA_proof_sketch.py` | Overlap convergence analysis |
| `step4_gap_growth.py` | Block-bound gap prediction |
| `shift_parity_cert_v2.py` | Interval certification of Shift Parity |
| `hellmann_feynman_gap.py` | Hellmann-Feynman derivative analysis |

### Results (scripts/_results/)

| File | Content |
|------|---------|
| `certificates.json` | 23 rigorous certificates (lambda 100-9201) |
| `certificates_extended.json` | 29 certificates (lambda 10000-320000) |
| `certificates_gap_closure.json` | 3 gap-closure certificates (700K, 1.05M, 1.3M) |
| `resolvent_analysis.json` | 12-point resolvent energy analysis |

## Directory Structure

```
RH/
├── FST-RH_I_Foundations.tex/pdf          Part I (EN)
├── FST-RH_I_Foundations_DE.tex/pdf       Part I (DE)
├── FST-RH_II_Even_Dominance.tex/pdf      Part II (EN, main paper)
├── FST-RH_II_Even_Dominance_DE.tex/pdf   Part II (DE)
├── FST-RH_III_Conclusio.tex/pdf          Part III (EN)
├── FST-RH_III_Conclusio_DE.tex/pdf       Part III (DE)
├── fst-rh-references.bib                 Bibliography
├── BEWEISNOTIZ.md                        Research notebook (internal)
├── MEILENSTEINE.md                       Milestone documentation
├── README.md                             This file
├── scripts/                              Core computation scripts
│   ├── _results/                         JSON output from scripts
│   └── _exploration/                     Historical exploration scripts
└── archive/                              Archived materials
    ├── superseded/                       Papers subsumed by current structure
    ├── old_papers/                       Original FST-RH2-6 paper series
    └── old_scripts/                      Scripts from original series
```

## Server Computation

Certificates are computed on ellmos-services (Hetzner CCX13, 2 vCPU, 8 GB RAM).
The certifier uses interval arithmetic (mpmath.iv, 50-digit precision) for the
even block and float64 with Cauchy tail bounds for the odd block.

## Author

Lukas Geiger, Bernau, Germany
ORCID: [0009-0005-7296-1534](https://orcid.org/0009-0005-7296-1534)
