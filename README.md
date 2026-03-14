# FST-RH: Even Dominance of the Weil Quadratic Form

A research project investigating the Riemann Hypothesis via Connes' spectral
program (arXiv:2602.04022). The central result is a computer-assisted
verification of even dominance for the Weil quadratic form QW_lambda, plus
an analytical framework (Leading-Mode Cancellation Lemma) that explains
*why* even dominance holds asymptotically.

## Paper Series (4 Parts)

| Paper | Title | Pages | Content |
|-------|-------|-------|---------|
| **Part I** | The Landscape | 7 | Thermodynamic foundation: Prime Hub Graphs, Li coefficients, archimedean/finite decomposition |
| **Part II** | The Path and the Dead Ends | 6 | Critical errors K1-K4, failed approaches, reorientation to Connes' program |
| **Part III** | Proof Closure | 37 | **Main paper.** Even dominance, Shift Parity Lemma, resolvent-damped M1'' framework, Leading-Mode Cancellation (c=2+sqrt(2)), 29 computer-assisted certificates |
| **Part IV** | Conclusio | 9 | Synthesis: catalog of proven results (R1-R9), excluded paths (K1-K4), open problems |

## Key Results

1. **Shift Parity Lemma**: S_cos(n,n,delta,L) - S_sin(n,n,delta,L) = -sin(pi*delta/L)/pi
   (every prime individually favors even eigenfunctions)

2. **29 Even Dominance Certificates**: lambda = 100 to 320,000, all rigorously verified
   via interval arithmetic. cert_gap grows as ~sqrt(lambda).

3. **Leading-Mode Cancellation Lemma**: The overlap differences cancel pairwise with
   exact constant c = 2 + sqrt(2), yielding (E_sin - E_cos)/D = O(1/L) -> 0.

4. **M1'' Corollary**: Resolvent-damped coupling subdominance implies Even Dominance
   for all sufficiently large lambda.

## Scripts

### Core (scripts/)

| Script | Purpose |
|--------|---------|
| `certifier_production.py` | Production certifier: lambda 200-10000, k_cos=4, interval arithmetic |
| `certifier_extended.py` | Extended certifier: lambda 10000-640000, geometric grid |
| `resolvent_analysis.py` | Dense-grid resolvent energy analysis (E_sin, E_cos, D) |
| `resolvent_R0K_test.py` | Neumann series convergence test: ||R_0 K|| < 1 |
| `partA_bounded_diff.py` | Mode decomposition of E_sin - E_cos |
| `partA_proof_sketch.py` | Overlap convergence analysis for leading modes |
| `step4_gap_growth.py` | Block-bound gap prediction (Schur complement) |
| `verify_H1_schranke.py` | Numerical verification of uniform H^1 bound |
| `shift_parity_cert_v2.py` | Interval-arithmetic certification of Shift Parity |
| `shift_parity_cert_v3_targeted.py` | Targeted certification at critical r-values |
| `hellmann_feynman_gap.py` | Hellmann-Feynman derivative analysis of the gap |
| `weighted_compactness_test.py` | H^1 bound, tightness, profile convergence tests |
| `weighted_compactness_server.py` | Server version of compactness tests |

### Results (scripts/_results/)

| File | Content |
|------|---------|
| `certificates.json` | 23 rigorous certificates (lambda 100-9201) |
| `certificates_extended.json` | 29 certificates (adds lambda 10000-320000) |
| `resolvent_analysis.json` | 12-point resolvent energy analysis |

### Exploration (scripts/_exploration/)

Historical exploration scripts from the development process. Not needed for
the current results but preserved for reference.

## Directory Structure

```
RH/
├── FST-RH_I_The_Landscape.tex/pdf      Part I
├── FST-RH_II_The_Path.tex/pdf           Part II
├── FST-RH_III_Proof_Closure.tex/pdf     Part III (main paper)
├── FST-RH_IV_Conclusio.tex/pdf          Part IV
├── fst-rh-references.bib                Bibliography
├── BEWEISNOTIZ.md                       Research notebook (internal)
├── scripts/                             Core computation scripts
│   ├── _results/                        JSON output from scripts
│   └── _exploration/                    Historical exploration scripts
└── archive/                             Archived materials
    ├── superseded/                      Papers subsumed by Part III
    ├── old_papers/                      Original FST-RH2-6 paper series
    └── old_scripts/                     Scripts from original series
```

## Server Computation

Certificates are computed on ellmos-services (Hetzner CCX13, 2 vCPU, 8 GB RAM).
The certifier uses interval arithmetic (mpmath.iv, 50-digit precision) for the
even block and float64 with Cauchy tail bounds for the odd block.

## Author

Lukas Geiger, Bernau, Germany
ORCID: [0009-0005-7296-1534](https://orcid.org/0009-0005-7296-1534)
