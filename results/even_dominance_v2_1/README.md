# Supplementary scripts for *Even Dominance* v2.1

This folder accompanies the paper *A Conditional Reduction of the Riemann
Hypothesis to Even Dominance of the Weil Quadratic Form*, version 2.1
(Zenodo record [10.5281/zenodo.21953431](https://doi.org/10.5281/zenodo.21953431),
concept DOI [10.5281/zenodo.19764771](https://doi.org/10.5281/zenodo.19764771)).

The paper names verification scripts that were not yet in this
repository. They are added here at the state the published version used.
The paper itself is unchanged, and so is its claim (conditional reduction, not
a proof of RH).

| Cited in v2.1 | File | Output |
|---|---|---|
| §2, §5 and §8.6, `verify_connes_vs_form.py` (values for λ = 100, 300, 1000) | [`scripts/verify_connes_vs_form.py`](../../scripts/verify_connes_vs_form.py) | [`verify_connes_vs_form_output.txt`](verify_connes_vs_form_output.txt) |
| §8.5 (diagonal identity, prime-summed entries for λ ∈ {100, 300, 1000}), `verify_diagonal_fast.py` | [`scripts/verify_diagonal_fast.py`](../../scripts/verify_diagonal_fast.py) | [`verify_diagonal_fast_output.txt`](verify_diagonal_fast_output.txt) |
| §5, `verify_lambda_star.py` (threshold λ* ≈ 175,000) | [`scripts/verify_lambda_star.py`](../../scripts/verify_lambda_star.py) (already present) | – |

## Reproduction

Python 3 with `numpy` and `sympy`. Each script runs in about one second:

```bash
python scripts/verify_connes_vs_form.py
python scripts/verify_diagonal_fast.py
```

The outputs in this folder were produced on 2026-11-22 with Python 3.12 and
NumPy 2.4.4. The values for λ = 1000, n = 1 (D₁ ≈ 11.89, S₁ ≈ 8.63, cosine
diagonal ≈ 3.26, sine diagonal ≈ 20.52) match the numbers printed in the paper.

## Changes against the version used for the paper

- `verify_connes_vs_form.py`: two lines added after `import numpy as np`, because
  NumPy 2.0 removed `np.trapz`. Under older NumPy versions the script is
  unchanged. Numerics and results are identical.
- `verify_diagonal_fast.py`: unchanged. Its closing interpretation text
  reflects the intermediate state of 14 May 2026. The paper (§8.6, update of
  15 May 2026) explains how this question was resolved with the Connes–van
  Suijlekom operator form.

## Not included

The paper also names working notes (`K0_*.md`). They contain intermediate
statuses and open follow-up steps that the published version has since
resolved or superseded. Under the project's rules for releasing proof notes
they are not published.
