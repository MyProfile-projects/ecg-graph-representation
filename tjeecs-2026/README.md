# Reversible graph representation of ECG dynamics — version 2026 (TJEECS submission)

This folder contains the complete, reproducible analysis for the revised version of the paper
*Reversible graph representation of ECG dynamics for detecting functional state changes*,
submitted to the **Turkish Journal of Electrical Engineering & Computer Sciences** (2026).

The notebook `article_code_ENG.ipynb` in the root of this repository belongs to the **earlier
version** of the paper and is kept unchanged. Everything for the current version is here.

## What is new in this version

* **Within-phase null control.** The state occupancy shift *D* is non-negative by construction,
  so it is compared not with zero but with the shift between two halves of the same phase
  (rest A vs rest B, tilt A vs tilt B), on segments of equal duration.
* **Rate-independent control.** The analysis is repeated without the cycle duration and with
  features taken from a fixed window (250 ms before to 450 ms after the R wave), which do not
  depend on the RR interval.
* **Drift control.** A null with the same time gap as between the rest and tilt segments.
* **Robustness.** Five random initialisations, 6 and 20 harmonics, SOM vs k-means.
* **One-command build.** All numbers, tables and figures of the paper are produced by one script.

## Key results (from `results/numbers.txt`)

| Analysis | Shift across the tilt | Within-phase null | Subjects | p (paired Wilcoxon) |
|---|---|---|---|---|
| State vector with cycle duration | 0.50 ± 0.29 | 0.15 ± 0.10 | 15 / 20 | 2.6 × 10⁻⁴ |
| Without cycle duration | 0.54 ± 0.29 | 0.15 ± 0.11 | 15 / 20 | 2.6 × 10⁻⁴ |
| Fixed window, independent of RR | 0.70 ± 0.28 | 0.10 ± 0.09 | 19 / 20 | 1.9 × 10⁻⁶ |
| Null with the same time gap | 0.48 ± 0.29 | 0.18 ± 0.13 | 15 / 20 | 1.7 × 10⁻³ |

Reconstruction error 2.2 ± 0.6 % of the signal range; compression ≈ 2800-fold;
SDNN did not change with tilt (p = 0.105).

## Structure

```
tjeecs-2026/
├── code/
│   ├── build_all.py        # entry point: data -> numbers, tables, figures (-> manuscript)
│   ├── ecg_graph.py        # core methods (annotation, Fourier state vectors, SOM, graphs)
│   ├── figures_tjeecs.py   # figures at true print size (Times, 7.5-9 pt, lines 0.5-1 pt)
│   ├── tex2docx.py         # optional: Word version of a compiled LaTeX manuscript
│   ├── check_refs.py       # optional: checks every DOI of the reference list against Crossref
│   └── fill_authors.py     # optional: first five authors by DOI (Crossref)
├── supplementary/
│   └── Supplementary_material.pdf  # Tables S1–S3, Figures S1–S2 (not in the 15-page journal PDF)
├── results/
│   ├── numbers.txt / .json # every number quoted in the paper
│   ├── tables/             # Table 1, Table 2, Supplementary Tables S1-S3 (CSV)
│   └── figures/            # Figures 1-5, Supplementary Figures S1-S2 (PDF)
└── requirements.txt
```

| Paper | File |
|---|---|
| Figure 1 — annotation of P, R, T waves | `results/figures/fig1.pdf` |
| Figure 2 — scheme of the phase comparison | `results/figures/fig2.pdf` |
| Figure 3 — occupancy shift vs null, state stability | `results/figures/fig3.pdf` |
| Figure 4 — forward and inverse transforms | `results/figures/fig4.pdf` |
| Figure 5 — full transformation cycle | `results/figures/fig5.pdf` |
| Supplementary Figures S1, S2 | `results/figures/figS1.pdf`, `figS2.pdf`; together with Tables S1–S3 in `supplementary/Supplementary_material.pdf` |
| Table 1 — annotation accuracy, synthetic ECG | `results/tables/table1.csv` |
| Table 2 — cardiac cycle parameters by phase | `results/tables/table2.csv` |
| Supplementary Tables S1–S3 | `results/tables/tableS1.csv` … `tableS3.csv` |

## How to run

Requires Python ≥ 3.9. Missing packages are installed automatically.

```bash
python code/build_all.py --data DIR_WITH_TILT_RECORDINGS --roh DIR_WITH_ROHMINE_RECORDINGS --out results
```

In Google Colab (one cell):

```python
!python /content/tjeecs-2026/code/build_all.py --data /content/data --roh /content/roh --out /content/results
```

Add `--fast` to skip the 5-seed robustness runs (about 2 min instead of 4).
With `--tex DIR` the script also copies figures and tables into a LaTeX manuscript folder,
checks that the key numbers quoted in the text match the computation, and compiles it.

## Data

* **Tilt-test recordings** (20 files `00NN_ecg.edf`, EDF+, 3 leads, 500 Hz) were provided by a
  third party for research use and contain data on human subjects. They are not public and are
  available from the corresponding author upon reasonable request with the permission of the
  providing institution.
* **Test-database recordings** (four files beginning with `07_`, `10_`, `01_`, `03_`) belong to the
  test database of critical conditions of the Russian Society of Holter Monitoring and Noninvasive
  Electrophysiology (ROHMINE, https://rohmine.org). File names may be in Cyrillic.

## Citation

The citation will be added after the review process is completed.
