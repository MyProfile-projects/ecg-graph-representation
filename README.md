# Reversible graph representation of ECG dynamics

**for detecting functional state changes**

Code reproducing all numerical results, tables and figures of the paper.

---

## What the work is about

A long-term electrocardiogram recording is converted into a weighted directed graph:
vertices correspond to typical cardiac cycle shapes, edges to the probabilities of
transitions between them. Such a representation describes neither individual cardiac
cycles nor the recording as a whole, but the **dynamics of functional state changes** —
exactly what is lost in both conventional approaches.

The key property is **reversibility**: an electrocardiogram is reconstructed from the
graph, preserving both the waveform and the rhythm character of the original recording.

## Main results

| Quantity | Value |
| --- | --- |
| Cardiac cycles processed | 13 909 |
| Sensitivity of R wave detection | 100% |
| Cardiac cycle reconstruction error | 2.2 ± 0.6% of the signal range |
| Compression of a recording into a graph | ≈ 2800-fold |
| Shift in state occupancy under orthostasis | 0.59 ± 0.28, p < 10⁻⁵ |
| Conventional SDNN index | no change detected (p = 0.105) |

A functional state change was detected in 15 of the 20 subjects — where the integral
variability index stays silent.

## Installation

```
pip install numpy scipy pyedflib PyWavelets scikit-learn matplotlib
```

Python 3.9 or newer.

## Running

### Google Colab / Jupyter

Open `article_code.ipynb` and run the code cell: the annotation is validated on
synthetic signals (no external data required). Then:

```python
run_all('/content/data', roh_dir='/content/roh')       # full computation, 6 stages
make_figures('/content/data', roh_dir='/content/roh')  # the figures of the paper
```

### Command line

```
python ecg_graph_full.py --validate                    # validation, no data needed
python ecg_graph_full.py --data <folder>               # computation
python ecg_graph_full.py --data <folder> --roh <folder> --figures   # figures
```

## Computation stages

| Stage | Content |
| --- | --- |
| 1 | ECG annotation: R, P and T waves |
| 2 | Variability of the RR, PR and RT intervals and of the wave amplitudes |
| 3 | Graphs of the test phases, occupancy shift, reproducibility check |
| 4 | Comparison with k-means |
| 5 | Inverse transform: reconstruction accuracy and compression |
| 6 | Graphs of the test-database recordings with expert reports |

Separately: validation of the annotation on synthetic signals with known wave positions,
and construction of the figures of the paper.

## Data

The code expects two sets of recordings:

- **`data`** — 20 passive head-up tilt test recordings (`0011_ecg.edf` … `0030_ecg.edf`),
  three standard leads, sampling rate 500 Hz, EDF+ format with phase annotations
- **`roh`** — recordings from the test database of critical conditions
  (`07_VORO.edf`, `10_MITI.edf`, `01_GUSA.edf`, `03_SPI2.edf`), 200 Hz

The `roh_dir` argument is optional: without it stage 6 is skipped. The validation on
synthetic signals requires no external data at all.

The recordings themselves are not included in this repository: they were provided for
research use and contain data on human subjects.

## Reproducibility

The random number generator is initialised with a fixed value (`SEED = 42`). Robustness
of the result is additionally checked for five seeds: 0, 1, 7, 42, 123.

All algorithm parameters are specified in physical units — milliseconds and hertz, not
in samples. This matters: parameters tuned at 200 Hz and expressed in samples lead,
at 500 Hz, to a twofold overestimation of the heart rate.

## Structure of the code

A single file; nothing has to be installed or assembled separately.

| Section | Purpose |
| --- | --- |
| `detect_r` | R wave detection: the Pan–Tompkins scheme with a locally adaptive threshold |
| `detect_pt` | P and T wave detection: QRS suppression and a half-wave wavelet |
| `fourier_vec` | Fourier expansion of a cardiac cycle, the state vector |
| `cluster`, `build_graph` | clustering with a Kohonen map, the state digraph |
| `vec_to_cycle`, `graph_to_ecg` | inverse transform |
| `stage1` … `stage6` | computation stages |
| `validate`, `make_figures` | annotation check and construction of the figures |

## A note on the sample

The PR and RT indices are computed over all cardiac cycles in which the corresponding
wave was detected. Requiring the simultaneous presence of P and T would reduce the
sample without need: the P wave is not detectable in every recording — in lead II it is
weakly expressed in some subjects.

## Material outside the paper (in the same repository)

These files do not belong to the paper and do not reproduce its results; they are kept
here only for convenience, as separate self-contained material:

| File | What it is |
| --- | --- |
| `combined_rr_pp_pr.py` | Comparison of the variability of the RR (or PP) and PR series in the frequency bands 0.04–0.15 and 0.16–0.4 Hz |
| `pr_variability.ipynb` | Decomposition of PR variability into a slow and a fast component |
| `lead_selection_pipeline.py` | Automatic lead selection by the quality of P, Q, R, S and T wave detection |
| `pp_pr_standalone.py` | Wave annotation and computation of the PP and PR series; used by the two scripts above |
| `lead_selection.ipynb` | Colab notebook: lead selection and figures for the report, runs without the other files |

All of them were produced in the course of separate work preceding the thesis defence
and are unrelated to reproducing the paper.

## Related material

Material from the dissertation research, including the wavelet algorithms for wave
detection, is available in the repository
[Wavelet. P, R and T peaks detection](https://github.com/MyProfile-projects/Wavelet.-P--R--and-T-peaks-detection).
