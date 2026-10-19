# -*- coding: utf-8 -*-
"""
ONE-CLICK BUILD of the paper
"Reversible graph representation of ECG dynamics for detecting functional state changes"

    python build_all.py --data DIR_TILT --roh DIR_ROHMINE [--tex DIR_MANUSCRIPT]

or in Colab / Jupyter (one cell):

    !python build_all.py --data /content/data --roh /content/roh --tex /content/manuscript

What it does, from start to finish:
  1. validation on synthetic ECG ................... Table 1
  2. annotation of the 20 tilt-test recordings ..... Table 2, Supplementary Table S1
  3. phase graphs + within-phase NULL CONTROL ...... Figure "shift", robustness (5 seeds, 6/20 harmonics)
  4. comparison with k-means
  5. inverse transform (reconstruction, compression)
  6. test-database recordings (ROHMINE) ........... Table 4 / Supplementary Table S2
  7. all figures as PDF (Times, line weight 0.5-1 pt at 16 cm width), as TJEECS requires
  8. tables as ready LaTeX files + CSV, all numbers in results/numbers.txt
  9. with --tex: copies figures and tables into the manuscript folder, checks that the
     numbers quoted in the text match the computation, and compiles the PDFs (if pdflatex exists)

Files expected
  --data : the tilt-test recordings 00NN_ecg.edf
  --roh  : the four ROHMINE recordings; any names starting with 07_, 10_, 01_, 03_
           (Cyrillic names such as 07_ВОРО.edf are fine)
"""
import argparse, csv, glob, json, os, re, shutil, subprocess, sys, time

# ---------------------------------------------------------------- dependencies
def _ensure(pkgs):
    import importlib
    missing = []
    for mod, pip_name in pkgs:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print('Installing:', ' '.join(missing))
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])

_ensure([('numpy', 'numpy'), ('scipy', 'scipy'), ('pyedflib', 'pyedflib'),
         ('pywt', 'PyWavelets'), ('sklearn', 'scikit-learn'), ('matplotlib', 'matplotlib')])

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def _find_here():
    """Folder with ecg_graph.py. Works both as a script (__file__ exists) and when the
    code is pasted into a Colab/Jupyter cell (no __file__ there)."""
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        pass
    cwd = os.getcwd()
    for cand in (cwd, os.path.join(cwd, 'pipeline'), '/content/paper/pipeline', '/content/pipeline', '/content'):
        if os.path.exists(os.path.join(cand, 'ecg_graph.py')):
            return cand
    for root, _dirs, fs in os.walk(cwd):
        if 'ecg_graph.py' in fs:
            return root
    raise SystemExit('ecg_graph.py not found. Put it in the same folder as build_all.py '
                     '(or in the current folder) and run again.')


HERE = _find_here()
sys.path.insert(0, HERE)
import ecg_graph as eg                     # the original code of the paper
from ecg_graph import (SEED, N_HARM, CH_TILT, cluster, build_graph, vectors_with_time,
                       read_with_marks, phase_bounds, detect_r, detect_pt,
                       generate_synthetic_ecg, match_peaks, se_pp_f1, SCENARIOS, TOL, DELAY,
                       stage1_detection, stage5_inverse, _graph_data, _draw_graph,
                       fig1_scheme, fig2_waves, fig6_roh, fig7_reversibility, fig9_cycle,
                       DARK, ACC, BLUE, GREY)

import figures_tjeecs as F                  # figures drawn at true print size

# ================================================================ within-phase null control
def occupancy_shift(lab, a, b, n):
    wa = np.array([(lab[a] == c).mean() for c in range(n)])
    wb = np.array([(lab[b] == c).mean() for c in range(n)])
    return 0.5 * np.abs(wa - wb).sum()


def segments(tt, p1e, p2b, gapped=False):
    """Four equal segments: last 2L s of rest (A, B) and first 2L s of tilt (A, B).
    gapped=True: inside each phase A and B are separated by the same time gap that
    separates rest B from tilt A (control for slow drift over time)."""
    r0, r1 = 30.0, p1e - 5.0
    t0, t1 = p2b + 15.0, p2b + 275.0
    if not gapped:
        L = min(r1 - r0, t1 - t0) / 2.0
        return dict(rA=(tt > r1 - 2 * L) & (tt <= r1 - L), rB=(tt > r1 - L) & (tt <= r1),
                    tA=(tt > t0) & (tt <= t0 + L), tB=(tt > t0 + L) & (tt <= t0 + 2 * L),
                    eR=(tt > r1 - L) & (tt <= r1), eT=(tt > t0) & (tt <= t0 + L))
    g = t0 - r1
    L = (min(r1 - r0, t1 - t0) - g) / 2.0
    return dict(rA=(tt > r0) & (tt <= r0 + L), rB=(tt > r0 + L + g) & (tt <= r0 + 2 * L + g),
                tA=(tt > t0) & (tt <= t0 + L), tB=(tt > t0 + L + g) & (tt <= t0 + 2 * L + g),
                eR=(tt > r1 - L) & (tt <= r1), eT=(tt > t0) & (tt <= t0 + L))


def null_one(X, tt, p1e, p2b, seed=SEED, gapped=False):
    s = segments(tt, p1e, p2b, gapped)
    sel = np.zeros(len(tt), bool)
    for m in s.values():
        sel |= m
    C, lab = cluster(X[sel], n_neurons=30, merge_frac=0.18, min_size=15, seed=seed)
    n = len(C); m = {k: v[sel] for k, v in s.items()}
    return dict(null_rest=occupancy_shift(lab, m['rA'], m['rB'], n),
                null_tilt=occupancy_shift(lab, m['tA'], m['tB'], n),
                effect=occupancy_shift(lab, m['eR'], m['eT'], n))


def null_summary(feats, seed=SEED, gapped=False):
    """feats: {subject: (X, tt, p1e, p2b)}"""
    names = sorted(feats)
    rows = [null_one(*feats[k], seed=seed, gapped=gapped) for k in names]
    eff = np.array([r['effect'] for r in rows])
    nr = np.array([r['null_rest'] for r in rows]); nt = np.array([r['null_tilt'] for r in rows])
    null = (nr + nt) / 2
    return dict(effect_mean=eff.mean(), effect_sd=eff.std(ddof=1), null_mean=null.mean(), null_sd=null.std(ddof=1),
                wins=int((eff > np.maximum(nr, nt)).sum()), n=len(eff), p=float(stats.wilcoxon(eff, null)[1]),
                names=names, effect=eff, null_max=np.maximum(nr, nt))


def window_features(path, pre=0.25, post=0.45, step=0.02):
    """Shape features independent of the RR interval: the signal from 250 ms before
    to 450 ms after each R wave, sampled every 20 ms (baseline = first 50 ms)."""
    f = eg.pyedflib.EdfReader(path); fs = float(f.getSampleFrequency(CH_TILT))
    sig = f.readSignal(CH_TILT).astype(float) / 1000.0; f.close()
    bad, hf = eg.bad_mask(sig, fs); r, _ = detect_r(sig, fs)
    a, b, st = int(pre * fs), int(post * fs), int(step * fs)
    X, tt = [], []
    for i in range(len(r) - 1):
        i0, i1 = r[i], r[i + 1]; d = (i1 - i0) / fs
        if d < 0.3 or d > 2.0 or bad[i0:i1].any() or i0 - a < 0 or i0 + b >= len(hf) or bad[i0 - a:i0 + b].any():
            continue
        w = hf[i0 - a:i0 + b]; X.append(w[::st] - w[:int(0.05 * fs)].mean()); tt.append(i0 / fs)
    return np.array(X), np.array(tt)


# ================================================================ helpers for tables
def pm(m, s, d):
    return f'{m:.{d}f} $\\pm$ {s:.{d}f}'


def pfmt(p):
    return '$<$ 0.001' if p < 0.001 else f'{p:.3f}'


def sepp(se, pp):
    f = lambda v: '100' if abs(v - 100) < 1e-9 else f'{v:.1f}'
    return f'{f(se)} / {f(pp)}'


def tex_table(caption, label, colspec, header, body_rows, hlines_after=(), small=True):
    L = ['\\begin{table}[htbp]']
    if label and label.startswith('S'):
        L.append('\\renewcommand{\\thetable}{%s}' % label)
        L.append('\\caption{%s}' % caption)
    else:
        L.append('\\caption{%s}' % caption)
        if label:
            L.append('\\label{%s}' % label)
    L += ['\\begin{center}' + ('\\small' if small else ''), '\\begin{tabular}{%s}' % colspec, '\\hline',
          ' & '.join(header) + ' \\\\', '\\hline']
    for i, r in enumerate(body_rows):
        L.append(' & '.join(r) + ' \\\\')
        if i in hlines_after:
            L.append('\\hline')
    L += ['\\hline', '\\end{tabular}', '\\end{center}' + ('' if (label or '').startswith('S') else '\\vs{-4mm}'),
          '\\end{table}']
    return '\n'.join(L) + '\n'


def write_csv(path, header, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def strip_tex(s):
    return (s.replace('$\\pm$', '±').replace('$<$', '<').replace('$', '').replace('\\%', '%')
             .replace('\\_', '_').replace('$^{\\circ}$', '°'))


# ================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='folder with the tilt-test recordings 00NN_ecg.edf')
    ap.add_argument('--roh', help='folder with the four ROHMINE recordings')
    ap.add_argument('--out', default='results', help='output folder (default: results)')
    ap.add_argument('--tex', help='manuscript folder: figures/tables are copied there and the PDFs compiled')
    ap.add_argument('--fast', action='store_true', help='skip the 5-seed / 20-harmonic robustness runs')
    args = ap.parse_args(argv)

    t_start = time.time()
    OUT = os.path.abspath(args.out); FIG = os.path.join(OUT, 'figures'); TAB = os.path.join(OUT, 'tables')
    os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
    N = {}                                       # every number of the paper

    files = sorted(glob.glob(os.path.join(args.data, '00*_ecg.edf'))) or \
        sorted(glob.glob(os.path.join(args.data, '*.edf')))
    if not files:
        sys.exit(f'No .edf recordings found in {args.data}')
    print(f'Tilt-test recordings: {len(files)}')

    roh = {}
    if args.roh:
        names = {'07': ('07_VORO', 'bradycardia, narrow QRS'), '10': ('10_MITI', 'bradycardia, narrow QRS'),
                 '01': ('01_GUSA', 'sudden tachycardia, narrow QRS'), '03': ('03_SPI2', 'sudden tachycardia, narrow QRS')}
        for f in sorted(glob.glob(os.path.join(args.roh, '*.edf'))):
            pre = os.path.basename(f)[:2]
            if pre in names:
                roh[pre] = (names[pre][0], names[pre][1], f)
        print(f'ROHMINE recordings: {len(roh)} of 4')

    # ------------------------------------------------------------ 1. synthetic validation -> Table 1
    print('\n[1/6] Validation on synthetic ECG')
    agg = {k: [0, 0, 0] for k in 'RPT'}; t1_rows = []
    for s in SCENARIOS:
        sig, fs_, true_r = generate_synthetic_ecg(fs=500.0, duration_s=120.0, hr_bpm=s['hr'],
                                                  hrv_pct=s['hrv'], noise_level=s['noise'], seed=s['seed'])
        det, _ = detect_r(sig, fs_); Pd, Td, _, _ = detect_pt(sig, det, fs_)
        truth = dict(R=true_r, P=(true_r + DELAY['P'] * fs_).astype(int), T=(true_r + DELAY['T'] * fs_).astype(int))
        found = dict(R=det, P=Pd[Pd > 0], T=Td[Td > 0])
        row = [s['name'], str(s['hr'])]
        for k in 'RPT':
            TP, FN, FP, _ = match_peaks(truth[k], np.asarray(found[k]), fs_, tol_ms=TOL[k])
            se, pp, _ = se_pp_f1(TP, FN, FP)
            row.append(sepp(se * 100, pp * 100))
            agg[k][0] += TP; agg[k][1] += FN; agg[k][2] += FP
        t1_rows.append(row)
    n_syn = agg['R'][0] + agg['R'][1]
    tot = [f'Total ({n_syn} cycles)', '--']
    for k in 'RPT':
        se, pp, _ = se_pp_f1(*agg[k]); tot.append(sepp(se * 100, pp * 100))
        N[f'syn_{k}_Se'] = round(se * 100, 1); N[f'syn_{k}_PPV'] = round(pp * 100, 1)
    N['syn_cycles'] = n_syn
    t1_rows.append(tot)
    hdr1 = ['Scenario', 'HR, bpm', 'R: Se / PPV, \\%', 'P: Se / PPV, \\%', 'T: Se / PPV, \\%']
    open(os.path.join(TAB, 'table1.tex'), 'w').write(tex_table(
        'Annotation accuracy on synthetic electrocardiograms.', 'tab:1', '|l|c|c|c|c|', hdr1, t1_rows,
        hlines_after=(len(t1_rows) - 2,)))
    write_csv(os.path.join(TAB, 'table1.csv'), [strip_tex(h) for h in hdr1], t1_rows)

    # ------------------------------------------------------------ 2. annotation -> Table 2, Table S1
    print('\n[2/6] Annotation of the tilt-test recordings')
    rows = stage1_detection(files)
    N['cycles_total'] = int(sum(r['R'] for r in rows))
    N['P_detected_pct'] = round(100 * sum(r['P'] for r in rows) / N['cycles_total'], 1)
    N['T_detected_pct'] = round(100 * sum(r['T'] for r in rows) / N['cycles_total'], 1)
    # heart rate = 60 / median RR of each phase
    hr = {1: [], 2: []}
    for path in files:
        sig, fs, marks = read_with_marks(path); p1e, p2b = phase_bounds(marks)
        r, _ = detect_r(sig, fs); t_r = r / fs; rr = np.diff(r) / fs
        hr[1].append(60 / np.median(rr[(t_r[:-1] > 30) & (t_r[:-1] < p1e - 5)]))
        hr[2].append(60 / np.median(rr[(t_r[:-1] > p2b + 15) & (t_r[:-1] < p2b + 275)]))

    def pair(grp, key):
        x = np.array([r.get(f'{grp}1', {}).get(key, np.nan) for r in rows], float)
        y = np.array([r.get(f'{grp}2', {}).get(key, np.nan) for r in rows], float)
        m = np.isfinite(x) & np.isfinite(y)
        return x[m], y[m]

    spec2 = [('RR, ms', ('rr', 'mean'), 0), ('HR, bpm', None, 1), ('PR, ms', ('pr', 'mean'), 0),
             ('RT, ms', ('rt', 'mean'), 0), ('pNN50, \\%', ('rr', 'pNN50'), 1), ('RMSSD, ms', ('rr', 'RMSSD'), 1),
             ('SDNN, ms', ('rr', 'SDNN'), 1), ('T amplitude, mV', ('ampT', 'mean'), 1), ('SD of RT, ms', ('rt', 'SD'), 1)]
    t2_rows = []; n_pr = None
    for lab, key, d in spec2:
        x, y = (np.array(hr[1]), np.array(hr[2])) if key is None else pair(*key)
        if key == ('pr', 'mean'):
            n_pr = len(x)
        p = stats.wilcoxon(x, y)[1]; dl = y.mean() - x.mean()
        dd = 0 if lab.startswith('RR') else 1
        t2_rows.append([lab, pm(x.mean(), x.std(ddof=1), d), pm(y.mean(), y.std(ddof=1), d),
                        f'${dl:+.{dd}f}$'.replace('+', '+').replace('-', '-'), pfmt(p)])
        N['T2 ' + strip_tex(lab)] = dict(rest=round(x.mean(), 3), tilt=round(y.mean(), 3), delta=round(dl, 3), p=p, n=len(x))
    hdr2 = ['Parameter', 'Rest', 'Tilt 45$^{\\circ}$', '$\\Delta$', '$p$']
    no_pr = [r['file'] for r in rows if not (r.get('pr1') and r.get('pr2'))]
    no_pr_rest = [r['file'] for r in rows if (not r.get('pr1')) and r.get('pr2')]
    no_pr_both = [f for f in no_pr if f not in no_pr_rest]
    why = []
    if no_pr_both:
        why.append('in recording%s %s the P wave was practically not detected in lead II' %
                   ('s' if len(no_pr_both) > 1 else '', ' and '.join(no_pr_both)))
    if no_pr_rest:
        why.append('in %s it was not detected in the rest phase' % ' and '.join(no_pr_rest))
    N['PR_excluded'] = dict(both=no_pr_both, rest_only=no_pr_rest)
    cap2 = ('Cardiac cycle parameters across test phases ($n = 20$, paired Wilcoxon test). '
            'PR: $n = %d$, since %s (Section 3.2).' % (n_pr, ', and '.join(why)))
    open(os.path.join(TAB, 'table2.tex'), 'w').write(tex_table(cap2, 'tab:2', '|l|c|c|c|c|', hdr2, t2_rows))
    write_csv(os.path.join(TAB, 'table2.csv'), [strip_tex(h) for h in hdr2], [[strip_tex(c) for c in r] for r in t2_rows])

    specS1 = [('RR intervals', 'rr', [('SDNN, ms', 'SDNN', 1), ('RMSSD, ms', 'RMSSD', 1), ('pNN50, \\%', 'pNN50', 1), ('CV, \\%', 'CV', 1)]),
              ('PR intervals', 'pr', [('SD, ms', 'SD', 1), ('RMSSD, ms', 'RMSSD', 1), ('CV, \\%', 'CV', 1)]),
              ('RT intervals', 'rt', [('SD, ms', 'SD', 1), ('RMSSD, ms', 'RMSSD', 1), ('CV, \\%', 'CV', 1)]),
              ('R amplitude', 'ampR', [('mean, mV', 'mean', 3), ('SD, mV', 'SD', 3)]),
              ('P amplitude', 'ampP', [('mean, mV', 'mean', 3), ('SD, mV', 'SD', 3)]),
              ('T amplitude', 'ampT', [('mean, mV', 'mean', 3), ('SD, mV', 'SD', 3)])]
    s1_rows, hl = [], []
    for series, grp, items in specS1:
        for j, (lab, key, d) in enumerate(items):
            x, y = pair(grp, key)
            s1_rows.append([series if j == 0 else '', lab, pm(x.mean(), x.std(ddof=1), d),
                            pm(y.mean(), y.std(ddof=1), d), pfmt(stats.wilcoxon(x, y)[1]), str(len(x))])
        hl.append(len(s1_rows) - 1)
    hdrS1 = ['Series', 'Index', 'Rest', 'Tilt 45$^{\\circ}$', '$p$', '$n$']
    open(os.path.join(TAB, 'tableS1.tex'), 'w').write(tex_table(
        'Variability of interval and amplitude series across test phases (paired Wilcoxon test).', 'S1', '|l|l|c|c|c|c|', hdrS1, s1_rows,
        hlines_after=tuple(hl[:-1])))
    write_csv(os.path.join(TAB, 'tableS1.csv'), [strip_tex(h) for h in hdrS1], [[strip_tex(c) for c in r] for r in s1_rows])

    # ------------------------------------------------------------ 3. phase graphs + null control
    print('\n[3/6] Phase graphs and within-phase null control')
    vecs6 = {os.path.basename(p)[:4]: vectors_with_time(p, ch=CH_TILT, n_harm=6) for p in files}
    sh, s1, s2, nstates = [], [], [], []
    for name in sorted(vecs6):
        V, tt, p1e, p2b = vecs6[name]
        m1 = (tt > 30) & (tt < p1e - 5); m2 = (tt > p2b + 15) & (tt < p2b + 275)
        sel = m1 | m2; t = tt[sel]; in1 = (t > 30) & (t < p1e - 5)
        C, lab = cluster(V[sel], n_neurons=30, merge_frac=0.18, min_size=15, seed=SEED)
        n = len(C); nstates.append(n); sh.append(occupancy_shift(lab, in1, ~in1, n))
        for mask, acc in ((in1, s1), (~in1, s2)):
            L = lab[mask]; M, P = build_graph(L, n)
            w = np.array([(L == c).mean() for c in range(n)])
            acc.append(float(sum(w[i] * P[i, i] for i in range(n))))
    sh = np.array(sh)
    N['states_tilt'] = (int(min(nstates)), int(np.median(nstates)), int(max(nstates)))
    N['D_full_phases'] = (round(sh.mean(), 2), round(sh.std(ddof=1), 2))
    N['stability'] = (round(np.mean(s1), 2), round(np.mean(s2), 2), float(stats.wilcoxon(s1, s2)[1]),
                      int((np.array(s2) > np.array(s1)).sum()))

    feats = {'full': vecs6,
             'noT': {k: (v[0][:, 1:], v[1], v[2], v[3]) for k, v in vecs6.items()}}
    win = {}
    for p in files:
        X, tw = window_features(p); _, _, p1e, p2b = vecs6[os.path.basename(p)[:4]]
        win[os.path.basename(p)[:4]] = (X, tw, p1e, p2b)
    feats['window'] = win
    dhr = np.array([60 / np.median(vecs6[k][0][(vecs6[k][1] > vecs6[k][3] + 15) & (vecs6[k][1] < vecs6[k][3] + 275), 0]) -
                    60 / np.median(vecs6[k][0][(vecs6[k][1] > 30) & (vecs6[k][1] < vecs6[k][2] - 5), 0])
                    for k in sorted(vecs6)])
    main_ = {k: null_summary(f) for k, f in feats.items()}
    main_['gapped'] = null_summary(feats['full'], gapped=True)
    rho = stats.spearmanr(np.abs(dhr), main_['window']['effect'])
    N['window_vs_dHR'] = (float(rho[0]), float(rho[1]))
    for k, r in main_.items():
        N['null_' + k] = {q: v for q, v in r.items() if q not in ('names', 'effect', 'null_max')}
        print(f"    {k:7s}: effect {r['effect_mean']:.2f} ± {r['effect_sd']:.2f} vs null "
              f"{r['null_mean']:.2f} ± {r['null_sd']:.2f}, p = {r['p']:.1e}, {r['wins']}/{r['n']} subjects")
    print(f"    fixed window: Spearman rho(|dHR|, effect) = {rho[0]:.2f} (p = {rho[1]:.2f})")

    # Supplementary Table S3: per-subject values
    s3 = []
    for i, nm_ in enumerate(main_['full']['names']):
        row = [nm_]
        for k in ('full', 'noT', 'window'):
            row += [f"{main_[k]['null_max'][i]:.2f}", f"{main_[k]['effect'][i]:.2f}"]
        s3.append(row)
    hdrS3 = ['Subject', 'Null', 'Effect', 'Null', 'Effect', 'Null', 'Effect']
    open(os.path.join(TAB, 'tableS3.tex'), 'w').write(tex_table(
        'Within-phase null control by subject: state vector with the cycle duration (columns 2--3), '
        'without it (columns 4--5) and fixed-window features independent of the RR interval (columns 6--7). '
        'Null is the larger of the two within-phase values; effect is the shift between rest~B and tilt~A.',
        'S3', '|l|c|c|c|c|c|c|', hdrS3, s3))
    open(os.path.join(TAB, 'tableS3_full.tex'), 'w').write(          # same table, numbered S2 in the full version
        open(os.path.join(TAB, 'tableS3.tex')).read().replace('\\renewcommand{\\thetable}{S3}', '\\renewcommand{\\thetable}{S2}'))
    write_csv(os.path.join(TAB, 'tableS3.csv'),
              ['Subject', 'Null full', 'Effect full', 'Null noT', 'Effect noT', 'Null window', 'Effect window'], s3)

    robust = {}
    if not args.fast:
        print('    robustness over 5 seeds ...')
        vecs20 = {os.path.basename(p)[:4]: vectors_with_time(p, ch=CH_TILT, n_harm=20) for p in files}
        variants = {'6 harmonics, with T': (feats['full'], False), '6 harmonics, without T': (feats['noT'], False),
                    '20 harmonics, with T': (vecs20, False),
                    '20 harmonics, without T': ({k: (v[0][:, 1:], v[1], v[2], v[3]) for k, v in vecs20.items()}, False),
                    'null with the same time gap': (feats['full'], True), 'fixed window': (feats['window'], False)}
        for label, (ff, gp) in variants.items():
            rs = [null_summary(ff, seed=sd, gapped=gp) for sd in (0, 1, 7, SEED, 123)]
            robust[label] = dict(effect=(min(r['effect_mean'] for r in rs), max(r['effect_mean'] for r in rs)),
                                 null=(min(r['null_mean'] for r in rs), max(r['null_mean'] for r in rs)),
                                 wins=(min(r['wins'] for r in rs), max(r['wins'] for r in rs)),
                                 p_max=max(r['p'] for r in rs))
            q = robust[label]
            print(f"      {label:30s} effect {q['effect'][0]:.2f}-{q['effect'][1]:.2f}, null {q['null'][0]:.2f}-{q['null'][1]:.2f}, "
                  f"subjects {q['wins'][0]}-{q['wins'][1]}, p <= {q['p_max']:.1e}")
    N['robustness'] = robust

    # ------------------------------------------------------------ 4. k-means
    print('\n[4/6] Comparison with k-means')
    from sklearn.cluster import KMeans
    som, km = [], []
    for name in sorted(vecs6):
        V, tt, p1e, p2b = vecs6[name]
        m1 = (tt > 30) & (tt < p1e - 5); m2 = (tt > p2b + 15) & (tt < p2b + 275)
        sel = m1 | m2; t = tt[sel]; in1 = (t > 30) & (t < p1e - 5); X = V[sel]
        C, lab = cluster(X, n_neurons=30, merge_frac=0.18, min_size=15, seed=SEED); n = len(C)
        som.append(occupancy_shift(lab, in1, ~in1, n))
        mu, sd = X.mean(0), X.std(0) + 1e-12
        lk = KMeans(n_clusters=n, n_init=10, random_state=SEED).fit_predict((X - mu) / sd)
        km.append(occupancy_shift(lk, in1, ~in1, n))
    N['kmeans'] = (round(np.mean(som), 2), round(np.mean(km), 2), stats.wilcoxon(som, km)[1])

    # ------------------------------------------------------------ 5. inverse transform
    print('\n[5/6] Inverse transform')
    inv = stage5_inverse(files)
    N['recon_vec'] = (round(inv[:, 0].mean(), 1), round(inv[:, 0].std(ddof=1), 1))
    N['recon_centre'] = (round(inv[:, 1].mean(), 1), round(inv[:, 1].std(ddof=1), 1))
    N['compression'] = float(np.median(inv[:, 2]))

    # ------------------------------------------------------------ 6. ROHMINE -> Table 4 / S2
    if len(roh) == 4:
        print('\n[6/6] Test-database recordings')
        prof, t4_rows, diags = [], [], []
        roh_graphs = {}
        for pre in ('07', '10', '01', '03'):
            name, diag, path = roh[pre]
            g = _graph_data(path); C, lab, P = g['C'], g['lab'], g['P']; n = len(C); roh_graphs[pre] = g
            occ = np.array([(lab == k).mean() for k in range(n)])
            hrs = np.array([60 / C[k][0] for k in range(n)])
            stab = float(sum(occ[i] * P[i, i] for i in range(n)))
            H = float(sum(occ[i] * -(P[i][P[i] > 0] * np.log2(P[i][P[i] > 0])).sum() for i in range(n)))
            k0 = int(np.argmax(occ))
            t4_rows.append([name.replace('_', '\\_'), diag, str(n), f'{hrs[k0]:.0f}', f'{100 * occ[k0]:.0f}\\%',
                            f'{stab:.2f}', f'{H:.2f}'])
            prof.append([hrs[k0], float((hrs * occ).sum()), hrs.max() - hrs.min(), stab, H]); diags.append(diag)
        M = np.array(prof); Mn = (M - M.mean(0)) / (M.std(0) + 1e-12)
        inner = [np.linalg.norm(Mn[i] - Mn[j]) for i in range(4) for j in range(i + 1, 4) if diags[i] == diags[j]]
        outer = [np.linalg.norm(Mn[i] - Mn[j]) for i in range(4) for j in range(i + 1, 4) if diags[i] != diags[j]]
        N['roh_dist'] = (round(np.mean(inner), 2), round(np.mean(outer), 2))
        hdr4 = ['Recording', 'Database section', 'States',
                '\\begin{tabular}[c]{@{}c@{}}HR of dominant\\\\ state, bpm\\end{tabular}',
                '\\begin{tabular}[c]{@{}c@{}}Share of\\\\ time\\end{tabular}', 'Stability',
                '\\begin{tabular}[c]{@{}c@{}}Transition\\\\ entropy\\end{tabular}']
        cap4 = 'Graph characteristics for recordings with different rhythm disturbances.'
        open(os.path.join(TAB, 'table4.tex'), 'w').write(tex_table(cap4, 'tab:4', '|l|l|c|c|c|c|c|', hdr4, t4_rows))
        open(os.path.join(TAB, 'tableS2.tex'), 'w').write(tex_table(cap4, 'S2', '|l|l|c|c|c|c|c|', hdr4, t4_rows))
        write_csv(os.path.join(TAB, 'table4.csv'), ['Recording', 'Section', 'States', 'HR dominant', 'Share', 'Stability', 'Entropy'],
                  [[strip_tex(c) for c in r] for r in t4_rows])
    else:
        print('\n[6/6] ROHMINE recordings not found - Table 4/S2 and figures 6, 8 skipped')

    # appendix versions of the supplementary tables (numbered automatically A1, A2, ... in the manuscript)
    for src, lab in (('tableS1', 'tab:Avar'), ('tableS2', 'tab:Aroh'), ('tableS3', 'tab:Anull')):
        pth = os.path.join(TAB, src + '.tex')
        if os.path.exists(pth):
            txt = re.sub(r'\\renewcommand\{\\thetable\}\{S\d\}\n', '', open(pth).read())
            txt = txt.replace('\n\\begin{center}', '\n\\label{%s}\n\\begin{center}' % lab, 1)
            open(os.path.join(TAB, src.replace('tableS', 'tableA') + '.tex'), 'w').write(txt)

    # ------------------------------------------------------------ figures
    print('\nFigures (PDF at true print size + PNG for Word):')
    path0 = files[0]
    sig, fs, marks = read_with_marks(path0); p1e0, p2b0 = phase_bounds(marks)
    r0, _ = detect_r(sig, fs); P0, T0, _, _ = detect_pt(sig, r0, fs)
    F.fig_waves(sig, fs, r0, P0, T0, 60, int(p2b0) + 90, FIG)
    F.fig_phases(FIG)
    F.fig_shift(main_['full']['names'], main_['full']['effect'], main_['full']['null_max'], s1, s2,
                N['stability'][2], FIG)
    V, tt, p1e, p2b = vecs6[os.path.basename(path0)[:4]]
    m1 = (tt > 30) & (tt < p1e - 5); m2 = (tt > p2b + 15) & (tt < p2b + 275); sel = m1 | m2
    C, lab = cluster(V[sel], n_neurons=30, merge_frac=0.18, min_size=15, seed=SEED)
    t = tt[sel]; F.fig_membership(t, lab, len(C), p1e, p2b, (t > 30) & (t < p1e - 5), FIG)
    n_all = list(nstates) + ([len(g['C']) for g in roh_graphs.values()] if len(roh) == 4 else [])
    F.fig_reversibility(N['cycles_total'], 2 * N_HARM + 2, round(N['compression'], -2), N['recon_vec'][0],
                        f'{min(n_all)}–{max(n_all)}', FIG)
    F.fig_pipeline(FIG)
    if len(roh) == 4:
        panels = []
        for pre, title in (('07', '07_VORO (slow rhythm)'), ('01', '01_GUSA (fast rhythm)')):
            g = roh_graphs[pre]; fsg = g['fs']; rr_ = g['r']; idx = g['idx']; k0 = min(40, len(idx) - 11)
            orig = g['hf'][rr_[idx[k0]]:rr_[idx[k0 + 8]]]
            rec = np.concatenate([eg.vec_to_cycle(g['V'][k], fsg) for k in range(k0, k0 + 8)])
            syn, _ = eg.graph_to_ecg(g['C'], g['P'], fsg, n_beats=8, start=int(np.bincount(g['lab']).argmax()), seed=3)
            panels.append(dict(title=title, fs=fsg, orig=orig, rec=rec, syn=syn, C=g['C'], lab=g['lab'], P=g['P']))
        F.fig_cycle(panels, FIG)
        graphs = []
        for pre in ('07', '10', '01', '03'):
            g = roh_graphs[pre]; n = len(g['C'])
            occ = np.array([(g['lab'] == k).mean() for k in range(n)])
            stab = float(sum(occ[i] * g['P'][i, i] for i in range(n)))
            graphs.append((f'{roh[pre][0]}: {roh[pre][1]}\nstates {n}, stability {stab:.2f}', g['C'], g['lab'], g['P']))
        F.fig_roh(graphs, FIG)

    # ------------------------------------------------------------ numbers.txt
    nm, ns, nw, ng = N['null_full'], N['null_noT'], N['null_window'], N['null_gapped']
    lines = [
        'NUMBERS OF THE PAPER (recomputed)', '',
        f"Synthetic validation: {N['syn_cycles']} cycles; Se/PPV R {N['syn_R_Se']}/{N['syn_R_PPV']}, "
        f"T {N['syn_T_Se']}/{N['syn_T_PPV']}, P {N['syn_P_Se']}/{N['syn_P_PPV']} %",
        f"Tilt-test cycles: {N['cycles_total']}; T detected {N['T_detected_pct']} %, P detected {N['P_detected_pct']} %",
        f"PR excluded: P not detected {N['PR_excluded']['both']}, not detected at rest {N['PR_excluded']['rest_only']}",
        f"Number of states (tilt test): {N['states_tilt'][0]}-{N['states_tilt'][2]}, median {N['states_tilt'][1]}",
        f"State occupancy shift over full phases: {N['D_full_phases'][0]:.2f} ± {N['D_full_phases'][1]:.2f}",
    ]
    for lab_, q in (('Null control, with T', nm), ('Null control, without T', ns),
                    ('Null control, fixed window', nw), ('Null with the same time gap', ng)):
        lines.append(f"{lab_}: effect {q['effect_mean']:.2f} ± {q['effect_sd']:.2f} vs null {q['null_mean']:.2f} ± "
                     f"{q['null_sd']:.2f}, p = {q['p']:.2e}, {q['wins']}/{q['n']} subjects")
    lines += [
        f"Fixed window: Spearman rho(|dHR|, effect) = {N['window_vs_dHR'][0]:.2f}, p = {N['window_vs_dHR'][1]:.2f}",
        f"State stability: {N['stability'][0]:.2f} -> {N['stability'][1]:.2f}, p = {N['stability'][2]:.3f}, "
        f"increase in {N['stability'][3]} subjects",
        f"k-means: SOM {N['kmeans'][0]:.2f} vs k-means {N['kmeans'][1]:.2f}, p = {N['kmeans'][2]:.2f}",
        f"Reconstruction: vector->cycle {N['recon_vec'][0]} ± {N['recon_vec'][1]} %, "
        f"cycle->centre {N['recon_centre'][0]} ± {N['recon_centre'][1]} %, compression {N['compression']:.0f}x",
    ]
    for lab_, q in N['robustness'].items():
        lines.append(f"Robustness ({lab_}, 5 seeds): effect {q['effect'][0]:.2f}-{q['effect'][1]:.2f}, null "
                     f"{q['null'][0]:.2f}-{q['null'][1]:.2f}, subjects {q['wins'][0]}-{q['wins'][1]}, p <= {q['p_max']:.1e}")
    if 'roh_dist' in N:
        lines.append(f"Test database: distance same diagnosis {N['roh_dist'][0]:.2f}, different {N['roh_dist'][1]:.2f}, "
                     f"ratio {N['roh_dist'][1] / N['roh_dist'][0]:.1f}")
    open(os.path.join(OUT, 'numbers.txt'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    json.dump(N, open(os.path.join(OUT, 'numbers.json'), 'w'), indent=1, default=float)
    print('\n' + '\n'.join(lines))

    # ------------------------------------------------------------ manuscript
    if args.tex:
        texdir = os.path.abspath(args.tex)
        for f in glob.glob(os.path.join(FIG, '*.pdf')) + glob.glob(os.path.join(FIG, '*.png')):
            shutil.copy(f, texdir)
        for f in glob.glob(os.path.join(TAB, '*.tex')):
            shutil.copy(f, texdir)
        # do the numbers quoted in the text match the computation?
        def tx(p):                                   # 2.6e-04 -> $p = 2.6\times10^{-4}$
            m, e = f'{p:.1e}'.split('e')
            return f'{m}\\times10^{{{int(e)}}}'
        expect = {
            'effect, with T': f"{nm['effect_mean']:.2f} $\\pm$ {nm['effect_sd']:.2f}",
            'null, with T': f"{nm['null_mean']:.2f} $\\pm$ {nm['null_sd']:.2f}",
            'p, with T': tx(nm['p']),
            'effect, without T': f"{ns['effect_mean']:.2f} $\\pm$ {ns['effect_sd']:.2f} versus {ns['null_mean']:.2f} $\\pm$ {ns['null_sd']:.2f}",
            'fixed window': f"{nw['effect_mean']:.2f} $\\pm$ {nw['effect_sd']:.2f} versus {nw['null_mean']:.2f} $\\pm$ {nw['null_sd']:.2f}",
            'fixed window p': tx(nw['p']),
            'fixed window subjects': f"{nw['wins']} of 20 subjects",
            'Spearman rho': f"\\rho = {N['window_vs_dHR'][0]:.2f}",
            'D over full phases': f"{N['D_full_phases'][0]:.2f} $\\pm$ {N['D_full_phases'][1]:.2f}",
            'stability': f"from {N['stability'][0]:.2f} at rest to {N['stability'][1]:.2f}",
            'k-means p': f"$p = {N['kmeans'][2]:.2f}$",
            'reconstruction': f"{N['recon_vec'][0]} $\\pm$ {N['recon_vec'][1]}",
            'centroid error': f"{N['recon_centre'][0]} $\\pm$ {N['recon_centre'][1]}",
            'cycles': f"{N['cycles_total']:,}".replace(',', '\\,'),
        }
        if N['robustness']:
            expect['robustness, gap'] = tx(N['robustness']['null with the same time gap']['p_max'])
            expect['robustness, window'] = tx(N['robustness']['fixed window']['p_max'])
            expect['robustness, 6 harmonics'] = tx(N['robustness']['6 harmonics, with T']['p_max'])
        if 'roh_dist' in N:
            expect['test-database distances'] = f"{N['roh_dist'][0]:.2f}"
        targets = [f for f in ('manuscript.tex', 'manuscript_full.tex') if os.path.exists(os.path.join(texdir, f))]
        print('\nCheck of the numbers quoted in the manuscript:')
        for t in targets:
            body = open(os.path.join(texdir, t), encoding='utf-8').read()
            bad = [k for k, v in expect.items() if v not in body]
            print(f'  {t}: ' + ('all key numbers match' if not bad else 'CHECK: ' + ', '.join(bad)))
        if shutil.which('pdflatex'):
            for t in targets + [f for f in ('titlepage.tex',)
                                if os.path.exists(os.path.join(texdir, f))]:
                for _ in range(2):
                    subprocess.run(['pdflatex', '-interaction=nonstopmode', t], cwd=texdir,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f'  compiled {t[:-4]}.pdf')
        else:
            print('  pdflatex not found: upload the manuscript folder to Overleaf and press Recompile')

    print(f'\nDone in {time.time() - t_start:.0f} s. Results: {OUT}')


def run(data, roh=None, out='results', tex=None, fast=False):
    """Call from a notebook cell:  run('/content/data', '/content/roh', tex='/content/paper/manuscript')"""
    argv = ['--data', data, '--out', out]
    if roh:
        argv += ['--roh', roh]
    if tex:
        argv += ['--tex', tex]
    if fast:
        argv.append('--fast')
    main(argv)


if __name__ == '__main__':
    if 'ipykernel' in sys.modules and '--data' not in sys.argv:
        # the file was pasted into a notebook cell: do not parse the kernel's own arguments
        print("Code loaded. Now run in the next cell, for example:\n"
              "  run('/content/data', '/content/roh', out='/content/paper/results', "
              "tex='/content/paper/manuscript')")
    else:
        main()
