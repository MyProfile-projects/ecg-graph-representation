# -*- coding: utf-8 -*-
"""
Объединённый расчёт по вопросу оппонента про частотные диапазоны
0,04-0,15 Гц и 0,16-0,4 Гц. Считает оба возможных прочтения письма
сразу: RR-vs-PR и PP-vs-PR (в письме первое обозначение написано
кириллическими буквами, что по начертанию совпадает с латинской P,
а не с R, отсюда и неоднозначность).

Строит: сводную таблицу (медианы LF/HF по фазам, p по Уилкоксону)
и сравнительный график из трёх панелей (RR, PP, PR).
"""

import os
import sys
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
sys.path.insert(0, _HERE)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.interpolate import CubicSpline
from scipy import stats

import pp_pr_standalone as m

FS_RESAMPLE = 4.0
LF_BAND = (0.04, 0.15)
HF_BAND = (0.16, 0.40)
TRAPZ = getattr(np, 'trapezoid', None) or np.trapz


def resample_phase(t, x, t_lo, t_hi, fs=FS_RESAMPLE):
    mask = (t >= t_lo) & (t <= t_hi) & np.isfinite(x)
    if mask.sum() < 20:
        return None
    t_seg, x_seg = t[mask], x[mask]
    cs = CubicSpline(t_seg, x_seg)
    t_u = np.arange(t_seg[0], t_seg[-1], 1 / fs)
    return cs(t_u) - cs(t_u).mean()


def band_power(x, fs, band):
    f, pw = welch(x, fs=fs, nperseg=min(256, len(x)))
    mask = (f >= band[0]) & (f < band[1])
    return TRAPZ(pw[mask], f[mask]) if mask.any() else np.nan


def lfhf(x):
    if x is None:
        return None
    lf = band_power(x, FS_RESAMPLE, LF_BAND)
    hf = band_power(x, FS_RESAMPLE, HF_BAND)
    return lf / hf


def summarize(rows, key):
    rest_d = {r['name']: r for r in rows if r['phase'] == 'rest' and r[key] is not None}
    tilt_d = {r['name']: r for r in rows if r['phase'] == 'tilt' and r[key] is not None}
    names = sorted(set(rest_d) & set(tilt_d))
    rv = np.array([rest_d[n][key] for n in names])
    tv = np.array([tilt_d[n][key] for n in names])
    p = stats.wilcoxon(rv, tv)[1]
    return rv, tv, p, names


if __name__ == '__main__':
    cache = m.load_series(_HERE)

    rows = []
    for name, v in sorted(cache.items()):
        t_rr, rr, p1e, p2b = v['t_rr'], v['rr'], v['p1e'], v['p2b']
        t_r, pr, pp = v['t_r'], v['pr'], v['pp']

        for phase, (lo_rr, hi_rr) in [('rest', (t_rr[0], p1e)), ('tilt', (p2b, t_rr[-1]))]:
            lo_r, hi_r = (t_r[0], p1e) if phase == 'rest' else (p2b, t_r[-1])
            entry = dict(name=name, phase=phase)
            entry['lfhf_rr'] = lfhf(resample_phase(t_rr, rr, lo_rr, hi_rr))
            entry['lfhf_pp'] = lfhf(resample_phase(t_r, pp, lo_r, hi_r))
            entry['lfhf_pr'] = lfhf(resample_phase(t_r, pr, lo_r, hi_r))
            rows.append(entry)

    print(f"{'Ряд':6s}{'покой':>10s}{'наклон':>10s}{'n':>5s}{'p':>10s}")
    results = {}
    for key, label in [('lfhf_rr', 'RR'), ('lfhf_pp', 'PP'), ('lfhf_pr', 'PR')]:
        rv, tv, p, names = summarize(rows, key)
        results[label] = (rv, tv, p)
        print(f"{label:6s}{np.median(rv):10.3f}{np.median(tv):10.3f}{len(rv):5d}{p:10.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, label in zip(axes, ['RR', 'PP', 'PR']):
        rv, tv, p = results[label]
        for r, t in zip(rv, tv):
            ax.plot([0, 1], [r, t], '-o', color='#999', alpha=.5, ms=4)
        ax.plot([0, 1], [np.median(rv), np.median(tv)], '-o', color='#A62B2B', lw=3, ms=10, label='медиана')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['покой', 'наклон'])
        ax.set_yscale('log')
        ax.set_title(f"{label}: {np.median(rv):.3f}\u2192{np.median(tv):.3f}, p={p:.3f}")
        ax.legend(fontsize=8); ax.grid(alpha=.25); ax.set_ylabel('LF/HF')
    plt.tight_layout()
    plt.savefig(os.path.join(_HERE, 'combined_rr_pp_pr.png'), dpi=140)
    print("\nГрафик сохранён: combined_rr_pp_pr.png")

    # ---------- прямое сравнение сдвигов: отличается ли реакция RR/PP от реакции PR ----------
    rest_d = {r['name']: r for r in rows if r['phase'] == 'rest'}
    tilt_d = {r['name']: r for r in rows if r['phase'] == 'tilt'}
    print("\nПрямое сравнение сдвигов (наклон минус покой, лог. шкала):")
    for ref_key, ref_label in [('lfhf_rr', 'RR'), ('lfhf_pp', 'PP')]:
        names = sorted(n for n in set(rest_d) & set(tilt_d)
                        if rest_d[n][ref_key] is not None and rest_d[n]['lfhf_pr'] is not None)
        d_ref = np.array([np.log(tilt_d[n][ref_key]) - np.log(rest_d[n][ref_key]) for n in names])
        d_pr = np.array([np.log(tilt_d[n]['lfhf_pr']) - np.log(rest_d[n]['lfhf_pr']) for n in names])
        p_direct = stats.wilcoxon(d_ref, d_pr)[1]
        print(f"  {ref_label} против PR: n={len(names)}, p={p_direct:.4f}")
