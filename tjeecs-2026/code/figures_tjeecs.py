# -*- coding: utf-8 -*-
"""
Figures of the TJEECS version, drawn at their TRUE PRINT SIZE.

Each figure is created exactly as wide as it is printed (see WIDTH_CM), so the
font sizes below are the real sizes on paper (7.5-9 pt, Times) and every line is
0.5-1 pt, as the journal requires. In the manuscript the figures are included with
the same width, so nothing is scaled.

File names follow the figure numbers of manuscript.tex:
  fig1  annotation of P, R, T waves            fig4  forward and inverse transforms
  fig2  scheme of the phase comparison         fig5  full cycle ECG -> graph -> ECG
  fig3  occupancy shift vs null + stability
  figS1 assignment of cycles to states, one subject (Supplementary)
  figS2 graphs of the test-database recordings (Supplementary)
  fig_pipeline  scheme of the method (only in manuscript_full.tex)
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

CM = 1 / 2.54
WIDTH_CM = dict(fig1=12, fig2=12, fig3=13, fig4=12, fig5=13, figS1=12, figS2=13, fig_pipeline=14)

DARK = '#0F0F0F'; ACC = '#A62B2B'; BLUE = '#2e6da4'; GREY = '#EAEAEA'; RED = '#c0392b'

mpl.rcParams.update({
    'font.family': ['serif', 'DejaVu Serif'],      # DejaVu only as fallback for rare glyphs
    'font.serif': ['Times New Roman', 'Times', 'Liberation Serif', 'TeX Gyre Termes', 'Nimbus Roman'],
    'mathtext.fontset': 'stix', 'pdf.fonttype': 42,
    'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'axes.titleweight': 'normal',
    'lines.linewidth': 0.8, 'axes.linewidth': 0.6, 'grid.linewidth': 0.5, 'patch.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'lines.markersize': 3.5, 'legend.frameon': True, 'legend.framealpha': 0.9,
    'figure.dpi': 150, 'savefig.dpi': 600,
})


def _clip_lines(fig, lo=0.5, hi=1.0):
    """Every visible line 0.5-1 pt (the figure is already at print size)."""
    for a in fig.findobj(lambda o: hasattr(o, 'get_linewidth') and hasattr(o, 'set_linewidth')):
        try:
            lw = a.get_linewidth()
            arr = np.atleast_1d(np.asarray(lw, dtype=float))
            if arr.size == 0:
                continue
            new = np.where(arr > 0, np.clip(arr, lo, hi), 0.0)
            a.set_linewidth(new if np.ndim(lw) else float(new[0]))
        except Exception:
            pass


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    _clip_lines(fig)
    fig.savefig(os.path.join(out, name + '.pdf'))
    fig.savefig(os.path.join(out, name + '.png'), dpi=300)      # PNG copy for the Word version
    plt.close(fig)
    print(f'    {name}.pdf')


def _fig(name, h_cm, **kw):
    return plt.figure(figsize=(WIDTH_CM[name] * CM, h_cm * CM), **kw)


# ------------------------------------------------------------------ Figure 1
def fig_waves(sig, fs, r, P, T, t_rest, t_tilt, out):
    fig = _fig('fig1', 4.8, layout='constrained')
    axs = fig.subplots(2, 1)
    for ax, t0, ttl in zip(axs, (t_rest, t_tilt), ('rest', '45° tilt')):
        i0, i1 = int(t0 * fs), int((t0 + 6) * fs)
        ax.plot(np.arange(i0, i1) / fs, sig[i0:i1], lw=0.7, color=DARK)
        for idx, c, m, lab in [(r, 'red', 'o', 'R'), (P, 'green', '^', 'P'), (T, 'blue', 'v', 'T')]:
            sel = idx[(idx >= i0) & (idx < i1)]; sel = sel[sel > 0]
            ax.scatter(sel / fs, sig[sel], c=c, s=9, marker=m, zorder=5, label=lab, linewidths=0)
        seg = sig[i0:i1]; rng = seg.max() - seg.min()
        ax.set_ylim(seg.min() - 0.08 * rng, seg.max() + 0.45 * rng)
        ax.set_title(f'{ttl} phase, {t0:.0f}–{t0 + 6:.0f} s', pad=2)
        ax.set_ylabel('mV'); ax.grid(alpha=.3)
        ax.legend(loc='upper right', ncol=3, handletextpad=0.2, columnspacing=0.8, borderpad=0.3)
    axs[1].set_xlabel('time, s')
    save(fig, out, 'fig1')


# ------------------------------------------------------------------ Figure 2
def fig_phases(out):
    fig = _fig('fig2', 4.6)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 12); ax.set_ylim(-0.3, 4.75); ax.axis('off')
    ax.annotate('', xy=(11.3, 4.05), xytext=(0.3, 4.05), arrowprops=dict(arrowstyle='-|>', lw=0.8, color='#555'))
    ax.text(11.45, 4.05, 'time', fontsize=7.5, color='#555', va='center')
    ax.text(2.85, 4.42, 'rest (5 min)', ha='center', fontsize=8.5, color=BLUE, fontweight='bold')
    ax.text(8.75, 4.42, '45° tilt (5 min)', ha='center', fontsize=8.5, color=ACC, fontweight='bold')
    ax.add_patch(mp.Rectangle((0.5, 3.35), 4.7, 0.5, facecolor=BLUE, alpha=.85, edgecolor='none'))
    ax.add_patch(mp.Rectangle((5.2, 3.35), 1.2, 0.5, facecolor='#BBB', edgecolor='none'))
    ax.add_patch(mp.Rectangle((6.4, 3.35), 4.7, 0.5, facecolor=ACC, alpha=.85, edgecolor='none'))
    for xc, lab in [(1.68, 'rest A'), (4.03, 'rest B'), (7.58, 'tilt A'), (9.93, 'tilt B')]:
        ax.text(xc, 3.6, lab, ha='center', va='center', color='white', fontsize=8, fontweight='bold')
    for xm in (2.85, 8.75):
        ax.plot([xm, xm], [3.35, 3.85], color='white', lw=0.8, ls='--')
    ax.text(5.8, 3.6, 'table\ntilting', ha='center', va='center', fontsize=6.5, color='#333', linespacing=0.9)
    for x, col in [(2.85, BLUE), (8.75, ACC)]:
        ax.annotate('', xy=(x, 2.62), xytext=(x, 3.3), arrowprops=dict(arrowstyle='-|>', lw=0.9, color=col))
    ax.add_patch(mp.FancyBboxPatch((1.6, 1.85), 8.4, 0.72, boxstyle='round,pad=0.04', facecolor=GREY, edgecolor='none'))
    ax.text(5.8, 2.34, 'clustering on the combined data of all four segments', ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=DARK)
    ax.text(5.8, 2.04, r'common state space $S_0 \ldots S_n$', ha='center', va='center', fontsize=7.5, color='#444')
    for x, col, lab in [(2.85, BLUE, 'rest-phase graph'), (8.75, ACC, 'tilt-phase graph')]:
        ax.annotate('', xy=(x, 1.28), xytext=(x, 1.82), arrowprops=dict(arrowstyle='-|>', lw=0.9, color=col))
        ax.add_patch(mp.FancyBboxPatch((x - 1.7, 0.72), 3.4, 0.54, boxstyle='round,pad=0.04',
                                       facecolor=col, alpha=.85, edgecolor='none'))
        ax.text(x, 0.99, lab, ha='center', va='center', color='white', fontsize=8.5, fontweight='bold')
    ax.annotate('', xy=(6.95, 0.99), xytext=(4.65, 0.99), arrowprops=dict(arrowstyle='<|-|>', lw=0.9, color='#333'))
    ax.text(5.8, 0.42, 'compared: state occupancy shift D and state stability S', ha='center', va='center',
            fontsize=7.5, color='#333', style='italic')
    ax.text(5.8, 0.05, 'effect = D between rest B and tilt A;   null = D between the halves A and B of the same phase',
            ha='center', va='center', fontsize=7.5, color='#333', style='italic')
    save(fig, out, 'fig2')


# ------------------------------------------------------------------ Figure 3
def fig_shift(names, eff, nul, s1, s2, p_stab, out):
    fig = _fig('fig3', 5.2, layout='constrained')
    axs = fig.subplots(1, 2, gridspec_kw={'width_ratios': [1.55, 1]})
    eff, nul = np.asarray(eff), np.asarray(nul)
    order = np.argsort(eff); win = eff > nul
    axs[0].barh(range(len(eff)), eff[order], height=0.7, color=[ACC if win[i] else '#bbb' for i in order])
    axs[0].scatter(nul[order], range(len(eff)), marker='|', s=30, color='k', zorder=5, linewidths=0.9)
    axs[0].set_yticks(range(len(eff))); axs[0].set_yticklabels([names[k] for k in order], fontsize=6)
    axs[0].set_xlabel('state occupancy shift D'); axs[0].set_xlim(0, 1)
    axs[0].set_title(f'Shift exceeds the null in {int(win.sum())} of {len(eff)}', pad=3)
    axs[0].legend(handles=[Patch(color=ACC, label='exceeds null'), Patch(color='#bbb', label='does not exceed'),
                           Line2D([], [], color='k', marker='|', ls='none', ms=6, label='null (max)')],
                  loc='lower right', fontsize=6.5, handlelength=1.2, borderpad=0.3)
    axs[0].grid(alpha=.25, axis='x')
    for a, b in zip(s1, s2):
        axs[1].plot([0, 1], [a, b], '-o', color='#999', lw=0.5, ms=2.2, alpha=.7)
    axs[1].plot([0, 1], [np.mean(s1), np.mean(s2)], '-o', color=ACC, lw=1.0, ms=4.5, label='mean')
    axs[1].set_xticks([0, 1]); axs[1].set_xticklabels(['rest', '45° tilt'])
    axs[1].set_ylabel('state stability S'); axs[1].set_xlim(-0.3, 1.3)
    axs[1].set_title(f'{np.mean(s1):.2f} → {np.mean(s2):.2f}, p = {p_stab:.3f}', pad=3)
    axs[1].legend(loc='lower right'); axs[1].grid(alpha=.25, axis='y')
    save(fig, out, 'fig3')


# ------------------------------------------------------------------ Supplementary Figure S1
def fig_membership(t, lab, n, p1e, p2b, in_rest, out):
    fig = _fig('figS1', 6.0, layout='constrained')
    axs = fig.subplots(2, 1, gridspec_kw={'height_ratios': [1.6, 1]})
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    for c in range(n):
        m = lab == c
        axs[0].scatter(t[m] / 60, np.full(m.sum(), c), s=2.5, color=cols[c], alpha=.7, linewidths=0)
    axs[0].axvspan(p1e / 60, p2b / 60, color='#ccc', alpha=.6, lw=0)
    axs[0].set_yticks(range(n)); axs[0].set_yticklabels([f'S{c}' for c in range(n)])
    axs[0].set_ylim(-0.6, n - 0.4)
    axs[0].set_ylabel('state'); axs[0].set_xlabel('time, min', labelpad=1); axs[0].grid(alpha=.25, axis='x')
    axs[0].set_title('Assignment of cardiac cycles to states', pad=3)
    w1 = [(lab[in_rest] == c).mean() for c in range(n)]
    w2 = [(lab[~in_rest] == c).mean() for c in range(n)]
    x = np.arange(n); wd = 0.36
    axs[1].bar(x - wd / 2, w1, wd, label='rest', color=BLUE)
    axs[1].bar(x + wd / 2, w2, wd, label='45° tilt', color=ACC)
    axs[1].set_xticks(x); axs[1].set_xticklabels([f'S{c}' for c in range(n)])
    axs[1].set_ylabel('fraction of time'); axs[1].legend(loc='upper left', ncol=2); axs[1].grid(alpha=.25, axis='y')
    sh = 0.5 * sum(abs(a - b) for a, b in zip(w1, w2))
    axs[1].set_ylim(0, max(max(w1), max(w2)) * 1.25)
    axs[1].set_title(f'State occupancy by phase, shift D = {sh:.2f}', pad=3)
    save(fig, out, 'figS1')


# ------------------------------------------------------------------ Figure 4
def fig_reversibility(n_cycles, n_comp, compression, err, n_vert, out):
    fig = _fig('fig4', 4.0)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 12); ax.set_ylim(0.15, 4.0); ax.axis('off')
    blocks = [('ECG', f'{n_cycles:,} cardiac cycles'.replace(',', ' ')),
              ('State vectors', f'{n_comp} components per cycle'),
              ('Digraph', f'{n_vert} vertices')]
    w, h, x0, gap = 2.6, 1.25, 0.2, 1.9
    for i, (t, s) in enumerate(blocks):
        x = x0 + i * (w + gap)
        face, tc, sc = (ACC, 'white', '#F5E2E2') if i == 2 else (GREY, DARK, '#2E2E2E')
        ax.add_patch(mp.FancyBboxPatch((x, 1.45), w, h, boxstyle='round,pad=0.04', facecolor=face, edgecolor='none'))
        ax.text(x + w / 2, 1.45 + h * 0.66, t, ha='center', va='center', fontsize=9, fontweight='bold', color=tc)
        ax.text(x + w / 2, 1.45 + h * 0.28, s, ha='center', va='center', fontsize=7.5, color=sc)
    for i in range(2):
        xa, xb = x0 + i * (w + gap) + w + 0.08, x0 + (i + 1) * (w + gap) - 0.08
        ax.annotate('', xy=(xb, 2.35), xytext=(xa, 2.35), arrowprops=dict(arrowstyle='-|>', lw=0.9, color=ACC))
        ax.text((xa + xb) / 2, 2.47, 'compression', ha='center', va='bottom', fontsize=7, color=ACC, style='italic')
        ax.annotate('', xy=(xa, 1.78), xytext=(xb, 1.78), arrowprops=dict(arrowstyle='-|>', lw=0.9, color=BLUE))
        ax.text((xa + xb) / 2, 1.66, ['Fourier series\nsummation', 'graph traversal\nby probabilities'][i],
                ha='center', va='top', fontsize=6.5, color=BLUE, style='italic', linespacing=1.0)
    ax.text(6, 3.55, f'forward transform: compression ≈ {compression:.0f}-fold', ha='center', fontsize=8.5,
            color=ACC, fontweight='bold')
    ax.text(6, 0.45, f'inverse transform: reconstruction error {err:.1f}%', ha='center', fontsize=8.5,
            color=BLUE, fontweight='bold')
    save(fig, out, 'fig4')


# ------------------------------------------------------------------ graphs of states
def draw_graph(ax, C, lab, P, title, layout='circle'):
    """Vertices = states (heart rate inside, share of time as a label), arrows = transitions
    with probability >= 5 %; arrow width and darkness grow with the probability."""
    n = len(C)
    occ = np.array([(lab == k).mean() for k in range(n)])
    hr = np.array([60 / C[k][0] for k in range(n)])
    if layout == 'row':                      # vertices in a row, ordered by heart rate
        xl, yl = (-1.95, 1.95), (-1.12, 0.9)
        keep = np.argsort(hr)
        xs = (np.linspace(-1.55, 1.55, n) if n <= 2 else np.linspace(-1.7, 1.7, n)) if n > 1 else np.array([0.0])
        pos = {k: np.array([x, 0.0]) for k, x in zip(keep, xs)}
        size = 0.2 if n <= 2 else 0.17
    else:
        xl, yl = (-1.9, 1.9), (-1.75, 1.85)
        keep = np.argsort(-occ)
        ang = np.linspace(0, 2 * np.pi, len(keep), endpoint=False) + np.pi / 2
        pos = {k: 1.1 * np.array([np.cos(a), np.sin(a)]) for k, a in zip(keep, ang)}
        size = 0.22
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect('equal'); ax.axis('off')
    # points per data unit, so that every arrow starts and ends exactly at the edge of its circle
    fig = ax.figure; fig.canvas.draw()
    bb = ax.get_window_extent()
    ppu = min(bb.width / (xl[1] - xl[0]), bb.height / (yl[1] - yl[0])) * 72 / fig.dpi
    if layout == 'row' and n > 2:            # floor on the radius so that three-digit rates fit inside
        rad_of = {k: 0.21 + 0.18 * occ[k] for k in keep}
    else:
        rad_of = {k: size + size * occ[k] for k in keep}
    for i in keep:
        for j in keep:
            if i == j or P[i, j] < 0.05:
                continue
            if layout == 'row' and n <= 2:   # two-vertex graph: keep the approved look
                ap = dict(arrowstyle='-|>', lw=0.5 + 0.5 * P[i, j], color='#8B2222', alpha=.4 + .6 * P[i, j],
                          connectionstyle='arc3,rad=0.45', mutation_scale=7, shrinkA=12, shrinkB=12)
            else:                            # arrows start and end exactly at the circle edges
                ap = dict(arrowstyle='-|>', lw=0.5 + 0.5 * P[i, j], color='#7A1F1F', alpha=.55 + .45 * P[i, j],
                          connectionstyle='arc3,rad=0.45', mutation_scale=8,
                          shrinkA=rad_of[i] * ppu + 1.5, shrinkB=rad_of[j] * ppu + 1.5)
            ax.annotate('', xy=pos[j], xytext=pos[i], arrowprops=ap, zorder=3)
    for k in keep:
        rr = rad_of[k]
        ax.add_patch(plt.Circle(pos[k], rr, color=BLUE if hr[k] < 60 else RED, alpha=.9, zorder=5, lw=0))
        ax.text(*pos[k], f'{hr[k]:.0f}', color='white', ha='center', va='center',
                fontsize=(6.3 if hr[k] >= 99.5 else 7) if layout == 'row' else 7.5, fontweight='bold', zorder=6)
        pct = '<1%' if occ[k] < 0.005 else f'{100 * occ[k]:.0f}%'
        if layout == 'row':                  # under the lowest arc, aligned with the vertex
            ax.text(pos[k][0], -0.86, pct, ha='center', va='top', fontsize=6.5, color='#333', zorder=7)
        else:
            below = pos[k][1] <= 0.5
            ax.text(pos[k][0], pos[k][1] - rr - 0.06 if below else pos[k][1] + rr + 0.06, pct, ha='center',
                    va='top' if below else 'bottom', fontsize=6.5, color='#333', zorder=7)
    ax.set_title(title, pad=2)


# ------------------------------------------------------------------ Figure 5
def fig_cycle(panels, out):
    """panels: list of 2 dicts with keys title, fs, orig, rec, syn, C, lab, P."""
    fig = _fig('fig5', 9.4, layout='constrained')
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1.55, 1])
    for col, g in enumerate(panels):
        fs = g['fs']
        for row, (sig, colr, ttl) in enumerate([(g['orig'], DARK, '(a) original ECG'),
                                                 (g['rec'], BLUE, '(b) reconstruction from state vectors')]):
            ax = fig.add_subplot(gs[row, col]); ax.plot(np.arange(len(sig)) / fs, sig, lw=0.6, color=colr)
            ax.set_title((g['title'] + '\n' if row == 0 else '') + ttl, pad=2); ax.grid(alpha=.25)
            ax.set_ylabel('mV', labelpad=1)
        ax = fig.add_subplot(gs[2, col]); draw_graph(ax, g['C'], g['lab'], g['P'], f'(c) state graph ({len(g["C"])} vertices, bpm)', layout='row')
        ax = fig.add_subplot(gs[3, col]); ax.plot(np.arange(len(g['syn'])) / fs, g['syn'], lw=0.6, color=RED)
        ax.set_title('(d) ECG synthesised by traversing the graph', pad=2); ax.grid(alpha=.25)
        ax.set_xlabel('time, s', labelpad=1); ax.set_ylabel('mV', labelpad=1)
    save(fig, out, 'fig5')


# ------------------------------------------------------------------ Supplementary Figure S2
def fig_roh(graphs, out):
    """graphs: list of 4 tuples (title, C, lab, P)."""
    fig = _fig('figS2', 11.5, layout='constrained')
    axs = fig.subplots(2, 2).ravel()
    for ax, (title, C, lab, P) in zip(axs, graphs):
        draw_graph(ax, C, lab, P, title)
    save(fig, out, 'figS2')


# ------------------------------------------------------------------ scheme of the method (full version only)
def fig_pipeline(out):
    fig = _fig('fig_pipeline', 3.0)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 14); ax.set_ylim(0, 3.0); ax.axis('off')
    steps = [('ECG', 'recording of\narbitrary duration'), ('Annotation', 'wavelet detection\nof R, P, T'),
             ('State vector', 'Fourier series:\n' + r'$T, a_0, (a_k, b_k)$'), ('Clusters', 'Kohonen map,\nunsupervised'),
             ('Digraph', 'vertices: states\nedges: probabilities')]
    w, gap, x0 = 2.35, 0.48, 0.25
    for i, (t, s) in enumerate(steps):
        x = x0 + i * (w + gap)
        face, tc, sc = (ACC, 'white', '#F5E2E2') if i == 4 else (GREY, DARK, '#2E2E2E')
        ax.add_patch(mp.FancyBboxPatch((x, 0.35), w, 2.3, boxstyle='round,pad=0.04', facecolor=face, edgecolor='none'))
        ax.text(x + w / 2, 2.05, t, ha='center', va='center', fontsize=9, fontweight='bold', color=tc)
        ax.text(x + w / 2, 1.05, s, ha='center', va='center', fontsize=7.5, color=sc, linespacing=1.2)
        if i < 4:
            ax.annotate('', xy=(x + w + gap - 0.05, 1.5), xytext=(x + w + 0.05, 1.5),
                        arrowprops=dict(arrowstyle='-|>', lw=0.9, color=ACC))
    save(fig, out, 'fig_pipeline')
