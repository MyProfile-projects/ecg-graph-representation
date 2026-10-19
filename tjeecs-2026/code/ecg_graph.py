# -*- coding: utf-8 -*-
# Core computations of the paper (unchanged code from the repository
# MyProfile-projects/ecg-graph-representation, article_code_ENG.ipynb).
# build_all.py imports this module; you normally do not run it directly.
# -*- coding: utf-8 -*-
"""
════════════════════════════════════════════════════════════════════════════
Reversible graph representation of ECG dynamics
for detecting functional state changes

SINGLE FILE. Reproduces all numerical results reported in the paper.
No need to split it into modules or to add anything.

Installation:
    pip install numpy scipy pyedflib PyWavelets scikit-learn

RUNNING IN JUPYTER / GOOGLE COLAB
    Execute this cell in full, then in a new cell:

        validate()                        # synthetic-data check, no recordings needed
        run_all('/path/to/recordings')    # full computation over the 20 recordings
        run_all('/path/to/recordings', roh_dir='/path/to/database')   # + test database
        make_figures('/path/to/recordings')   # builds all figures of the paper

RUNNING FROM THE COMMAND LINE
        python ecg_graph_full.py --validate
        python ecg_graph_full.py --data /path/to/recordings
        python ecg_graph_full.py --data /path/to/recordings --figures

Expected data: <data>/0011_ecg.edf … 0030_ecg.edf
               20 head-up tilt test recordings, 500 Hz, EDF+ format
════════════════════════════════════════════════════════════════════════════
"""

import argparse
import glob
import json
import math
import os
import pickle

import numpy as np
import pywt
from scipy import signal as sg
from scipy.signal import butter, filtfilt, find_peaks
from scipy.spatial.distance import pdist
from scipy import stats

try:
    import pyedflib
except ImportError:
    pyedflib = None

SEED = 42        # random number generator seed
N_HARM = 20      # number of harmonics in the state vector
CH_TILT = 1      # lead II in the tilt test recordings


from collections import Counter



# ──────────────────────────────────────────────────────────────────────────
# detect3
# ──────────────────────────────────────────────────────────────────────────

def detect_r(sig, fs, band=(5.0, 20.0), integ_ms=120,
             min_rr_ms=300, win_s=10.0, k=0.35, refine_ms=100):
    x = np.asarray(sig, float)
    x = x - np.median(x)

    # 1) band-pass filter - the QRS energy band
    ny = fs/2
    b, a = butter(3, [band[0]/ny, min(band[1]/ny, 0.99)], btype='band')
    xf = filtfilt(b, a, x)

    # 2) derivative and squaring (emphasises the steep QRS slopes)
    d = np.diff(xf, append=xf[-1])
    sq = d**2

    # 3) moving-window integration ~ QRS duration
    w = max(3, int(round(integ_ms/1000*fs)))
    integ = np.convolve(sq, np.ones(w)/w, mode='same')

    # 4) LOCALLY adaptive threshold: a fraction of the local maximum
    #    (window win_s seconds, step half the window)
    n = len(integ)
    W = max(int(win_s*fs), w*4)
    thr = np.zeros(n)
    for start in range(0, n, W//2):
        end = min(start+W, n)
        seg = integ[start:end]
        if len(seg) == 0:
            continue
        local = k*np.percentile(seg, 98)      # 98th percentile ~ QRS level
        thr[start:end] = np.maximum(thr[start:end], local)
    thr[thr == 0] = k*np.percentile(integ, 98)

    # 5) peak search with a refractory period
    dist = max(1, int(round(min_rr_ms/1000*fs)))
    cand, _ = find_peaks(integ, height=thr, distance=dist)

    # 6) position refinement at the maximum of |band-pass signal|
    half = max(2, int(round(refine_ms/1000*fs/2)))
    out = []
    for c in cand:
        a0, b0 = max(0, c-half), min(len(xf), c+half+1)
        out.append(a0 + int(np.argmax(np.abs(xf[a0:b0]))))
    return np.array(sorted(set(out)), dtype=int), xf



# ──────────────────────────────────────────────────────────────────────────
# pt_detect
# ──────────────────────────────────────────────────────────────────────────

def suppress_qrs(sig, r_idx, fs, before_ms=50, after_ms=100):
    out = sig.copy()
    b = int(round(before_ms/1000*fs)); a = int(round(after_ms/1000*fs))
    for r in r_idx:
        i0, i1 = max(0, r-b), min(len(sig)-1, r+a)
        if i1-i0 < 2: continue
        out[i0:i1+1] = np.interp(np.arange(i0, i1+1), [i0, i1], [sig[i0], sig[i1]])
    return out


def halfwave(lam):
    """Half-wave wavelet: a positive half-period of a sine
       with symmetric negative wings (zero mean)."""
    if lam % 2 == 0: lam += 1
    pos = np.sin(np.linspace(0, math.pi, lam))
    area = float(np.trapezoid(pos)) if hasattr(np, 'trapezoid') else float(np.trapz(pos))
    wing = max(1, lam//4)
    w = np.zeros(lam + 2*wing)
    w[:wing] = -area/(2*wing)
    w[wing:wing+lam] = pos
    w[wing+lam:] = -area/(2*wing)
    return w - w.mean()


def detect_pt(sig, r_idx, fs, wave_ms=85, thr_frac=0.3):
    """Returns the indices of P and T (one per cardiac cycle, where detected)."""
    r_idx = np.asarray(r_idx, int)
    ecg = suppress_qrs(sig, r_idx, fs)
    lam = max(5, int(round(wave_ms/1000*fs)))
    w = halfwave(lam)
    conv = np.convolve(ecg, w[::-1], mode='same')

    # threshold over usable segments (abnormally long RR intervals excluded)
    rr = np.diff(r_idx)
    med = np.median(rr) if len(rr) else 0
    ok = np.ones(len(conv), bool)
    for k in range(len(rr)):
        if med and rr[k] > 1.8*med:
            ok[r_idx[k]:r_idx[k+1]] = False
    pos = conv[(conv > 0) & ok]
    thr = thr_frac*np.percentile(pos, 99) if len(pos) else 0

    P, T = [], []
    for k in range(len(r_idx)):
        r = r_idx[k]
        r_prev = r_idx[k-1] if k > 0 else max(0, r-int(fs))
        r_next = r_idx[k+1] if k < len(r_idx)-1 else min(len(sig)-1, r+int(fs))

        # P: immediately before the QRS. The window is bounded both by a fraction of RR
        # and by physiological limits (P precedes R by 100-300 ms),
        # otherwise the T wave of the preceding cycle falls inside the window.
        rr_p = r - r_prev
        lo = max(r_prev + rr_p//2, r - int(0.35*fs))
        ps, pe = lo, r - max(1, int(0.06*fs))
        if ps < pe <= len(conv):
            seg = conv[ps:pe]
            if len(seg) and seg.max() > thr:
                P.append(ps + int(np.argmax(seg)))
            else: P.append(-1)
        else: P.append(-1)

        # T: after the QRS, before the next cycle begins
        rr_n = r_next - r
        ts = r + max(int(0.12*fs), rr_n//8)
        te = min(r + int(0.6*rr_n), r_next - int(0.08*fs))
        if ts < te <= len(conv):
            seg = conv[ts:te]
            if len(seg) and seg.max() > thr:
                T.append(ts + int(np.argmax(seg)))
            else: T.append(-1)
        else: T.append(-1)

    return np.array(P), np.array(T), ecg, conv



# ──────────────────────────────────────────────────────────────────────────
# variability
# ──────────────────────────────────────────────────────────────────────────

def hrv_metrics(rr_s):
    """Conventional heart rate variability indices from the RR series (in seconds)."""
    rr = np.asarray(rr_s, float)*1000.0          # -> ms
    if len(rr) < 5: return {}
    d = np.diff(rr)
    return dict(
        mean=float(rr.mean()),
        SDNN=float(rr.std(ddof=1)),
        RMSSD=float(np.sqrt(np.mean(d**2))),
        pNN50=float(100*np.mean(np.abs(d) > 50)),
        CV=float(100*rr.std(ddof=1)/rr.mean()),
    )

def series_metrics(x, unit='ms'):
    """Generic variability indices for an arbitrary series."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5: return {}
    d = np.diff(x)
    return dict(
        mean=float(x.mean()),
        SD=float(x.std(ddof=1)),
        RMSSD=float(np.sqrt(np.mean(d**2))),
        CV=float(100*x.std(ddof=1)/abs(x.mean())) if x.mean() else np.nan,
    )



# ──────────────────────────────────────────────────────────────────────────
# pipeline
# ──────────────────────────────────────────────────────────────────────────

import pyedflib


# ---------- 1. reading ----------
def read_ecg(path, ch=0):
    f = pyedflib.EdfReader(path)
    fs = float(f.getSampleFrequency(ch))
    sig = f.readSignal(ch).astype(float)
    f.close()
    return sig, fs


# ---------- 2. rejection of unusable segments ----------
def bad_mask(sig, fs, k=100, n_sigma=2.0):
    """Low-frequency component used as a noise indicator (procedure of section 1)."""
    b, a = butter(3, 0.5/(fs/2), btype='high')
    hf = filtfilt(b, a, sig)
    lf = sig - hf                      # low-frequency component
    d = np.empty_like(lf); d[0] = 0
    d[1:] = k*np.diff(lf)
    A, D = np.abs(lf), np.abs(d)
    lim_a = A.mean() + n_sigma*A.std()
    lim_d = D.mean() + n_sigma*D.std()
    return (A > lim_a) | (D > lim_d), hf


# ---------- 3. state vectors ----------
def fourier_vec(seg, fs, n_harm):
    m = len(seg)
    t = np.arange(m)/fs
    T = m/fs
    a0 = np.trapezoid(seg, t)/T if hasattr(np,'trapezoid') else np.trapz(seg, t)/T
    a, b = [], []
    for h in range(1, n_harm+1):
        w = 2*math.pi*h/T
        a.append(2*np.trapezoid(seg*np.cos(w*t), t)/T)
        b.append(2*np.trapezoid(seg*np.sin(w*t), t)/T)
    return T, a0, np.array(a), np.array(b)


def build_vectors(path, ch=0, n_harm=10):
    sig, fs = read_ecg(path, ch)
    bad, hf = bad_mask(sig, fs)
    r, _ = detect_r(sig, fs)
    V, meta = [], []
    for i in range(len(r)-1):
        i0, i1 = r[i], r[i+1]
        dur = (i1-i0)/fs
        if dur < 0.25 or dur > 3.0:      # physiological filter
            continue
        if bad[i0:i1].any():             # segment unusable
            continue
        T, a0, a, b = fourier_vec(hf[i0:i1], fs, n_harm)
        V.append(np.concatenate(([T, a0], a, b)))
        meta.append((i0/fs, dur))
    return np.array(V), np.array(meta), fs, len(r)



# ──────────────────────────────────────────────────────────────────────────
# graphs
# ──────────────────────────────────────────────────────────────────────────

class SOM:
    """Self-organizing map. The training schedule is defined by the fraction of
    progress (0..1) and therefore does not depend on the sample size."""
    def __init__(self, X, n, seed=42, v_start=0.5, v_end=0.01,
                 r_start=0.3, r_end=0.02, scale=1.0):
        rs = np.random.RandomState(seed)
        # weights initialised with real observations -- neurons start inside the data
        idx = rs.choice(len(X), n, replace=(len(X) < n))
        self.W = X[idx].astype(float).copy()
        self.rs = rs
        self.v_start, self.v_end = v_start, v_end
        self.r_start, self.r_end = r_start*scale, r_end*scale

    def _sched(self, a, b, p):
        return a*((b/a)**p)                     # geometric decay

    def fit(self, X, eras=20):
        total = eras*len(X); step = 0
        for _ in range(eras):
            for i in self.rs.permutation(len(X)):
                p = step/max(total-1, 1)
                v = self._sched(self.v_start, self.v_end, p)
                r = self._sched(self.r_start, self.r_end, p)
                x = X[i]
                w = int(np.argmin(np.linalg.norm(self.W - x, axis=1)))
                d = np.linalg.norm(self.W - self.W[w], axis=1)
                h = np.exp(-d**2/(2*r*r))
                self.W += (v*h)[:, None]*(x - self.W)
                step += 1
        return self


def cluster(V, n_neurons=40, merge_frac=0.15, min_size=10, seed=42):
    """Returns the cluster centres (in the original units) and the vector labels."""
    mu, sd = V.mean(0), V.std(0) + 1e-12
    Z = (V - mu)/sd
    rs = np.random.RandomState(0)
    sub = Z[rs.choice(len(Z), min(400, len(Z)), replace=False)]
    scale = float(np.median(pdist(sub)))

    W = SOM(Z, n_neurons, seed=seed, scale=scale).fit(Z).W

    # merging of nearby neurons
    thr = merge_frac*scale
    used = np.zeros(len(W), bool); cents = []
    for i in range(len(W)):
        if used[i]: continue
        grp = [i]; used[i] = True
        for j in range(i+1, len(W)):
            if not used[j] and np.linalg.norm(W[i]-W[j]) < thr:
                grp.append(j); used[j] = True
        cents.append(W[grp].mean(0))
    C = np.array(cents)

    lab = np.argmin(((Z[:, None, :] - C[None, :, :])**2).sum(-1), axis=1)
    # discard clusters with too few members
    cnt = Counter(lab.tolist())
    keep = sorted([c for c in cnt if cnt[c] >= min_size]) or sorted(cnt)
    Ck = C[keep]
    lab = np.argmin(((Z[:, None, :] - Ck[None, :, :])**2).sum(-1), axis=1)
    return Ck*sd + mu, lab


def build_graph(lab, n):
    M = np.zeros((n, n))
    for a, b in zip(lab[:-1], lab[1:]):
        M[a, b] += 1
    P = np.zeros_like(M)
    for i in range(n):
        s = M[i].sum()
        if s: P[i] = M[i]/s
    return M, P



# ──────────────────────────────────────────────────────────────────────────
# phase_graphs
# ──────────────────────────────────────────────────────────────────────────

def vectors_with_time(path, ch=1, n_harm=6, scale=1000.0):
    f = pyedflib.EdfReader(path)
    fs = float(f.getSampleFrequency(ch))
    sig = f.readSignal(ch).astype(float)/scale
    ann = f.readAnnotations(); f.close()
    marks = [float(t) for t in ann[0]]
    mid = [t for t in marks if 250 < t < 340]
    p1e = mid[0] if mid else 300.0
    p2b = mid[1] if len(mid) > 1 else p1e+15

    bad, hf = bad_mask(sig, fs)
    r, _ = detect_r(sig, fs)
    V, tt = [], []
    for i in range(len(r)-1):
        i0, i1 = r[i], r[i+1]
        d = (i1-i0)/fs
        if d < 0.3 or d > 2.0: continue
        if bad[i0:i1].any(): continue
        T, a0, a, b = fourier_vec(hf[i0:i1], fs, n_harm)
        V.append(np.concatenate(([T, a0], a, b)))
        tt.append(i0/fs)
    return np.array(V), np.array(tt), p1e, p2b


def entropy(P, w):
    """Weighted mean entropy of the rows of the transition matrix."""
    e = 0.0
    for i in range(len(P)):
        p = P[i][P[i] > 0]
        if len(p): e += w[i]*(-(p*np.log2(p)).sum())
    return float(e)


def analyse(path):
    V, tt, p1e, p2b = vectors_with_time(path)
    m1 = (tt > 30) & (tt < p1e-5)
    m2 = (tt > p2b+15) & (tt < p2b+275)
    if m1.sum() < 50 or m2.sum() < 50: return None

    C, lab = cluster(V[m1 | m2], n_neurons=30, merge_frac=0.18, min_size=15)
    n = len(C)
    sel = np.where(m1 | m2)[0]
    t_sel = tt[sel]
    in1 = (t_sel > 30) & (t_sel < p1e-5)

    res = {}
    for tag, mask in [('rest', in1), ('tilt', ~in1)]:
        L = lab[mask]
        M, P = build_graph(L, n)
        w = np.array([(L == c).mean() for c in range(n)])
        res[tag] = dict(
            occ=w,
            n_used=int((w > 0.01).sum()),
            edges=int((M > 0).sum()),
            self_p=float(np.sum([w[i]*P[i, i] for i in range(n)])),
            H=entropy(P, w),
        )
    res['n_states'] = n
    res['occ_shift'] = float(0.5*np.abs(res['rest']['occ']-res['tilt']['occ']).sum())
    return res



# ──────────────────────────────────────────────────────────────────────────
# inverse
# ──────────────────────────────────────────────────────────────────────────

def vec_to_cycle(v, fs, n_harm=None):
    """Reconstructs the cardiac cycle shape from its state vector."""
    T, a0 = float(v[0]), float(v[1])
    rest = v[2:]
    n = len(rest)//2 if n_harm is None else n_harm
    a, b = rest[:n], rest[n:2*n]
    m = max(2, int(round(T*fs)))
    t = np.arange(m)/fs
    w = 2*math.pi/T
    y = np.full(m, a0)
    for k in range(n):
        y += a[k]*np.cos((k+1)*w*t) + b[k]*np.sin((k+1)*w*t)
    return y


def graph_to_ecg(C, P, fs, n_beats=20, start=0, seed=0, smooth=True):
    """Synthesises an ECG by traversing the graph according to transition probabilities."""
    rs = np.random.RandomState(seed)
    cur = start
    parts, states = [], []
    for _ in range(n_beats):
        parts.append(vec_to_cycle(C[cur], fs))
        states.append(cur)
        p = P[cur]
        cur = rs.choice(len(P), p=p) if p.sum() > 0 else rs.randint(len(P))
    sig = np.concatenate(parts)
    if smooth:                      # smoothing of the joints between cycles
        pos = 0
        for seg in parts[:-1]:
            pos += len(seg)
            k = min(3, pos, len(sig)-pos)
            if k > 1:
                sig[pos-k:pos+k] = np.convolve(sig[pos-k:pos+k], np.ones(3)/3, mode='same')
    return sig, states


def reconstruction_error(orig, rec):
    """NRMSE as a percentage of the range of the original signal."""
    n = min(len(orig), len(rec))
    o, r = orig[:n], rec[:n]
    rng = o.max()-o.min()
    return float(100*np.sqrt(np.mean((o-r)**2))/rng) if rng else float('nan')



# ──────────────────────────────────────────────────────────────────────────
# synth
# ──────────────────────────────────────────────────────────────────────────

def _gauss(t, center, width, amp):
    return amp * np.exp(-0.5 * ((t - center) / width) ** 2)



def make_beat_template(fs, amp_scale=1.0):
    """Shape of a single beat (P, Q, R, S, T as a sum of Gaussian bumps) relative to
    the R peak (t=0). Amplitudes are set to the order of magnitude of a
    physiological ECG: P~0.15 mV, R~1.2 mV, T~0.3 mV."""
    def one_beat(t_ms):
        s = np.zeros_like(t_ms)
        s += _gauss(t_ms, -160, 25, 0.15 * amp_scale)   # P
        s += _gauss(t_ms, -40, 8, -0.10 * amp_scale)    # Q
        s += _gauss(t_ms, 0, 10, 1.20 * amp_scale)      # R
        s += _gauss(t_ms, 35, 9, -0.25 * amp_scale)     # S
        s += _gauss(t_ms, 250, 45, 0.30 * amp_scale)    # T
        return s
    return one_beat



def generate_synthetic_ecg(fs=500.0, duration_s=120.0, hr_bpm=70.0,
                            hrv_pct=3.0, noise_level="clean", seed=0):
    """Synthetic single-lead ECG with a known (prescribed) sequence of R peaks.
    noise_level: 'clean' | 'realistic' | 'noisy'.
    Returns (signal_mV, fs, true_r_idx)."""
    rng = np.random.default_rng(seed)
    mean_rr_s = 60.0 / hr_bpm
    n_samples = int(duration_s * fs)

    r_times = [0.3]
    while r_times[-1] < duration_s - 0.5:
        rr = mean_rr_s * (1 + rng.normal(0, hrv_pct / 100.0))
        rr = max(0.3, rr)
        r_times.append(r_times[-1] + rr)
    r_times = np.array(r_times[:-1])
    true_r_idx = np.round(r_times * fs).astype(int)
    true_r_idx = true_r_idx[true_r_idx < n_samples]

    one_beat = make_beat_template(fs)
    t_axis_ms = (np.arange(n_samples) / fs) * 1000.0
    signal = np.zeros(n_samples)

    half_lo_ms, half_hi_ms = 300, 450
    for r in true_r_idx:
        r_ms = r / fs * 1000.0
        lo = max(0, int(r - half_lo_ms / 1000.0 * fs))
        hi = min(n_samples, int(r + half_hi_ms / 1000.0 * fs))
        signal[lo:hi] += one_beat(t_axis_ms[lo:hi] - r_ms)

    if noise_level == "clean":
        pass
    elif noise_level == "realistic":
        signal += rng.normal(0, 0.02, n_samples)
        signal += 0.10 * np.sin(2 * np.pi * 0.05 * np.arange(n_samples) / fs + rng.uniform(0, 2 * np.pi))
    elif noise_level == "noisy":
        signal += rng.normal(0, 0.06, n_samples)
        signal += 0.25 * np.sin(2 * np.pi * 0.08 * np.arange(n_samples) / fs + rng.uniform(0, 2 * np.pi))
        n_art = int(duration_s / 20)
        for _ in range(n_art):
            pos = rng.integers(0, n_samples - 100)
            signal[pos:pos + 100] += rng.normal(0, 0.4, 100)
    else:
        raise ValueError(noise_level)

    return signal, fs, true_r_idx



def match_peaks(true_idx, det_idx, fs, tol_ms):
    """Greedy matching of detected peaks against the reference ones (each reference
    peak is used at most once). Returns TP, FN, FP and the list of position
    errors (ms) for the matches found."""
    tol = tol_ms / 1000.0 * fs
    true_idx, det_idx = np.sort(true_idx), np.sort(det_idx)
    used_true = np.zeros(len(true_idx), dtype=bool)
    used_det = np.zeros(len(det_idx), dtype=bool)
    errors_ms = []
    j = 0
    for i, t in enumerate(true_idx):
        best_j, best_d = -1, None
        k = j
        while k < len(det_idx) and det_idx[k] < t - tol:
            k += 1
        m = k
        while m < len(det_idx) and det_idx[m] <= t + tol:
            if not used_det[m]:
                d = abs(det_idx[m] - t)
                if best_d is None or d < best_d:
                    best_d, best_j = d, m
            m += 1
        if best_j >= 0:
            used_det[best_j] = True
            used_true[i] = True
            errors_ms.append((det_idx[best_j] - t) / fs * 1000.0)
        j = k
    TP = int(used_true.sum())
    FN = int((~used_true).sum())
    FP = int((~used_det).sum())
    return TP, FN, FP, errors_ms



def se_pp_f1(TP, FN, FP):
    se = TP / (TP + FN) if (TP + FN) > 0 else float('nan')
    pp = TP / (TP + FP) if (TP + FP) > 0 else float('nan')
    f1 = 2 * se * pp / (se + pp) if (se + pp) > 0 else float('nan')
    return se, pp, f1


# ──────────────────────────────────────────────────────────────────────────
# Computation stages
# ──────────────────────────────────────────────────────────────────────────






# ---------------------------------------------------------------- stage 1
def stage1_detection(files):
    """Annotation of R, P, T waves and cardiac cycle parameters by test phase."""
    rows = []
    for path in files:
        name = os.path.basename(path).replace('_ecg.edf', '')
        sig, fs, marks = read_with_marks(path)
        p1e, p2b = phase_bounds(marks)

        r, _ = detect_r(sig, fs)
        P, T, _, _ = detect_pt(sig, r, fs)
        t_r = r/fs
        rr = np.diff(r)/fs

        # Phase masks. For every series all cardiac cycles are used
        # in which the corresponding wave was detected: requiring the presence of
        # both P and T would reduce the sample without need.
        ph1 = (t_r[:-1] > 30) & (t_r[:-1] < p1e-5)
        ph2 = (t_r[:-1] > p2b+15) & (t_r[:-1] < p2b+275)
        in1 = (t_r > 30) & (t_r < p1e-5)
        in2 = (t_r > p2b+15) & (t_r < p2b+275)
        hasP, hasT = P > 0, T > 0
        p1m, p2m = in1 & hasP, in2 & hasP        # cycles with a detected P wave
        t1m, t2m = in1 & hasT, in2 & hasT        # cycles with a detected T wave

        rows.append(dict(
            file=name, R=len(r), P=int(hasP.sum()), T=int(hasT.sum()),
            rr1=hrv_metrics(rr[ph1]), rr2=hrv_metrics(rr[ph2]),
            pr1=series_metrics((r[p1m]-P[p1m])/fs*1000),
            pr2=series_metrics((r[p2m]-P[p2m])/fs*1000),
            rt1=series_metrics((T[t1m]-r[t1m])/fs*1000),
            rt2=series_metrics((T[t2m]-r[t2m])/fs*1000),
            ampR1=series_metrics(sig[r[in1]]), ampR2=series_metrics(sig[r[in2]]),
            ampP1=series_metrics(sig[P[p1m]]), ampP2=series_metrics(sig[P[p2m]]),
            ampT1=series_metrics(sig[T[t1m]]), ampT2=series_metrics(sig[T[t2m]]),
        ))
        print(f'  {name}: R={len(r)}, P={int((P>0).sum())}, T={int((T>0).sum())}')
    return rows


def read_with_marks(path):
    f = pyedflib.EdfReader(path)
    fs = float(f.getSampleFrequency(CH_TILT))
    sig = f.readSignal(CH_TILT).astype(float)/1000.0     # uV -> mV
    ann = f.readAnnotations()
    f.close()
    return sig, fs, [float(t) for t in ann[0]]


def phase_bounds(marks):
    mid = [t for t in marks if 250 < t < 340]
    p1e = mid[0] if mid else 300.0
    p2b = mid[1] if len(mid) > 1 else p1e+15
    return p1e, p2b


# ---------------------------------------------------------------- stage 2
def stage2_variability(rows):
    """Comparison of indices between phases (paired Wilcoxon test)."""
    def col(grp, key, ph):
        return np.array([r.get(f'{grp}{ph}', {}).get(key, np.nan) for r in rows], float)

    tests = [('RR','mean'),('RR','SDNN'),('RR','RMSSD'),('RR','pNN50'),('RR','CV'),
             ('PR','mean'),('PR','SD'),('PR','RMSSD'),('PR','CV'),
             ('RT','mean'),('RT','SD'),('RT','RMSSD'),('RT','CV'),
             ('ampR','mean'),('ampR','SD'),
             ('ampP','mean'),('ampP','SD'),
             ('ampT','mean'),('ampT','SD')]
    key = {'RR':'rr','PR':'pr','RT':'rt','ampR':'ampR','ampP':'ampP','ampT':'ampT'}
    out = {}
    print(f'\n  {"index":16s} {"rest":>16s} {"tilt":>16s} {"p":>10s}')
    for grp, m in tests:
        x, y = col(key[grp], m, '1'), col(key[grp], m, '2')
        msk = np.isfinite(x) & np.isfinite(y)
        if msk.sum() < 6:
            continue
        x, y = x[msk], y[msk]
        p = stats.wilcoxon(x, y)[1]
        out[f'{grp}.{m}'] = dict(rest=[x.mean(), x.std(ddof=1)],
                                 tilt=[y.mean(), y.std(ddof=1)], p=p, n=int(msk.sum()))
        print(f'  {grp+" "+m:16s} {x.mean():8.2f}±{x.std(ddof=1):6.2f} '
              f'{y.mean():8.2f}±{y.std(ddof=1):6.2f} {p:10.4f}')
    return out


# ---------------------------------------------------------------- stage 3
def stage3_phase_graphs(files, seeds=(0, 1, 7, SEED, 123)):
    """Phase graphs in a common state space, with a reproducibility check."""
    cache = {}
    for path in files:
        name = os.path.basename(path).replace('_ecg.edf', '')
        V, tt, p1e, p2b = vectors_with_time(path, ch=CH_TILT, n_harm=6)
        m1 = (tt > 30) & (tt < p1e-5)
        m2 = (tt > p2b+15) & (tt < p2b+275)
        if m1.sum() < 50 or m2.sum() < 50:
            continue
        sel = m1 | m2
        t = tt[sel]
        cache[name] = (V[sel], (t > 30) & (t < p1e-5))

    summary = []
    for sd in seeds:
        sh, s1, s2 = [], [], []
        for name, (V, in1) in cache.items():
            C, lab = cluster(V, n_neurons=30, merge_frac=0.18, min_size=15, seed=sd)
            n = len(C)
            w1 = np.array([(lab[in1] == c).mean() for c in range(n)])
            w2 = np.array([(lab[~in1] == c).mean() for c in range(n)])
            sh.append(0.5*np.abs(w1-w2).sum())
            for mask, acc in ((in1, s1), (~in1, s2)):
                L = lab[mask]
                M, P = build_graph(L, n)
                w = np.array([(L == c).mean() for c in range(n)])
                acc.append(float(sum(w[i]*P[i, i] for i in range(n))))
        sh, s1, s2 = map(np.array, (sh, s1, s2))
        summary.append(dict(seed=sd, shift=sh.mean(), p=stats.wilcoxon(sh)[1],
                            n_sig=int((sh > 0.3).sum()),
                            stab_rest=s1.mean(), stab_tilt=s2.mean(),
                            p_stab=stats.wilcoxon(s1, s2)[1]))
        print(f'  seed={sd:3d}: shift={sh.mean():.2f}, p={stats.wilcoxon(sh)[1]:.1e}, '
              f'significant={int((sh>0.3).sum())}/{len(sh)}, '
              f'stability {s1.mean():.2f}->{s2.mean():.2f}')
    return cache, summary


# ---------------------------------------------------------------- stage 4
def stage4_baseline(cache):
    """Comparison with k-means for an equal number of clusters."""
    from sklearn.cluster import KMeans
    som, km = [], []
    for name, (V, in1) in cache.items():
        C, lab = cluster(V, n_neurons=30, merge_frac=0.18, min_size=15, seed=SEED)
        n = len(C)
        som.append(shift(lab, in1, n))
        mu, sd = V.mean(0), V.std(0)+1e-12
        lk = KMeans(n_clusters=n, n_init=10, random_state=SEED).fit_predict((V-mu)/sd)
        km.append(shift(lk, in1, n))
    som, km = np.array(som), np.array(km)
    p = stats.wilcoxon(som, km)[1]
    print(f'  SOM {som.mean():.2f} | k-means {km.mean():.2f} | difference p={p:.3f}')
    return dict(som=som.mean(), kmeans=km.mean(), p=p)


def shift(lab, in1, n):
    w1 = np.array([(lab[in1] == c).mean() for c in range(n)])
    w2 = np.array([(lab[~in1] == c).mean() for c in range(n)])
    return 0.5*np.abs(w1-w2).sum()


# ---------------------------------------------------------------- stage 5
def stage5_inverse(files):
    """Inverse transform: reconstruction accuracy and compression."""
    res = []
    for path in files:
        name = os.path.basename(path).replace('_ecg.edf', '')
        sig, fs, _ = read_with_marks(path)
        bad, hf = bad_mask(sig, fs)
        r, _ = detect_r(sig, fs)
        V, segs = [], []
        for i in range(len(r)-1):
            i0, i1 = r[i], r[i+1]
            d = (i1-i0)/fs
            if d < 0.3 or d > 2.0 or bad[i0:i1].any() or (i1-i0)//2 < N_HARM:
                continue
            T, a0, a, b = fourier_vec(hf[i0:i1], fs, N_HARM)
            V.append(np.concatenate(([T, a0], a, b)))
            segs.append(hf[i0:i1])
        if len(V) < 100:
            continue
        V = np.array(V)
        e_vec = [reconstruction_error(segs[k], vec_to_cycle(V[k], fs))
                 for k in range(0, len(V), 5)]
        C, lab = cluster(V, n_neurons=30, merge_frac=0.18, min_size=15, seed=SEED)
        e_cl = [reconstruction_error(segs[k], vec_to_cycle(C[lab[k]], fs))
                for k in range(0, len(V), 5)]
        n_samp = sum(len(s) for s in segs)
        n_graph = len(C)*(2*N_HARM+2) + len(C)**2
        res.append((np.median(e_vec), np.median(e_cl), n_samp/n_graph))
        print(f'  {name}: vector->cycle {np.median(e_vec):.2f}%, '
              f'cycle->centre {np.median(e_cl):.1f}%, compression {n_samp/n_graph:.0f}x')
    a = np.array(res)
    print(f'\n  TOTAL: {a[:,0].mean():.2f}±{a[:,0].std(ddof=1):.2f}% | '
          f'{a[:,1].mean():.1f}±{a[:,1].std(ddof=1):.1f}% | compression {np.median(a[:,2]):.0f}x')
    return a


# ---------------------------------------------------------------- main


# ──────────────────────────────────────────────────────────────────────────
# Validation on synthetic signals
# ──────────────────────────────────────────────────────────────────────────

# Scenarios from table 1 of the paper
SCENARIOS = [
    dict(name='Clean signal',      hr=60,  noise='clean',     hrv=1, seed=1),
    dict(name='Clean signal',      hr=100, noise='clean',     hrv=1, seed=2),
    dict(name='Realistic noise',   hr=70,  noise='realistic', hrv=5, seed=3),
    dict(name='Realistic noise',   hr=45,  noise='realistic', hrv=5, seed=4),
    dict(name='Realistic noise',   hr=130, noise='realistic', hrv=8, seed=5),
    dict(name='Noise and artefacts',    hr=75,  noise='noisy',     hrv=6, seed=6),
    dict(name='Noise and artefacts',    hr=160, noise='noisy',     hrv=6, seed=7),
]

# Matching tolerances, ms
TOL = dict(R=50, P=40, T=60)

# Wave delays relative to R in the generator template, s
DELAY = dict(P=-0.160, T=+0.250)


def validate_scenarios(fs=500.0, duration_s=120.0):
    """Table 1: sensitivity and positive predictive value by scenario."""
    agg = {k: [0, 0, 0] for k in 'RPT'}
    print(f'{"Scenario":22s} {"HR":>5s} | {"R: Se/+P":>13s} | {"P: Se/+P":>13s} | {"T: Se/+P":>13s}')
    for s in SCENARIOS:
        sig, fs_, true_r = generate_synthetic_ecg(
            fs=fs, duration_s=duration_s, hr_bpm=s['hr'],
            hrv_pct=s['hrv'], noise_level=s['noise'], seed=s['seed'])
        det, _ = detect_r(sig, fs_)
        Pd, Td, _, _ = detect_pt(sig, det, fs_)

        truth = dict(R=true_r,
                     P=(true_r + DELAY['P']*fs_).astype(int),
                     T=(true_r + DELAY['T']*fs_).astype(int))
        found = dict(R=det, P=Pd[Pd > 0], T=Td[Td > 0])

        out = {}
        for k in 'RPT':
            TP, FN, FP, _ = match_peaks(truth[k], np.asarray(found[k]), fs_, tol_ms=TOL[k])
            se, pp, _ = se_pp_f1(TP, FN, FP)
            out[k] = (se*100, pp*100)
            agg[k][0] += TP; agg[k][1] += FN; agg[k][2] += FP
        print(f'{s["name"]:22s} {s["hr"]:5d} | '
              f'{out["R"][0]:5.1f}/{out["R"][1]:6.1f} | '
              f'{out["P"][0]:5.1f}/{out["P"][1]:6.1f} | '
              f'{out["T"][0]:5.1f}/{out["T"][1]:6.1f}')

    print()
    for k in 'RPT':
        TP, FN, FP = agg[k]
        se, pp, f1 = se_pp_f1(TP, FN, FP)
        print(f'{k}: Se = {se*100:.2f} %, +P = {pp*100:.2f} %, F1 = {f1*100:.2f} % '
              f'(TP={TP}, FN={FN}, FP={FP})')
    return agg


def validate_tolerance(fs=500.0):
    """Check: the result does not depend on the matching tolerance."""
    sig, fs_, true_r = generate_synthetic_ecg(fs=fs, duration_s=120.0, hr_bpm=75,
                                              hrv_pct=6, noise_level='noisy', seed=6)
    det, _ = detect_r(sig, fs_)
    print(f'\n{"Tolerance, ms":>14s} {"Se, %":>7s} {"+P, %":>7s} {"error, ms":>11s}')
    for tol in (50, 20, 10, 5, 2):
        TP, FN, FP, errs = match_peaks(true_r, det, fs_, tol_ms=tol)
        se, pp, _ = se_pp_f1(TP, FN, FP)
        me = np.mean(np.abs(errs)) if errs else float('nan')
        print(f'{tol:11d} {se*100:7.2f} {pp*100:7.2f} {me:11.2f}')


def validate_noise(fs=500.0):
    """Robustness limit: increasing noise level."""
    base, fs_, true_r = generate_synthetic_ecg(fs=fs, duration_s=120.0, hr_bpm=75,
                                               hrv_pct=6, noise_level='clean', seed=6)
    amp = base.max() - base.min()
    rng = np.random.default_rng(0)
    print(f'\n{"noise σ, mV":>12s} {"SNR, dB":>9s} {"Se, %":>7s} {"+P, %":>7s}')
    for sd in (0.02, 0.06, 0.12, 0.25, 0.40, 0.60, 0.90):
        sig = base + rng.normal(0, sd, len(base))
        snr = 20*np.log10(amp/(sd*6))
        det, _ = detect_r(sig, fs_)
        TP, FN, FP, _ = match_peaks(true_r, det, fs_, tol_ms=50)
        se, pp, _ = se_pp_f1(TP, FN, FP)
        print(f'{sd:11.2f} {snr:9.1f} {se*100:7.2f} {pp*100:7.2f}')

# ──────────────────────────────────────────────────────────────────────────
# Building the figures of the paper
# ──────────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp

DARK='#0F0F0F'; ACC='#A62B2B'; BLUE='#2e6da4'; GREY='#EAEAEA'

# ─────────────────────────── diagrams (no data needed) ─────────────────────
def fig1_scheme(out):
    fig, ax = plt.subplots(figsize=(13.5, 3.4))
    steps = [('ECG', 'recording of\narbitrary\nduration'),
             ('Annotation', 'wavelet detection\nof R, P, T'),
             ('State\nvector', 'Fourier series:\nT, a₀, (aₙ,bₙ)'),
             ('Clusters', 'Kohonen map,\nunsupervised'),
             ('Digraph', 'vertices: states\nedges: probabilities')]
    w, h, gap = 2.3, 1.72, 0.42
    for i, (t, s) in enumerate(steps):
        x = i*(w+gap)
        face = ACC if i == 4 else GREY
        tc = 'white' if i == 4 else DARK
        sc = '#F5E2E2' if i == 4 else '#2E2E2E'
        ax.add_patch(mp.FancyBboxPatch((x, 0.15), w, h, boxstyle='round,pad=0.05',
                                       facecolor=face, edgecolor='none'))
        ax.text(x+w/2, 0.15+h*0.71, t, ha='center', va='center',
                fontsize=19, fontweight='bold', color=tc)
        ax.text(x+w/2, 0.15+h*0.26, s, ha='center', va='center',
                fontsize=13, color=sc, linespacing=1.4)
        if i < 4:
            ax.annotate('', xy=(x+w+gap-0.04, 0.15+h/2), xytext=(x+w+0.04, 0.15+h/2),
                        arrowprops=dict(arrowstyle='-|>', lw=2.6, color=ACC))
    ax.set_xlim(-0.3, 13.3); ax.set_ylim(0, 2.05); ax.axis('off')
    _save(fig, out, 'fig1_scheme.png')


def fig3_phases(out):
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.annotate('', xy=(10.4, 3.55), xytext=(0.2, 3.55),
                arrowprops=dict(arrowstyle='-|>', lw=1.6, color='#555'))
    ax.text(10.5, 3.55, 'time', fontsize=11, color='#555', va='center')
    ax.add_patch(mp.Rectangle((0.4, 3.0), 4.3, 0.42, facecolor=BLUE, alpha=.85, edgecolor='none'))
    ax.text(2.55, 3.21, 'rest (5 min)', ha='center', va='center',
            color='white', fontsize=12, fontweight='bold')
    ax.add_patch(mp.Rectangle((4.7, 3.0), 0.75, 0.42, facecolor='#BBB', edgecolor='none'))
    ax.text(5.07, 2.80, 'table\ntilting', ha='center', va='top', fontsize=9.5, color='#555')
    ax.add_patch(mp.Rectangle((5.45, 3.0), 4.3, 0.42, facecolor=ACC, alpha=.85, edgecolor='none'))
    ax.text(7.6, 3.21, '45° tilt (5 min)', ha='center', va='center',
            color='white', fontsize=12, fontweight='bold')
    for x, col in [(2.55, BLUE), (7.6, ACC)]:
        ax.annotate('', xy=(x, 2.05), xytext=(x, 2.72),
                    arrowprops=dict(arrowstyle='-|>', lw=2, color=col))
    ax.add_patch(mp.FancyBboxPatch((2.1, 1.25), 6.0, 0.78, boxstyle='round,pad=0.05',
                                   facecolor=GREY, edgecolor='none'))
    ax.text(5.1, 1.64, 'clustering on the COMBINED data of both phases',
            ha='center', va='center', fontsize=11, fontweight='bold', color=DARK)
    ax.text(5.1, 1.37, 'common state space  S₀ … Sₙ',
            ha='center', va='center', fontsize=10, color='#444')
    for x, col, lab in [(2.55, BLUE, 'rest-phase graph'), (7.6, ACC, 'tilt-phase graph')]:
        ax.annotate('', xy=(x, 0.68), xytext=(x, 1.20),
                    arrowprops=dict(arrowstyle='-|>', lw=2, color=col))
        ax.add_patch(mp.FancyBboxPatch((x-1.45, 0.10), 2.9, 0.56, boxstyle='round,pad=0.04',
                                       facecolor=col, alpha=.85, edgecolor='none'))
        ax.text(x, 0.38, lab, ha='center', va='center', color='white',
                fontsize=11.5, fontweight='bold')
    ax.annotate('', xy=(6.10, 0.38), xytext=(4.05, 0.38),
                arrowprops=dict(arrowstyle='<|-|>', lw=1.8, color='#333'))
    ax.text(5.1, 0.62, 'comparison: occupancy shift D, stability S',
            ha='center', fontsize=11, color='#333', style='italic')
    ax.set_xlim(-0.1, 11.4); ax.set_ylim(-0.35, 3.95); ax.axis('off')
    _save(fig, out, 'fig3_phases.png')


def fig7_reversibility(out, n_cycles=13909):
    fig, ax = plt.subplots(figsize=(12, 4.4))
    blocks = [('ECG', f'{n_cycles}\ncardiac cycles'),
              ('State\nvectors', f'{2*N_HARM+2} components\nper cycle'),
              ('Digraph', '2–5 vertices')]
    w, h, gap = 2.5, 1.45, 1.35
    total = 3*w + 2*gap
    for i, (t, s) in enumerate(blocks):
        x = i*(w+gap)
        face = ACC if i == 2 else GREY
        tc = 'white' if i == 2 else DARK
        sc = '#F5E2E2' if i == 2 else '#2E2E2E'
        ax.add_patch(mp.FancyBboxPatch((x, 1.30), w, h, boxstyle='round,pad=0.05',
                                       facecolor=face, edgecolor='none'))
        ax.text(x+w/2, 1.30+h*0.68, t, ha='center', va='center',
                fontsize=15, fontweight='bold', color=tc)
        ax.text(x+w/2, 1.30+h*0.24, s, ha='center', va='center', fontsize=11, color=sc)
    for i in range(2):
        x = i*(w+gap)
        ax.annotate('', xy=(x+w+gap-0.10, 2.28), xytext=(x+w+0.10, 2.28),
                    arrowprops=dict(arrowstyle='-|>', lw=2.4, color=ACC))
        ax.text(x+w+gap/2, 2.44, 'compression', ha='center', fontsize=11, color=ACC, style='italic')
        ax.annotate('', xy=(x+w+0.10, 1.62), xytext=(x+w+gap-0.10, 1.62),
                    arrowprops=dict(arrowstyle='-|>', lw=2.4, color=BLUE))
    for i, lab in enumerate(['Fourier series\nsummation', 'graph traversal\nby probabilities']):
        ax.text(i*(w+gap)+w+gap/2, 1.24, lab, ha='center', va='top',
                fontsize=10.5, color=BLUE, style='italic', linespacing=1.3)
    ax.text(total/2, 3.28, 'forward transform: compression ≈ 2800-fold',
            ha='center', fontsize=13, color=ACC, fontweight='bold')
    ax.text(total/2, 0.34, 'inverse transform: reconstruction error 2.2%',
            ha='center', fontsize=13, color=BLUE, fontweight='bold')
    ax.set_xlim(-0.4, total+0.4); ax.set_ylim(0.05, 3.62); ax.axis('off')
    _save(fig, out, 'fig7_reversibility.png')


# ─────────────────────────── figures from the data ─────────────────────────
def fig2_waves(path, out):
    """Annotation of P, R and T waves in the two phases."""
    sig, fs, marks = read_with_marks(path)
    p1e, p2b = phase_bounds(marks)
    r, _ = detect_r(sig, fs)
    P, T, _, _ = detect_pt(sig, r, fs)
    fig, axs = plt.subplots(2, 1, figsize=(13, 6))
    for ax, t0 in zip(axs, [60, int(p2b)+90]):
        i0, i1 = int(t0*fs), int((t0+6)*fs)
        ax.plot(np.arange(i0, i1)/fs, sig[i0:i1], lw=1, color=DARK)
        for idx, c, m, lab in [(r, 'red', 'o', 'R'), (P, 'green', '^', 'P'), (T, 'blue', 'v', 'T')]:
            sel = idx[(idx >= i0) & (idx < i1)]
            sel = sel[sel > 0]
            if len(sel):
                ax.scatter(sel/fs, sig[sel], c=c, s=45, marker=m, zorder=5, label=lab)
        ax.set_title(f'{os.path.basename(path)}, {t0}–{t0+6} s')
        ax.grid(alpha=.3); ax.legend(fontsize=8, loc='upper right')
    _save(fig, out, 'fig2_waves.png')


def fig8_harmonics(path, out):
    """Reconstruction accuracy and compression as a function of the number of harmonics."""
    sig, fs, _ = read_with_marks(path)
    bad, hf = bad_mask(sig, fs)
    r, _ = detect_r(sig, fs)
    ns, errs, comps = [], [], []
    for n in (3, 5, 8, 10, 15, 20, 30):
        e = []
        for i in range(10, 110):
            i0, i1 = r[i], r[i+1]
            if (i1-i0)//2 < n or bad[i0:i1].any():
                continue
            T, a0, a, b = fourier_vec(hf[i0:i1], fs, n)
            e.append(reconstruction_error(hf[i0:i1], vec_to_cycle(np.concatenate(([T, a0], a, b)), fs)))
        if e:
            ns.append(n); errs.append(np.median(e))
            comps.append(np.median(np.diff(r[10:110]))/(2*n+2))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(ns, errs, 'o-', color=ACC, lw=2, ms=7, label='reconstruction error')
    ax.set_xlabel('number of harmonics'); ax.set_ylabel('error, % of range', color=ACC)
    ax.tick_params(axis='y', labelcolor=ACC); ax.grid(alpha=.3)
    ax2 = ax.twinx()
    ax2.plot(ns, comps, 's--', color=BLUE, lw=1.6, ms=6, label='cycle compression')
    ax2.set_ylabel('cycle compression, fold', color=BLUE); ax2.tick_params(axis='y', labelcolor=BLUE)
    ax.axvline(N_HARM, color='#888', ls=':', lw=1.2)
    l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, lb1+lb2, fontsize=9, loc='upper center')
    _save(fig, out, 'fig8_harmonics.png')


def fig4_dynamics(path, out):
    """Assignment of cardiac cycles to states over time."""
    V, tt, p1e, p2b = vectors_with_time(path, ch=CH_TILT, n_harm=6)
    m1 = (tt > 30) & (tt < p1e-5); m2 = (tt > p2b+15) & (tt < p2b+275)
    sel = m1 | m2
    C, lab = cluster(V[sel], n_neurons=30, merge_frac=0.18, min_size=15, seed=SEED)
    t = tt[sel]; n = len(C)
    fig, axs = plt.subplots(2, 1, figsize=(11, 5.6),
                            gridspec_kw={'height_ratios': [2, 1.1]})
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    for c in range(n):
        m = lab == c
        axs[0].scatter(t[m]/60, np.full(m.sum(), c), s=5, color=cols[c], alpha=.65)
    axs[0].axvspan(p1e/60, p2b/60, color='#ccc', alpha=.5)
    axs[0].set_yticks(range(n)); axs[0].set_yticklabels([f'S{c}' for c in range(n)])
    axs[0].set_ylabel('state'); axs[0].grid(alpha=.25, axis='x')
    axs[0].set_title('Assignment of cardiac cycles to states over time')
    in1 = (t > 30) & (t < p1e-5)
    w1 = [(lab[in1] == c).mean() for c in range(n)]
    w2 = [(lab[~in1] == c).mean() for c in range(n)]
    x = np.arange(n); wd = 0.38
    axs[1].bar(x-wd/2, w1, wd, label='rest', color=BLUE)
    axs[1].bar(x+wd/2, w2, wd, label='45° tilt', color=ACC)
    axs[1].set_xticks(x); axs[1].set_xticklabels([f'S{c}' for c in range(n)])
    axs[1].set_ylabel('fraction of time'); axs[1].legend(fontsize=9); axs[1].grid(alpha=.25, axis='y')
    sh = 0.5*sum(abs(a-b) for a, b in zip(w1, w2))
    axs[1].set_title(f'State occupancy by phase, shift = {sh:.2f}')
    _save(fig, out, 'fig4_dynamics.png')


def fig5_shift(files, out):
    """Occupancy shift in all subjects and the change in state stability."""
    from scipy import stats
    names, sh, s1, s2 = [], [], [], []
    for path in files:
        V, tt, p1e, p2b = vectors_with_time(path, ch=CH_TILT, n_harm=6)
        m1 = (tt > 30) & (tt < p1e-5); m2 = (tt > p2b+15) & (tt < p2b+275)
        if m1.sum() < 50 or m2.sum() < 50:
            continue
        sel = m1 | m2; t = tt[sel]; in1 = (t > 30) & (t < p1e-5)
        C, lab = cluster(V[sel], n_neurons=30, merge_frac=0.18, min_size=15, seed=SEED)
        n = len(C)
        w1 = np.array([(lab[in1] == c).mean() for c in range(n)])
        w2 = np.array([(lab[~in1] == c).mean() for c in range(n)])
        sh.append(0.5*np.abs(w1-w2).sum())
        for mask, acc in ((in1, s1), (~in1, s2)):
            L = lab[mask]; M, P = build_graph(L, n)
            w = np.array([(L == c).mean() for c in range(n)])
            acc.append(float(sum(w[i]*P[i, i] for i in range(n))))
        names.append(os.path.basename(path)[:4])
    sh, s1, s2 = map(np.array, (sh, s1, s2))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={'width_ratios': [1.5, 1]})
    order = np.argsort(sh)
    axs[0].barh(range(len(sh)), sh[order],
                color=[ACC if v > 0.3 else '#bbb' for v in sh[order]])
    axs[0].axvline(0.3, color='#444', ls='--', lw=1.2)
    axs[0].set_yticks(range(len(sh)))
    axs[0].set_yticklabels([names[i] for i in order], fontsize=7.5)
    axs[0].set_xlabel('shift in state occupancy'); axs[0].set_xlim(0, 1)
    axs[0].set_title(f'Shift (significant in {int((sh>0.3).sum())} of {len(sh)})')
    axs[0].grid(alpha=.25, axis='x')
    for a, b in zip(s1, s2):
        axs[1].plot([0, 1], [a, b], '-o', color='#999', lw=.9, ms=4, alpha=.6)
    axs[1].plot([0, 1], [s1.mean(), s2.mean()], '-o', color=ACC, lw=2.6, ms=9, label='mean')
    axs[1].set_xticks([0, 1]); axs[1].set_xticklabels(['rest', '45° tilt'])
    axs[1].set_ylabel('state stability'); axs[1].set_xlim(-0.25, 1.25)
    p = stats.wilcoxon(s1, s2)[1]
    axs[1].set_title(f'{s1.mean():.2f} → {s2.mean():.2f}, p = {p:.3f}')
    axs[1].legend(fontsize=9); axs[1].grid(alpha=.25, axis='y')
    _save(fig, out, 'fig5_shift.png')


def _graph_data(path, ch=0, scale=1.0, n_harm=N_HARM):
    """State vectors, clusters and graph for a single recording (for figures 6 and 8)."""
    import pyedflib
    f = pyedflib.EdfReader(path)
    fs = float(f.getSampleFrequency(ch))
    sig = f.readSignal(ch).astype(float)/scale
    f.close()
    bad, hf = bad_mask(sig, fs)
    r, _ = detect_r(sig, fs)
    V, segs, idx = [], [], []
    for i in range(len(r)-1):
        i0, i1 = r[i], r[i+1]; dur = (i1-i0)/fs
        if dur < 0.25 or dur > 3.0 or bad[i0:i1].any() or (i1-i0)//2 < n_harm:
            continue
        T, a0, a, b = fourier_vec(hf[i0:i1], fs, n_harm)
        V.append(np.concatenate(([T, a0], a, b))); segs.append(hf[i0:i1]); idx.append(i)
    V = np.array(V)
    C, lab = cluster(V, n_neurons=30, merge_frac=0.18, min_size=15, seed=SEED)
    M, P = build_graph(lab, len(C))
    return dict(V=V, segs=segs, idx=idx, r=r, hf=hf, fs=fs, C=C, lab=lab, P=P)


def _draw_graph(ax, C, lab, P, title, max_v=4):
    n = len(C)
    occ = np.array([(lab == k).mean() for k in range(n)])
    hr = np.array([60/C[k][0] for k in range(n)])
    keep = np.argsort(-occ)[:max_v]
    ang = np.linspace(0, 2*np.pi, len(keep), endpoint=False) + np.pi/2
    pos = {k: 1.15*np.array([np.cos(a), np.sin(a)]) for k, a in zip(keep, ang)}
    for i in keep:
        for j in keep:
            if i == j or P[i, j] < 0.05:
                continue
            ax.annotate('', xy=pos[j]*0.62, xytext=pos[i]*0.62,
                        arrowprops=dict(arrowstyle='-|>', lw=1+4*P[i, j], color='#8B2222',
                                        alpha=.7, connectionstyle='arc3,rad=0.2'))
    for k in keep:
        rr = 0.24 + 0.20*occ[k]
        ax.add_patch(plt.Circle(pos[k], rr, color=BLUE if hr[k] < 60 else '#c0392b',
                                alpha=.9, zorder=5))
        ax.text(*pos[k], f'{hr[k]:.0f}', color='white', ha='center', va='center',
                fontsize=14, fontweight='bold', zorder=6)
        ax.text(pos[k][0], pos[k][1]-rr-0.22, f'{100*occ[k]:.0f} %', ha='center',
                va='top', fontsize=10.5, color='#333')
    ax.set_xlim(-2.1, 2.1); ax.set_ylim(-2.2, 1.9); ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold')


def fig6_roh(paths, out):
    """Graphs of the test-database recordings, grouped by expert report."""
    fig, axs = plt.subplots(2, 2, figsize=(13, 11.6)); axs = axs.ravel()
    for ax, (name, diag, path) in zip(axs, paths):
        g = _graph_data(path)
        stab = float(sum(np.array([(g['lab'] == k).mean() for k in range(len(g['C']))])[i]*g['P'][i, i]
                         for i in range(len(g['C']))))
        _draw_graph(ax, g['C'], g['lab'], g['P'],
                    f'{name} — {diag}\nstates {len(g["C"])}, stability {stab:.2f}')
    _save(fig, out, 'fig6_roh.png')


def fig9_cycle(pairs, out):
    """The full cycle ECG -> graph -> ECG for two recordings."""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1.6, 1], hspace=0.75, wspace=0.22)
    for col, (title, path) in enumerate(pairs):
        g = _graph_data(path)
        fs = g['fs']; r = g['r']; idx = g['idx']
        k0 = min(40, len(idx)-11)
        orig = g['hf'][r[idx[k0]]:r[idx[k0+8]]]
        rec = np.concatenate([vec_to_cycle(g['V'][k], fs) for k in range(k0, k0+8)])
        syn, _ = graph_to_ecg(g['C'], g['P'], fs, n_beats=8,
                              start=int(np.bincount(g['lab']).argmax()), seed=3)
        ax = fig.add_subplot(gs[0, col]); ax.plot(np.arange(len(orig))/fs, orig, lw=1, color=DARK)
        ax.set_title(f'{title}\n(a) original ECG', fontsize=11.5); ax.grid(alpha=.25); ax.set_ylabel('mV')
        ax = fig.add_subplot(gs[1, col]); ax.plot(np.arange(len(rec))/fs, rec, lw=1, color=BLUE)
        ax.set_title('(b) reconstruction from state vectors', fontsize=11.5)
        ax.grid(alpha=.25); ax.set_ylabel('mV')
        ax = fig.add_subplot(gs[2, col])
        _draw_graph(ax, g['C'], g['lab'], g['P'], f'(c) state graph ({len(g["C"])} vertices)')
        ax = fig.add_subplot(gs[3, col]); ax.plot(np.arange(len(syn))/fs, syn, lw=1, color='#c0392b')
        ax.set_title('(d) ECG synthesised by traversing the graph', fontsize=11.5)
        ax.grid(alpha=.25); ax.set_xlabel('time, s'); ax.set_ylabel('mV')
    _save(fig, out, 'fig9_cycle.png')


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name), dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {name}')


# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────


def stage6_roh(roh_dir):
    """Graphs of the test-database recordings with expert reports (table 4).

    Reproduces the graph characteristics of the four recordings and the
    quantitative comparison: the distance between graphs within one report
    against the distance between graphs of different reports.
    """
    spec = [('07_VORO', 'bradycardia, narrow QRS', '07_VORO.edf'),
            ('10_MITI', 'bradycardia, narrow QRS', '10_MITI.edf'),
            ('01_GUSA', 'sudden tachycardia', '01_GUSA.edf'),
            ('03_SPI2', 'sudden tachycardia', '03_SPI2.edf')]
    have = [(n, d, os.path.join(roh_dir, f)) for n, d, f in spec
            if os.path.exists(os.path.join(roh_dir, f))]
    if len(have) < 2:
        print('  test-database recordings not found - stage skipped')
        return None

    print(f'  {"record":9s} {"expert report":24s} {"states":>7s} {"main HR":>8s} '
          f'{"share":>7s} {"stab.":>6s} {"entr.":>6s}')
    prof, names, diags = [], [], []
    for name, diag, path in have:
        g = _graph_data(path)
        C, lab, P = g['C'], g['lab'], g['P']
        n = len(C)
        occ = np.array([(lab == k).mean() for k in range(n)])
        hr = np.array([60/C[k][0] for k in range(n)])
        stab = float(sum(occ[i]*P[i, i] for i in range(n)))
        H = 0.0
        for i in range(n):
            p = P[i][P[i] > 0]
            if len(p):
                H += occ[i]*(-(p*np.log2(p)).sum())
        k0 = int(np.argmax(occ))
        print(f'  {name:9s} {diag:24s} {n:5d} {hr[k0]:8.0f} '
              f'{100*occ[k0]:5.0f}% {stab:6.2f} {H:6.2f}')
        prof.append([hr[k0], float((hr*occ).sum()), hr.max()-hr.min(), stab, H])
        names.append(name); diags.append(diag)

    if len(prof) < 4:
        return None

    # normalised profile and pairwise distances
    M = np.array(prof, float)
    Mn = (M - M.mean(0))/(M.std(0) + 1e-12)
    inner, outer = [], []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            dist = float(np.linalg.norm(Mn[i] - Mn[j]))
            (inner if diags[i] == diags[j] else outer).append(dist)
    print()
    print(f'  distance within the same report:   {np.mean(inner):.2f}')
    print(f'  distance between different reports: {np.mean(outer):.2f}')
    print(f'  ratio:                              {np.mean(outer)/np.mean(inner):.1f}x')
    return dict(names=names, diags=diags, profile=M.tolist(),
                inner=float(np.mean(inner)), outer=float(np.mean(outer)))


def run_all(data_dir, out_dir='results', roh_dir=None):
    """Full computation over the head-up tilt test recordings."""
    if pyedflib is None:
        raise SystemExit('pyedflib is not installed:  pip install pyedflib')
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(data_dir, '00*_ecg.edf')))
    if not files:
        raise SystemExit(f'No files of the form 00NN_ecg.edf found in {data_dir}')
    print(f'Recordings found: {len(files)}\n')

    print('STAGE 1. ECG annotation')
    rows = stage1_detection(files)
    json.dump(rows, open(os.path.join(out_dir, 'detection.json'), 'w'),
              ensure_ascii=False, indent=1)

    print('\nSTAGE 2. Variability of cardiac cycle elements')
    var = stage2_variability(rows)
    json.dump(var, open(os.path.join(out_dir, 'variability.json'), 'w'),
              ensure_ascii=False, indent=1, default=float)

    print('\nSTAGE 3. Phase graphs and reproducibility of the result')
    cache, summ = stage3_phase_graphs(files)
    pickle.dump(cache, open(os.path.join(out_dir, 'vectors.pkl'), 'wb'))

    print('\nSTAGE 4. Comparison with k-means')
    stage4_baseline(cache)

    print('\nSTAGE 5. Inverse transform')
    stage5_inverse(files)

    if roh_dir:
        print('\nSTAGE 6. Graphs of the test-database recordings')
        roh = stage6_roh(roh_dir)
        if roh:
            json.dump(roh, open(os.path.join(out_dir, 'roh_graphs.json'), 'w'),
                      ensure_ascii=False, indent=1)

    print(f'\nDone. Results saved to {out_dir}/')



def make_figures(data_dir=None, roh_dir=None, out='figures'):
    """Builds the figures of the paper.

    The diagrams (figures 1, 3, 7) are built without data.
    data_dir is the folder with the tilt test recordings 00NN_ecg.edf (figures 2, 4, 5).
    roh_dir  is the folder with the test-database recordings (figures 6, 8).
    """
    print('Diagrams (no data needed):')
    fig1_scheme(out); fig3_phases(out); fig7_reversibility(out)

    if data_dir:
        files = sorted(glob.glob(os.path.join(data_dir, '00*_ecg.edf')))
        if files:
            print('\nFigures from the tilt test recordings:')
            fig2_waves(files[0], out)
            fig4_dynamics(files[0], out)
            fig8_harmonics(files[0], out)
            fig5_shift(files, out)

    if roh_dir:
        spec = [('VORO', 'bradycardia, narrow QRS', '07_VORO.edf'),
                ('MITI', 'bradycardia, narrow QRS', '10_MITI.edf'),
                ('GUSA', 'sudden tachycardia', '01_GUSA.edf'),
                ('SPI2', 'sudden tachycardia', '03_SPI2.edf')]
        have = [(n, d, os.path.join(roh_dir, f)) for n, d, f in spec
                if os.path.exists(os.path.join(roh_dir, f))]
        if have:
            print('\nFigures from the test-database recordings:')
        if len(have) == 4:
            fig6_roh(have, out)
        if len(have) >= 3:
            fig9_cycle([('07_VORO — bradycardia', have[0][2]),
                        ('01_GUSA — tachycardia', have[2][2])], out)

    print(f'\nFigures saved to {out}/')


def validate():
    """Validation of the annotation on synthetic signals. No external data needed."""
    print('ANNOTATION CHECK ON SYNTHETIC SIGNALS\n')
    validate_scenarios()
    print('\nROBUSTNESS TO THE MATCHING TOLERANCE')
    validate_tolerance()
    print('\nROBUSTNESS TO NOISE')
    validate_noise()


def _in_notebook():
    """Whether the code runs inside Jupyter or Colab."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ != 'TerminalInteractiveShell'
    except Exception:
        return False


