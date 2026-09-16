# -*- coding: utf-8 -*-
"""
Пайплайн выбора лучшего отведения из массива ЭКГ по критерию чистоты
определения P-, Q-, R-, S- и T-пиков. (п.7 "Итогов второй встречи")

В каждой записи 3 отведения (ECG, ECG2, ECG3, все 500 Гц). До сих пор во
всех расчётах использовалось жёстко заданное ECG2 (CH_TILT = 1, "отведение
II" по комментарию в коде) — задание требует не считать это заранее
известным, а выбирать отведение по каждой записи отдельно, по объективному
критерию.

Критерий (сформулирован до расчёта): для R детектор почти всегда находит
комплекс независимо от отведения (это самый крупный пик ЭКГ) — поэтому
основной источник различий между отведениями это P, T (уже известно:
детектор P/T зависит от отведения — см. историю с записями 0011/0012/0022,
где из-за низкой предсердной волны P почти не находился) и Q, S (глубина
зубцов сильно зависит от угла отведения). Итоговый показатель чистоты
отведения — произведение двух долей: (а) доля кардиоциклов ВНУТРИ найденных
детектором R, где все четыре остальных зубца (P,Q,S,T) нашлись одновременно
(не среднее по отдельности — среднее скрывает случай, когда отведение хорошо
ловит P, но плохо T), и (б) доля самих R-пиков, найденных на этом отведении,
относительно отведения с максимальным их числом в этой записи — иначе
отведение, которое просто теряет часть кардиоциклов из-за плохой видимости
R, ошибочно выглядело бы «чистым» по остальным зубцам.

Детектор Q и S в исходном коде отсутствовал (был только R и P/T) — написан
здесь заново, тем же приёмом, что и P/T: локальный минимум в окне
физиологической длительности от R, "найден", если провал превышает порог,
заданный по локальному шуму изолинии, а не фиксированной величиной в мкВ.
"""

import sys
import numpy as np
from scipy.signal import butter, filtfilt
import glob

import os
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
sys.path.insert(0, _HERE)
import pp_pr_standalone as m

DATA_DIR = _HERE  # EDF-файлы лежат в той же папке, что и сам скрипт (или в рабочей папке блокнота)
CHANNELS = [0, 1, 2]
CHANNEL_NAMES = {0: 'ECG', 1: 'ECG2', 2: 'ECG3'}


def detect_qs(sig, r_idx, fs, q_ms=40, s_ms=60, k=2.5, baseline_ms=40):
    """Q — локальный минимум в окне q_ms перед R, S — в окне s_ms после R.
    "Найден", если провал относительно ближайшей изолинии (участок TP,
    оценённый как медиана сигнала в окне baseline_ms до Q-окна) превышает
    k * (локальный шум, MAD за тот же участок изолинии)."""
    r_idx = np.asarray(r_idx, int)
    qw = max(2, int(round(q_ms/1000*fs)))
    sw = max(2, int(round(s_ms/1000*fs)))
    bw = max(2, int(round(baseline_ms/1000*fs)))
    Q, S = [], []
    for r in r_idx:
        # --- Q ---
        b0, b1 = max(0, r-qw-bw), max(0, r-qw)
        q0, q1 = max(0, r-qw), r
        if b1 > b0 and q1 > q0:
            base = np.median(sig[b0:b1])
            noise = np.median(np.abs(sig[b0:b1]-base)) * 1.4826 + 1e-9
            seg = sig[q0:q1]
            j = int(np.argmin(seg))
            depth = base - seg[j]
            Q.append(q0+j if depth > k*noise else -1)
        else:
            Q.append(-1)
        # --- S ---
        s0, s1 = r, min(len(sig), r+sw)
        b0s, b1s = min(len(sig), r+sw), min(len(sig), r+sw+bw)
        if s1 > s0 and b1s > b0s:
            base = np.median(sig[b0s:b1s])
            noise = np.median(np.abs(sig[b0s:b1s]-base)) * 1.4826 + 1e-9
            seg = sig[s0:s1]
            j = int(np.argmin(seg))
            depth = base - seg[j]
            S.append(s0+j if depth > k*noise else -1)
        else:
            S.append(-1)
    return np.array(Q), np.array(S)


def lead_quality(path, ch, max_n_for_record):
    sig, fs = m.read_ecg(path, ch=ch)
    r_idx, xf = m.detect_r(sig, fs)
    n = len(r_idx)
    rate_r = n / max_n_for_record if max_n_for_record else 1.0
    if n < 10:
        return dict(n=n, rate_p=0, rate_q=0, rate_r=rate_r, rate_s=0, rate_t=0,
                    rate_all=0, score=0)
    P, T, _, _ = m.detect_pt(sig, r_idx, fs)
    Q, S = detect_qs(xf, r_idx, fs)
    found_p, found_q, found_s, found_t = P >= 0, Q >= 0, S >= 0, T >= 0
    all5 = found_p & found_q & found_s & found_t
    rate_all = all5.mean()
    return dict(
        n=n,
        rate_p=found_p.mean(), rate_q=found_q.mean(), rate_r=rate_r,
        rate_s=found_s.mean(), rate_t=found_t.mean(),
        rate_all=rate_all,
        score=rate_all * rate_r,   # чистота ВНУТРИ найденных циклов x полнота охвата циклов
    )


if __name__ == '__main__':
    files = sorted(glob.glob(f'{DATA_DIR}/*_ecg.edf'))
    rows = []
    for path in files:
        name = path.split('/')[-1][:4]
        # предварительный проход: сколько R нашлось на каждом канале, чтобы знать max
        counts = {}
        for ch in CHANNELS:
            sig, fs = m.read_ecg(path, ch=ch)
            r_idx, _ = m.detect_r(sig, fs)
            counts[ch] = len(r_idx)
        max_n = max(counts.values())
        per_ch = {ch: lead_quality(path, ch, max_n) for ch in CHANNELS}
        best_ch = max(CHANNELS, key=lambda c: per_ch[c]['score'])
        rows.append((name, per_ch, best_ch))

    print("=" * 110)
    print(f"{'запись':7s}{'n':>5s}   " +
          "  |  ".join(f"{CHANNEL_NAMES[c]:^36s}" for c in CHANNELS) + "   выбрано")
    print(" " * 14 + "  |  ".join(f"{'P':>5s}{'Q':>5s}{'R':>5s}{'S':>5s}{'T':>5s}{'ВСЕ5':>6s}{'ИТОГ':>6s}" for _ in CHANNELS))
    print("-" * 145)
    agree_with_hardcoded = 0
    for name, per_ch, best_ch in rows:
        n = per_ch[1]['n']
        line = f"{name:7s}{n:5d}   "
        parts = []
        for c in CHANNELS:
            q = per_ch[c]
            parts.append(f"{q['rate_p']*100:5.0f}{q['rate_q']*100:5.0f}{q['rate_r']*100:5.0f}"
                          f"{q['rate_s']*100:5.0f}{q['rate_t']*100:5.0f}{q['rate_all']*100:6.0f}{q['score']*100:6.0f}")
        line += "  |  ".join(parts) + f"   {CHANNEL_NAMES[best_ch]}"
        print(line)
        if best_ch == 1:
            agree_with_hardcoded += 1

    print("-" * 145)
    print(f"\nСовпадение выбора пайплайна с жёстко заданным ECG2: {agree_with_hardcoded} из {len(rows)} записей")
