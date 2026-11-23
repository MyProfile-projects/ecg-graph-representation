# -*- coding: utf-8 -*-
# WTSEE R-wave detector (wavelet denoising + Shannon energy envelope), Section A of the authors' notebook
# Universal_ECG_20_files_final_WTSEE.ipynb (repository "Wavelet. P, R and T peaks detection"), unchanged.
"""
ecg_full_toolkit_FINAL.py
================================================================================
ЕДИНЫЙ файл, объединяющий весь код, использованный при разработке и проверке
итогового пайплайна детекции R/P/T-зубцов ЭКГ.

Структура файла (4 раздела):

  РАЗДЕЛ A — ФИНАЛЬНЫЙ АЛГОРИТМ (universal_ecg_pt_FINAL.py)
             Основной, рекомендуемый к использованию код. В этой финальной
             версии R-зубцы определяются методом WTSEE: вейвлет-денойзинг +
             модифицированная энергетическая огибающая Шеннона + уточнение
             координаты R-пика в локальном окне. Прежняя версия с простым
             адаптивным порогом заменена.

  РАЗДЕЛ B — НЕЗАВИСИМЫЙ ДЕТЕКТОР ДЛЯ ПЕРЕКРЁСТНОЙ ПРОВЕРКИ (ecg_pipeline.py)
             Второй, независимо реализованный (по другим принципам —
             Pan-Tompkins-подобная схема) детектор R/P/T. Используется не
             как основной инструмент анализа, а как "внешний свидетель":
             если оба независимых алгоритма совпадают — это содержательное
             свидетельство корректности на записях без готового эталона.
             Функции — с приставкой crossval_.

  РАЗДЕЛ C — СИНТЕТИЧЕСКИЙ ЭТАЛОН (synthetic_ecg.py)
             Генератор ЭКГ-подобного сигнала с ТОЧНО известными (заданными
             как параметр генерации, а не измеренными) положениями R-зубцов.
             Единственный способ получить численные Se/+P/F1 без доступа к
             аннотированным клиническим базам ввиду недоступности physionet.org и т.п.

  РАЗДЕЛ D — РАСЧЁТ МЕТРИК И ВАЛИДАЦИЯ (validate_r_peak_accuracy.py)
             Сопоставление найденных пиков с эталоном/друг с другом,
             расчёт Se/+P/F1, запуск полной проверки (синтетика + реальные
             записи) одной командой.

Пример использования — см. блок if __name__ == "__main__" в конце файла.
================================================================================
"""
import glob
import json
import math
import os
import warnings

import numpy as np
import pyedflib
import pywt
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, cheby2, filtfilt, welch
from scipy import signal as _sig


# ══════════════════════════════════════════════════════════════════════════
# РАЗДЕЛ A. ФИНАЛЬНЫЙ АЛГОРИТМ  (universal_ecg_pt_WTSEE_FINAL.py)
# ══════════════════════════════════════════════════════════════════════════

_UNIT_TO_MV = {
    "v": 1000.0, "volt": 1000.0, "volts": 1000.0,
    "mv": 1.0, "millivolt": 1.0, "millivolts": 1.0,
    "uv": 0.001, "µv": 0.001, "microvolt": 0.001, "microvolts": 0.001,
}


def find_ecg_channel(labels, preferred=None):
    """Находит индекс ЭКГ-канала по названию. Если preferred задан и
    найден — берёт его; иначе первый канал, содержащий 'ECG'/'ЭКГ'."""
    if preferred is not None:
        for i, lab in enumerate(labels):
            if preferred.lower() in lab.lower():
                return i
    for i, lab in enumerate(labels):
        if "ecg" in lab.lower() or "экг" in lab.lower():
            return i
    return 0


def read_edf_channel_mV(path: str, channel_name: str = None):
    """Универсальное чтение одного канала EDF в мВ: находит ЭКГ-канал по
    названию, приводит единицы измерения к мВ по метаданным заголовка."""
    f = pyedflib.EdfReader(path)
    try:
        labels = f.getSignalLabels()
        ch = find_ecg_channel(labels, channel_name)
        fs = float(f.getSampleFrequency(ch))
        sig = f.readSignal(ch).astype(np.float64)
        unit = f.getPhysicalDimension(ch).strip().lower()
        scale = _UNIT_TO_MV.get(unit)
        if scale is None:
            warnings.warn(f"Неизвестная единица измерения '{unit}' — "
                           f"сигнал возвращён БЕЗ масштабирования, проверьте вручную!")
            scale = 1.0
        return sig * scale, fs, labels[ch]
    finally:
        f.close()


def select_best_lead(path: str) -> str:
    """[v3] Автоматический выбор ЭКГ-отведения по максимуму доли
    спектральной энергии сигнала в полосе QRS (5-15 Гц) — вместо первого
    канала, где просто есть подстрока 'ecg'/'экг' в названии."""
    f = pyedflib.EdfReader(path)
    try:
        labels = f.getSignalLabels()
        fs = float(f.getSampleFrequency(0))
        raw = [f.readSignal(i).astype(np.float64) for i in range(f.signals_in_file)]
    finally:
        f.close()

    def qrs_energy_ratio(x):
        freqs, pxx = welch(x, fs=fs, nperseg=int(4 * fs))
        band = (freqs >= 5) & (freqs <= 15)
        return np.sum(pxx[band]) / (np.sum(pxx) + 1e-12)

    scores = [qrs_energy_ratio(x) for x in raw]
    return labels[int(np.argmax(scores))]


def _sine_wavelet(length: int) -> np.ndarray:
    t = np.linspace(0, math.pi, length)
    w = np.sin(t)
    return w - w.mean()


def filter_ecg(sig_mV: np.ndarray, fs: float, fc: float = 0.5, order: int = 5):
    """Фильтр Чебышёва II рода (ФВЧ, fc=0.5 Гц) — устраняет дрейф изолинии
    перед детекцией R, не искажая форму QRS-комплекса."""
    nyq = fs / 2.0
    b, a = cheby2(order, rs=40, Wn=fc / nyq, btype="high", analog=False)
    return filtfilt(b, a, sig_mV)


def _wavelet_image(sig: np.ndarray, w: np.ndarray) -> np.ndarray:
    L, N, half = len(w), len(sig), len(w) // 2
    out = np.zeros(N)
    for i in range(half, N - half):
        out[i] = float(np.dot(sig[i - half:i - half + L], w))
    return out


def _merge_close_peaks(peaks: np.ndarray, score: np.ndarray, min_gap: int) -> np.ndarray:
    """Оставляет один, наиболее выраженный пик в группе слишком близких кандидатов."""
    if len(peaks) == 0:
        return np.array([], dtype=int)
    peaks = np.asarray(sorted(set(int(p) for p in peaks)), dtype=int)
    groups = [[peaks[0]]]
    for p in peaks[1:]:
        if p - groups[-1][-1] < min_gap:
            groups[-1].append(p)
        else:
            groups.append([p])
    out = []
    for g in groups:
        g = np.asarray(g, dtype=int)
        out.append(int(g[np.argmax(score[g])]))
    return np.asarray(out, dtype=int)


def detect_r_peaks(ecg_mV: np.ndarray, fs: float, min_rr_ms: float = 300,
                   search_half_ms: float = 80.0,
                   see_ms: float = 65.0,
                   pee_ms: float = 85.0,
                   min_prominence_frac: float = 0.08) -> np.ndarray:
    """Финальная детекция R-зубцов методом WTSEE.

    WTSEE (Wavelet Transform and Shannon Energy Envelope) объединяет:
    1) вейвлет-денойзинг исходного ЭКГ-сигнала;
    2) построение модифицированной энергетической огибающей Шеннона;
    3) выделение кандидатов R-зубцов по огибающей пиковой энергии;
    4) уточнение координаты каждого R-зубца по максимуму исходного
       отфильтрованного ЭКГ-сигнала в локальном окне.

    В отличие от прежней версии с простым адаптивным порогом по отклику
    синус-вейвлета, здесь порог не является основным механизмом детекции:
    R-кандидаты формируются по энергетической огибающей Шеннона, что лучше
    соответствует заявленному методу WTSEE.

    Parameters
    ----------
    ecg_mV : np.ndarray
        ЭКГ-сигнал после предварительной фильтрации, мВ.
    fs : float
        Частота дискретизации, Гц.
    min_rr_ms : float
        Минимально допустимый интервал между соседними R-зубцами.
    search_half_ms : float
        Полуширина окна уточнения R-пика по исходному сигналу.
    see_ms, pee_ms : float
        Ширины сглаживающих окон для Shannon Energy Envelope и Peak Energy
        Envelope соответственно.
    min_prominence_frac : float
        Мягкая отсечка по выраженности пиков PEE относительно максимума.
    """
    x = np.asarray(ecg_mV, dtype=float)
    if len(x) < max(8, int(0.5 * fs)):
        return np.array([], dtype=int)

    # 1. Wavelet Transform denoising: sym5 + soft-thresholding detail coeffs.
    cA1, cD1 = pywt.dwt(x, "sym5", mode="symmetric")
    sigma = np.median(np.abs(cD1)) / 0.6745 if len(cD1) else 0.0
    thr = sigma * np.sqrt(2 * np.log(max(len(cD1), 2)))
    cD1t = np.sign(cD1) * np.maximum(np.abs(cD1) - thr, 0.0)
    ef = pywt.idwt(cA1, cD1t, "sym5", mode="symmetric")[:len(x)]

    # 2. Modified Shannon Energy Envelope.
    d = np.diff(ef, append=ef[-1])
    dmax = np.max(np.abs(d))
    if not np.isfinite(dmax) or dmax <= 1e-12:
        return np.array([], dtype=int)
    dn = d / dmax
    dn2 = dn ** 2
    se = -dn2 * np.log(dn2 + 1e-10)
    see_win = max(3, int(round(see_ms / 1000.0 * fs)))
    if see_win % 2 == 0:
        see_win += 1
    ss = np.convolve(se, np.ones(see_win) / see_win, mode="same")

    # 3. Peak Energy Envelope.
    ds = np.diff(ss, append=ss[-1])
    dsmax = np.max(np.abs(ds))
    if not np.isfinite(dsmax) or dsmax <= 1e-12:
        return np.array([], dtype=int)
    dnds = ds / dsmax
    pe = dnds ** 2
    pee_win = max(3, int(round(pee_ms / 1000.0 * fs)))
    if pee_win % 2 == 0:
        pee_win += 1
    ps = np.convolve(pe, np.ones(pee_win) / pee_win, mode="same")

    min_gap = max(1, int(min_rr_ms / 1000.0 * fs))
    prominence = float(np.max(ps)) * min_prominence_frac if len(ps) else None
    init_peaks, _ = find_peaks(ps, distance=max(1, int(0.20 * fs)), prominence=prominence)
    if len(init_peaks) == 0:
        init_peaks, _ = find_peaks(ps, distance=max(1, int(0.20 * fs)))
    if len(init_peaks) == 0:
        return np.array([], dtype=int)

    # 4. Уточнение: истинный R-пик берём как максимум EF в окрестности кандидата.
    search_half = max(1, int(round(search_half_ms / 1000.0 * fs)))
    refined = []
    for idx in init_peaks:
        lo = max(0, int(idx) - search_half)
        hi = min(len(ef), int(idx) + search_half + 1)
        if hi <= lo:
            continue
        refined.append(lo + int(np.argmax(ef[lo:hi])))

    return _merge_close_peaks(np.asarray(refined, dtype=int), ef, min_gap)


def check_double_counting(r_idx: np.ndarray, fs: float, tol: float = 0.15) -> bool:
    """Самопроверка на задвоение: True, если RR-интервалы регулярно
    чередуются короткий/длинный (признак того, что порог всё ещё ловит
    два срабатывания на один удар)."""
    if len(r_idx) < 5:
        return False
    rr = np.diff(r_idx) / fs
    short, long_ = rr[0::2], rr[1::2]
    n = min(len(short), len(long_))
    if n < 2:
        return False
    is_alt = (np.std(short[:n]) / np.mean(short[:n]) < tol and
              np.std(long_[:n]) / np.mean(long_[:n]) < tol and
              np.mean(short[:n]) < 0.6 * np.mean(long_[:n]))
    return bool(is_alt)


