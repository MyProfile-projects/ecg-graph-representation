"""Independent check of R-wave detection: the main detector (Pan-Tompkins-type, ecg_graph.detect_r)
versus the WTSEE detector (wavelet denoising + Shannon energy envelope, authors' notebook),
both on the SAME lead (II), detections matched by position within a tolerance."""
import sys, glob, os, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ecg_graph as eg
import wtsee as W

ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--tol', type=float, default=50)
ap.add_argument('--tol0026', type=float, default=100, help='tolerance for recording 0026 (fiducial point on S vs r)')
args = ap.parse_args()
rows = []
for p in sorted(glob.glob(os.path.join(args.data, '00*_ecg.edf'))):
    nm = os.path.basename(p)[:4]
    sig, fs, _ = eg.read_with_marks(p)                    # lead II, mV
    r_main, _ = eg.detect_r(sig, fs)
    r_ws = W.detect_r_peaks(W.filter_ecg(sig, fs), fs)    # WTSEE on the same lead
    auto = W.select_best_lead(p)
    TP, FN, FP, err = eg.match_peaks(r_main, r_ws, fs, tol_ms=args.tol0026 if nm == '0026' else args.tol)
    err = np.abs(err)
    rows.append(dict(rec=nm, auto_lead=auto, n_main=len(r_main), n_wtsee=len(r_ws), TP=TP, only_main=FN, only_wtsee=FP,
                     med_err=float(np.median(err)), p95_err=float(np.percentile(err, 95))))
    print(f"{nm}  WTSEE auto-lead={auto:5s}  main={len(r_main):4d}  WTSEE(II)={len(r_ws):4d}  matched={TP:4d}  "
          f"only main={FN:3d}  only WTSEE={FP:3d}  |dt| median={np.median(err):.1f} ms, 95%={np.percentile(err,95):.1f} ms")
T = {k: sum(r[k] for r in rows) for k in ('n_main', 'n_wtsee', 'TP', 'only_main', 'only_wtsee')}
print('\nTOTAL', T)
print(f"agreement: {100*T['TP']/T['n_main']:.2f}% of main detections confirmed by WTSEE; "
      f"{100*T['TP']/T['n_wtsee']:.2f}% of WTSEE detections confirmed by main")
allerr = []
json.dump(rows, open('wtsee_check.json', 'w'), indent=1)
