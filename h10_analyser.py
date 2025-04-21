#!/usr/bin/env python3
"""
Robust Polar H10 CSV analyser (2–4 columns), incorporating community review.
This is a crappy early version, ignre it

Improvements
------------
* **fs estimator** – mode of Δt (10 µs bin) → robust to lost packets.
* **SciPy find_peaks** – native height + distance → faster.
* **Single HR_DIP per episode** (no duplicates).
* **Vectorised QRS‑width** with threshold crossing cache.
* **NaN‑coercion** (`errors='coerce'`) and float32 dtypes.
* CLI flags: --outdir  --brady-drop  --qrs-wide  --plot.

USAGE
-----
    python h10_analyser_v4.py  file.csv  [options]
"""
import argparse, os, csv, numpy as np, pandas as pd, scipy.signal as sg, matplotlib.pyplot as plt

# ------------- defaults / tunables -----------------
BP_L, BP_H   = 5, 20          # QRS band
REFRACT_S    = 0.25
ROLL_N       = 3
BRADY_DROP_D = 30             # bpm
BRADY_CONS   = 3
QRS_WIDE_D   = 120            # ms
DT_BIN_US    = 10             # mode‑bin for fs

# ---------------- helpers --------------------------
def load_var_csv(path):
    cols=[[],[],[],[]]
    with open(path,newline="") as fh:
        for row in csv.reader(fh):
            if len(row)<2: continue
            for i in range(4):
                try: cols[i].append(float(row[i]))
                except (IndexError,ValueError): cols[i].append(np.nan)
    arrs=[np.asarray(c,dtype=np.float64) for c in cols]
    mask = ~np.isnan(arrs[0]) & ~np.isnan(arrs[1])   # keep rows with time & ecg
    return [a[mask] for a in arrs]
    #return [np.asarray(c,dtype=np.float64) for c in cols]

def fs_mode(time_ns):
    dt = np.diff(time_ns)
    if dt.size == 0:                  # failsafe for tiny file
        return 1
    bins = (dt // (DT_BIN_US * 1000))     # integer μs bins
    mode = np.bincount(bins.astype(np.int64)).argmax() * DT_BIN_US  # seconds
    return 1.0 / (mode * 1e-6)  

def bandpass(sig, fs, lo=BP_L, hi=BP_H, order=2):
    b,a=sg.butter(order,[lo/(fs/2),hi/(fs/2)],'band')
    return sg.filtfilt(b,a,sig)

def qrs_widths(sig, peaks, fs):
    thr = 0.3 * sig[peaks]              # vector threshold
    left = np.searchsorted(sig < thr[:,None], True, side='left')
    right= np.searchsorted(sig < thr[:,None], True, side='right')
    return (right-left)/fs*1000         # ms

# ---------------- main -----------------------------
def run(path, outdir, brady_drop, qrs_wide, plot):
    t_ns, ecg, hr_col, rr_col = load_var_csv(path)
    if np.isnan(ecg).all(): raise ValueError("ECG column empty")
    t_sec = (t_ns - t_ns[0]) / 1e9
    fs = fs_mode(t_ns)
    ecg_f = bandpass(ecg.astype(np.float32), fs)

    peaks,_ = sg.find_peaks(ecg_f, height=np.median(np.abs(ecg_f))*0.7,
                            distance=int(REFRACT_S*fs))

    rr = np.diff(t_sec[peaks])
    hr = 60/rr
    hr_t = t_sec[peaks][1:]

    hr_sm = pd.Series(hr).rolling(ROLL_N,center=True,min_periods=1).median().to_numpy()
    baseline = pd.Series(hr_sm).rolling(20,center=True,min_periods=1).median().to_numpy()

    # HR_DIP episodes
    dips = (baseline - hr_sm) > brady_drop
    edges = np.where(np.diff(dips.astype(int))==1)[0] + 1   # rising edges
    events=[(hr_t[i],"HR_DIP") for i in edges]              # one per episode

    # QRS width
    widths = []
    thr = 0.3*ecg_f[peaks]
    for pk,th in zip(peaks,thr):
        l=pk; r=pk
        while l>0 and ecg_f[l]>th: l-=1
        while r<len(ecg_f)-1 and ecg_f[r]>th: r+=1
        widths.append((r-l)/fs*1000)
    widths=np.asarray(widths)
    wide_idx = np.where(widths>qrs_wide)[0]
    events += [(t_sec[peaks[i]],"WIDE_QRS") for i in wide_idx]

    stem=os.path.splitext(os.path.basename(path))[0]
    out_base=os.path.join(outdir, stem)
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame({"timestamp_s":hr_t,"HR_BPM":hr_sm}).to_csv(out_base+"_hr.csv", index=False)
    if events:
        pd.DataFrame(events,columns=["timestamp_s","event"]).to_csv(out_base+"_events.csv",index=False)

    if plot:
        plt.figure(figsize=(12,4))
        plt.plot(hr_t, hr_sm,label='HR')
        for ts,ev in events:
            plt.axvline(ts,color='r' if ev=="HR_DIP" else 'purple', ls='--')
        plt.title(stem); plt.xlabel("Time (s)"); plt.ylabel("BPM"); plt.tight_layout(); plt.show()

    print(f"{path}: beats={len(peaks)}, HR_DIP={sum(e[1]=='HR_DIP' for e in events)}, "
          f"WIDE_QRS={sum(e[1]=='WIDE_QRS' for e in events)}, fs≈{fs:.1f} Hz")

# ---------------- CLI ------------------------------
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("csv", nargs='?', help="Polar ECG CSV")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--brady-drop", type=float, default=BRADY_DROP_D)
    ap.add_argument("--qrs-wide",  type=float, default=QRS_WIDE_D)
    args=ap.parse_args()

    if not args.csv:
        print(__doc__); exit(0)
    run(args.csv, args.outdir, args.brady_drop, args.qrs_wide, args.plot)
