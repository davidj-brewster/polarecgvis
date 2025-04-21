#!/usr/bin/env python3
"""
h10_viewer_v11.py – Polar H10 interactive viewer with Breathing‑Rate proxy
Requires: numpy pandas scipy plotly pytz
"""
import argparse, csv, os
import numpy as np, pandas as pd, scipy.signal as sg, plotly.graph_objs as go
import plotly.offline as po, pytz

# ---- Parameters ----
BP_L, BP_H = 5, 20              # QRS band
REFRACT_S  = 0.25
K_THRESH   = 0.8
ROLL_N     = 3
PVC_THRESH = 0.6
QRS_WIDE   = 130                # ms
DT_BIN_US  = 10
TZ         = "Europe/Berlin"
BR_FS      = 4.0                # Hz for RR->BR resample
BR_LO, BR_HI = 0.15, 0.40       # Resp band (Hz)

# ---- CSV loader ----
def load_csv(path):
    cols=[[],[],[],[]]
    with open(path,newline='') as fh:
        for r in csv.reader(fh):
            if len(r)<2: continue
            for i in range(4):
                try: cols[i].append(float(r[i]))
                except (ValueError,IndexError): cols[i].append(np.nan)
    return [np.asarray(c,float) for c in cols]

# ---- Utility ----
def fs_mode(ns):
    dt=np.diff(ns); bins=(dt//(DT_BIN_US*1000)).astype(int)
    mode_us=np.bincount(bins).argmax()*DT_BIN_US
    return 1/(mode_us*1e-6)

def bandpass(x,fs,lo=BP_L,hi=BP_H):
    b,a=sg.butter(2,[lo/(fs/2),hi/(fs/2)],'band')
    return sg.filtfilt(b,a,x)

def detect_peaks(sig,fs):
    thr=K_THRESH*np.median(np.abs(sig[:int(5*fs)]))
    pks,_=sg.find_peaks(sig,height=thr,distance=int(REFRACT_S*fs))
    return pks
# ----------------------------------------------------------------------
# ECG helpers that were still missing
# ----------------------------------------------------------------------
def qrs_width(ecg_f, r_idx, fs, frac=0.5, max_ms=200):
    """
    Estimate QRS duration (ms) around a detected R‑peak.

    The algorithm walks left and right from the peak until the signal
    falls below `frac` · |peak| or until `max_ms` is reached.

    Parameters
    ----------
    ecg_f : ndarray
        Band‑pass–filtered ECG signal
    r_idx : int
        Index of the R‑peak sample
    fs    : float
        Sampling frequency (Hz)
    frac  : float
        Threshold as fraction of the peak amplitude
    max_ms: int
        Half window (ms) in which to search

    Returns
    -------
    float : width in milliseconds
    """
    win = int(max_ms / 1000 * fs)
    pk_amp = abs(ecg_f[r_idx])
    thr    = frac * pk_amp

    # walk left
    l = r_idx
    while l > 0 and abs(ecg_f[l]) > thr and r_idx - l < win:
        l -= 1
    # walk right
    r = r_idx
    while r < len(ecg_f) - 1 and abs(ecg_f[r]) > thr and r - r_idx < win:
        r += 1

    return (r - l) / fs * 1000.0


def detect_p_wave(ecg_f, t, peak_idx, fs, window_ms=200):
    """Detect P-wave before a QRS complex.
    
    Parameters:
    -----------
    ecg_f : array
        Filtered ECG signal
    t : array
        Time points corresponding to ECG samples
    peak_idx : int
        Index of R peak in the signal
    fs : float
        Sampling frequency in Hz
    window_ms : int
        Window size in ms to look for P wave before QRS complex
    
    Returns:
    --------
    p_present : bool
        Whether a P-wave was detected
    p_amplitude : float
        Amplitude of P-wave
    pr_interval : float
        PR interval in ms
    """
    # Convert window from ms to samples
    window_samples = int((window_ms / 1000) * fs)
    
    # Define search window before R peak
    start_idx = max(0, peak_idx - window_samples)
    
    # Extract the window before the R peak
    pre_qrs = ecg_f[start_idx:peak_idx]
    
    if len(pre_qrs) < fs // 10:  # Ensure we have enough data
        return False, 0, 0
    
    # Apply low-pass filter to focus on P waves (0.5-10Hz)
    b, a = sg.butter(2, [0.5/(fs/2), 10/(fs/2)], 'band')
    p_signal = sg.filtfilt(b, a, pre_qrs)
    
    # Find peaks in the filtered signal (potential P waves)
    p_peaks, p_props = sg.find_peaks(p_signal, height=0.05*np.max(np.abs(p_signal)), 
                                    distance=fs//20, prominence=0.1*np.max(np.abs(p_signal)))
    
    if len(p_peaks) == 0:
        return False, 0, 0
    
    # Get the last peak before QRS (most likely to be P wave)
    p_peak = p_peaks[-1]
    p_amplitude = p_props['peak_heights'][-1]
    
    # Calculate PR interval in ms
    pr_interval = (peak_idx - (start_idx + p_peak)) / fs * 1000
    
    # P wave is typically present if the peak is more than 50ms before QRS
    # and less than 200ms before QRS
    p_present = 50 < pr_interval < 200
    
    return p_present, p_amplitude, pr_interval


def beat_morphology_similarity(beat1, beat2, fs):
    """Calculate similarity between two cardiac beats using correlation.
    
    Parameters:
    -----------
    beat1, beat2 : array
        ECG segments representing individual beats
    fs : float
        Sampling frequency
        
    Returns:
    --------
    similarity : float
        Correlation coefficient between beats
    """
    # Ensure beats are the same length
    min_len = min(len(beat1), len(beat2))
    beat1 = beat1[:min_len]
    beat2 = beat2[:min_len]
    
    # Apply same preprocessing to both beats
    beat1 = sg.detrend(beat1)
    beat2 = sg.detrend(beat2)
    
    # Calculate correlation
    return np.corrcoef(beat1, beat2)[0, 1]

def detect_arrhythmias(ecg_f, t, pk, fs):
    # Initialize output arrays
    pvc_x, pvc_y = [], []
    pac_x, pac_y = [], []
    wq_x, wq_y = [], []
    dip_x, dip_y = [], []

    # Need sufficient data
    if len(pk) < 10:
        return pvc_x, pvc_y, pac_x, pac_y, wq_x, wq_y, dip_x, dip_y
    
    # Get RR intervals
    rr = np.diff(t[pk])
    
    # Calculate RR statistics
    median_rr = np.median(rr)
    
    # First identify "definitely normal" beats (middle 70%)
    sorted_idx = np.argsort(rr)
    mid_start = int(len(sorted_idx) * BR_LO )

    mid_end = int(len(sorted_idx) * 0.85)
    normal_idx = sorted_idx[mid_start:mid_end]
    
    # Calculate normal RR and QRS width
    normal_rr = np.median([rr[i] for i in normal_idx if i < len(rr)])
    normal_width = np.median([qrs_width(ecg_f, pk[i+1], fs) for i in normal_idx if i+1 < len(pk)])
    
    # Set much stricter thresholds
    rr_threshold = 0.65  # 25% early for premature beat
    width_threshold = 1.5 * normal_width  # 50% wider for wide QRS
    
    # Track previous classifications to avoid clusters of detections
    prev_detection = 0  # 0=none, 1=PVC, 2=PAC
    
    # Analyze each beat with context
    for i in range(1, len(pk)-2):
        # Skip if we already found an arrhythmia recently (within 2 beats)
        if prev_detection > 0 and i < prev_detection + 3:
            continue
            
        # Get current beat info
        beat_idx = pk[i]
        next_beat_idx = pk[i+1]
        ts = t[beat_idx]
        amp = ecg_f[beat_idx]
        
        # Calculate current interval and next interval
        curr_rr = rr[i-1] if i-1 < len(rr) else normal_rr
        next_rr = rr[i] if i < len(rr) else normal_rr
        
        # Measure QRS width
        width = qrs_width(ecg_f, beat_idx, fs)
        
        # Define clear conditions
        is_premature = curr_rr < rr_threshold * normal_rr
        is_wide = width > width_threshold
        has_compensatory = next_rr > 1.5 * normal_rr
        
        # Only detect strong, clear signals
        if is_premature and is_wide and has_compensatory:
            # Classic PVC - premature, wide QRS, compensatory pause
            pvc_x.append(ts)
            pvc_y.append(amp)
            prev_detection = i
        elif is_premature and not is_wide and not has_compensatory:
            # Classic PAC - premature, normal QRS, no compensatory pause
            # Let's add an extra check - make sure this is an isolated early beat
            if i > 1:
                prev_rr = rr[i-2] if i-2 < len(rr) else normal_rr
                is_isolated = abs(prev_rr - normal_rr) / normal_rr < 0.2
                
                if is_isolated:
                    pac_x.append(ts)
                    pac_y.append(amp)
                    prev_detection = i
        elif is_wide and width > 2 * normal_width:
            # Very wide QRS complex
            wq_x.append(ts)
            wq_y.append(amp)
            prev_detection = i
    
    # HR dip detection
    hr_inst = 60/rr
    hr_t = t[pk][1:]
    hr_peak = pd.Series(hr_inst).rolling(ROLL_N, center=True, min_periods=1).median().to_numpy()
    baseline = pd.Series(hr_peak).rolling(20, center=True, min_periods=1).median().to_numpy()
    
    # Higher threshold for HR dips
    dip_edges = np.where(np.diff(((baseline-hr_peak) > 35).astype(int)) == 1)[0] + 1
    for i in dip_edges:
        if i+1 < len(t[pk]):
            dip_x.append(t[pk][i+1])
            dip_y.append(ecg_f[pk[i+1]])
    
    return pvc_x, pvc_y, pac_x, pac_y, wq_x, wq_y, dip_x, dip_y

def old_another_detect_arrhythmias(ecg_f, t, pk, fs):
    """Arrhythmia detection with adaptive thresholds"""
    # Calculate RR intervals
    if len(pk) < 5:
        return [], [], [], [], [], [], [], []
        
    rr = np.diff(t[pk])
    
    # Output arrays
    pvc_x, pvc_y = [], []
    pac_x, pac_y = [], []
    wq_x, wq_y = [], []
    dip_x, dip_y = [], []
    
    # Calculate robust statistics
    median_rr = np.median(rr)
    rr_iqr = np.percentile(rr, 75) - np.percentile(rr, 25)
    
    # Adaptive threshold based on RR variability
    prematurity_threshold = max(0.85, 1.0 - 2.0 * (rr_iqr / median_rr))
    
    # Establish normal QRS width from the data
    widths = [qrs_width(ecg_f, idx, fs) for idx in pk[1:-1]]
    median_width = np.median(widths)
    width_iqr = np.percentile(widths, 75) - np.percentile(widths, 25)
    wide_qrs_threshold = median_width + 2 * width_iqr
    
    # Analyze each beat
    for i in range(1, len(pk)-1):
        if i-1 >= len(rr) or i >= len(rr):
            continue
            
        beat_idx = pk[i]
        ts = t[beat_idx]
        amp = ecg_f[beat_idx]
        
        # Check prematurity
        is_premature = rr[i-1] < prematurity_threshold * median_rr
        
        # Check QRS width
        width = qrs_width(ecg_f, beat_idx, fs)
        is_wide = width > wide_qrs_threshold
        
        # Check for compensatory pause
        has_compensatory_pause = i < len(rr) and rr[i] > 1.2 * median_rr
        
        # Basic classification
        if is_premature and is_wide:
            pvc_x.append(ts)
            pvc_y.append(amp)
        elif is_premature and not is_wide and not has_compensatory_pause:
            pac_x.append(ts)
            pac_y.append(amp)
        elif is_wide and not is_premature:
            wq_x.append(ts)
            wq_y.append(amp)
    
    # Detect HR dips
    hr_inst = 60/rr
    hr_t = t[pk][1:]
    hr_peak = pd.Series(hr_inst).rolling(ROLL_N, center=True, min_periods=1).median().to_numpy()
    baseline = pd.Series(hr_peak).rolling(20, center=True, min_periods=1).median().to_numpy()
    dip_edges = np.where(np.diff(((baseline-hr_peak) > 30).astype(int)) == 1)[0] + 1
    for i in dip_edges:
        if i+1 < len(t[pk]):
            dip_x.append(t[pk][i+1])
            dip_y.append(ecg_f[pk[i+1]])
    
    return pvc_x, pvc_y, pac_x, pac_y, wq_x, wq_y, dip_x, dip_y

def old_old_detect_arrhythmias(ecg_f, t, pk, fs):
    """Comprehensive arrhythmia detection with balanced sensitivity and specificity"""
    # Calculate RR intervals
    if len(pk) < 5:
        return [], [], [], [], [], [], [], []
        
    rr = np.diff(t[pk])
    
    # Output arrays
    pvc_x, pvc_y = [], []
    pac_x, pac_y = [], []
    wq_x, wq_y = [], []
    dip_x, dip_y = [], []
    
    # Calculate robust statistics for RR intervals
    median_rr = np.median(rr)
    rr_sorted = np.sort(rr)
    q1 = np.percentile(rr, 25)
    q3 = np.percentile(rr, 75)
    iqr = q3 - q1
    
    # Create template of normal beats
    normal_beat_indices = []
    for i in range(1, len(pk)-2):
        # Only use beats with normal RR intervals (within ±20% of median)
        if (0.85 * median_rr <= rr[i-1] <= 1.15 * median_rr and
            0.85 * median_rr <= rr[i] <= 1.15 * median_rr):
            normal_beat_indices.append(i)
    
    # Extract templates
    normal_templates = []
    window_samples = int(0.2 * fs)  # 120ms window
    
    for idx in normal_beat_indices[:min(10, len(normal_beat_indices))]:
        beat_idx = pk[idx]
        start_idx = max(0, beat_idx - window_samples)
        end_idx = min(len(ecg_f), beat_idx + window_samples)
        
        if end_idx - start_idx == 2 * window_samples:
            beat = ecg_f[start_idx:end_idx]
            # Normalize amplitude
            beat_norm = (beat - np.mean(beat)) / (np.std(beat) if np.std(beat) > 0 else 1)
            normal_templates.append(beat_norm)
    
    # Create mean template
    normal_template = np.mean(normal_templates, axis=0) if len(normal_templates) >= 3 else None
    
    # Calculate normal QRS width
    normal_widths = []
    for idx in normal_beat_indices[:min(10, len(normal_beat_indices))]:
        width = qrs_width(ecg_f, pk[idx], fs)
        normal_widths.append(width)
    
    # Use actual measured QRS widths rather than arbitrary threshold
    if len(normal_widths) >= 3:
        mean_normal_width = np.mean(normal_widths)
        std_normal_width = np.std(normal_widths)
        qrs_wide_threshold = mean_normal_width + 2 * std_normal_width
    else:
        qrs_wide_threshold = QRS_WIDE  # Fallback to default
    
    # Analyze each beat
    for i in range(1, len(pk)-1):
        if i-1 >= len(rr) or i >= len(rr):
            continue
            
        beat_idx = pk[i]
        ts = t[beat_idx]
        amp = ecg_f[beat_idx]
        
        # --- Measure key features ---
        
        # 1. Prematurity
        prev_rr = rr[i-1]
        next_rr = rr[i]
        
        # Use relative and absolute prematurity criteria
        prematurity_ratio = prev_rr / median_rr
        is_premature = prematurity_ratio < 0.9  # 10% early
        is_very_premature = prematurity_ratio < 0.8  # 20% early
        
        # 2. QRS width
        width_ms = qrs_width(ecg_f, beat_idx, fs)
        
        # Compare to normal QRS width
        is_wide_qrs = width_ms > qrs_wide_threshold
        
        # 3. Morphology comparison
        morphology_score = 0  # Default
        if normal_template is not None:
            # Extract current beat
            start_idx = max(0, beat_idx - window_samples)
            end_idx = min(len(ecg_f), beat_idx + window_samples)
            
            if end_idx - start_idx == 2 * window_samples:
                current_beat = ecg_f[start_idx:end_idx]
                # Normalize
                current_beat_norm = (current_beat - np.mean(current_beat)) / (np.std(current_beat) if np.std(current_beat) > 0 else 1)
                
                # Cross-correlation for better alignment before comparison
                xcorr = np.correlate(normal_template, current_beat_norm, mode='full')
                max_xcorr = np.max(np.abs(xcorr))
                morphology_score = max_xcorr / len(normal_template)
        
        # 4. Compensatory pause
        is_compensatory = (next_rr > 1.1 * median_rr)
        is_fully_compensatory = (prev_rr + next_rr) > (1.9 * median_rr)  # Classic full compensation
        
        # 5. P-wave detection (if we have enough signal)
        has_p_wave = False
        if beat_idx > fs // 4:  # At least 250ms of data before R peak
            # Look back up to 250ms before R peak
            pre_qrs_window = int(0.25 * fs)
            pre_qrs_start = max(0, beat_idx - pre_qrs_window)
            pre_qrs = ecg_f[pre_qrs_start:beat_idx]
            
            if len(pre_qrs) >= 5:  # Need enough data
                # Use bandpass filter for P wave detection (5-15 Hz)
                try:
                    b, a = sg.butter(2, [5/(fs/2), 15/(fs/2)], 'band')
                    p_signal = sg.filtfilt(b, a, pre_qrs)
                    
                    # Find potential P waves
                    min_distance = max(1, int(0.04 * fs))  # At least 40ms between peaks
                    p_peaks, _ = sg.find_peaks(p_signal, distance=min_distance)
                    
                    # P-wave typically occurs 120-200ms before QRS
                    # Convert peak indices to time from QRS
                    p_times = [(beat_idx - (pre_qrs_start + p_idx)) / fs * 1000 for p_idx in p_peaks]
                    
                    # Check if any peaks are in the right time range
                    p_in_range = [50 <= p_time <= 250 for p_time in p_times]
                    has_p_wave = any(p_in_range)
                except:
                    # Fallback if filtering fails
                    has_p_wave = False
        
        # --- Classification rules ---
        
        # PVC scoring - more nuanced approach
        pvc_score = 0
        
        # Core PVC criteria with weighted importance
        if is_premature:
            pvc_score += 0.3
            if is_very_premature:
                pvc_score += 0.1
        
        if is_wide_qrs:
            pvc_score += 0.3
        
        if morphology_score < 0.7:  # Different morphology
            pvc_score += 0.1
            if morphology_score < 0.5:
                pvc_score += 0.1
        
        if is_compensatory:
            pvc_score += 0.1
            if is_fully_compensatory:
                pvc_score += 0.1
        
        if not has_p_wave:
            pvc_score += 0.1
        
        # PAC scoring
        pac_score = 0
        
        # Core PAC criteria
        if is_premature:
            pac_score += 0.3
            if is_very_premature:
                pac_score += 0.1
        
        if not is_wide_qrs:  # Normal QRS
            pac_score += 0.3
        
        if morphology_score > 0.7:  # Similar to normal QRS
            pac_score += 0.2
        
        if has_p_wave:  # Has P-wave
            pac_score += 0.2
        
        if not is_compensatory:  # No compensatory pause
            pac_score += 0.1
        
        # Final classification with context-aware thresholds
        
        # For high confidence classifications, use higher thresholds
        pvc_threshold = 0.7
        pac_threshold = 0.7
        
        # Add contextual requirement - isolated abnormality
        is_isolated = True
        if i > 1 and i < len(rr) - 1:
            prev_prev_rr = rr[i-2]
            next_next_rr = rr[i+1] if i+1 < len(rr) else median_rr
            
            # Check if surrounding beats are normal
            if prev_prev_rr < 0.9 * median_rr or next_next_rr < 0.9 * median_rr:
                is_isolated = False
                # Increase threshold for non-isolated events to avoid clusters of false positives
                pvc_threshold = 0.8
                pac_threshold = 0.8
        
        # Apply classifications
        if pvc_score >= pvc_threshold and pvc_score > pac_score:
            pvc_x.append(ts)
            pvc_y.append(amp)
        elif pac_score >= pac_threshold and pac_score > pvc_score:
            # Extra check for PACs - must be premature
            if is_premature:
                pac_x.append(ts)
                pac_y.append(amp)
        elif is_wide_qrs and width_ms > 1.5 * qrs_wide_threshold:
            # Detect very wide QRS that aren't classified as PVCs
            wq_x.append(ts)
            wq_y.append(amp)
    
    # Detect HR dips (unchanged)
    hr_inst = 60/rr
    hr_t = t[pk][1:]
    hr_peak = pd.Series(hr_inst).rolling(ROLL_N, center=True, min_periods=1).median().to_numpy()
    baseline = pd.Series(hr_peak).rolling(20, center=True, min_periods=1).median().to_numpy()
    dip_edges = np.where(np.diff(((baseline-hr_peak) > 30).astype(int)) == 1)[0] + 1
    for i in dip_edges:
        if i+1 < len(t[pk]):
            dip_x.append(t[pk][i+1])
            dip_y.append(ecg_f[pk[i+1]])
    
    return pvc_x, pvc_y, pac_x, pac_y, wq_x, wq_y, dip_x, dip_y

def older_still_detect_arrhythmias(ecg_f, t, pk, fs):
    """Fixed arrhythmia detection with better calibration to reduce false positives.
    """
    # Calculate RR intervals
    rr = np.diff(t[pk])
    
    # Ensure we have enough data to proceed
    if len(rr) < 4:
        return [], [], [], [], [], [], [], []
    
    # Calculate robust statistics
    rr_50 = np.median(rr)  # median
    
    # Establish a baseline by taking the middle 60% of RR intervals
    sorted_rr = np.sort(rr)
    baseline_start = int(len(sorted_rr) * 0.2)
    baseline_end = int(len(sorted_rr) * 0.8)
    baseline_rr = sorted_rr[baseline_start:baseline_end]
    
    if len(baseline_rr) < 3:
        baseline_rr = sorted_rr  # Use all if not enough data
    
    baseline_median = np.median(baseline_rr)
    baseline_std = np.std(baseline_rr)
    
    # Calculate normalized variability
    cv = baseline_std / baseline_median if baseline_median > 0 else 0
    
    # Define strict prematurity threshold - at least 3 standard deviations from baseline
    prematurity_threshold = max(0.8, 1 - 3 * cv)
    
    # Create output arrays
    pvc_x, pvc_y = [], []
    pac_x, pac_y = [], []
    wq_x, wq_y = [], []
    dip_x, dip_y = [], []
    
    # Collect normal beats to create template (more discriminating criteria)
    normal_templates = []
    window_samples = int(0.15 * fs)
    
    # First pass - identify definite normal beats
    for i in range(1, len(pk)-1):
        if i >= len(rr) or i-1 >= len(rr):
            continue
            
        curr_rr = rr[i-1]
        next_rr = rr[i] if i < len(rr) else rr[i-1]
        
        # Only use beats with very normal timing
        if abs(curr_rr - baseline_median) < baseline_std and abs(next_rr - baseline_median) < baseline_std:
            beat_idx = pk[i]
            start_idx = max(0, beat_idx - window_samples)
            end_idx = min(len(ecg_f), beat_idx + window_samples)
            
            if end_idx - start_idx == 2 * window_samples:
                beat = ecg_f[start_idx:end_idx]
                # Normalize
                beat = (beat - np.mean(beat)) / (np.std(beat) if np.std(beat) > 0 else 1)
                normal_templates.append(beat)
    
    # Create template only if we have enough good beats
    if len(normal_templates) >= 3:
        normal_template = np.mean(normal_templates, axis=0)
    else:
        # Not enough normal beats to create reliable template
        normal_template = None
    
    # Second pass - analyze each beat with stricter criteria
    for i in range(1, len(pk)-1):
        if i >= len(rr) or i-1 >= len(rr):
            continue
            
        beat_idx = pk[i]
        ts = t[beat_idx]
        amp = ecg_f[beat_idx]
        
        # Measure beat width
        width_ms = qrs_width(ecg_f, beat_idx, fs)
        
        # Get surrounding RR intervals
        prev_rr = rr[i-1] if i-1 < len(rr) else rr[0]
        next_rr = rr[i] if i < len(rr) else rr[-1]
        
        # 1. STRICT prematurity detection (more than 20% early AND statistical outlier)
        prematurity_ratio = prev_rr / baseline_median
        is_premature = (prematurity_ratio < prematurity_threshold and 
                       (prev_rr < baseline_median - 2 * baseline_std))
        
        # 2. Check for compensatory pause
        has_compensatory_pause = next_rr > (baseline_median + baseline_std)
        
        # 3. Extract and normalize current beat for morphology comparison
        start_idx = max(0, beat_idx - window_samples)
        end_idx = min(len(ecg_f), beat_idx + window_samples)
        
        if end_idx - start_idx != 2 * window_samples:
            continue
            
        current_beat = ecg_f[start_idx:end_idx]
        current_beat_norm = (current_beat - np.mean(current_beat)) / (np.std(current_beat) if np.std(current_beat) > 0 else 1)
        
        # Check morphology difference
        morphology_diff = 1.0  # Default to maximum difference
        if normal_template is not None:
            # Cross-correlation for better comparison
            corr = np.corrcoef(normal_template, current_beat_norm)[0, 1]
            morphology_diff = 1 - max(0, corr)
        
        # 4. Check for P-wave (if we have enough signal before QRS)
        has_p_wave = False
        if start_idx > window_samples // 2:
            pre_qrs_start = max(0, start_idx - window_samples // 2)
            pre_qrs = ecg_f[pre_qrs_start:start_idx]
            
            # Simple P wave detection
            if len(pre_qrs) > fs // 20:
                # Adaptive window size for Savitzky-Golay filter
                window_length = min(11, len(pre_qrs) - 2)
                # Ensure window_length is odd
                window_length = window_length - 1 if window_length % 2 == 0 and window_length > 1 else window_length
                
                if window_length > 2:  # Need at least 3 points for savgol_filter with polyorder=2
                    # Smooth the signal with adaptive window
                    pre_qrs_smooth = sg.savgol_filter(pre_qrs, window_length, min(2, window_length-1))
                    p_peaks, _ = sg.find_peaks(pre_qrs_smooth, distance=fs//40)
                    has_p_wave = len(p_peaks) > 0
                else:
                    # If segment too small, use simpler peak detection
                    p_peaks, _ = sg.find_peaks(pre_qrs, distance=fs//40)
                    has_p_wave = len(p_peaks) > 0
        
        # Calculate confidence scores with STRICTER criteria
        
        # PVC criteria - MUST be premature AND have wide QRS or abnormal morphology
        pvc_score = 0
        if is_premature:  # Must be premature
            pvc_score += 0.4
            if width_ms > QRS_WIDE:  # Wide QRS
                pvc_score += 0.3
            if morphology_diff > 0.5:  # Very different morphology
                pvc_score += 0.2
            if has_compensatory_pause:  # Has compensatory pause
                pvc_score += 0.1
            if not has_p_wave:  # No P wave
                pvc_score += 0.1
        
        # PAC criteria - MUST be premature, have normal QRS, and (usually) a P wave
        pac_score = 0
        if is_premature:  # Must be premature
            pac_score += 0.4
            if width_ms <= QRS_WIDE:  # Normal QRS width
                pac_score += 0.3
            if morphology_diff < 0.3:  # Similar to normal QRS
                pac_score += 0.2
            if has_p_wave:  # Has P wave
                pac_score += 0.1
            if not has_compensatory_pause:  # No compensatory pause
                pac_score += 0.1
        
        # Apply very strict thresholds to avoid false positives
        if pvc_score >= 0.8 and pvc_score > pac_score:
            pvc_x.append(ts)
            pvc_y.append(amp)
        elif pac_score >= 0.8 and pac_score > pvc_score:
            pac_x.append(ts)
            pac_y.append(amp)
        elif width_ms > QRS_WIDE * 1.3:  # Much wider than normal
            wq_x.append(ts)
            wq_y.append(amp)
    
    # Detect HR dips (unchanged)
    hr_inst = 60/rr
    hr_t = t[pk][1:]
    hr_peak = pd.Series(hr_inst).rolling(ROLL_N, center=True, min_periods=1).median().to_numpy()
    baseline = pd.Series(hr_peak).rolling(20, center=True, min_periods=1).median().to_numpy()
    dip_edges = np.where(np.diff(((baseline-hr_peak) > 30).astype(int)) == 1)[0] + 1
    for i in dip_edges:
        if i+1 < len(t[pk]):
            dip_x.append(t[pk][i+1])
            dip_y.append(ecg_f[pk[i+1]])
    
    return pvc_x, pvc_y, pac_x, pac_y, wq_x, wq_y, dip_x, dip_y
def older_detect_arrhythmias(ecg_f, t, pk, fs):
    """Improved arrhythmia detection with better sensitivity/specificity balance.
    
    Parameters:
    -----------
    ecg_f : array
        Filtered ECG signal
    t : array
        Time points corresponding to ECG samples
    pk : array
        Indices of R peaks
    fs : float
        Sampling frequency in Hz
        
    Returns:
    --------
    pvc_x, pvc_y, pac_x, pac_y, wq_x, wq_y, dip_x, dip_y : arrays
        X and Y coordinates of various cardiac events
    """
    # Calculate RR intervals
    rr = np.diff(t[pk])
    
    # Calculate more robust RR statistics using percentiles instead of median
    # to handle outliers better
    rr_25 = np.percentile(rr, 25)
    rr_50 = np.percentile(rr, 50)  # median
    rr_75 = np.percentile(rr, 75)
    
    # Define adaptive prematurity threshold based on RR variability
    # Use tighter threshold if low RR variability, looser if high variability
    rr_var = (rr_75 - rr_25) / rr_50  # normalized IQR
    
    # Adaptive thresholds based on RR variability
    if rr_var < 0.05:  # Very regular rhythm
        prematurity_threshold = 0.85  # 15% early
    elif rr_var < 0.10:  # Moderately regular rhythm
        prematurity_threshold = 0.80  # 20% early
    else:  # Irregular rhythm
        prematurity_threshold = 0.75  # 25% early
    
    # Use array operations for efficiency
    rr_array = np.array(rr)
    pk_array = np.array(pk)
    
    # Create output arrays
    pvc_x, pvc_y = [], []
    pac_x, pac_y = [], []
    wq_x, wq_y = [], []
    dip_x, dip_y = [], []
    
    # Analyze normal beats to create template
    normal_beat_indices = []
    for i in range(1, len(pk)-2):
        # Consider beats with normal RR intervals before and after
        if (rr_array[i-1] >= prematurity_threshold * rr_50 and 
            rr_array[i] >= prematurity_threshold * rr_50):
            normal_beat_indices.append(i)
    
    # Create templates from normal beats
    if len(normal_beat_indices) >= 3:
        # Window size for beat (±150ms around R peak)
        window_samples = int(0.15 * fs)
        normal_templates = []
        
        for idx in normal_beat_indices[:min(10, len(normal_beat_indices))]:
            beat_idx = pk_array[idx]
            start_idx = max(0, beat_idx - window_samples)
            end_idx = min(len(ecg_f), beat_idx + window_samples)
            if end_idx - start_idx == 2 * window_samples:
                # Normalize beat for better comparison
                beat = ecg_f[start_idx:end_idx]
                beat = (beat - np.mean(beat)) / (np.std(beat) if np.std(beat) > 0 else 1)
                normal_templates.append(beat)
        
        # Create average template
        if normal_templates:
            normal_template = np.mean(normal_templates, axis=0)
        else:
            normal_template = None
    else:
        normal_template = None
    
    # Calculate rolling RR intervals for better context
    if len(rr) >= 5:
        rolling_rr = np.convolve(rr, np.ones(5)/5, mode='valid')
        rolling_rr = np.pad(rolling_rr, (2, 2), mode='edge')  # Pad to match original size
    else:
        rolling_rr = rr
    
    # Analyze each beat
    for i in range(1, len(pk)-1):  # Skip first and last beat for better context
        beat_idx = pk_array[i]
        ts = t[beat_idx]
        amp = ecg_f[beat_idx]
        
        # Extract the current beat
        window_samples = int(0.15 * fs)
        start_idx = max(0, beat_idx - window_samples)
        end_idx = min(len(ecg_f), beat_idx + window_samples)
        
        if end_idx - start_idx != 2 * window_samples:
            continue  # Skip beats near the edges
            
        current_beat = ecg_f[start_idx:end_idx]
        current_beat_norm = (current_beat - np.mean(current_beat)) / (np.std(current_beat) if np.std(current_beat) > 0 else 1)
        
        # Measure QRS width
        width_ms = qrs_width(ecg_f, beat_idx, fs)
        
        # Calculate 3-point local RR context
        if i > 0 and i < len(rr):
            # Previous RR interval
            prev_rr = rr_array[i-1]
            # Current RR interval
            curr_rr = rr_array[i]
        else:
            continue  # Skip beats without full context
        
        # Define strict criteria for PAC and PVC
        
        # Feature 1: Prematurity - how early is this beat compared to local context?
        prematurity_ratio = prev_rr / rolling_rr[i-1]
        is_premature = prematurity_ratio < prematurity_threshold
        
        # Feature 2: Morphology difference - how different is this beat from normal template?
        morphology_diff = 0
        if normal_template is not None:
            corr = np.corrcoef(normal_template, current_beat_norm)[0, 1]
            morphology_diff = 1 - max(0, corr)  # 0 (identical) to 1 (completely different)
        
        # Feature 3: Post-beat pause - compensatory or not?
        pause_ratio = curr_rr / rolling_rr[i]
        has_compensatory_pause = pause_ratio > 1.2
        
        # Feature 4: P-wave detection in 250ms before QRS
        p_window_samples = int(0.25 * fs)
        pre_qrs_start = max(0, beat_idx - p_window_samples)
        pre_qrs = ecg_f[pre_qrs_start:beat_idx]
        
        # Apply bandpass filter focused on P-wave frequencies (5-15 Hz)
        if len(pre_qrs) > fs // 10:
            b, a = sg.butter(2, [5/(fs/2), 15/(fs/2)], 'band')
            p_signal = sg.filtfilt(b, a, pre_qrs)
            
            # Find peaks that might be P waves
            p_peaks, _ = sg.find_peaks(p_signal, distance=fs//50)
            has_p_wave = len(p_peaks) > 0
            
            if has_p_wave:
                # Check if P wave is in normal position (50-200ms before QRS)
                last_p_idx = p_peaks[-1]
                pr_interval = (beat_idx - (pre_qrs_start + last_p_idx)) / fs * 1000
                normal_p_timing = 50 <= pr_interval <= 200
            else:
                normal_p_timing = False
        else:
            has_p_wave = False
            normal_p_timing = False
        
        # Calculate confidence scores for PAC and PVC
        
        # PAC confidence score (0-1):
        # - Premature beat (+0.3)
        # - Normal QRS width (+0.3)
        # - P wave present but potentially abnormal (+0.3)
        # - No significant morphology change in QRS (+0.1)
        # - No prolonged compensatory pause (+0.2)
        # For a sure PAC, need confidence > 0.7
        
        pac_confidence = 0
        if is_premature:
            pac_confidence += 0.3
        if width_ms <= QRS_WIDE:
            pac_confidence += 0.3
        if has_p_wave:
            pac_confidence += 0.3
        if morphology_diff < 0.4:
            pac_confidence += 0.1
        if not has_compensatory_pause:
            pac_confidence += 0.2
            
        # PVC confidence score (0-1):
        # - Premature beat (+0.3)
        # - Wide QRS (+0.3)
        # - Different morphology from normal beats (+0.2)
        # - Often has compensatory pause (+0.1)
        # - No clear P wave before QRS (+0.1)
        # For a sure PVC, need confidence > 0.7
        
        pvc_confidence = 0
        if is_premature:
            pvc_confidence += 0.3
        if width_ms > QRS_WIDE:
            pvc_confidence += 0.3
        if morphology_diff > 0.4:
            pvc_confidence += 0.2
        if has_compensatory_pause:
            pvc_confidence += 0.1
        if not has_p_wave or not normal_p_timing:
            pvc_confidence += 0.1
            
        # Final classification with confidence thresholds
        # Higher thresholds reduce false positives
        if pvc_confidence > 0.75 and pvc_confidence > pac_confidence:
            pvc_x.append(ts)
            pvc_y.append(amp)
        elif pac_confidence > 0.75 and pac_confidence > pvc_confidence:
            pac_x.append(ts)
            pac_y.append(amp)
        elif width_ms > QRS_WIDE and pvc_confidence < 0.7:
            # Wide QRS that doesn't fully meet PVC criteria
            wq_x.append(ts)
            wq_y.append(amp)
    
    # Detect HR dips (unchanged from original)
    hr_inst = 60/rr
    hr_t = t[pk][1:]
    hr_peak = pd.Series(hr_inst).rolling(ROLL_N, center=True, min_periods=1).median().to_numpy()
    baseline = pd.Series(hr_peak).rolling(20, center=True, min_periods=1).median().to_numpy()
    dip_edges = np.where(np.diff(((baseline-hr_peak) > 30).astype(int)) == 1)[0] + 1
    for i in dip_edges:
        if i+1 < len(t[pk]):
            dip_x.append(t[pk][i+1])
            dip_y.append(ecg_f[pk[i+1]])
    
    return pvc_x, pvc_y, pac_x, pac_y, wq_x, wq_y, dip_x, dip_y


# Main anomaly detection logic
def old_detect_arrhythmias(ecg_f, t, pk, fs):
    """Detect cardiac arrhythmias in ECG signal.
    
    Parameters:
    -----------
    ecg_f : array
        Filtered ECG signal
    t : array
        Time points corresponding to ECG samples
    pk : array
        Indices of R peaks
    fs : float
        Sampling frequency in Hz
        
    Returns:
    --------
    pvc_x, pvc_y : arrays
        X and Y coordinates of PVCs
    pac_x, pac_y : arrays
        X and Y coordinates of PACs
    wq_x, wq_y : arrays
        X and Y coordinates of other wide QRS complexes
    dip_x, dip_y : arrays
        X and Y coordinates of HR dips
    """
    # Calculate RR intervals
    rr = np.diff(t[pk])
    median_rr = np.median(rr)
    
    # Create arrays to store anomalies
    pvc_x, pvc_y = [], []
    pac_x, pac_y = [], []
    wq_x, wq_y = [], []
    dip_x, dip_y = [], []

    # Create templates for normal beats
    beat_width_samples = int(0.12 * fs)  # 120ms window for beat template
    normal_beats = []
    
    # Collect normal beats (not premature, not first or last)
    for i in range(1, len(pk)-1):
        if i > 0 and i < len(rr) and rr[i-1] >= 0.9 * median_rr:
            # Extract beat template centered on R peak
            beat_start = max(0, pk[i] - beat_width_samples)
            beat_end = min(len(ecg_f), pk[i] + beat_width_samples)
            beat = ecg_f[beat_start:beat_end]
            if len(beat) == 2 * beat_width_samples:
                normal_beats.append(beat)
    
    # Create normal template by averaging if we have enough beats
    normal_template = np.mean(normal_beats, axis=0) if len(normal_beats) > 3 else None
    
    # Analyze each beat
    for i in range(len(pk)-1):
        # Skip first beat as we need the previous beat for context
        if i == 0:
            continue
            
        # Calculate beat prematurity
        premature = False
        if i < len(rr):
            premature = rr[i-1] < PVC_THRESH * median_rr
            
        # Extract beat features
        ts = t[pk][i]
        amp = ecg_f[pk[i]]
        
        # Calculate QRS width
        width_ms = qrs_width(ecg_f, pk[i], fs)
        
        # Check post-beat pause (compensatory or not)
        has_compensatory_pause = False
        if i < len(rr):
            has_compensatory_pause = rr[i] > (1.2 * median_rr)
            
        # Extract beat morphology for comparison
        beat_start = max(0, pk[i] - beat_width_samples)
        beat_end = min(len(ecg_f), pk[i] + beat_width_samples)
        current_beat = ecg_f[beat_start:beat_end]
        
        # Calculate morphology similarity with normal template
        morphology_similarity = 0
        if normal_template is not None and len(current_beat) == len(normal_template):
            morphology_similarity = beat_morphology_similarity(current_beat, normal_template, fs)
        
        # Check for P wave
        has_p_wave, p_amplitude, pr_interval = detect_p_wave(ecg_f, t, pk[i], fs)
        
        # PVC criteria: premature, wide QRS, different morphology, often has compensatory pause, often no P wave
        is_pvc = (premature and 
                  width_ms > QRS_WIDE and 
                  (morphology_similarity < 0.7 or not has_p_wave) and
                  (has_compensatory_pause or morphology_similarity < 0.5))
        
        # PAC criteria: premature, normal QRS width, has P wave but potentially abnormal PR interval
        is_pac = (premature and 
                  width_ms <= QRS_WIDE and 
                  has_p_wave and 
                  morphology_similarity > 0.6 and
                  not is_pvc)
        
        # Assign to appropriate category
        if is_pvc:
            pvc_x.append(ts)
            pvc_y.append(amp)
        elif is_pac:
            pac_x.append(ts)
            pac_y.append(amp)
        elif width_ms > QRS_WIDE:
            # Wide QRS that's not a PVC
            wq_x.append(ts)
            wq_y.append(amp)
    
    # Detect HR dips (unchanged from original)
    hr_inst = 60/rr
    hr_t = t[pk][1:]
    hr_peak = pd.Series(hr_inst).rolling(ROLL_N, center=True, min_periods=1).median().to_numpy()
    baseline = pd.Series(hr_peak).rolling(20, center=True, min_periods=1).median().to_numpy()
    dip_edges = np.where(np.diff(((baseline-hr_peak) > 30).astype(int)) == 1)[0] + 1
    for i in dip_edges:
        dip_x.append(t[pk][i+1])
        dip_y.append(ecg_f[pk[i+1]])
    
    return pvc_x, pvc_y, pac_x, pac_y, wq_x, wq_y, dip_x, dip_y

def qrs_width(sig,idx,fs):
    thr=0.3*sig[idx]; l=idx; r=idx
    while l>0 and sig[l]>thr:l-=1
    while r<len(sig)-1 and sig[r]>thr:r+=1
    return (r-l)/fs*1000

# ---- RR -> Breathing‑rate proxy ----
def rr_to_br(rr_ms, rr_times):
    # --- 1. valid finite & unique times ----------- [no changes]
    mask = np.isfinite(rr_ms) & np.isfinite(rr_times)
    rr_ms, rr_times = rr_ms[mask], rr_times[mask]
    rr_times, idx = np.unique(rr_times, return_index=True)
    rr_ms = rr_ms[idx]
    if rr_ms.size < 5: return None, None

    # resample RR tachogram to 4 Hz ---------------- [no changes]
    t_uniform = np.arange(rr_times[0], rr_times[-1], 1/BR_FS)
    rr_interp = np.interp(t_uniform, rr_times, rr_ms)

    # --- 2. band‑pass RSA band -------------------- [no changes]
    b, a = sg.butter(2, [BR_LO/(BR_FS/2), BR_HI/(BR_FS/2)], 'band')
    resp = sg.filtfilt(b, a, sg.detrend(rr_interp))

    # --- 3. Analyze quality using sliding window approach ---
    window_size = int(10 * BR_FS)  # 10-second windows
    overlap = int(window_size * 0.8)  # 80% overlap
    quality_mask = np.ones_like(t_uniform, dtype=bool)
    
    # Create windows with sufficient data
    if len(rr_interp) > window_size:
        for i in range(0, len(rr_interp) - window_size + 1, window_size - overlap):
            window_end = min(i + window_size, len(rr_interp))
            window = rr_interp[i:window_end]
            window_resp = resp[i:window_end]
            
            # Calculate metrics in this window
            window_var = np.var(window)
            rsa_var = np.var(window_resp)
            
            # Original quality check but applied to window
            if window_var > 0 and rsa_var < 0.02 * window_var:
                quality_mask[i:window_end] = False
    else:
        # For short signals, use the original approach
        total_var = np.var(rr_interp)
        rsa_power = np.var(resp)
        if rsa_power < 0.02 * total_var:
            return t_uniform, np.full_like(t_uniform, np.nan)

    # --- 4. Hilbert phase smoothed --------------- [no changes]
    analytic = sg.hilbert(resp)
    phase = np.unwrap(np.angle(analytic))
    phase_smooth = sg.savgol_filter(phase, window_length=31, polyorder=3)

    inst_freq = np.gradient(phase_smooth) / (2*np.pi) * BR_FS  # Hz
    inst_freq = np.clip(inst_freq, 0, 0.7)        # physiologic band

    br_bpm = pd.Series(inst_freq*60).rolling(5, center=True,
                                            min_periods=1).median()

    # --- 5. Apply quality mask -------------------
    br_filled = br_bpm.to_numpy()
    br_filled[~quality_mask] = np.nan  # Mark low-quality segments as NaN
    
    # --- 6. forward‑fill short NaN gaps ---------- [no changes]
    isnan = np.isnan(br_filled)
    if isnan.any():
        good = np.where(~isnan)[0]
        if good.size:
            br_filled = np.interp(np.arange(len(br_filled)),
                                good, br_filled[good])
    return t_uniform, br_filled

# ---- Main viewer ----
def build(path, down, out_html, auto_open):
    ns,ecg,hr_col,rr_col = load_csv(path)
    mask=np.isfinite(ns)&np.isfinite(ecg); ns,ecg=ns[mask],ecg[mask]
    t=(ns-ns[0])/1e9; fs=fs_mode(ns)
    ecg_f=bandpass(ecg.astype(np.float32),fs); pk=detect_peaks(ecg_f,fs)
    rr=np.diff(t[pk]); hr_inst=60/rr; hr_t=t[pk][1:]
    hr_peak=pd.Series(hr_inst).rolling(ROLL_N,center=True,min_periods=1).median().to_numpy()

    # external cols
    hr_strap = pd.Series(hr_col).interpolate('linear').to_numpy() if np.isfinite(hr_col).sum() else None
    #rr_ms_col= pd.Series(rr_col).interpolate('linear').to_numpy() if np.isfinite(rr_col).sum() else None
    rr_ms_col = pd.Series(rr_col[mask]).interpolate('linear').to_numpy() if np.isfinite(rr_col[mask]).sum() else None
    rr_ms = rr_ms_col if rr_ms_col is not None else rr*1000
    rr_times = t[pk][1:] if rr_ms_col is None else t

    # breathing‑rate proxy
    br_t, br_bpm = rr_to_br(rr_ms, rr_times)

    # anomaly classification
    pvc_x=[]
    pvc_y=[]
    pac_x=[]
    pac_y=[]
    wq_x=[]
    wq_y=[]
    dip_x=[]
    dip_y=[]
    
    median_rr = np.median(rr)
    
    pvc_x, pvc_y, pac_x, pac_y, wq_x, wq_y, dip_x, dip_y = detect_arrhythmias(ecg_f, t, pk, fs)

    #for i in range(len(rr)):        # rr[i] is interval between pk[i] and pk[i+1]
    #    # Check for premature beats
    #    if rr[i] < PVC_THRESH * median_rr:  # Early/premature beat detected
    #        # Get the timestamp and amplitude of the premature beat
    #        ts = t[pk][i + 1]                     # time‑stamp of premature beat
    #        amp = ecg_f[np.searchsorted(t, ts)]   # y‑position on filtered ECG
    #        
    #        # Measure QRS width of the premature beat
    #        width_ms = qrs_width(ecg_f, pk[i + 1], fs)
    #        
    #        # Check if the following RR interval is compensatory
    #        next_rr_ok = (
    #            i + 1 < len(rr) and                    # have RR after the premature
    #            rr[i + 1] < (1+PVC_THRESH) * median_rr  # non‑compensatory pause
    #        )
    #        
    #        # Classify the premature beat
    #        if width_ms > QRS_WIDE:
    #            # Wide QRS and premature → PVC
    #            pvc_x.append(ts)
    #            pvc_y.append(amp)
    #        elif next_rr_ok:
    #            # Normal width, premature, and no compensatory pause → PAC
    #            pac_x.append(ts)
    #            pac_y.append(amp)
    #
    ## Detect other wide QRS complexes (not premature)
    #for idx in pk:
    #    if qrs_width(ecg_f, idx, fs) > QRS_WIDE and t[idx] not in pvc_x:
    #        wq_x.append(t[idx])
    #        wq_y.append(ecg_f[np.searchsorted(t, t[idx])])
    
    # Detect HR dips
    baseline = pd.Series(hr_peak).rolling(20, center=True, min_periods=1).median().to_numpy()
    dip_edges = np.where(np.diff(((baseline-hr_peak) > 30).astype(int)) == 1)[0] + 1
    dip_ts = t[pk][dip_edges]
    for ts in dip_ts:
        dip_x.append(ts)
        dip_y.append(ecg_f[np.searchsorted(t, ts)])

    # datetime axis
    start=pd.to_datetime(ns[0],unit='ns',utc=True).tz_convert(TZ)
    dt=start+pd.to_timedelta(t,unit='s')
    ds=max(1,int(down))
    traces=[
        go.Scatter(x=dt[::ds], y=ecg[::ds], name='Raw ECG',
                   line=dict(color='rgba(90,90,90,0.4)'), yaxis='y1'),
        go.Scatter(x=dt[::ds], y=ecg_f[::ds], name='Filtered 5‑20 Hz',
                   line=dict(color='blue',width=1), yaxis='y1')
    ]
    # HR curves
    idx_peak=len(traces)
    traces.append(go.Scatter(x=start+pd.to_timedelta(hr_t,unit='s'),
                             y=hr_peak, name='Peak‑HR', yaxis='y2', visible=False))
    idx_strap=idx_br=None
    if hr_strap is not None:
        idx_strap=len(traces)
        traces.append(go.Scatter(x=dt[::ds], y=hr_strap[::ds],
                                 name='Strap HR', yaxis='y2', visible=True))
    if br_bpm is not None:
        idx_br=len(traces)
        traces.append(go.Scatter(x=start+pd.to_timedelta(br_t,unit='s'),
                                 y=br_bpm, name='Inst BR (bpm)',
                                 yaxis='y3', visible=True))
    # markers aggregated
    def agg(xs, sym, col, name):
        ys=ecg_f[np.searchsorted(t,xs)] #if xs else []
        traces.append(go.Scatter(x=start+pd.to_timedelta(xs,unit='s'), y=ys,
                                 mode='markers',
                                 marker=dict(symbol=sym,color=col,size=10),
                                 name=name, yaxis='y1'))
    agg(pvc_x,'x','red','PVC')
    agg(pac_x,'circle-open','teal','PAC')
    agg(wq_x,'triangle-up','purple','WIDE QRS')
    agg(dip_x,'triangle-down','orange','HR DIP')

    # dropdown vis
    def vis(sel):
        v=[True]*len(traces)
        for idx in [idx_peak, idx_strap, idx_br]:
            if idx is not None: v[idx]=False
        if sel=='Peak' and idx_peak is not None: v[idx_peak]=True
        if sel=='Strap' and idx_strap is not None: v[idx_strap]=True
        if sel=='BR'   and idx_br   is not None: v[idx_br]=True
        return v
    buttons=[dict(label=l,method='update',args=[{'visible':vis(k)}])
             for l,k in [('Peak‑HR','Peak'),('Strap HR','Strap'),('BR (bpm)','BR')]]

    layout=go.Layout(
        title=os.path.basename(path),
        updatemenus=[dict(type='dropdown',x=1.05,y=0.9,buttons=buttons)],
        xaxis=dict(title=f'Time ({TZ})'),
        yaxis=dict(title='ECG',domain=[0.55,1]),
        yaxis2=dict(title='HR',domain=[0.3,0.54]),
        yaxis3=dict(title='Breaths/min',domain=[0,0.25]),
        template='plotly_white',height=850,legend=dict(orientation='h')
    )
    po.plot(go.Figure(data=traces,layout=layout),
            filename=out_html,auto_open=auto_open)
    print("Saved",out_html)

# ---- CLI ----
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--down",type=int,default=5)
    ap.add_argument("--out",default="ecg_view.html"); ap.add_argument("--open",action="store_true")
    a=ap.parse_args(); build(a.csv,a.down,a.out,a.open)
