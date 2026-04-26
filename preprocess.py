"""
Preprocessing pipeline — v3.0  (Simulation Edition)
=====================================================
New in v3:
  - 5 phenotypes: healthy, insomnia, circadian_delayed, JET_LAG, NIGHT_SHIFT
  - New raw features: event_type, tz_shift_hrs, shift_week, chronotype_score
  - New engineered features: event_jetlag, event_nightshift, tz_shift_norm,
    shift_week_norm, days_since_event_norm  →  NUM_FEATURES = 19
  - New labels:
      label_recovery_days  (regression, 0–21)
      label_insomnia_7d    (7-day insomnia risk sequence, shape (7,))
      label_strategy       (multiclass: 0=maintain, 1=light_therapy,
                            2=melatonin, 3=sleep_restriction, 4=gradual_shift)
  - create_windows() builds (X, y_dur, y_ins, y_rec, y_ins7d, y_strategy)
  - run_pipeline() saves splits.npz with all labels
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os


# ── Feature definitions ──────────────────────────────────────────────────────

RAW_FEATURES_V3 = [
    # Original 12
    "sleep_duration_hrs", "sleep_efficiency", "heart_rate_resting",
    "rmssd", "sdnn", "steps", "light_exposure_lux", "bedtime_hour",
    "isi_score", "phq_score", "gad_score", "meq_score",
    # New event context
    "event_type",          # 0=none, 1=jet_lag, 2=night_shift
    "tz_shift_hrs",        # timezone shift in hours (–12 to +12)
    "shift_week",          # week number on night shift (0 if not night shift)
    "days_since_event",    # days elapsed since event started (0–60)
    "chronotype_score",    # 0=extreme evening, 1=extreme morning (MEQ-derived)
]

ENGINEERED_FEATURES_V3 = [
    # Original 14
    "sleep_duration_hrs", "sleep_efficiency",
    "heart_rate_norm", "rmssd_norm", "sdnn_norm",
    "steps_norm", "light_exposure_norm",
    "bedtime_sin", "bedtime_cos",
    "isi_norm", "phq_norm", "gad_norm", "meq_norm",
    "age_norm",
    # New 5
    "event_jetlag",        # binary: 1 if jet lag event
    "event_nightshift",    # binary: 1 if night shift event
    "tz_shift_norm",       # tz_shift_hrs / 12.0 (signed, –1 to +1)
    "shift_week_norm",     # shift_week / 12.0
    "days_since_event_norm", # days_since_event / 60.0
]

NUM_FEATURES = len(ENGINEERED_FEATURES_V3)   # 19
SEQ_LEN         = 7
INSOMNIA_ISI_THRESHOLD = 15

# Adaptation strategy classes
STRATEGY_LABELS = {
    0: "Maintain current routine",
    1: "Morning bright light therapy",
    2: "Melatonin + gradual phase advance",
    3: "Sleep restriction therapy",
    4: "Gradual shift schedule adaptation",
}


# ── Circadian encoding ───────────────────────────────────────────────────────

def encode_bedtime(hour_decimal: np.ndarray):
    angle = 2 * np.pi * hour_decimal / 24.0
    return np.sin(angle), np.cos(angle)


# ── Feature engineering v3 ──────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()

    # Original 14
    out["sleep_duration_hrs"]  = df["sleep_duration_hrs"].clip(0, 12) / 12.0
    out["sleep_efficiency"]    = df["sleep_efficiency"].clip(0, 1)
    out["heart_rate_norm"]     = (df["heart_rate_resting"].clip(40, 120) - 40) / 80.0
    out["rmssd_norm"]          = df["rmssd"].clip(0, 150) / 150.0
    out["sdnn_norm"]           = df["sdnn"].clip(0, 200) / 200.0
    out["steps_norm"]          = df["steps"].clip(0, 20000) / 20000.0
    out["light_exposure_norm"] = df["light_exposure_lux"].clip(0, 10000) / 10000.0

    sin_b, cos_b = encode_bedtime(df["bedtime_hour"].values)
    out["bedtime_sin"] = sin_b
    out["bedtime_cos"] = cos_b

    out["isi_norm"] = df["isi_score"].clip(0, 28) / 28.0
    out["phq_norm"] = df["phq_score"].clip(0, 27) / 27.0
    out["gad_norm"] = df["gad_score"].clip(0, 21) / 21.0
    out["meq_norm"] = (df["meq_score"].clip(16, 86) - 16) / 70.0

    if "age" in df.columns:
        out["age_norm"] = (df["age"].clip(18, 80) - 18) / 62.0
    else:
        out["age_norm"] = 0.4

    # New 5 event features
    event = df["event_type"].fillna(0).astype(int)
    out["event_jetlag"]     = (event == 1).astype(float)
    out["event_nightshift"] = (event == 2).astype(float)

    tz = df["tz_shift_hrs"].fillna(0).clip(-12, 12)
    out["tz_shift_norm"] = tz / 12.0                          # signed –1 to +1

    sw = df["shift_week"].fillna(0).clip(0, 12)
    out["shift_week_norm"] = sw / 12.0

    dse = df["days_since_event"].fillna(0).clip(0, 60)
    out["days_since_event_norm"] = dse / 60.0

    # Labels
    out["label_duration"]       = df["sleep_duration_hrs"].clip(0, 12)
    out["label_insomnia"]       = (df["isi_score"] >= INSOMNIA_ISI_THRESHOLD).astype(float)
    out["label_recovery_days"]  = df["label_recovery_days"].clip(0, 21)
    out["label_strategy"]       = df["label_strategy"].astype(int)

    # Forward-fill 7-day insomnia risk labels (one per row; window builder uses them)
    out["label_insomnia_risk"]  = df["label_insomnia_risk"].clip(0.0, 1.0)

    if "subject_id" in df.columns:
        out["subject_id"] = df["subject_id"]
    if "date" in df.columns:
        out["date"] = pd.to_datetime(df["date"])

    return out


# ── Sliding window creator v3 ───────────────────────────────────────────────

def create_windows(df: pd.DataFrame, seq_len: int = SEQ_LEN):
    """
    Returns:
      X            (N, 7, 19)  — feature windows
      y_dur        (N,)        — sleep duration at window end
      y_ins        (N,)        — insomnia binary at window end
      y_rec        (N,)        — recovery days (regression)
      y_ins7d      (N, 7)      — 7-day insomnia risk trajectory
      y_strategy   (N,)        — adaptation strategy class
    """
    feat_cols = ENGINEERED_FEATURES_V3
    X_list, yd_list, yi_list, yr_list, yi7_list, ys_list = [], [], [], [], [], []

    if "subject_id" in df.columns:
        groups = df.groupby("subject_id")
    else:
        groups = [("all", df)]

    for _, grp in groups:
        if "date" in grp.columns:
            grp = grp.sort_values("date").reset_index(drop=True)

        arr    = grp[feat_cols].values.astype(np.float32)
        yd     = grp["label_duration"].values.astype(np.float32)
        yi     = grp["label_insomnia"].values.astype(np.float32)
        yr     = grp["label_recovery_days"].values.astype(np.float32)
        yi_risk = grp["label_insomnia_risk"].values.astype(np.float32)
        ys     = grp["label_strategy"].values.astype(np.int64)

        n = len(arr)
        for i in range(n - seq_len + 1):
            end = i + seq_len - 1
            X_list.append(arr[i : i + seq_len])
            yd_list.append(yd[end])
            yi_list.append(yi[end])
            yr_list.append(yr[end])
            ys_list.append(ys[end])
            # 7-day insomnia risk trajectory starting at window end
            fut_end = min(end + seq_len, n)
            traj    = yi_risk[end : fut_end]
            # Pad if near end of subject
            if len(traj) < seq_len:
                traj = np.pad(traj, (0, seq_len - len(traj)), mode="edge")
            yi7_list.append(traj)

    return (
        np.stack(X_list),
        np.array(yd_list),
        np.array(yi_list),
        np.array(yr_list),
        np.stack(yi7_list),
        np.array(ys_list),
    )


# ── Synthetic dataset v3 ────────────────────────────────────────────────────

def _recovery_days(phenotype, tz_shift, shift_week, age, meq_score, isi_base):
    """
    Clinically-grounded recovery day estimate.
    Reference: Waterhouse et al. (2007), Åkerstedt (2003)
    """
    if phenotype == "jet_lag":
        # Eastward harder than westward; 1h/day rule with age penalty
        direction_factor = 1.3 if tz_shift > 0 else 1.0
        base = abs(tz_shift) * direction_factor
        age_penalty = max(0, (age - 30) / 30) * 2
        isi_penalty = (isi_base / 28) * 3
        return float(np.clip(base + age_penalty + isi_penalty + np.random.normal(0, 0.8), 1, 21))
    elif phenotype == "night_shift":
        # Misalignment builds over weeks; harder for morning chronotypes
        base = 3 + shift_week * 0.5
        chronotype_penalty = (meq_score - 16) / 70 * 5  # high MEQ = morning = worse for nights
        return float(np.clip(base + chronotype_penalty + np.random.normal(0, 1.0), 2, 21))
    else:
        return float(np.clip(np.random.normal(1.5, 0.5), 0, 4))


def _strategy(phenotype, tz_shift, shift_week, isi_score, eff):
    """Determine the recommended adaptation strategy."""
    if phenotype == "jet_lag":
        if tz_shift > 3:
            return 2   # melatonin + gradual advance
        elif tz_shift < -3:
            return 1   # morning bright light
        else:
            return 0   # mild shift, maintain
    elif phenotype == "night_shift":
        if shift_week > 4:
            return 4   # gradual shift schedule
        else:
            return 1   # morning bright light
    elif phenotype == "insomnia" or isi_score >= 15:
        if eff < 0.75:
            return 3   # sleep restriction
        else:
            return 2   # melatonin
    else:
        return 0       # maintain


def _insomnia_risk_trajectory(phenotype, tz_shift, shift_week, base_isi, recovery_days, rng):
    """
    Generate a 60-day insomnia risk time series that peaks at event and decays.
    """
    n = 60
    risk = np.zeros(n)

    if phenotype == "jet_lag":
        peak_day  = 1
        peak_risk = 0.3 + abs(tz_shift) / 12 * 0.6 + (base_isi / 28) * 0.2
        decay     = recovery_days / n
        for d in range(n):
            risk[d] = peak_risk * np.exp(-decay * max(0, d - peak_day))
        risk += rng.normal(0, 0.05, n)

    elif phenotype == "night_shift":
        # Risk builds over weeks then plateaus
        for d in range(n):
            week = d // 7
            build = min(1.0, week / 8)
            peak_risk = 0.3 + build * 0.5 + (base_isi / 28) * 0.2
            risk[d]   = peak_risk
        risk += rng.normal(0, 0.05, n)

    elif phenotype == "insomnia":
        base = base_isi / 28
        risk = np.full(n, base) + rng.normal(0, 0.05, n)

    elif phenotype == "healthy":
        risk = np.full(n, 0.1) + rng.normal(0, 0.03, n)

    else:  # delayed
        risk = np.full(n, 0.25) + rng.normal(0, 0.05, n)

    return np.clip(risk, 0.0, 1.0).astype(np.float32)


def generate_synthetic_dataset(
    n_subjects: int = 300,
    days_per_subject: int = 60,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate a clinically plausible synthetic dataset with 5 phenotypes:
      healthy (30%), insomnia (25%), circadian_delayed (15%),
      jet_lag (20%), night_shift (10%)
    """
    rng = np.random.default_rng(seed)
    records = []

    def nc(val, lo, hi):
        return float(np.clip(val, lo, hi))

    phenotype_dist = ["healthy"] * 30 + ["insomnia"] * 25 + \
                     ["delayed"] * 15 + ["jet_lag"] * 20 + ["night_shift"] * 10
    # Shuffle for randomness
    rng.shuffle(phenotype_dist)

    for subj in range(n_subjects):
        phenotype = rng.choice(
            ["healthy", "insomnia", "delayed", "jet_lag", "night_shift"],
            p=[0.30, 0.25, 0.15, 0.20, 0.10]
        )
        age       = int(rng.integers(18, 65))
        gender    = rng.choice(["male", "female"])
        meq_score_base = int(rng.integers(16, 86))

        # Event-specific parameters
        tz_shift   = 0.0
        shift_week = 0
        event_type = 0

        if phenotype == "jet_lag":
            event_type = 1
            tz_shift   = float(rng.choice([-10,-8,-6,-5,-3,3,5,6,8,10]))
        elif phenotype == "night_shift":
            event_type = 2
            shift_week = int(rng.integers(1, 13))

        # Base ISI for this subject
        if phenotype in ("healthy",):
            isi_base = int(rng.integers(0, 7))
        elif phenotype in ("insomnia",):
            isi_base = int(rng.integers(15, 28))
        elif phenotype in ("delayed",):
            isi_base = int(rng.integers(8, 18))
        elif phenotype == "jet_lag":
            isi_base = int(rng.integers(5, 20))
        else:  # night_shift
            isi_base = int(rng.integers(10, 22))

        rec_days = _recovery_days(phenotype, tz_shift, shift_week, age, meq_score_base, isi_base)
        strategy = _strategy(phenotype, tz_shift, shift_week, isi_base, 0.75)
        risk_trajectory = _insomnia_risk_trajectory(
            phenotype, tz_shift, shift_week, isi_base, rec_days, rng
        )

        for day in range(days_per_subject):
            date = pd.Timestamp("2023-01-01") + pd.Timedelta(days=day)
            # How far into recovery are we?
            recovery_progress = min(1.0, day / max(rec_days, 1))

            r = {
                "subject_id": f"S{subj:04d}",
                "date": date,
                "age": age,
                "gender": gender,
                "event_type": event_type,
                "tz_shift_hrs": tz_shift,
                "shift_week": shift_week,
                "days_since_event": day,
                "chronotype_score": (meq_score_base - 16) / 70.0,
                "label_recovery_days": rec_days,
                "label_strategy": strategy,
                "label_insomnia_risk": float(risk_trajectory[day]),
            }

            # ── Phenotype-specific physiological trajectory ──────────────
            if phenotype == "healthy":
                r["sleep_duration_hrs"] = nc(rng.normal(7.5, 0.5), 5.5, 10)
                r["sleep_efficiency"]   = float(rng.beta(9, 1.5))
                r["heart_rate_resting"] = nc(rng.normal(62, 5), 48, 80)
                r["rmssd"]              = nc(rng.normal(55, 12), 20, 120)
                r["sdnn"]               = nc(rng.normal(60, 15), 20, 130)
                r["steps"]              = nc(rng.normal(9000, 1800), 2000, 18000)
                r["light_exposure_lux"] = nc(rng.normal(3000, 700), 300, 8000)
                r["bedtime_hour"]       = float(rng.normal(23.0, 0.4) % 24)
                r["isi_score"]          = int(rng.integers(0, 7))
                r["phq_score"]          = int(rng.integers(0, 4))
                r["gad_score"]          = int(rng.integers(0, 4))
                r["meq_score"]          = meq_score_base

            elif phenotype == "insomnia":
                r["sleep_duration_hrs"] = nc(rng.normal(5.0, 0.8), 2, 7)
                r["sleep_efficiency"]   = float(rng.beta(3, 3))
                r["heart_rate_resting"] = nc(rng.normal(73, 8), 55, 100)
                r["rmssd"]              = nc(rng.normal(26, 9), 5, 55)
                r["sdnn"]               = nc(rng.normal(29, 11), 5, 65)
                r["steps"]              = nc(rng.normal(5000, 1800), 500, 11000)
                r["light_exposure_lux"] = nc(rng.normal(1500, 500), 100, 4500)
                r["bedtime_hour"]       = float(rng.normal(1.5, 1.0) % 24)
                r["isi_score"]          = int(np.clip(rng.integers(15, 28), 0, 28))
                r["phq_score"]          = int(rng.integers(8, 20))
                r["gad_score"]          = int(rng.integers(8, 18))
                r["meq_score"]          = meq_score_base

            elif phenotype == "delayed":
                r["sleep_duration_hrs"] = nc(rng.normal(7.0, 0.7), 4, 10)
                r["sleep_efficiency"]   = float(rng.beta(6, 2))
                r["heart_rate_resting"] = nc(rng.normal(65, 7), 50, 90)
                r["rmssd"]              = nc(rng.normal(42, 12), 10, 90)
                r["sdnn"]               = nc(rng.normal(48, 14), 10, 100)
                r["steps"]              = nc(rng.normal(7000, 1800), 500, 14000)
                r["light_exposure_lux"] = nc(rng.normal(2000, 600), 100, 5500)
                r["bedtime_hour"]       = float(rng.normal(3.0, 0.7) % 24)
                r["isi_score"]          = int(rng.integers(8, 18))
                r["phq_score"]          = int(rng.integers(4, 14))
                r["gad_score"]          = int(rng.integers(4, 12))
                r["meq_score"]          = meq_score_base

            elif phenotype == "jet_lag":
                # Acute phase (day 0–3): severe disruption
                # Recovery phase (day 4+): gradual normalisation
                if day < 4:
                    disruption = 1.0
                else:
                    disruption = max(0.0, 1.0 - (day - 3) / rec_days)

                # Shift bedtime toward destination
                base_bed   = 23.0
                target_bed = (base_bed + tz_shift) % 24
                current_bed = base_bed + (target_bed - base_bed) * (1 - disruption)

                r["sleep_duration_hrs"] = nc(rng.normal(7.5 - 2.5 * disruption, 0.8), 2.5, 10)
                r["sleep_efficiency"]   = float(rng.beta(max(1, 9 - int(7 * disruption)), max(1, 1 + int(5 * disruption))))
                r["heart_rate_resting"] = nc(rng.normal(65 + 15 * disruption, 6), 48, 100)
                r["rmssd"]              = nc(rng.normal(55 - 30 * disruption, 10), 5, 110)
                r["sdnn"]               = nc(rng.normal(60 - 35 * disruption, 12), 5, 120)
                r["steps"]              = nc(rng.normal(8000 - 4000 * disruption, 1500), 500, 16000)
                r["light_exposure_lux"] = nc(rng.normal(3000, 700), 100, 8000)
                r["bedtime_hour"]       = float(current_bed % 24)
                r["isi_score"]          = int(np.clip(isi_base * disruption + rng.integers(0, 5), 0, 28))
                r["phq_score"]          = int(np.clip(int(7 * disruption) + rng.integers(0, 4), 0, 27))
                r["gad_score"]          = int(np.clip(int(5 * disruption) + rng.integers(0, 3), 0, 21))
                r["meq_score"]          = meq_score_base

            else:  # night_shift
                # Simulate circadian misalignment increasing with shift weeks
                misalignment = min(1.0, shift_week / 8.0)
                inverted_bed = (8.0 + rng.normal(0, 0.5)) % 24   # sleeping during day

                r["sleep_duration_hrs"] = nc(rng.normal(7.0 - 1.5 * misalignment, 0.8), 2.5, 9)
                r["sleep_efficiency"]   = float(rng.beta(max(1, 8 - int(5 * misalignment)), max(1, 1 + int(4 * misalignment))))
                r["heart_rate_resting"] = nc(rng.normal(67 + 10 * misalignment, 7), 48, 98)
                r["rmssd"]              = nc(rng.normal(50 - 25 * misalignment, 10), 5, 100)
                r["sdnn"]               = nc(rng.normal(55 - 28 * misalignment, 12), 5, 110)
                r["steps"]              = nc(rng.normal(7000, 1500), 500, 14000)
                r["light_exposure_lux"] = nc(rng.normal(800, 400), 50, 3000)  # less daylight
                r["bedtime_hour"]       = float(inverted_bed)
                r["isi_score"]          = int(np.clip(isi_base + int(8 * misalignment) + rng.integers(0, 4), 0, 28))
                r["phq_score"]          = int(np.clip(int(10 * misalignment) + rng.integers(0, 5), 0, 27))
                r["gad_score"]          = int(np.clip(int(7 * misalignment) + rng.integers(0, 4), 0, 21))
                r["meq_score"]          = meq_score_base

            records.append(r)

    return pd.DataFrame(records)


# ── Full pipeline v3 ─────────────────────────────────────────────────────────

def run_pipeline(
    csv_path: str    = None,
    save_dir: str    = ".",
    seq_len: int     = SEQ_LEN,
    val_size: float  = 0.15,
    test_size: float = 0.15,
    seed: int        = 42,
):
    os.makedirs(save_dir, exist_ok=True)

    if csv_path and os.path.exists(csv_path):
        print(f"Loading real data from {csv_path}")
        raw = pd.read_csv(csv_path, parse_dates=["date"])
    else:
        print("Generating synthetic dataset v3 (300 subjects × 60 days)…")
        print("  Phenotypes: healthy / insomnia / delayed / jet_lag / night_shift")
        raw = generate_synthetic_dataset()

    print("Engineering features v3…")
    eng = engineer_features(raw)
    out_csv = os.path.join(save_dir, "final_tcn_dataset.csv")
    eng.to_csv(out_csv, index=False)
    print(f"  Saved {out_csv}  ({len(eng)} rows, {NUM_FEATURES} features)")

    phenotype_counts = raw.groupby("event_type")["subject_id"].nunique()
    print(f"  Event types — {dict(phenotype_counts)}")

    print(f"Creating {seq_len}-day sliding windows…")
    X, y_dur, y_ins, y_rec, y_ins7d, y_strat = create_windows(eng, seq_len)
    print(f"  X shape       : {X.shape}")
    print(f"  y_ins7d shape : {y_ins7d.shape}")
    print(f"  Insomnia prev : {y_ins.mean():.1%}")
    strat_dist = {STRATEGY_LABELS[k]: int((y_strat == k).sum()) for k in range(5)}
    print(f"  Strategy dist : {strat_dist}")

    idx = np.arange(len(X))
    tr_idx, tmp_idx = train_test_split(
        idx, test_size=val_size + test_size,
        random_state=seed, stratify=y_ins.astype(int)
    )
    val_ratio = val_size / (val_size + test_size)
    va_idx, te_idx = train_test_split(
        tmp_idx, test_size=1 - val_ratio,
        random_state=seed, stratify=y_ins[tmp_idx].astype(int)
    )

    print(f"  Train: {len(tr_idx)}  Val: {len(va_idx)}  Test: {len(te_idx)}")

    np.savez(
        os.path.join(save_dir, "splits.npz"),
        X_train     =X[tr_idx],  y_dur_train =y_dur[tr_idx],
        y_ins_train =y_ins[tr_idx], y_rec_train=y_rec[tr_idx],
        y_ins7d_train=y_ins7d[tr_idx], y_strat_train=y_strat[tr_idx],
        X_val       =X[va_idx],  y_dur_val   =y_dur[va_idx],
        y_ins_val   =y_ins[va_idx],   y_rec_val  =y_rec[va_idx],
        y_ins7d_val =y_ins7d[va_idx], y_strat_val=y_strat[va_idx],
        X_test      =X[te_idx],  y_dur_test  =y_dur[te_idx],
        y_ins_test  =y_ins[te_idx],   y_rec_test =y_rec[te_idx],
        y_ins7d_test=y_ins7d[te_idx], y_strat_test=y_strat[te_idx],
    )
    print("  Saved splits.npz")

    return (X[tr_idx], y_dur[tr_idx], y_ins[tr_idx], y_rec[tr_idx],
            y_ins7d[tr_idx], y_strat[tr_idx],
            X[va_idx],  y_dur[va_idx],  y_ins[va_idx],  y_rec[va_idx],
            y_ins7d[va_idx], y_strat[va_idx])


if __name__ == "__main__":
    run_pipeline(save_dir=".")
    print(f"Preprocessing v3 complete. NUM_FEATURES = {NUM_FEATURES}")