"""
export_onnx.py
=================
Converts tcn_edge.pt → circadian_edge.onnx for browser Edge AI inference.

Run ONCE after training:
    python export_onnx.py

Then place circadian_edge.onnx next to index.html and serve with:
    python -m http.server 8080

Changes from export_onnx.py:
  - Input shape: (1, 7, 19)  — was (1, 7, 13)
  - Output names: sleep_duration, insomnia_prob, recovery_days,
                  insomnia_trajectory, strategy_logits  — was 2 outputs
  - Opset 14, dynamic batch axis
"""

import torch
import torch.onnx
import os
import json

# ── Config ───────────────────────────────────────────────────────────────────
SEQ_LEN       = 7
NUM_FEATURES  = 19           # v3: 14 original + 5 event context features
EDGE_PT_PATH  = "tcn_edge.pt"
ONNX_OUT      = "circadian_edge.onnx"
FORECAST_DAYS = 7
N_STRATEGY    = 5

STRATEGY_LABELS = {
    0: "Maintain current routine",
    1: "Morning bright light therapy",
    2: "Melatonin + gradual phase advance",
    3: "Sleep restriction therapy",
    4: "Gradual shift schedule adaptation",
}

FEATURE_NAMES = [
    # Original 14
    "sleep_duration_hrs", "sleep_efficiency",
    "heart_rate_norm", "rmssd_norm", "sdnn_norm",
    "steps_norm", "light_exposure_norm",
    "bedtime_sin", "bedtime_cos",
    "isi_norm", "phq_norm", "gad_norm", "meq_norm",
    "age_norm",
    # New 5
    "event_jetlag", "event_nightshift",
    "tz_shift_norm", "shift_week_norm", "days_since_event_norm",
]


def export():
    print(f"Loading TorchScript edge model from '{EDGE_PT_PATH}' …")
    if not os.path.exists(EDGE_PT_PATH):
        raise FileNotFoundError(
            f"'{EDGE_PT_PATH}' not found. Run train_tcn_model.py first."
        )

    model = torch.jit.load(EDGE_PT_PATH, map_location="cpu")
    model.eval()

    # Dummy input: (batch=1, seq_len=7, features=19)
    dummy = torch.randn(1, SEQ_LEN, NUM_FEATURES)

    # Verify the model runs before export
    with torch.no_grad():
        dur, ins, rec, traj, strat = model(dummy)
    print(f"  Model verified:")
    print(f"    duration   : {dur.shape}  → {dur[0].item():.2f} hrs")
    print(f"    insomnia   : {ins.shape}  → {ins[0].item():.4f}")
    print(f"    recovery   : {rec.shape}  → {rec[0].item():.1f} days")
    print(f"    trajectory : {traj.shape}")
    print(f"    strategy   : {strat.shape}")

    print(f"\nExporting to ONNX: '{ONNX_OUT}' …")
    torch.onnx.export(
        model,
        dummy,
        ONNX_OUT,
        input_names=["input"],
        output_names=[
            "sleep_duration",
            "insomnia_prob",
            "recovery_days",
            "insomnia_trajectory",
            "strategy_logits",
        ],
        dynamic_axes={
            "input":               {0: "batch"},
            "sleep_duration":      {0: "batch"},
            "insomnia_prob":       {0: "batch"},
            "recovery_days":       {0: "batch"},
            "insomnia_trajectory": {0: "batch"},
            "strategy_logits":     {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,   # force legacy exporter — required for PyTorch >= 2.6
    )

    size_kb = os.path.getsize(ONNX_OUT) / 1024
    print(f"\n✓ Exported '{ONNX_OUT}'  ({size_kb:.0f} KB)")

    # ── Write model metadata JSON (used by index.html to parse outputs) ──────
    meta = {
        "version":       "3.0",
        "seq_len":       SEQ_LEN,
        "num_features":  NUM_FEATURES,
        "feature_names": FEATURE_NAMES,
        "forecast_days": FORECAST_DAYS,
        "n_strategy":    N_STRATEGY,
        "strategy_labels": STRATEGY_LABELS,
        "outputs": {
            "sleep_duration":      "float, hours [0–12]",
            "insomnia_prob":       "float, probability [0–1]",
            "recovery_days":       "float, days [0–21]",
            "insomnia_trajectory": f"float array [{FORECAST_DAYS}], daily risk [0–1]",
            "strategy_logits":     f"float array [{N_STRATEGY}], apply softmax + argmax",
        },
        "event_type_map": {
            "0": "none",
            "1": "jet_lag",
            "2": "night_shift",
        },
        "disclaimer": (
            "SCREENING TOOL ONLY. Not a medical diagnostic device. "
            "Consult a qualified healthcare professional."
        ),
    }
    meta_path = "circadian_edge_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✓ Metadata written to '{meta_path}'")

    # ── How to use in browser (JavaScript snippet) ────────────────────────────
    print("""
── Browser usage (JavaScript) ────────────────────────────────────────────────

  import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js';

  const session = await ort.InferenceSession.create('circadian_edge.onnx');

  // input: Float32Array of shape [1, 7, 19]
  const inputTensor = new ort.Tensor('float32', inputData, [1, 7, 19]);
  const results     = await session.run({ input: inputTensor });

  const sleepDuration      = results['sleep_duration'].data[0];        // hrs
  const insomniaProbability = results['insomnia_prob'].data[0];         // 0–1
  const recoveryDays        = results['recovery_days'].data[0];         // days
  const trajectory          = Array.from(results['insomnia_trajectory'].data);  // [7]
  const strategyLogits      = Array.from(results['strategy_logits'].data);      // [5]
  const strategyId          = strategyLogits.indexOf(Math.max(...strategyLogits));

──────────────────────────────────────────────────────────────────────────────

Next steps:
  1. Copy circadian_edge.onnx + circadian_edge_meta.json → same folder as index.html
  2. Serve locally:   python -m http.server 8080
  3. Open:            http://localhost:8080
""")


if __name__ == "__main__":
    export()