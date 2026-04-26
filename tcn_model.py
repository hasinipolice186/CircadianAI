"""
tcn_model.py — CircadianTCN (Simulation Edition)
=======================================================
New output heads:
  1. duration_head    → predicted sleep duration (regression, hrs)
  2. insomnia_head    → insomnia binary probability (sigmoid)
  3. recovery_head    → recovery days (regression, 0–21)
  4. trajectory_head  → 7-day insomnia risk sequence (7 sigmoids)
  5. strategy_head    → adaptation strategy (5-class softmax)

New inputs: 19 features (14 original + 5 event context features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Causal convolution ────────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size,
            padding=self.padding, dilation=dilation
        )
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]


class TCNResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(out_ch)
        self.norm2 = nn.LayerNorm(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.residual_proj = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)
        out = F.gelu(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)
        out = F.gelu(out)
        out = self.drop(out)
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        return F.gelu(out + residual)


# ── Main model ────────────────────────────────────────────────────────────────

class CircadianTCN(nn.Module):
    """
    Multi-task TCN v3 with 5 output heads:
      - sleep duration regression
      - insomnia binary classification
      - recovery days regression (NEW)
      - 7-day insomnia risk trajectory (NEW)
      - adaptation strategy classification (NEW, 5 classes)

    Architecture:
      Input Projection (19→64)
      → 4 TCN Residual Blocks (dilations 1,2,4,8)
      → Multi-Head Attention (circadian phase attention)
      → Global Average Pool
      → Shared FC (128→64)
      → 5 task heads
    """

    def __init__(
        self,
        num_features: int = 19,
        seq_len: int      = 7,
        channels: list    = [64, 128, 128, 64],
        kernel_size: int  = 3,
        dropout: float    = 0.2,
        n_strategy: int   = 5,
        forecast_days: int = 7,
    ):
        super().__init__()
        self.num_features  = num_features
        self.seq_len       = seq_len
        self.n_strategy    = n_strategy
        self.forecast_days = forecast_days

        # ── Input projection ──────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Conv1d(num_features, channels[0], 1),
            nn.GELU(),
        )

        # ── TCN blocks ────────────────────────────────────────────────
        dilations = [1, 2, 4, 8]
        blocks = []
        in_ch = channels[0]
        for i, out_ch in enumerate(channels):
            blocks.append(TCNResidualBlock(in_ch, out_ch, kernel_size,
                                           dilation=dilations[i], dropout=dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*blocks)

        # ── Circadian attention ───────────────────────────────────────
        self.attn      = nn.MultiheadAttention(channels[-1], num_heads=4,
                                               dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(channels[-1])

        # ── Event-aware gating (learns when jet lag / shift features matter) ──
        # Takes the 5 event features (last 5 of 19) and gates the shared repr
        self.event_gate = nn.Sequential(
            nn.Linear(5, 32),
            nn.GELU(),
            nn.Linear(32, channels[-1]),
            nn.Sigmoid(),
        )

        # ── Shared FC ─────────────────────────────────────────────────
        self.shared_fc = nn.Sequential(
            nn.Linear(channels[-1], 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
        )

        # ── Head 1: Sleep duration (regression, 0–12h) ────────────────
        self.duration_head = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 1),  nn.Sigmoid()
        )

        # ── Head 2: Insomnia risk (binary) ────────────────────────────
        self.insomnia_head = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 1),  nn.Sigmoid()
        )

        # ── Head 3: Recovery days (regression, 0–21) ──────────────────
        self.recovery_head = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 1),  nn.Sigmoid()   # × 21 at output
        )

        # ── Head 4: 7-day insomnia risk trajectory ────────────────────
        self.trajectory_head = nn.Sequential(
            nn.Linear(64, 64), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, forecast_days), nn.Sigmoid()
        )

        # ── Head 5: Adaptation strategy (5-class) ─────────────────────
        self.strategy_head = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, n_strategy)
            # No softmax here — use CrossEntropyLoss which includes it
        )

    def forward(self, x):
        """
        x: (B, T, F) — batch, 7 days, 19 features

        Returns:
            duration    (B,)      hours [0, 12]
            insomnia    (B,)      probability [0, 1]
            recovery    (B,)      days [0, 21]
            trajectory  (B, 7)   daily insomnia risk [0, 1]
            strategy    (B, 5)   raw logits for CrossEntropyLoss
        """
        B, T, F = x.shape

        # ── Event context features (last 5 columns) ───────────────────
        # Average over time dimension for gate input
        event_ctx = x[:, -1, -5:]   # (B, 5) — use last day's event context

        # ── TCN forward ───────────────────────────────────────────────
        out = x.permute(0, 2, 1)         # (B, F, T)
        out = self.input_proj(out)        # (B, 64, T)
        out = self.tcn(out)               # (B, 64, T)
        out = out.permute(0, 2, 1)        # (B, T, 64)

        # ── Attention ─────────────────────────────────────────────────
        attn_out, _ = self.attn(out, out, out)
        out = self.attn_norm(out + attn_out)    # (B, T, 64)

        # ── Global average pool ───────────────────────────────────────
        out = out.mean(dim=1)                   # (B, 64)

        # ── Event gating (modulates representation for event subjects) ─
        gate  = self.event_gate(event_ctx)      # (B, 64)
        out   = out * gate                       # element-wise modulation

        shared = self.shared_fc(out)            # (B, 64)

        # ── Heads ─────────────────────────────────────────────────────
        duration   = self.duration_head(shared).squeeze(-1) * 12.0   # hrs
        insomnia   = self.insomnia_head(shared).squeeze(-1)           # prob
        recovery   = self.recovery_head(shared).squeeze(-1) * 21.0   # days
        trajectory = self.trajectory_head(shared)                     # (B, 7)
        strategy   = self.strategy_head(shared)                       # (B, 5) logits

        return duration, insomnia, recovery, trajectory, strategy


# ── Edge model (TorchScript-compatible, no MHA) ───────────────────────────────

class EdgeTCN(nn.Module):
    """Lightweight edge model for browser ONNX export — 19 features, 5 outputs."""
    def __init__(self, num_features=19, seq_len=7, channels=[32,64,64,32],
                 kernel_size=3, dropout=0.1, n_strategy=5, forecast_days=7):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Conv1d(num_features, channels[0], 1), nn.GELU())
        blocks = []
        dilations = [1, 2, 4, 8]
        in_ch = channels[0]
        for i, out_ch in enumerate(channels):
            blocks.append(TCNResidualBlock(in_ch, out_ch, kernel_size, dilations[i], dropout))
            in_ch = out_ch
        self.tcn  = nn.Sequential(*blocks)
        self.fc   = nn.Sequential(
            nn.Linear(channels[-1], 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.GELU(),
        )
        self.out  = nn.Linear(32, 2 + 1 + forecast_days + n_strategy)

    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = self.input_proj(out)
        out = self.tcn(out)
        out = out.mean(dim=-1)
        out = self.fc(out)
        out = self.out(out)

        duration   = torch.sigmoid(out[:, 0]) * 12.0
        insomnia   = torch.sigmoid(out[:, 1])
        recovery   = torch.sigmoid(out[:, 2]) * 21.0
        trajectory = torch.sigmoid(out[:, 3:10])
        strategy   = out[:, 10:]          # logits
        return duration, insomnia, recovery, trajectory, strategy


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = CircadianTCN()
    x = torch.randn(4, 7, 19)
    dur, ins, rec, traj, strat = model(x)
    print(f"duration  : {dur.shape}  sample: {dur[0].item():.2f} hrs")
    print(f"insomnia  : {ins.shape}  sample: {ins[0].item():.4f}")
    print(f"recovery  : {rec.shape}  sample: {rec[0].item():.1f} days")
    print(f"trajectory: {traj.shape}  sample: {traj[0].tolist()}")
    print(f"strategy  : {strat.shape}  sample: {strat[0].tolist()}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")