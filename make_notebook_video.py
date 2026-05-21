"""
make_notebook_video.py
======================
Generates a 10-second MP4 video that walks through the 5 Project 8
notebooks — one scene per notebook — with animated KPI reveals,
chart thumbnails, and a scrolling bottom ticker.

Usage (from the P8/ directory):
    python make_notebook_video.py

Requirements:
    pip install imageio[ffmpeg] matplotlib Pillow numpy pandas

Output:
    results/ml_risk_notebook_video.mp4   (~1.2 MB)

Author : Niraj Neupane | github.com/nirajneupane17
Project: ML Risk Estimation & Forecasting (Project 8 / 10)
"""

# ── 0. Imports ────────────────────────────────────────────────────────────────
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                        # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from PIL import Image
import imageio
import io
warnings.filterwarnings("ignore")

# ── 1. Configuration ──────────────────────────────────────────────────────────

# Video parameters
FPS           = 30                           # frames per second
DURATION_SEC  = 10.0                         # total video length
TOTAL_FRAMES  = int(FPS * DURATION_SEC)      # 300 frames
WIDTH, HEIGHT = 1280, 720                    # resolution (px)
OUTPUT_PATH   = "results/ml_risk_notebook_video.mp4"

# Colour palette (Bloomberg-dark theme)
C = {
    "bg":      "#0a0e1a",   # deep navy background
    "panel":   "#111827",   # card panel
    "border":  "#1f2937",   # subtle border
    "text":    "#f0ece0",   # warm white text
    "muted":   "#6b7280",   # secondary labels
    "amber":   "#ff9900",   # primary accent
    "cyan":    "#00d4ff",   # secondary accent
    "green":   "#00e676",   # positive / pass
    "red":     "#ff4444",   # negative / fail / alert
    "purple":  "#a855f7",   # SHAP panel
    "teal":    "#14b8a6",   # feature importance panel
    "yellow":  "#fbbf24",   # anomaly highlight
}

# ── 2. Scene definitions ──────────────────────────────────────────────────────
# Each scene occupies 2 seconds (60 frames) = 5 scenes × 2s = 10s total.
# Structure: (scene_title, chart_file, accent_color, kpi_dict, subtitle)

RES = "results"
SCENES = [
    {
        "nb":       "01 · Volatility Forecasting",
        "subtitle": "Random Forest vs XGBoost vs GARCH Baseline",
        "chart":    f"{RES}/01_volatility_forecast.png",
        "color":    C["amber"],
        "kpis":     [
            ("RF RMSE",    "0.0104"),
            ("XGB RMSE",   "0.0107"),
            ("GARCH RMSE", "0.0097"),
            ("Horizon",    "21-day"),
        ],
        "tag": "SR 11-7  Challenger Benchmark",
    },
    {
        "nb":       "02 · ML VaR Estimation",
        "subtitle": "XGBoost Quantile Regression — 99% Confidence",
        "chart":    f"{RES}/02_var_ml_comparison.png",
        "color":    C["cyan"],
        "kpis":     [
            ("99% VaR",    "-3.09%"),
            ("Exc Rate",   "1.03%"),
            ("Expected",   "1.00%"),
            ("Kupiec",     "PASS ✓"),
        ],
        "tag": "Basel III  Kupiec POF",
    },
    {
        "nb":       "03 · Credit Risk Classification",
        "subtitle": "Logistic Regression  ·  RF  ·  XGBoost",
        "chart":    f"{RES}/03_credit_roc_auc.png",
        "color":    C["green"],
        "kpis":     [
            ("LR AUC",   "0.840"),
            ("RF AUC",   "0.824"),
            ("XGB AUC",  "0.785"),
            ("Gini RF",  "0.648"),
        ],
        "tag": "2000 loans  ·  13.6% default rate",
    },
    {
        "nb":       "04 · Anomaly Detection",
        "subtitle": "Isolation Forest  —  Market Stress Identification",
        "chart":    f"{RES}/04_anomaly_detection.png",
        "color":    C["red"],
        "kpis":     [
            ("Anomalies",   "130"),
            ("Rate",        "5.0%"),
            ("COVID cap.",  "~90%"),
            ("2022 cap.",   "~20%"),
        ],
        "tag": "2015–2024  ·  COVID + Rate Shock flagged",
    },
    {
        "nb":       "05 · SHAP Explainability",
        "subtitle": "SR 11-7 Model Transparency  ·  Feature Stability",
        "chart":    f"{RES}/05_shap_waterfall.png",
        "color":    C["purple"],
        "kpis":     [
            ("Top feature", "vol_21d"),
            ("Top-3 conc.", "~71%"),
            ("Models",      "RF+XGB"),
            ("Standard",    "SR 11-7"),
        ],
        "tag": "SHAP  ·  LIME  ·  Feature Importance Correlation",
    },
]

# Frames per scene (evenly distributed)
FRAMES_PER_SCENE = TOTAL_FRAMES // len(SCENES)   # 60 frames = 2 seconds each

# ── 3. Easing functions ───────────────────────────────────────────────────────

def ease_out(t):
    """Decelerate into final position — natural motion."""
    return 1 - (1 - t) ** 3

def ease_in_out(t):
    """Smooth start and end."""
    return t * t * (3 - 2 * t)

def pulse(t, freq=2.0):
    """Oscillating glow for live indicators."""
    return 0.5 + 0.5 * np.sin(t * freq * 2 * np.pi)

# ── 4. Pre-load chart thumbnails ──────────────────────────────────────────────

def load_chart(path, target_w=760, target_h=400):
    """Load a PNG chart and resize it for embedding."""
    if not os.path.exists(path):
        # Return a placeholder if chart not found
        img = Image.new("RGB", (target_w, target_h), color="#1a2035")
        return img
    img = Image.open(path).convert("RGB")
    img = img.resize((target_w, target_h), Image.LANCZOS)
    return img

charts = [load_chart(s["chart"]) for s in SCENES]
print(f"Loaded {len(charts)} chart thumbnails.")

# ── 5. Frame renderer ─────────────────────────────────────────────────────────

def render_frame(frame_idx: int) -> Image.Image:
    """
    Render a single video frame.

    Each scene = FRAMES_PER_SCENE frames.
    Within a scene: 0–0.3s = slide-in, 0.3–1.7s = hold, 1.7–2.0s = fade-out.
    """
    global_t = frame_idx / TOTAL_FRAMES           # 0 → 1 over whole video
    global_sec = frame_idx / FPS                  # seconds elapsed

    # Determine which scene we're in
    scene_idx = min(frame_idx // FRAMES_PER_SCENE, len(SCENES) - 1)
    scene = SCENES[scene_idx]
    scene_frame = frame_idx - scene_idx * FRAMES_PER_SCENE
    scene_t = scene_frame / FRAMES_PER_SCENE      # 0 → 1 within this scene

    # Scene timing stages
    SLIDE_IN_END  = 0.20   # slide-in: 0 → 0.20
    HOLD_END      = 0.85   # hold:     0.20 → 0.85
    FADE_OUT_END  = 1.00   # fade-out: 0.85 → 1.00

    if scene_t < SLIDE_IN_END:
        phase = "in"
        phase_t = ease_out(scene_t / SLIDE_IN_END)
    elif scene_t < HOLD_END:
        phase = "hold"
        phase_t = 1.0
    else:
        phase = "out"
        phase_t = 1.0 - ease_out((scene_t - HOLD_END) / (FADE_OUT_END - HOLD_END))

    alpha_main = max(0, min(1, phase_t))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), facecolor=C["bg"], dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=C["bg"])
    ax.set_xlim(0, WIDTH); ax.set_ylim(0, HEIGHT); ax.axis("off")

    # ── Background grid lines (subtle atmosphere) ─────────────────────────────
    for x in range(0, WIDTH, 48):
        ax.axvline(x, color=scene["color"], alpha=0.025 + 0.015 * np.sin(x / 60 + global_t * 5), lw=0.4)
    for y in range(0, HEIGHT, 48):
        ax.axhline(y, color=C["cyan"], alpha=0.018 + 0.012 * np.sin(y / 50 + global_t * 4), lw=0.3)

    # ── Top header ────────────────────────────────────────────────────────────
    # Header background bar
    ax.add_patch(plt.Rectangle((0, HEIGHT - 58), WIDTH, 58,
        facecolor=C["panel"], alpha=0.96))
    ax.plot([0, WIDTH], [HEIGHT - 58, HEIGHT - 58], color=scene["color"], lw=2.0, alpha=0.9)

    # Progress bar (fills across the full video)
    ax.add_patch(plt.Rectangle((0, HEIGHT - 5), WIDTH * global_t, 5,
        facecolor=scene["color"], alpha=0.9))

    # Scene progress dots (one per notebook)
    dot_y = HEIGHT - 32
    for i, s in enumerate(SCENES):
        dot_x = WIDTH - 180 + i * 32
        done   = i < scene_idx
        active = i == scene_idx
        col    = s["color"] if done or active else C["muted"]
        alph   = 1.0 if done or active else 0.35
        radius = 9 if active else 6
        circle = plt.Circle((dot_x, dot_y), radius, color=col, alpha=alph, zorder=5)
        ax.add_patch(circle)
        if active:
            # Pulsing ring
            ring = plt.Circle((dot_x, dot_y), radius + 4,
                color=col, fill=False, lw=1.5, alpha=0.4 * pulse(global_t), zorder=4)
            ax.add_patch(ring)

    # Notebook name (left)
    ax.text(18, HEIGHT - 22, scene["nb"],
        fontsize=15, fontfamily="monospace", fontweight="bold",
        color=scene["color"], va="center", alpha=alpha_main)
    ax.text(18, HEIGHT - 44, scene["subtitle"],
        fontsize=9.5, fontfamily="monospace",
        color=C["muted"], va="center", alpha=alpha_main * 0.85)

    # Repo tag (right of dots)
    ax.text(WIDTH - 195, HEIGHT - 30, "github.com/nirajneupane17",
        fontsize=8, fontfamily="monospace", color=C["muted"],
        ha="right", va="center", alpha=0.6)

    # ── Chart thumbnail (left side) ───────────────────────────────────────────
    chart_x = -30 + 30 * phase_t      # slides in from left
    chart_y = 58
    chart_w, chart_h = 762, 398

    ax.imshow(np.array(charts[scene_idx]),
        extent=[chart_x, chart_x + chart_w, chart_y, chart_y + chart_h],
        aspect="auto", zorder=2, alpha=alpha_main * 0.90)

    # Border around chart
    ax.add_patch(FancyBboxPatch((chart_x - 1, chart_y - 1), chart_w + 2, chart_h + 2,
        facecolor="none", edgecolor=scene["color"], linewidth=1.5,
        boxstyle="round,pad=1", alpha=alpha_main * 0.6, zorder=3))

    # ── KPI Cards (right panel) ───────────────────────────────────────────────
    kpi_x0 = 810                      # left edge of KPI column
    kpi_w  = 440
    card_h = 76
    card_gap = 12
    card_y0 = HEIGHT - 80             # start just below header

    for i, (label, val) in enumerate(scene["kpis"]):
        delay_t = max(0, (scene_t - 0.08 * i) / 0.25)
        card_alpha = ease_out(min(delay_t, 1.0)) * alpha_main
        if card_alpha < 0.01:
            continue

        cy = card_y0 - i * (card_h + card_gap)
        # Slide in from right
        cx = WIDTH + (1 - ease_out(min(delay_t, 1.0))) * 120
        cx = min(cx, kpi_x0)

        # Card body
        ax.add_patch(FancyBboxPatch((cx, cy - card_h), kpi_w, card_h,
            facecolor=C["panel"], edgecolor=scene["color"],
            linewidth=1.2, boxstyle="round,pad=3", alpha=card_alpha * 0.94, zorder=4))

        # Glow strip on top of card
        ax.add_patch(plt.Rectangle((cx + 3, cy - 3), kpi_w - 6, 3,
            facecolor=scene["color"], alpha=card_alpha * 0.85, zorder=5))

        # Label
        ax.text(cx + 16, cy - 20, label,
            fontsize=9, fontfamily="monospace", color=C["muted"],
            va="center", alpha=card_alpha, zorder=6)

        # Value
        ax.text(cx + 16, cy - card_h / 2 - 4, val,
            fontsize=21, fontfamily="monospace", fontweight="bold",
            color=scene["color"], va="center", alpha=card_alpha, zorder=6)

    # SR 11-7 / regulatory tag (below KPI cards)
    tag_y = card_y0 - 4 * (card_h + card_gap) - 18
    ax.add_patch(FancyBboxPatch((kpi_x0, tag_y - 24), kpi_w, 24,
        facecolor=C["border"], edgecolor=C["muted"],
        linewidth=0.8, boxstyle="round,pad=2", alpha=alpha_main * 0.6, zorder=4))
    ax.text(kpi_x0 + kpi_w / 2, tag_y - 12, scene["tag"],
        fontsize=8.5, fontfamily="monospace", color=C["muted"],
        ha="center", va="center", alpha=alpha_main * 0.85, zorder=5)

    # ── Bottom ticker ─────────────────────────────────────────────────────────
    ticker_h = 28
    ax.add_patch(plt.Rectangle((0, 0), WIDTH, ticker_h,
        facecolor=C["panel"], alpha=0.95))
    ax.plot([0, WIDTH], [ticker_h, ticker_h], color=scene["color"], lw=0.9, alpha=0.5)

    # Scrolling text
    ticker_text = (
        "  RF RMSE 0.0104  ·  XGB RMSE 0.0107  ·  GARCH 0.0097   ▌   "
        "Credit AUC: LR 0.840  RF 0.824  XGB 0.785   ▌   "
        "99%% VaR −3.09%%  exc_rate 1.03%%  Kupiec PASS   ▌   "
        "Anomalies 130 (2015-2024)  COVID flagged  Rate Shock flagged   ▌   "
        "SHAP vol_21d top feature  top-3 conc. 71%%  SR 11-7 COMPLIANT   ▌   "
    )
    scroll_x = WIDTH / 2 - (global_sec * 130) % WIDTH
    ax.text(scroll_x, ticker_h / 2, ticker_text * 3,
        fontsize=8.5, fontfamily="monospace", color=scene["color"],
        va="center", ha="center", alpha=0.88)

    # ── Scene number badge ────────────────────────────────────────────────────
    badge_x, badge_y = WIDTH - 45, HEIGHT - 82
    ax.add_patch(plt.Circle((badge_x, badge_y), 20, color=scene["color"], alpha=0.15, zorder=4))
    ax.add_patch(plt.Circle((badge_x, badge_y), 20, color="none",
        ec=scene["color"], lw=1.5, alpha=alpha_main * 0.8, zorder=5))
    ax.text(badge_x, badge_y, f"{scene_idx+1}/{len(SCENES)}",
        fontsize=8.5, fontfamily="monospace", fontweight="bold",
        color=scene["color"], ha="center", va="center", alpha=alpha_main, zorder=6)

    # ── Render frame to PIL Image ─────────────────────────────────────────────
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, facecolor=C["bg"], bbox_inches=None)
    buf.seek(0)
    pil = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return pil


# ── 6. Main render loop ───────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    print(f"Rendering {TOTAL_FRAMES} frames  ({DURATION_SEC}s @ {FPS}fps)  →  {OUTPUT_PATH}")
    print(f"Scenes ({len(SCENES)} × {FRAMES_PER_SCENE} frames each):")
    for i, s in enumerate(SCENES):
        print(f"  {i+1}. {s['nb']}")

    frames = []
    for f in range(TOTAL_FRAMES):
        frames.append(render_frame(f))
        if (f + 1) % 30 == 0:
            pct = (f + 1) / TOTAL_FRAMES * 100
            sec = (f + 1) / FPS
            print(f"  {f+1:3d}/{TOTAL_FRAMES} frames  ({pct:.0f}%  ·  {sec:.1f}s)")

    print("Writing MP4...")
    # imageio 2.30+ API — use output_params instead of codec/quality kwargs
    imageio.mimwrite(
        OUTPUT_PATH,
        [np.array(f) for f in frames],
        fps=FPS,
        output_params=["-vcodec", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"],
    )

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"\n✓ Done  →  {OUTPUT_PATH}  ({size_kb:.0f} KB)")
    print(f"  Resolution : {WIDTH}×{HEIGHT}")
    print(f"  Duration   : {DURATION_SEC}s  |  {FPS}fps  |  {TOTAL_FRAMES} frames")
    print(f"  Scenes     : {len(SCENES)} notebooks × 2 seconds each")
