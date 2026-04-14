#!/usr/bin/env python3
"""
===============================================================================
  SceneFactory — NeurIPS 2026 Experiments Figures
===============================================================================

Generates all publishable figures for §4 (Experiments) of the SceneFactory paper.

Style guide:
  • NeurIPS 2026 single-column format (text width ≈ 5.5 in)
  • Font: serif (Times / Computer Modern) to match LaTeX body
  • Palette: colorblind-safe (Wong 2011 + Tol muted)
  • 300 DPI raster, PDF vector primary output
  • Minimal chart-junk: no unnecessary grid, spines trimmed
  • Error visualisation: thin caps on bars, translucent bands on lines
  • All text ≥ 8 pt at final print size

Figures produced:
  1. fig_throughput_scaling    — CASPS vs. agent slots (log-y dual-line)
  2. fig_cross_simulator       — Cross-simulator throughput comparison (horizontal bar)
  3. fig_generalization         — Generalization study grouped bar chart
  4. fig_training_curve         — Reward / success-rate vs. training iteration
  5. fig_per_world_success      — Per-world success-rate distribution (histogram + strip)

Usage:
  python plot_experiments.py            # generates all figures
  python plot_experiments.py --only 1   # generates only figure 1

Requires: matplotlib ≥ 3.7, numpy
===============================================================================
"""

import argparse
import json
import pathlib
import textwrap

import matplotlib
matplotlib.use("Agg")                       # headless backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import FancyBboxPatch

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  GLOBAL STYLE                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════╝

# --- Colorblind-safe palette (Wong 2011 extended) -----------------------
C_BLUE      = "#0072B2"   # SceneFactory primary
C_ORANGE    = "#D55E00"   # Baseline / GPUDrive
C_GREEN     = "#009E73"   # Nocturne / secondary
C_YELLOW    = "#F0E442"   # accent (highlight bars)
C_SKYBLUE   = "#56B4E9"   # light accent
C_VERMILLION= "#E69F00"   # warm accent
C_PURPLE    = "#CC79A7"   # tertiary
C_GRAY      = "#999999"   # neutral / grid
C_DARKGRAY  = "#404040"   # text

# --- Matplotlib RC overrides -------------------------------------------
STYLE = {
    # Font
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":   "cm",
    "font.size":          9,
    "axes.labelsize":     10,
    "axes.titlesize":     11,
    "legend.fontsize":    8,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    # Axes
    "axes.linewidth":     0.6,
    "axes.edgecolor":     C_DARKGRAY,
    "axes.labelcolor":    C_DARKGRAY,
    "xtick.color":        C_DARKGRAY,
    "ytick.color":        C_DARKGRAY,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.minor.width":  0.4,
    "ytick.minor.width":  0.4,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    # Grid
    "axes.grid":          False,
    "grid.linewidth":     0.4,
    "grid.alpha":         0.3,
    # Legend
    "legend.frameon":     True,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "#cccccc",
    "legend.fancybox":    True,
    # Saving
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype":       42,       # TrueType in PDF (editable text)
    "ps.fonttype":        42,
}
plt.rcParams.update(STYLE)

OUTDIR = pathlib.Path(__file__).parent / "figures"
OUTDIR.mkdir(exist_ok=True)

def _save(fig, name):
    """Save figure as PDF (vector) and PNG (raster preview)."""
    fig.savefig(OUTDIR / f"{name}.pdf")
    fig.savefig(OUTDIR / f"{name}.png")
    print(f"  ✓ saved {OUTDIR / name}.{{pdf,png}}")
    plt.close(fig)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 1 — Throughput Scaling  (§4.2)                              ║
# ║                                                                     ║
# ║  Prompt:                                                            ║
# ║  "Dual-line plot on a log-scale y-axis showing controlled agent     ║
# ║  steps per second (CASPS) vs. total agent slots for two pipelines   ║
# ║  (SceneFactory and Baseline) at 4 scaling points (512, 1024, 2048,  ║
# ║  4096 slots). SceneFactory should use a bold blue line with circle  ║
# ║  markers and translucent ±1σ error band; Baseline should use a      ║
# ║  dashed orange line with square markers. Annotate the speedup       ║
# ║  factor (e.g. '127×') next to the rightmost SceneFactory point.    ║
# ║  Include a subtle horizontal reference line at the Nocturne         ║
# ║  throughput (15K ASPS) with a label. X-axis: 'Total agent slots',  ║
# ║  Y-axis: 'CASPS (log scale)'. Use serif fonts matching NeurIPS     ║
# ║  LaTeX template, colorblind-safe palette, and minimal spines.       ║
# ║  Figure width = 3.25 in (NeurIPS half-column). Output PDF+PNG."    ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def fig1_throughput_scaling():
    print("Figure 1: Throughput scaling")

    # Data
    worlds = np.array([32, 64, 128, 256])
    slots  = worlds * 16

    bl_mean = np.array([174.3, 158.6, 163.8, 152.3])
    bl_std  = np.array([26.6,  29.4,   1.0,   0.4])

    sf_mean = np.array([3869.8, 7172.9, 12225.3, 19249.7])
    sf_std  = np.array([  66.0,  113.9,  1519.9,  2984.0])

    speedup = sf_mean / bl_mean

    fig, ax = plt.subplots(figsize=(3.4, 2.5))

    # SceneFactory line + error band
    ax.fill_between(slots, sf_mean - sf_std, sf_mean + sf_std,
                    color=C_BLUE, alpha=0.15, linewidth=0)
    ax.plot(slots, sf_mean, '-o', color=C_BLUE, linewidth=1.8,
            markersize=5, markeredgecolor='white', markeredgewidth=0.6,
            label='SceneFactory (ours)', zorder=5)

    # Baseline line + error band
    ax.fill_between(slots, bl_mean - bl_std, bl_mean + bl_std,
                    color=C_ORANGE, alpha=0.15, linewidth=0)
    ax.plot(slots, bl_mean, '--s', color=C_ORANGE, linewidth=1.5,
            markersize=4.5, markeredgecolor='white', markeredgewidth=0.6,
            label='Baseline (Vehicle Wizard + SB3)', zorder=5)

    # Nocturne reference line
    ax.axhline(15000, color=C_GREEN, linewidth=0.8, linestyle=':', alpha=0.7, zorder=2)
    ax.text(550, 15000 * 1.18, 'Nocturne (15K ASPS, CPU)',
            fontsize=7, color=C_GREEN, va='bottom', style='italic')

    # Speedup annotations
    for i in range(len(slots)):
        label = f'{speedup[i]:.0f}×'
        ax.annotate(label,
                    xy=(slots[i], sf_mean[i]),
                    xytext=(0, 10), textcoords='offset points',
                    fontsize=7, fontweight='bold', color=C_BLUE,
                    ha='center', va='bottom',
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(slots)
    ax.set_xticklabels([f'{s:,}' for s in slots])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel('Total agent slots')
    ax.set_ylabel('CASPS (log scale)')
    ax.set_ylim(80, 40000)

    # Light grid
    ax.yaxis.grid(True, which='major', linewidth=0.3, alpha=0.4, color=C_GRAY)

    # Trim spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc='upper left', borderpad=0.4, handlelength=1.8)
    fig.tight_layout()
    _save(fig, 'fig_throughput_scaling')


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 2 — Cross-Simulator Comparison  (§4.2, Table 2)            ║
# ║                                                                     ║
# ║  Prompt:                                                            ║
# ║  "Horizontal bar chart comparing simulator throughputs on a log-x   ║
# ║  axis. Entries (top to bottom): GPUDrive (ASPS, 2.3M), GPUDrive    ║
# ║  training (CASPS, 200K–500K midpoint), SceneFactory (CASPS, 19.3K),║
# ║  Nocturne (ASPS, 15K). Color bars by physics model: blue for rigid- ║
# ║  body (SceneFactory), orange for kinematic (GPUDrive, Nocturne).    ║
# ║  Add a vertical dashed separator between kinematic and rigid-body   ║
# ║  regimes. Annotate each bar with the numeric value inside/beside.   ║
# ║  Include a text note 'kinematic ↔ rigid-body' at the boundary.     ║
# ║  Compact figure (3.25 in × 1.8 in). Serif fonts, NeurIPS style."  ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def fig2_cross_simulator():
    print("Figure 2: Cross-simulator comparison")

    labels     = ['Nocturne\n(CPU, kinematic)',
                  'SceneFactory\n(ours, 10-DOF rigid body)',
                  'GPUDrive training\n(A100, kinematic)',
                  'GPUDrive peak\n(A100, kinematic)']
    values     = [15_000, 19_250, 350_000, 2_300_000]   # midpoint for GPUDrive training
    colors     = [C_ORANGE, C_BLUE, C_ORANGE, C_ORANGE]
    hatches    = ['',  '',  '',  '']
    edge_colors= [C_ORANGE, C_BLUE, C_ORANGE, C_ORANGE]

    fig, ax = plt.subplots(figsize=(3.4, 2.0))

    bars = ax.barh(range(len(labels)), values, height=0.55,
                   color=colors, edgecolor=edge_colors, linewidth=0.6,
                   alpha=0.85, zorder=3)

    # Highlight SceneFactory bar
    bars[1].set_alpha(1.0)
    bars[1].set_linewidth(1.2)

    ax.set_xscale('log')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel('Throughput (agent steps / second)')
    ax.set_xlim(5000, 5_000_000)
    ax.invert_yaxis()

    # Value annotations
    for i, (bar, val) in enumerate(zip(bars, values)):
        if val >= 100_000:
            txt = f'{val/1000:.0f}K'
        else:
            txt = f'{val:,.0f}'
        ax.text(val * 1.3, i, txt, va='center', ha='left',
                fontsize=7, fontweight='bold',
                color=colors[i],
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # Physics model annotation
    ax.axvline(19250, color=C_GRAY, linewidth=0.5, linestyle='--', alpha=0.4, zorder=1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True, which='major', linewidth=0.3, alpha=0.3, color=C_GRAY)

    fig.tight_layout()
    _save(fig, 'fig_cross_simulator')


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 3 — Generalization Study  (§4.3, Table 3)                  ║
# ║                                                                     ║
# ║  Prompt:                                                            ║
# ║  "Grouped bar chart with 3 evaluation conditions on the x-axis:    ║
# ║  (1) Test scenes / Original OD, (2) Test scenes / Random OD,       ║
# ║  (3) Train scenes / Random OD. For each condition, show 4 grouped  ║
# ║  bars: Success rate (blue, tall), Collision rate (orange, short),   ║
# ║  Lane-forbidden rate (purple, short), and mean goal distance        ║
# ║  (green, on a secondary y-axis as diamond markers with connecting   ║
# ║  line). The success rate bars should be prominently taller. Add     ║
# ║  value labels atop each bar. Use a light horizontal dashed line at  ║
# ║  the 90% mark to highlight near-ceiling performance. Dual y-axes:  ║
# ║  left = 'Rate (%)' [0–100], right = 'Mean goal distance (m)'       ║
# ║  [0–12]. Figure width = 5.5 in (NeurIPS full-width). Serif fonts,  ║
# ║  colorblind-safe palette."                                          ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def fig3_generalization():
    print("Figure 3: Generalization study")

    conditions = ['Test scenes\nOriginal OD', 'Test scenes\nRandom OD', 'Train scenes\nRandom OD']
    success    = [71.6, 92.8, 92.5]
    collision  = [6.3,  4.5,  3.5]
    lane_forb  = [9.0,  6.9,  7.6]
    goal_dist  = [10.1, 3.7,  3.7]

    x = np.arange(len(conditions))
    w = 0.2  # bar width

    fig, ax1 = plt.subplots(figsize=(4.5, 2.8))

    # Bar groups
    b1 = ax1.bar(x - w, success,   w, color=C_BLUE,   alpha=0.85, label='Success ↑', zorder=3, edgecolor='white', linewidth=0.5)
    b2 = ax1.bar(x,     collision, w, color=C_ORANGE,  alpha=0.85, label='Collision ↓', zorder=3, edgecolor='white', linewidth=0.5)
    b3 = ax1.bar(x + w, lane_forb, w, color=C_PURPLE,  alpha=0.85, label='Lane-forbidden ↓', zorder=3, edgecolor='white', linewidth=0.5)

    # Value labels on bars
    for bars_group, vals in [(b1, success), (b2, collision), (b3, lane_forb)]:
        for bar, val in zip(bars_group, vals):
            ypos = bar.get_height() + 1.0
            ax1.text(bar.get_x() + bar.get_width() / 2, ypos,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=6.5, fontweight='medium')

    ax1.set_ylabel('Rate (%)')
    ax1.set_ylim(0, 105)
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, fontsize=8)
    ax1.axhline(90, color=C_GRAY, linewidth=0.5, linestyle='--', alpha=0.5, zorder=1)
    ax1.text(2.35, 91, '90%', fontsize=6, color=C_GRAY, va='bottom')

    # Secondary y-axis: goal distance
    ax2 = ax1.twinx()
    ax2.plot(x, goal_dist, 'D-', color=C_GREEN, markersize=6, linewidth=1.5,
             markeredgecolor='white', markeredgewidth=0.8, label=r'$\bar{d}_\mathrm{goal}$ ↓', zorder=6)
    for i, d in enumerate(goal_dist):
        ax2.annotate(f'{d:.1f} m', xy=(x[i], d), xytext=(6, 4),
                     textcoords='offset points', fontsize=6.5, color=C_GREEN, fontweight='bold',
                     path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    ax2.set_ylabel(r'Mean goal distance $\bar{d}_\mathrm{goal}$ (m)', color=C_GREEN)
    ax2.set_ylim(0, 14)
    ax2.tick_params(axis='y', colors=C_GREEN)
    ax2.spines['right'].set_color(C_GREEN)

    # Trim spines
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # Combined legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2,
               loc='upper right', ncol=2, fontsize=7, borderpad=0.3,
               columnspacing=0.8, handletextpad=0.4)

    fig.tight_layout()
    _save(fig, 'fig_generalization')


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 4 — Training Curve  (§4.3, TODO)                           ║
# ║                                                                     ║
# ║  Prompt:                                                            ║
# ║  "Training curve showing mean episode reward (or success rate)      ║
# ║  vs. PPO iteration for the best SceneFactory run (v6_patient,       ║
# ║  ~975 iterations). If TensorBoard CSV export is available, load     ║
# ║  it; otherwise use synthetic data with the correct scale. Plot a    ║
# ║  bold blue line for the running mean (window=20) with a            ║
# ║  translucent band for per-iteration raw values. Mark the selected  ║
# ║  checkpoint (model_975) with a vertical dashed line and star       ║
# ║  marker. X-axis: 'PPO iteration', Y-axis: 'Mean episode reward'.   ║
# ║  Add a secondary x-axis on top showing approximate total env        ║
# ║  steps (iteration × 4096 × 128). Include inset text box:          ║
# ║  '4,096 agents × 128 steps/iter ≈ 524K transitions/update'.       ║
# ║  Figure width = 3.25 in (half-column). Serif, minimal spines."    ║
# ║                                                                     ║
# ║  NOTE: This generates a TEMPLATE figure. Replace `reward_data`      ║
# ║  with actual TensorBoard export for the final version.              ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def fig4_training_curve():
    print("Figure 4: Training curve (template — replace with real data)")

    # ── Try loading real TensorBoard CSV export if available ──
    tb_csv = OUTDIR.parent / "data" / "training_curve_v6_patient.csv"
    if tb_csv.exists():
        import csv
        iters, rewards = [], []
        with open(tb_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                iters.append(float(row['Step']))
                rewards.append(float(row['Value']))
        iters = np.array(iters)
        rewards = np.array(rewards)
        print(f"  Loaded real data from {tb_csv} ({len(iters)} points)")
    else:
        # Synthetic placeholder — mimics typical PPO learning curve
        print(f"  ⚠ No real data at {tb_csv}; using synthetic placeholder.")
        print(f"    Export from TensorBoard: tensorboard → select run → download CSV")
        np.random.seed(42)
        iters = np.arange(0, 1000)
        # Saturating curve with noise
        base = 12.0 * (1 - np.exp(-iters / 200)) - 3.0
        noise = np.random.normal(0, 0.8, len(iters))
        rewards = base + noise

    # Smoothed curve (EMA)
    window = 20
    kernel = np.ones(window) / window
    if len(rewards) > window:
        smoothed = np.convolve(rewards, kernel, mode='valid')
        smooth_iters = iters[window-1:]
    else:
        smoothed = rewards
        smooth_iters = iters

    fig, ax = plt.subplots(figsize=(3.4, 2.4))

    # Raw data (translucent)
    ax.plot(iters, rewards, linewidth=0.3, alpha=0.25, color=C_BLUE, zorder=2)
    # Smoothed
    ax.plot(smooth_iters, smoothed, linewidth=1.6, color=C_BLUE, zorder=4,
            label=f'Mean reward (EMA-{window})')

    # Checkpoint marker
    ckpt_iter = 975
    if ckpt_iter < len(rewards):
        ckpt_reward = smoothed[min(ckpt_iter - window + 1, len(smoothed) - 1)] if ckpt_iter >= window else rewards[ckpt_iter]
    else:
        ckpt_reward = smoothed[-1]
    ax.axvline(ckpt_iter, color=C_ORANGE, linewidth=0.8, linestyle='--', alpha=0.7, zorder=3)
    ax.plot(ckpt_iter, ckpt_reward, '*', color=C_ORANGE, markersize=10,
            markeredgecolor='white', markeredgewidth=0.6, zorder=6,
            label=f'Selected checkpoint (iter {ckpt_iter})')

    ax.set_xlabel('PPO iteration')
    ax.set_ylabel('Mean episode reward')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, which='major', linewidth=0.3, alpha=0.3, color=C_GRAY)

    # Top axis: total env steps
    ax_top = ax.twiny()
    ax_top.set_xlim([x * 4096 * 128 / 1e6 for x in ax.get_xlim()])
    ax_top.set_xlabel('Total env steps (M)', fontsize=8, labelpad=4)
    ax_top.tick_params(labelsize=7)
    ax_top.spines['top'].set_linewidth(0.4)
    ax_top.spines['right'].set_visible(False)

    # Info box
    info = '4,096 agents × 128 steps/iter\n≈ 524K transitions/update'
    ax.text(0.03, 0.97, info, transform=ax.transAxes,
            fontsize=6, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=C_GRAY, alpha=0.8, linewidth=0.5))

    ax.legend(loc='lower right', fontsize=7, borderpad=0.3)
    fig.tight_layout()
    _save(fig, 'fig_training_curve')


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 5 — Per-World Success Distribution  (§4.3 supplement)       ║
# ║                                                                     ║
# ║  Prompt:                                                            ║
# ║  "Combined strip plot + histogram showing per-world success rates   ║
# ║  across the 64 held-out test scenes (original OD condition). Each   ║
# ║  dot represents one test world, x-axis = success rate [0–1],       ║
# ║  jittered vertically for visibility. Below the strip, overlay a     ║
# ║  KDE density curve. Color dots by number of spawned agents (use a   ║
# ║  sequential blue colormap, lighter = fewer agents). Add a vertical  ║
# ║  dashed line at the aggregate mean (71.6%) with annotation. Mark    ║
# ║  the worst 5 worlds with red edge rings. Figure width = 3.25 in.   ║
# ║  Minimal axes, serif fonts, NeurIPS style."                         ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def fig5_per_world_success():
    print("Figure 5: Per-world success distribution")

    # ── Try loading real eval data ──
    summary_paths = sorted(pathlib.Path(
        '/home/yz8733/Github/isaac-rl/logs/rsl_rl/scene_factory_goal_reaching_eval'
    ).glob('**/scene_factory_policy_eval_summary.json'))

    # Find the v6_patient test64 (original OD) run
    real_data = None
    for p in summary_paths:
        if 'v6_patient' in str(p) and 'random_od' not in str(p) and 'train' not in str(p):
            try:
                with open(p) as f:
                    data = json.load(f)
                if abs(data.get('success_rate', 0) - 0.7163) < 0.01:
                    real_data = data
                    print(f"  Loaded real eval data from {p}")
                    break
            except Exception:
                continue

    if real_data and 'per_world_summary_sorted_by_least_success' in real_data:
        worlds = real_data['per_world_summary_sorted_by_least_success']
        success_rates = np.array([w['success_rate'] for w in worlds])
        spawned_counts = np.array([w['spawned_count'] for w in worlds])
        agg_success = real_data['success_rate']
    else:
        print("  ⚠ Using synthetic data; update path to real eval JSON.")
        np.random.seed(42)
        success_rates = np.clip(np.random.beta(3.5, 1.5, 64), 0, 1)
        spawned_counts = np.random.randint(2, 17, 64)
        agg_success = 0.716

    fig, ax = plt.subplots(figsize=(3.4, 2.2))

    # Strip plot
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(success_rates))
    norm = plt.Normalize(vmin=spawned_counts.min(), vmax=spawned_counts.max())
    cmap = plt.cm.Blues

    scatter = ax.scatter(success_rates, jitter, c=spawned_counts, cmap=cmap, norm=norm,
                         s=25, alpha=0.8, edgecolors='white', linewidths=0.4, zorder=4)

    # Highlight worst 5
    worst_idx = np.argsort(success_rates)[:5]
    ax.scatter(success_rates[worst_idx], jitter[worst_idx],
               s=50, facecolors='none', edgecolors=C_ORANGE, linewidths=1.2, zorder=5)

    # KDE-like histogram at bottom
    from matplotlib.colors import to_rgba
    bins = np.linspace(0, 1, 20)
    counts, edges = np.histogram(success_rates, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Normalize to fit in lower portion
    counts_norm = counts / counts.max() * 0.3
    ax.bar(centers, -counts_norm, width=edges[1] - edges[0],
           bottom=-0.05, color=C_BLUE, alpha=0.3, edgecolor='none', zorder=2)

    # Aggregate mean line
    ax.axvline(agg_success, color=C_ORANGE, linewidth=1.0, linestyle='--', zorder=3)
    ax.text(agg_success + 0.02, 0.28, f'Mean: {agg_success*100:.1f}%',
            fontsize=7, color=C_ORANGE, fontweight='bold', va='center',
            path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    ax.set_xlabel('Per-world success rate')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.4, 0.35)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=15, pad=0.02)
    cbar.set_label('Spawned agents', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    fig.tight_layout()
    _save(fig, 'fig_per_world_success')


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 6 — Timing Breakdown  (§4.2 / Appendix)                    ║
# ║                                                                     ║
# ║  Prompt:                                                            ║
# ║  "Side-by-side stacked horizontal bar chart comparing per-step      ║
# ║  timing breakdown (ms) between Baseline and SceneFactory at 64      ║
# ║  worlds. Categories (bottom to top): Physics, Observations,         ║
# ║  Reward/Done, Action, Other. Use distinct muted colors for each     ║
# ║  category. Annotate total wall-clock time per step on the right     ║
# ║  edge. X-axis: 'Time per step (ms)'. Figure width = 3.25 in.       ║
# ║  This reveals that the Baseline bottleneck is Python-side loops     ║
# ║  (Observations + Reward), not physics."                             ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def fig6_timing_breakdown():
    print("Figure 6: Timing breakdown (64 worlds)")

    categories = ['Physics', 'Observations', 'Reward/Done', 'Action', 'Other']
    # Baseline @ 64 worlds (ms per 30-Hz step)
    bl_times = np.array([2.93, 1.46, 0.32, 0.10, 1.50])  # ~6.31 ms total
    # SceneFactory @ 64 worlds (ms per 30-Hz step)
    sf_times = np.array([0.054, 0.023, 0.012, 0.025, 0.025])  # ~0.139 ms total

    colors = [C_BLUE, C_SKYBLUE, C_VERMILLION, C_GREEN, C_GRAY]

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.8), sharey=True,
                              gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.15})

    for ax, times, title in [(axes[0], bl_times, 'Baseline'),
                              (axes[1], sf_times, 'SceneFactory')]:
        left = 0
        for i, (cat, t) in enumerate(zip(categories, times)):
            bar = ax.barh(0, t, left=left, height=0.5, color=colors[i],
                          edgecolor='white', linewidth=0.5, label=cat if ax == axes[0] else None)
            # Label inside bar if wide enough
            if t / times.sum() > 0.12:
                ax.text(left + t / 2, 0, f'{t:.2f}' if t < 1 else f'{t:.1f}',
                        ha='center', va='center', fontsize=6, color='white', fontweight='bold')
            left += t

        total = times.sum()
        ax.text(total + total * 0.03, 0, f'Σ = {total:.2f} ms',
                va='center', ha='left', fontsize=7, fontweight='bold', color=C_DARKGRAY)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xlim(0, max(bl_times.sum(), sf_times.sum()) * 1.25 if ax == axes[0]
                     else sf_times.sum() * 1.6)
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xlabel('Time per step (ms)', fontsize=8)

    axes[0].legend(loc='upper right', fontsize=6.5, ncol=1, borderpad=0.3,
                   handlelength=1.0, handletextpad=0.3)
    fig.tight_layout()
    _save(fig, 'fig_timing_breakdown')


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                               ║
# ╚═══════════════════════════════════════════════════════════════════════╝

FIGURES = {
    1: ("Throughput Scaling",       fig1_throughput_scaling),
    2: ("Cross-Simulator Comparison", fig2_cross_simulator),
    3: ("Generalization Study",      fig3_generalization),
    4: ("Training Curve",            fig4_training_curve),
    5: ("Per-World Success Dist.",   fig5_per_world_success),
    6: ("Timing Breakdown",          fig6_timing_breakdown),
}

def main():
    parser = argparse.ArgumentParser(description="Generate NeurIPS 2026 experiment figures")
    parser.add_argument('--only', type=int, nargs='+', default=None,
                        help='Generate only specific figure numbers (1-6)')
    args = parser.parse_args()

    targets = args.only if args.only else sorted(FIGURES.keys())
    print(f"═══ SceneFactory NeurIPS Figures ═══")
    print(f"Output directory: {OUTDIR}\n")

    for idx in targets:
        if idx not in FIGURES:
            print(f"⚠ Unknown figure number {idx}, skipping.")
            continue
        name, func = FIGURES[idx]
        print(f"── [{idx}/{len(FIGURES)}] {name} ──")
        func()
        print()

    print("Done. Import into LaTeX with:")
    print(r'  \includegraphics[width=\columnwidth]{figures/fig_<name>.pdf}')

if __name__ == '__main__':
    main()
