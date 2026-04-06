#!/usr/bin/env python3
"""Plot Experiment A throughput scaling results."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────
worlds = np.array([32, 64, 128, 256])
slots  = worlds * 16

baseline_mean = np.array([174.3, 158.6, 163.8, 152.3])
baseline_std  = np.array([26.6,  29.4,   1.0,   0.4])

sf_mean = np.array([3869.8, 7172.9, 12225.3, 19249.7])
sf_std  = np.array([  66.0,  113.9,  1519.9,  2984.0])

speedup = sf_mean / baseline_mean

# ── Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

BLUE   = "#2563EB"
ORANGE = "#E45932"
GREEN  = "#16A34A"
GRAY   = "#6B7280"

# =====================================================================
# Figure 1: CASPS scaling  (log-scale y)
# =====================================================================
fig, ax = plt.subplots(figsize=(6, 4))

ax.errorbar(slots, sf_mean, yerr=sf_std, fmt="o-", color=BLUE,
            linewidth=2, markersize=7, capsize=4, capthick=1.5,
            label="SceneFactory", zorder=3)
ax.errorbar(slots, baseline_mean, yerr=baseline_std, fmt="s--", color=ORANGE,
            linewidth=2, markersize=7, capsize=4, capthick=1.5,
            label="Baseline (Vehicle Wizard + SB3)", zorder=3)

# ideal linear scaling reference from SF 32w
ideal = sf_mean[0] * (slots / slots[0])
ax.plot(slots, ideal, ":", color=GRAY, linewidth=1.2, alpha=0.6,
        label="Ideal linear scaling")

ax.set_yscale("log")
ax.set_xscale("log", base=2)
ax.set_xticks(slots)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f"{int(x):,}"))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f"{int(x):,}"))

ax.set_xlabel("Agent slots (worlds × 16)")
ax.set_ylabel("CASPS (controlled agent steps / s)")
ax.set_title("Throughput Scaling — Single RTX PRO 6000 Blackwell")
ax.legend(loc="upper left", framealpha=0.9)
ax.grid(True, which="both", ls=":", alpha=0.3)

# annotate speedup on each point
for i, (x, ym) in enumerate(zip(slots, sf_mean)):
    ax.annotate(f"{speedup[i]:.0f}×",
                xy=(x, ym), xytext=(8, -4),
                textcoords="offset points", fontsize=9.5,
                color=BLUE, fontweight="bold")

fig.tight_layout()
fig.savefig("experiments/experiment_A/fig_throughput_scaling.pdf")
fig.savefig("experiments/experiment_A/fig_throughput_scaling.png")
print("Saved fig_throughput_scaling.pdf / .png")
plt.close(fig)

# =====================================================================
# Figure 2: Speedup bar chart
# =====================================================================
fig2, ax2 = plt.subplots(figsize=(5, 3.5))

bars = ax2.bar([f"{s:,}" for s in slots], speedup, color=BLUE,
               edgecolor="white", linewidth=0.8, width=0.6)
for bar, sp in zip(bars, speedup):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
             f"{sp:.0f}×", ha="center", va="bottom", fontsize=11,
             fontweight="bold", color=BLUE)

ax2.set_xlabel("Agent slots")
ax2.set_ylabel("Speedup (SceneFactory / Baseline)")
ax2.set_title("SceneFactory Speedup over Baseline")
ax2.set_ylim(0, max(speedup) * 1.2)
ax2.grid(axis="y", ls=":", alpha=0.3)

fig2.tight_layout()
fig2.savefig("experiments/experiment_A/fig_speedup.pdf")
fig2.savefig("experiments/experiment_A/fig_speedup.png")
print("Saved fig_speedup.pdf / .png")
plt.close(fig2)

# =====================================================================
# Figure 3: Per-step timing breakdown (stacked bar, both pipelines @ 64w)
# =====================================================================
# Baseline @ 64w (from TensorBoard expA_baseline_64w)
bl_physics   = 2933.8
bl_obs       = 1455.5
bl_geom_lane = 316.0
bl_ttc_road  = 236.4
bl_ttc_veh   = 190.6
bl_apply     = 299.0
bl_ttc_state = 136.2
bl_contact   = 65.8
bl_other     = 5682.7 - (bl_physics + bl_obs + bl_geom_lane + bl_ttc_road
                         + bl_ttc_veh + bl_apply + bl_ttc_state + bl_contact)

# SceneFactory @ 64w (from TensorBoard expA_sf_64w)
sf_physics = 33.0 + 15.9 + 4.9          # write + sim + update
sf_obs     = 23.3
sf_action  = 24.9
sf_reward  = 18.4
sf_done    = 11.5
sf_reset   = 32.4
sf_other_s = 135.0 - (sf_physics + sf_obs + sf_action + sf_reward
                       + sf_done + sf_reset)
# clamp negative
sf_other_s = max(sf_other_s, 0)

# Category mapping (simplified for clarity)
categories = ["Physics", "Observations", "Reward/Done", "Action", "Other"]

bl_vals = [bl_physics,
           bl_obs + bl_geom_lane,
           bl_ttc_road + bl_ttc_veh + bl_ttc_state + bl_contact,
           bl_apply,
           max(bl_other, 0)]
sf_vals = [sf_physics, sf_obs, sf_reward + sf_done, sf_action,
           sf_reset + sf_other_s]

colors = ["#3B82F6", "#F59E0B", "#10B981", "#8B5CF6", "#9CA3AF"]

fig3, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=False)

# Baseline
bottom = 0
for cat, val, col in zip(categories, bl_vals, colors):
    axes[0].barh(0, val, left=bottom, color=col, edgecolor="white",
                 height=0.5, label=cat)
    if val > 150:
        axes[0].text(bottom + val / 2, 0, f"{val:.0f}",
                     ha="center", va="center", fontsize=8.5, color="white",
                     fontweight="bold")
    bottom += val
axes[0].set_xlim(0, sum(bl_vals) * 1.05)
axes[0].set_xlabel("Time per env step (ms)")
axes[0].set_title(f"Baseline @ 64w\n({sum(bl_vals):.0f} ms total)", fontsize=11)
axes[0].set_yticks([])

# SceneFactory
bottom = 0
for cat, val, col in zip(categories, sf_vals, colors):
    axes[1].barh(0, val, left=bottom, color=col, edgecolor="white",
                 height=0.5, label=cat)
    if val > 8:
        axes[1].text(bottom + val / 2, 0, f"{val:.0f}",
                     ha="center", va="center", fontsize=8.5, color="white",
                     fontweight="bold")
    bottom += val
axes[1].set_xlim(0, sum(sf_vals) * 1.05)
axes[1].set_xlabel("Time per env step (ms)")
axes[1].set_title(f"SceneFactory @ 64w\n({sum(sf_vals):.0f} ms total)",
                  fontsize=11)
axes[1].set_yticks([])

# shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig3.legend(handles, labels, loc="lower center", ncol=5,
            fontsize=9.5, bbox_to_anchor=(0.5, -0.02))

fig3.suptitle("Per-Step Timing Breakdown (64 worlds, 1,024 agent slots)",
              fontsize=12, y=1.02)
fig3.tight_layout()
fig3.savefig("experiments/experiment_A/fig_timing_breakdown.pdf",
             bbox_inches="tight")
fig3.savefig("experiments/experiment_A/fig_timing_breakdown.png",
             bbox_inches="tight")
print("Saved fig_timing_breakdown.pdf / .png")
plt.close(fig3)

# =====================================================================
# Figure 4: CASPS linear scale (side by side bars)
# =====================================================================
fig4, ax4 = plt.subplots(figsize=(6, 4))

x = np.arange(len(slots))
w = 0.35

bars1 = ax4.bar(x - w/2, baseline_mean, w, yerr=baseline_std,
                color=ORANGE, capsize=3, label="Baseline", edgecolor="white")
bars2 = ax4.bar(x + w/2, sf_mean, w, yerr=sf_std,
                color=BLUE, capsize=3, label="SceneFactory", edgecolor="white")

ax4.set_xticks(x)
ax4.set_xticklabels([f"{s:,}" for s in slots])
ax4.set_xlabel("Agent slots (worlds × 16)")
ax4.set_ylabel("CASPS")
ax4.set_title("Throughput Comparison — Linear Scale")
ax4.legend()
ax4.grid(axis="y", ls=":", alpha=0.3)

# annotate SF bar values
for i, bar in enumerate(bars2):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + sf_std[i] + 300,
             f"{sf_mean[i]:,.0f}", ha="center", va="bottom", fontsize=8.5,
             color=BLUE, fontweight="bold")

fig4.tight_layout()
fig4.savefig("experiments/experiment_A/fig_throughput_linear.pdf")
fig4.savefig("experiments/experiment_A/fig_throughput_linear.png")
print("Saved fig_throughput_linear.pdf / .png")
plt.close(fig4)

print("\nAll plots generated successfully!")
print(f"\nSummary table:")
print(f"{'Worlds':>6} {'Slots':>6} {'Baseline':>12} {'SceneFactory':>16} {'Speedup':>8}")
print("-" * 52)
for i in range(len(worlds)):
    print(f"{worlds[i]:>6} {slots[i]:>6} {baseline_mean[i]:>8.1f}±{baseline_std[i]:<4.1f}"
          f"{sf_mean[i]:>10.1f}±{sf_std[i]:<7.1f} {speedup[i]:>6.0f}×")
