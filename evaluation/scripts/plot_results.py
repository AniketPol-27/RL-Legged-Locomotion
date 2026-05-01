"""Generate plots for the final report."""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

CSV_PATH = "evaluation/data/eval_results.csv"
OUT_DIR = "evaluation/plots"


def main():
    if not os.path.exists(CSV_PATH):
        print(f"❌ {CSV_PATH} not found. Run inference in Unity first.")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    # ---- Plot 1: Distance per trial ----
    plt.figure(figsize=(11, 5))
    colors = ['#E74C3C' if f else '#27AE60' for f in df['fell']]
    plt.bar(df['episode'], df['distance_m'], color=colors, edgecolor='black', linewidth=0.5)
    plt.axhline(5.0, color='blue', linestyle='--', linewidth=2, label='Target (5m)')
    plt.xlabel('Trial #', fontsize=12)
    plt.ylabel('Distance (m)', fontsize=12)
    plt.title('Distance Traveled per Evaluation Trial', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/distance_per_trial.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {OUT_DIR}/distance_per_trial.png")

    # ---- Plot 2: Reward distribution ----
    plt.figure(figsize=(9, 5))
    plt.hist(df['reward'], bins=20, color='#3498DB', edgecolor='black')
    plt.axvline(df['reward'].mean(), color='red', linestyle='--', linewidth=2,
                label=f"Mean: {df['reward'].mean():.2f}")
    plt.xlabel('Episodic Reward', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Reward Distribution Across Trials', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/reward_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {OUT_DIR}/reward_distribution.png")

    # ---- Plot 3: Stability vs distance scatter ----
    plt.figure(figsize=(9, 6))
    sc = plt.scatter(df['upright_time_s'], df['distance_m'],
                     c=df['fell'], cmap='RdYlGn_r', s=80, edgecolor='black', alpha=0.8)
    plt.colorbar(sc, label='Fell (1=yes)')
    plt.xlabel('Upright Time (s)', fontsize=12)
    plt.ylabel('Distance (m)', fontsize=12)
    plt.title('Stability vs. Performance', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/stability_vs_distance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {OUT_DIR}/stability_vs_distance.png")

    # ---- Plot 4: Success pie chart ----
    fig, ax = plt.subplots(figsize=(7, 7))
    succeeded = int((df['distance_m'] >= 5.0).sum())
    failed = len(df) - succeeded
    ax.pie([succeeded, failed],
           labels=[f'Success ({succeeded})', f'Failed ({failed})'],
           autopct='%1.1f%%',
           colors=['#27AE60', '#E74C3C'],
           startangle=90,
           textprops={'fontsize': 13, 'fontweight': 'bold'})
    ax.set_title('Success Rate (Distance ≥ 5m)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/success_pie.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {OUT_DIR}/success_pie.png")

    # ---- Plot 5: Distance over trials (line) ----
    plt.figure(figsize=(11, 5))
    plt.plot(df['episode'], df['distance_m'], marker='o', linewidth=2, color='#2C3E50')
    plt.axhline(5.0, color='blue', linestyle='--', label='Target (5m)')
    plt.fill_between(df['episode'], 0, df['distance_m'], alpha=0.2)
    plt.xlabel('Trial #', fontsize=12)
    plt.ylabel('Distance (m)', fontsize=12)
    plt.title('Performance Trend Across Trials', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/distance_trend.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {OUT_DIR}/distance_trend.png")

    print(f"\n🎨 All plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()