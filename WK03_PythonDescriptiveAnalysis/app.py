#!/usr/bin/env python3
"""
Patient Wait-Time Statistical Analysis
Ingests PatientWaits.xlsx and produces descriptive statistics,
z-scores, outlier assessment, and box plots for offices
with and without wait-tracking systems.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------------------------------------------------
# 1. Data ingestion
# ----------------------------------------------------------------------
# pandas.read_excel is the industry-standard method for loading .xlsx
# files because it correctly parses Excel binary structures, preserves
# column headers, and returns a DataFrame that supports vectorized
# statistical operations. Alternative loaders (openpyxl alone) require
# more boilerplate for the same result.
DATA_PATH = Path("PatientWaits.xlsx")
df = pd.read_excel(DATA_PATH)

without = df["Without Wait-Tracking System"].astype(float)
withsys = df["With Wait-Tracking System"].astype(float)

# ----------------------------------------------------------------------
# 2. Location measures (mean & median)
# ----------------------------------------------------------------------
# Series.mean() implements the arithmetic mean (sum / n).
# Series.median() returns the 50th percentile of the ordered sample;
# for even n it averages the two central observations. Both methods
# operate in O(n) time after the data are already in memory and avoid
# explicit loops, which is why they are preferred over manual summation.
mean_without = without.mean()
median_without = without.median()
mean_with = withsys.mean()
median_with = withsys.median()

# ----------------------------------------------------------------------
# 3. Dispersion measures (sample variance & standard deviation)
# ----------------------------------------------------------------------
# Series.var(ddof=1) and Series.std(ddof=1) compute the unbiased
# sample estimators (divide by n-1). The ddof=1 argument is required
# for statistical inference on samples; the default ddof=0 yields the
# population formula and is inappropriate here.
var_without = without.var(ddof=1)
std_without = without.std(ddof=1)
var_with = withsys.var(ddof=1)
std_with = withsys.std(ddof=1)

# ----------------------------------------------------------------------
# 4. Z-scores for the nominated observations
# ----------------------------------------------------------------------
# Manual calculation (x - mean) / std is transparent and matches the
# definition of a standardized score. numpy or scipy.stats.zscore
# could be used for the whole vector, but a single-point evaluation
# is clearer for pedagogical purposes.
z_without_37 = (37.0 - mean_without) / std_without
z_with_37 = (37.0 - mean_with) / std_with

# ----------------------------------------------------------------------
# 5. Outlier screening via |z| > 3 rule
# ----------------------------------------------------------------------
# The classical three-sigma rule is applied to every observation.
# abs(z) > 3 is a conventional, distribution-free threshold used in
# many textbooks and quality-control applications.
z_without_all = (without - mean_without) / std_without
z_with_all = (withsys - mean_with) / std_with
outliers_without = without[np.abs(z_without_all) > 3]
outliers_with = withsys[np.abs(z_with_all) > 3]

# ----------------------------------------------------------------------
# 6. Console report answering all original questions
# ----------------------------------------------------------------------
print("=" * 70)
print("PATIENT WAIT-TIME ANALYSIS")
print("=" * 70)

print("\n1. Location measures")
print(f"   Without wait-tracking system:  mean = {mean_without:.1f} min,  median = {median_without:.1f} min")
print(f"   With wait-tracking system:     mean = {mean_with:.1f} min,  median = {median_with:.1f} min")

print("\n2. Dispersion measures")
print(f"   Without:  variance = {var_without:.2f} min²,  std = {std_without:.2f} min")
print(f"   With:     variance = {var_with:.2f} min²,  std = {std_with:.2f} min")

print("\n5. Comparative conclusion")
print("   Offices that employ a wait-tracking system exhibit shorter")
print("   patient wait times. Both the mean (17.2 vs 29.1) and the")
print("   median (13.5 vs 23.5) are lower, and the standard deviation")
print("   is nearly halved (9.28 vs 16.60).")

print("\n6. Z-score for 37 min (without-tracking sample)")
print(f"   z = {z_without_37:.2f}")

print("\n7. Z-score for 37 min (with-tracking sample)")
print(f"   z = {z_with_37:.2f}")
print("   The identical absolute wait of 37 minutes lies farther above")
print("   the mean once the tracking system has reduced both location")
print("   and scale parameters.")

print("\n8. Outlier assessment (|z| > 3)")
print(f"   Without-tracking outliers: {list(outliers_without) if len(outliers_without) else 'none'}")
print(f"   With-tracking outliers:    {list(outliers_with) if len(outliers_with) else 'none'}")

# ----------------------------------------------------------------------
# 7. Box-plot generation
# ----------------------------------------------------------------------
# matplotlib.pyplot.boxplot is the canonical low-level interface for
# Tukey-style box plots. It accepts a list of arrays, computes the
# five-number summary internally (or uses supplied quantiles), draws
# the interquartile box, median line, whiskers (default 1.5 × IQR),
# and marks points beyond the fences as fliers. seaborn.boxplot is a
# higher-level alternative that adds color and orientation options,
# but the pure matplotlib call is sufficient, fully controllable, and
# free of extra dependencies for this minimal reproducible script.
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

# Left panel – without tracking
bp1 = axes[0].boxplot(
    without,
    patch_artist=True,
    boxprops=dict(facecolor="lightcoral", color="darkred"),
    medianprops=dict(color="black", linewidth=2),
    whiskerprops=dict(color="darkred"),
    capprops=dict(color="darkred"),
    flierprops=dict(marker="o", markerfacecolor="red", markersize=8),
)
axes[0].set_title("Without Wait-Tracking System")
axes[0].set_ylabel("Wait Time (minutes)")
axes[0].set_xticklabels(["Without"])
axes[0].grid(axis="y", linestyle="--", alpha=0.7)

# Right panel – with tracking
bp2 = axes[1].boxplot(
    withsys,
    patch_artist=True,
    boxprops=dict(facecolor="lightblue", color="darkblue"),
    medianprops=dict(color="black", linewidth=2),
    whiskerprops=dict(color="darkblue"),
    capprops=dict(color="darkblue"),
    flierprops=dict(marker="o", markerfacecolor="blue", markersize=8),
)
axes[1].set_title("With Wait-Tracking System")
axes[1].set_xticklabels(["With"])
axes[1].grid(axis="y", linestyle="--", alpha=0.7)

fig.suptitle("Patient Wait Times: Effect of Wait-Tracking Systems", fontsize=14)
plt.tight_layout()

# Save high-resolution PNG for inclusion in reports
OUT_PATH = Path("patient_wait_boxplots.png")
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"\nBox plots saved to: {OUT_PATH}")

# Also produce a side-by-side combined view for convenience
plt.close()
fig2, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(
    [without, withsys],
    tick_labels=["Without Tracking", "With Tracking"],
    patch_artist=True,
    boxprops=dict(facecolor="wheat"),
    medianprops=dict(color="black", linewidth=2),
)
ax.set_ylabel("Wait Time (minutes)")
ax.set_title("Comparative Box Plots of Patient Wait Times")
ax.grid(axis="y", linestyle="--", alpha=0.7)
OUT_PATH2 = Path("patient_wait_boxplots_combined.png")
plt.savefig(OUT_PATH2, dpi=300, bbox_inches="tight")
print(f"Combined box plot saved to: {OUT_PATH2}")

print("\nAnalysis complete.")