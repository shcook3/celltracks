"""
TrackMate _allspots.csv -> migration metrics + representative squiggle

Outputs:
    1. <name>_track_metrics.csv
    2. <name>_representative_track.csv
    3. <name>_speed_colored_tracks.png
    4. <name>_representative_track.svg
    5. <name>_representative_track.png

The script is designed for TrackMate *_allspots.csv files exported in the
format shown in the example file.

Speed colors are EDITABLE below:
    0-20 um/min   = green
    20-40         = blue
    40-60         = orange
    >=60          = red

The representative track is the REAL track closest to the group mean for:
    - net displacement
    - total path length
    - mean speed

The three metrics are z-scored first so their different numerical scales
do not bias the selection.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# USER SETTINGS
# ============================================================

INPUT_FILE = r"/Users/shcook/Desktop/newtracks/atma16-oldb-m1-l1-s2-v1_allspots.csv"

# Speed bins in um/min.
SPEED_BINS = [0, 10, 20, 30, 40, 50, np.inf]

SPEED_COLORS = [
    
 "#AAFF00",  # 0–10
        "#B478FB",  # 10–20
    "#4A1A84",  # 20–30
    "#59D0FB", # 30–40
  "#008587", # 40–50
        "#FF286F", # 50+
]

# Whether to normalize every track so its first point is (0,0)
NORMALIZE_START = True

# Minimum number of spots required for a track to be analyzed
MIN_SPOTS = 2

# Line widths
ALL_TRACK_LINEWIDTH = 0.9
REPRESENTATIVE_LINEWIDTH = 3.0


# ============================================================
# FUNCTIONS
# ============================================================

def load_trackmate_allspots(filename):
    """
    Load a TrackMate *_allspots.csv file.

    TrackMate exports several metadata/header rows before the actual data.
    We therefore read the file normally and remove rows that cannot be
    converted to numeric values in TRACK_ID / POSITION_X / POSITION_Y.
    """

    df = pd.read_csv(filename)

    # Convert the fields needed for tracking to numeric.
    required = ["TRACK_ID", "POSITION_X", "POSITION_Y", "POSITION_T"]

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove TrackMate metadata/header rows.
    df = df.dropna(subset=required).copy()

    return df


def calculate_track_metrics(df):
    """
    Calculate migration metrics for each TrackMate TRACK_ID.

    Mean speed is calculated as:
        total path length / track duration

    with units of um/min because POSITION_X/Y are in microns and
    POSITION_T is in seconds in the supplied file.
    """

    results = []

    for track_id, g in df.groupby("TRACK_ID"):

        g = g.sort_values("POSITION_T")

        x = g["POSITION_X"].to_numpy(dtype=float)
        y = g["POSITION_Y"].to_numpy(dtype=float)
        t = g["POSITION_T"].to_numpy(dtype=float)

        if len(g) < MIN_SPOTS:
            continue

        dx = np.diff(x)
        dy = np.diff(y)
        dt = np.diff(t)

        # Distance traveled between consecutive spots.
        step_distance = np.sqrt(dx**2 + dy**2)

        # Only use intervals with positive time.
        valid = dt > 0

        total_path_length = step_distance[valid].sum()

        duration_s = t[-1] - t[0]

        # Net displacement from first to last position.
        displacement = np.sqrt(
            (x[-1] - x[0])**2 +
            (y[-1] - y[0])**2
        )

        if duration_s > 0:
            mean_speed_um_min = (
                total_path_length / duration_s * 60.0
            )
        else:
            mean_speed_um_min = np.nan

        if total_path_length > 0:
            straightness = displacement / total_path_length
        else:
            straightness = np.nan

        results.append({
            "TRACK_ID": track_id,
            "N_SPOTS": len(g),
            "DURATION_s": duration_s,
            "TOTAL_PATH_LENGTH_um": total_path_length,
            "DISPLACEMENT_um": displacement,
            "MEAN_SPEED_um_min": mean_speed_um_min,
            "STRAIGHTNESS": straightness
        })

    return pd.DataFrame(results)


def add_speed_bin(metrics):
    """Assign each track to a user-defined speed bin."""

    labels = []

    for i in range(len(SPEED_BINS) - 1):
        low = SPEED_BINS[i]
        high = SPEED_BINS[i + 1]

        if np.isinf(high):
            label = f"{low:g}+ um/min"
        else:
            label = f"{low:g}-{high:g} um/min"

        labels.append(label)

    metrics["SPEED_BIN"] = pd.cut(
        metrics["MEAN_SPEED_um_min"],
        bins=SPEED_BINS,
        labels=labels,
        right=False,
        include_lowest=True
    )

    return metrics


def select_representative_track(metrics):
    """
    Select the actual track closest to the group mean using:
        displacement
        total path length
        mean speed

    Each variable is z-scored first.
    """

    selection_columns = [
        "DISPLACEMENT_um",
        "TOTAL_PATH_LENGTH_um",
        "MEAN_SPEED_um_min"
    ]

    clean = metrics.dropna(subset=selection_columns).copy()

    # Z-score each metric.
    z = (
        clean[selection_columns] -
        clean[selection_columns].mean()
    ) / clean[selection_columns].std(ddof=0)

    # Euclidean distance from the population centroid.
    clean["DISTANCE_FROM_MEAN"] = np.sqrt(
        (z**2).sum(axis=1)
    )

    representative_row = clean.loc[
        clean["DISTANCE_FROM_MEAN"].idxmin()
    ]

    representative_id = representative_row["TRACK_ID"]

    metrics = metrics.copy()
    metrics["DISTANCE_FROM_MEAN"] = np.nan
    metrics.loc[
        metrics["TRACK_ID"] == representative_id,
        "DISTANCE_FROM_MEAN"
    ] = representative_row["DISTANCE_FROM_MEAN"]

    return metrics, representative_id


def get_speed_color(speed):
    """Return the color corresponding to the speed bin."""

    for i in range(len(SPEED_BINS) - 1):

        low = SPEED_BINS[i]
        high = SPEED_BINS[i + 1]

        if low <= speed < high:
            return SPEED_COLORS[i]

    return SPEED_COLORS[-1]


def plot_all_tracks(df, metrics, representative_id, output_file):
    """
    Plot every cell track.

    Each track is colored according to that cell's mean speed.
    The representative track is highlighted with a thicker black outline.
    """

    fig, ax = plt.subplots(figsize=(7, 7))

    for track_id, g in df.groupby("TRACK_ID"):

        row = metrics[metrics["TRACK_ID"] == track_id]

        if row.empty:
            continue

        speed = row["MEAN_SPEED_um_min"].iloc[0]
        color = get_speed_color(speed)

        g = g.sort_values("POSITION_T")

        x = g["POSITION_X"].to_numpy()
        y = g["POSITION_Y"].to_numpy()

        if NORMALIZE_START:
            x = x - x[0]
            y = y - y[0]

        # Representative track gets a black underlay.
        if track_id == representative_id:

            ax.plot(
                x, y,
                color="black",
                linewidth=REPRESENTATIVE_LINEWIDTH + 2,
                alpha=0.9,
                zorder=10
            )

            ax.plot(
                x, y,
                color=color,
                linewidth=REPRESENTATIVE_LINEWIDTH,
                zorder=11
            )

        else:

            ax.plot(
                x, y,
                color=color,
                linewidth=ALL_TRACK_LINEWIDTH,
                alpha=0.45
            )

    # Speed legend.
    legend_handles = []

    for i in range(len(SPEED_BINS) - 1):

        low = SPEED_BINS[i]
        high = SPEED_BINS[i + 1]

        if np.isinf(high):
            label = f"{low:g}+ µm/min"
        else:
            label = f"{low:g}-{high:g} µm/min"

        legend_handles.append(
            Line2D(
                [0], [0],
                color=SPEED_COLORS[i],
                linewidth=3,
                label=label
            )
        )

    legend_handles.append(
        Line2D(
            [0], [0],
            color="black",
            linewidth=5,
            label="Representative track"
        )
    )

    ax.legend(
        handles=legend_handles,
        title="Mean cell speed",
        loc="best",
        frameon=False
    )

    ax.set_xlabel("X displacement (µm)")
    ax.set_ylabel("Y displacement (µm)")
    ax.set_title("Cell migration tracks colored by mean speed")
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(output_file, dpi=600, bbox_inches="tight")
    plt.close()


def plot_representative_track(
    df,
    metrics,
    representative_id,
    output_png,
    output_svg
):
    """Plot only the selected representative track."""

    g = df[df["TRACK_ID"] == representative_id].copy()
    g = g.sort_values("POSITION_T")

    x = g["POSITION_X"].to_numpy()
    y = g["POSITION_Y"].to_numpy()

    if NORMALIZE_START:
        x = x - x[0]
        y = y - y[0]

    row = metrics[metrics["TRACK_ID"] == representative_id].iloc[0]

    speed = row["MEAN_SPEED_um_min"]
    color = get_speed_color(speed)

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot(
        x, y,
        color=color,
        linewidth=3
    )

    ax.scatter(
        x[0], y[0],
        color="black",
        s=35,
        zorder=5
    )

    ax.scatter(
        x[-1], y[-1],
        color="black",
        s=35,
        marker="s",
        zorder=5
    )

    ax.set_xlabel("X displacement (µm)")
    ax.set_ylabel("Y displacement (µm)")

    ax.set_title(
        f"Representative track\n"
        f"Track {int(representative_id)} | "
        f"Mean speed = {speed:.2f} µm/min"
    )

    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()

    plt.savefig(
        output_png,
        dpi=600,
        bbox_inches="tight"
    )

    plt.savefig(
        output_svg,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nCould not find:\n{INPUT_FILE}\n\n"
        "Change INPUT_FILE at the top of the script to the location "
        "of your TrackMate *_allspots.csv file."
    )

print("\nLoading:", INPUT_FILE)

df = load_trackmate_allspots(INPUT_FILE)

print(f"Found {df['TRACK_ID'].nunique()} tracks.")
print(f"Found {len(df)} usable spots.")

# Calculate metrics.
metrics = calculate_track_metrics(df)

# Add speed categories.
metrics = add_speed_bin(metrics)

# Find the real track closest to the population mean.
metrics, representative_id = select_representative_track(metrics)

# Output folder = same folder as input CSV.
output_folder = os.path.dirname(os.path.abspath(INPUT_FILE))
base = os.path.splitext(os.path.basename(INPUT_FILE))[0]

metrics_file = os.path.join(
    output_folder,
    base + "_track_metrics.csv"
)

representative_file = os.path.join(
    output_folder,
    base + "_representative_track.csv"
)

all_tracks_png = os.path.join(
    output_folder,
    base + "_speed_colored_tracks.png"
)

representative_png = os.path.join(
    output_folder,
    base + "_representative_track.png"
)

representative_svg = os.path.join(
    output_folder,
    base + "_representative_track.svg"
)

# Save metrics.
metrics.to_csv(metrics_file, index=False)

# Save representative track coordinates.
representative_track = df[
    df["TRACK_ID"] == representative_id
].sort_values("POSITION_T").copy()

representative_track.to_csv(
    representative_file,
    index=False
)

# Make figures.
plot_all_tracks(
    df,
    metrics,
    representative_id,
    all_tracks_png
)

plot_representative_track(
    df,
    metrics,
    representative_id,
    representative_png,
    representative_svg
)

# Print useful summary.
rep = metrics[
    metrics["TRACK_ID"] == representative_id
].iloc[0]

print("\n========================================")
print("REPRESENTATIVE TRACK")
print("========================================")
print(f"Track ID:          {int(representative_id)}")
print(f"Spots:             {int(rep['N_SPOTS'])}")
print(f"Duration:          {rep['DURATION_s']:.2f} s")
print(f"Displacement:      {rep['DISPLACEMENT_um']:.2f} µm")
print(f"Path length:       {rep['TOTAL_PATH_LENGTH_um']:.2f} µm")
print(f"Mean speed:        {rep['MEAN_SPEED_um_min']:.2f} µm/min")
print(f"Straightness:      {rep['STRAIGHTNESS']:.3f}")
print(f"Speed bin:         {rep['SPEED_BIN']}")
print("========================================")

print("\nFiles saved:")
print(metrics_file)
print(representative_file)
print(all_tracks_png)
print(representative_png)
print(representative_svg)
