import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ---------- USER SETTINGS ----------
INPUT_FILE = r"atma16-oldb-m1-l1-s2-v1_allspots.csv"
SPEED_BINS = [0, 10, 20, 30, 40, 50, np.inf]
SPEED_COLORS = [
    
 "#AAFF00",  # 0–10
        "#B478FB",  # 10–20
    "#4A1A84",  # 20–30
    "#59D0FB", # 30–40
  "#008587", # 40–50
        "#FF286F", # 50+
]

NORMALIZE_START = True
MIN_SPOTS = 2

# Fixed canvas: every exported graph is exactly 6 x 6 inches.
CANVAS_INCHES = 6.0
FIGURE_SIZE = (CANVAS_INCHES, CANVAS_INCHES)
EXPORT_DPI = 600  # PNG = 3600 x 3600 px; SVG = 432 x 432 pt (6 x 6 in)
EXPECTED_PX = int(round(CANVAS_INCHES * EXPORT_DPI))

# Fixed SQUARE plotting area, identical on every figure (fractions of canvas).
# [left, bottom, width, height]; width == height -> square axes box.
AXES_RECT = [0.15, 0.14, 0.77, 0.77]

# Arial throughout, including axes, ticks, and title.
plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "svg.fonttype": "none",       # editable text in SVG
    "figure.figsize": FIGURE_SIZE,
    "figure.dpi": 100,
    "savefig.dpi": EXPORT_DPI,
    "savefig.bbox": "standard",   # never crop/resize the canvas on save
    "savefig.pad_inches": 0,
    "figure.autolayout": False,
    "figure.constrained_layout.use": False,
})


def load_allspots(filename):
    df = pd.read_csv(filename)
    required = ["TRACK_ID", "POSITION_X", "POSITION_Y", "POSITION_T"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Found columns: {list(df.columns)}"
        )
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required).copy()
    if df.empty:
        raise ValueError("No usable spot rows found in this _allspots.csv.")
    return df


def calculate_metrics(df):
    rows = []
    for track_id, g in df.groupby("TRACK_ID"):
        g = g.sort_values("POSITION_T")
        if len(g) < MIN_SPOTS:
            continue

        x = g["POSITION_X"].to_numpy(float)
        y = g["POSITION_Y"].to_numpy(float)
        t = g["POSITION_T"].to_numpy(float)

        dt = np.diff(t)
        step = np.hypot(np.diff(x), np.diff(y))
        path = float(step[dt > 0].sum())
        duration = float(t[-1] - t[0])
        displacement = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
        speed = path / duration * 60 if duration > 0 else np.nan
        straightness = displacement / path if path > 0 else np.nan

        rows.append({
            "TRACK_ID": track_id,
            "N_SPOTS": len(g),
            "DURATION_s": duration,
            "TOTAL_PATH_LENGTH_um": path,
            "DISPLACEMENT_um": displacement,
            "MEAN_SPEED_um_min": speed,
            "STRAIGHTNESS": straightness,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No tracks with at least two usable spots were found.")
    return result


def add_speed_bins(metrics):
    out = metrics.copy()
    out["SPEED_BIN"] = pd.cut(
        out["MEAN_SPEED_um_min"],
        bins=SPEED_BINS,
        labels=["0-10", "10-20", "20-30", "30-40", "40-50", "50+"],
        right=False,
        include_lowest=True,
    )
    return out


def choose_representative(metrics):
    # Select the actual track closest to the group centroid for displacement,
    # path length, and mean speed after standardizing each metric.
    cols = ["DISPLACEMENT_um", "TOTAL_PATH_LENGTH_um", "MEAN_SPEED_um_min"]
    valid = metrics.dropna(subset=cols).copy()
    if valid.empty:
        raise ValueError("Cannot select representative track; metrics are missing.")

    sd = valid[cols].std(ddof=0).replace(0, np.nan)
    z = ((valid[cols] - valid[cols].mean()) / sd).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)
    distances = np.sqrt((z ** 2).sum(axis=1))
    chosen_index = distances.idxmin()
    representative_id = valid.loc[chosen_index, "TRACK_ID"]

    out = metrics.copy()
    out["DISTANCE_FROM_MEAN"] = np.nan
    out.loc[valid.index, "DISTANCE_FROM_MEAN"] = distances
    return out, representative_id


def get_color(speed):
    if pd.isna(speed) or speed < 0:
        return "#808080"
    for i in range(len(SPEED_BINS) - 1):
        if SPEED_BINS[i] <= speed < SPEED_BINS[i + 1]:
            return SPEED_COLORS[i]
    return SPEED_COLORS[-1]


def track_xy(g):
    g = g.sort_values("POSITION_T")
    x = g["POSITION_X"].to_numpy(float)
    y = g["POSITION_Y"].to_numpy(float)
    if NORMALIZE_START and len(x):
        x, y = x - x[0], y - y[0]
    return x, y


def make_fixed_figure():
    # Exactly 6 x 6 inch canvas with a manually placed, identical square
    # axes box on every figure (no automatic layout).
    fig = plt.figure(figsize=FIGURE_SIZE)
    fig.set_size_inches(*FIGURE_SIZE, forward=True)
    ax = fig.add_axes(AXES_RECT)
    return fig, ax


def style_axes(ax):
    # adjustable="datalim": keep the axes BOX fixed and expand the data
    # limits instead, so x and y stay equal-scaled without resizing the box.
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("X displacement (µm)", fontname="Arial")
    ax.set_ylabel("Y displacement (µm)", fontname="Arial")
    ax.tick_params(axis="both", labelsize=8)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname("Arial")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def save_fixed_square(fig, png_path, svg_path=None):
    # Do NOT use bbox_inches='tight': it changes the canvas size.
    fig.set_size_inches(*FIGURE_SIZE)
    fig.savefig(png_path, dpi=EXPORT_DPI, facecolor="white", bbox_inches=None)
    if svg_path:
        fig.savefig(svg_path, facecolor="white", bbox_inches=None)
    plt.close(fig)


def plot_all_tracks(df, metrics, representative_id, output_png):
    lookup = metrics.set_index("TRACK_ID")
    fig, ax = make_fixed_figure()

    for track_id, g in df.groupby("TRACK_ID"):
        if track_id not in lookup.index:
            continue
        color = get_color(lookup.loc[track_id, "MEAN_SPEED_um_min"])
        x, y = track_xy(g)

        if track_id == representative_id:
            ax.plot(x, y, color="black", linewidth=4.0, zorder=10,
                    solid_capstyle="round")
            ax.plot(x, y, color=color, linewidth=2.6, zorder=11,
                    solid_capstyle="round")
        else:
            ax.plot(x, y, color=color, linewidth=0.9, alpha=0.55,
                    solid_capstyle="round")

    style_axes(ax)
    ax.set_title("Cell migration tracks by mean speed", fontname="Arial", pad=10)
    save_fixed_square(fig, output_png)


def plot_representative(df, metrics, representative_id, png_path, svg_path):
    g = df[df["TRACK_ID"] == representative_id].copy()
    x, y = track_xy(g)
    row = metrics.loc[metrics["TRACK_ID"] == representative_id].iloc[0]
    speed = row["MEAN_SPEED_um_min"]
    color = get_color(speed)

    fig, ax = make_fixed_figure()
    ax.plot(x, y, color=color, linewidth=2.8,
            solid_capstyle="round", solid_joinstyle="round")
    if len(x):
        ax.scatter(x[0], y[0], color="black", s=24, zorder=5)
        ax.scatter(x[-1], y[-1], color="black", s=24, marker="s", zorder=5)

    style_axes(ax)
    ax.set_title(
        f"Representative track | Mean speed = {speed:.2f} µm/min",
        fontname="Arial", pad=10
    )
    save_fixed_square(fig, png_path, svg_path)


def verify_dimensions(png_paths, svg_path):
    print("\nDimension check (expected "
          f"{EXPECTED_PX} x {EXPECTED_PX} px = {CANVAS_INCHES} x {CANVAS_INCHES} in "
          f"at {EXPORT_DPI} dpi):")
    for p in png_paths:
        with Image.open(p) as im:
            w, h = im.size
        status = "OK" if (w, h) == (EXPECTED_PX, EXPECTED_PX) else "MISMATCH"
        print(f"  {os.path.basename(p)}: {w} x {h} px [{status}]")

    with open(svg_path, "r", encoding="utf-8") as f:
        head = f.read(2000)
    m = re.search(r'width="([\d.]+)pt"\s+height="([\d.]+)pt"', head)
    if m:
        w_in = float(m.group(1)) / 72
        h_in = float(m.group(2)) / 72
        status = ("OK" if abs(w_in - CANVAS_INCHES) < 0.01
                  and abs(h_in - CANVAS_INCHES) < 0.01 else "MISMATCH")
        print(f"  {os.path.basename(svg_path)}: {w_in:.2f} x {h_in:.2f} in [{status}]")
    else:
        print(f"  {os.path.basename(svg_path)}: could not read SVG size")


def main():
    if INPUT_FILE == "PUT_YOUR_ALLSPOTS_CSV_HERE.csv" or not os.path.isfile(INPUT_FILE):
        raise FileNotFoundError(
            "Edit INPUT_FILE near the top of the script and set it to the full "
            "path of your TrackMate *_allspots.csv file."
        )

    print("Loading:", INPUT_FILE)
    df = load_allspots(INPUT_FILE)
    print(f"Usable spots: {len(df)} | Tracks: {df['TRACK_ID'].nunique()}")

    metrics = add_speed_bins(calculate_metrics(df))
    metrics, representative_id = choose_representative(metrics)

    folder = os.path.dirname(os.path.abspath(INPUT_FILE))
    base = os.path.splitext(os.path.basename(INPUT_FILE))[0]
    paths = {
        "metrics": os.path.join(folder, base + "_track_metrics.csv"),
        "representative_csv": os.path.join(folder, base + "_representative_track.csv"),
        "all_tracks_png": os.path.join(folder, base + "_speed_colored_tracks.png"),
        "representative_png": os.path.join(folder, base + "_representative_track.png"),
        "representative_svg": os.path.join(folder, base + "_representative_track.svg"),
    }

    metrics.to_csv(paths["metrics"], index=False)
    df[df["TRACK_ID"] == representative_id].sort_values("POSITION_T").to_csv(
        paths["representative_csv"], index=False
    )
    plot_all_tracks(df, metrics, representative_id, paths["all_tracks_png"])
    plot_representative(
        df, metrics, representative_id,
        paths["representative_png"], paths["representative_svg"]
    )

    rep = metrics.loc[metrics["TRACK_ID"] == representative_id].iloc[0]
    print("\nRepresentative track")
    print(f"Track ID: {int(representative_id)}")
    print(f"Displacement: {rep['DISPLACEMENT_um']:.2f} µm")
    print(f"Path length: {rep['TOTAL_PATH_LENGTH_um']:.2f} µm")
    print(f"Mean speed: {rep['MEAN_SPEED_um_min']:.2f} µm/min")
    print(f"Straightness: {rep['STRAIGHTNESS']:.3f}")
    print(f"Speed category: {rep['SPEED_BIN']}")

    verify_dimensions(
        [paths["all_tracks_png"], paths["representative_png"]],
        paths["representative_svg"],
    )

    print("\nSaved files:")
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
