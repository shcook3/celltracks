import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage import filters, morphology


# ============================================================
# SETTINGS
# ============================================================

# Folder containing your images
image_folder = "/Users/shcook/Desktop/whole node"
# Example:
# LN_images/
#     Young_01.png
#     Young_02.png
#     Young_03.png
#     Old_01.png
#     Old_02.png
#     Old_03.png

# Fluorescence channels
# Change these if your image channels are different
RED_CHANNEL = 0       # B220
GREEN_CHANNEL = 1     # CD3

# Gaussian smoothing
SIGMA = 2

# Remove small segmentation objects
MIN_OBJECT_SIZE = 500


# ============================================================
# ANALYSIS FUNCTION
# ============================================================

def analyze_image(image_path):

    img = np.array(
        Image.open(image_path).convert("RGB")
    ).astype(float)

    # --------------------------------------------------------
    # Extract B220 and CD3
    # --------------------------------------------------------

    red = img[:, :, RED_CHANNEL]
    green = img[:, :, GREEN_CHANNEL]

    # Smooth image slightly
    red = gaussian_filter(red, SIGMA)
    green = gaussian_filter(green, SIGMA)

    # --------------------------------------------------------
    # Tissue mask
    # --------------------------------------------------------

    total_signal = red + green

    tissue_threshold = filters.threshold_otsu(total_signal)

    tissue = total_signal > tissue_threshold

    tissue = morphology.remove_small_objects(
        tissue,
        min_size=MIN_OBJECT_SIZE
    )

    tissue = morphology.remove_small_holes(
        tissue,
        area_threshold=MIN_OBJECT_SIZE
    )

    # --------------------------------------------------------
    # Normalize channels within tissue
    # --------------------------------------------------------

    red_norm = red / (np.percentile(red[tissue], 99) + 1e-8)
    green_norm = green / (np.percentile(green[tissue], 99) + 1e-8)

    red_norm = np.clip(red_norm, 0, 1)
    green_norm = np.clip(green_norm, 0, 1)

    # ========================================================
    # METRIC 1: RED-GREEN MIXING
    # ========================================================

    # 0 = purely red or purely green
    # 1 = equal red and green

    mixing = (
        2 * np.minimum(red_norm, green_norm) /
        (red_norm + green_norm + 1e-8)
    )

    mixing_index = np.mean(
        mixing[tissue]
    )

    # ========================================================
    # METRIC 2: TRUE RED/GREEN OVERLAP
    # ========================================================

    red_threshold = filters.threshold_otsu(
        red_norm[tissue]
    )

    green_threshold = filters.threshold_otsu(
        green_norm[tissue]
    )

    red_mask = (
        red_norm > red_threshold
    ) & tissue

    green_mask = (
        green_norm > green_threshold
    ) & tissue

    overlap = red_mask & green_mask

    overlap_fraction = (
        np.sum(overlap) /
        np.sum(red_mask | green_mask)
    )

    # ========================================================
    # METRIC 3: BOUNDARY CONTACT
    # ========================================================

    # Dilate both compartments.
    # Where the two dilated regions touch,
    # B220 and CD3 compartments are spatially close.

    red_dilated = morphology.binary_dilation(
        red_mask,
        morphology.disk(3)
    )

    green_dilated = morphology.binary_dilation(
        green_mask,
        morphology.disk(3)
    )

    boundary_contact = (
        red_dilated &
        green_dilated &
        tissue
    )

    boundary_contact_fraction = (
        np.sum(boundary_contact) /
        np.sum(tissue)
    )

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
        "File": os.path.basename(image_path),
        "Mixing index": mixing_index,
        "Red-green overlap": overlap_fraction,
        "Boundary contact": boundary_contact_fraction
    }, {
        "image": img,
        "red": red_norm,
        "green": green_norm,
        "tissue": tissue,
        "red_mask": red_mask,
        "green_mask": green_mask,
        "mixing": mixing,
        "boundary": boundary_contact
    }


# ============================================================
# ANALYZE ALL IMAGES
# ============================================================

results = []
processed_images = {}

for filename in os.listdir(image_folder):

    if filename.lower().endswith(
        (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    ):

        path = os.path.join(
            image_folder,
            filename
        )

        result, processed = analyze_image(path)

        results.append(result)
        processed_images[filename] = processed


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\nRESULTS:")
print(results_df)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    os.path.join(
        image_folder,
        "B220_CD3_disorganization_results.csv"
    ),
    index=False
)


# ============================================================
# VISUALIZE EACH IMAGE
# ============================================================

for filename, data in processed_images.items():

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 5)
    )

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    axes[0].imshow(
        data["image"].astype(np.uint8)
    )

    axes[0].set_title(filename)
    axes[0].axis("off")

    # --------------------------------------------------------
    # B220/CD3 segmentation
    # --------------------------------------------------------

    axes[1].imshow(
        data["red_mask"],
        cmap="Reds",
        alpha=0.7
    )

    axes[1].imshow(
        data["green_mask"],
        cmap="Greens",
        alpha=0.5
    )

    axes[1].set_title(
        "B220 / CD3 segmentation"
    )

    axes[1].axis("off")

    # --------------------------------------------------------
    # Mixing map
    # --------------------------------------------------------

    im = axes[2].imshow(
        data["mixing"],
        cmap="inferno",
        vmin=0,
        vmax=1
    )

    axes[2].set_title(
        "B220-CD3 mixing"
    )

    axes[2].axis("off")

    plt.colorbar(
        im,
        ax=axes[2],
        label="Mixing"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# COMPARE YOUNG VS OLD
# ============================================================

# Automatically classify files based on filename

results_df["Age"] = np.where(
    results_df["File"].str.contains(
        "young",
        case=False
    ),
    "Young",
    np.where(
        results_df["File"].str.contains(
            "old",
            case=False
        ),
        "Old",
        "Unknown"
    )
)

print("\nGROUP SUMMARY:")

summary = results_df.groupby("Age")[
    [
        "Mixing index",
        "Red-green overlap",
        "Boundary contact"
    ]
].agg(
    ["mean", "std", "count"]
)

print(summary)


# ============================================================
# PLOT YOUNG VS OLD
# ============================================================

plt.figure(figsize=(6, 5))

for age in ["Young", "Old"]:

    values = results_df.loc[
        results_df["Age"] == age,
        "Mixing index"
    ]

    x = np.random.normal(
        ["Young", "Old"].index(age) + 1,
        0.04,
        size=len(values)
    )

    plt.scatter(
        x,
        values,
        s=50
    )

plt.xticks(
    [1, 2],
    ["Young", "Old"]
)

plt.ylabel("B220-CD3 mixing index")
plt.title("B/T compartment disorganization")

plt.tight_layout()
plt.show()
