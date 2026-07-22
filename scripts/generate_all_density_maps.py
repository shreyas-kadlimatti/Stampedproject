import os
import cv2
import numpy as np

from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from tqdm import tqdm


# ===================================================
# PROJECT ROOT
# ===================================================
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)


# ===================================================
# DATASET PATHS
# ===================================================
image_dir = os.path.join(
    BASE_DIR,
    "dataset",
    "ShanghaiTech",
    "part_A",
    "train_data",
    "images"
)

gt_dir = os.path.join(
    BASE_DIR,
    "dataset",
    "ShanghaiTech",
    "part_A",
    "train_data",
    "ground-truth"
)

output_dir = os.path.join(
    BASE_DIR,
    "density_maps"
)

os.makedirs(output_dir, exist_ok=True)


# ===================================================
# VERIFY PATHS
# ===================================================
if not os.path.exists(image_dir):
    raise FileNotFoundError(
        f"Image directory not found: {image_dir}"
    )

if not os.path.exists(gt_dir):
    raise FileNotFoundError(
        f"Ground-truth directory not found: {gt_dir}"
    )


# ===================================================
# GEOMETRY-ADAPTIVE DENSITY MAP
# ===================================================
def generate_density_map(height, width, points):

    density = np.zeros(
        (height, width),
        dtype=np.float32
    )

    number_of_points = len(points)

    if number_of_points == 0:
        return density

    # ------------------------------------------------
    # SINGLE PERSON
    # ------------------------------------------------
    if number_of_points == 1:

        x = int(round(points[0][0]))
        y = int(round(points[0][1]))

        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)

        impulse = np.zeros(
            (height, width),
            dtype=np.float32
        )

        impulse[y, x] = 1.0

        density += gaussian_filter(
            impulse,
            sigma=15,
            mode="constant"
        )

        return density

    # ------------------------------------------------
    # FIND NEAREST NEIGHBOURS
    # ------------------------------------------------
    tree = KDTree(points)

    # k cannot be greater than the number of points
    neighbour_count = min(
        4,
        number_of_points
    )

    distances, _ = tree.query(
        points,
        k=neighbour_count
    )

    # ------------------------------------------------
    # CREATE ADAPTIVE GAUSSIAN FOR EACH POINT
    # ------------------------------------------------
    for index, point in enumerate(points):

        x = int(round(point[0]))
        y = int(round(point[1]))

        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)

        if number_of_points >= 4:

            # Average distance to the three nearest
            # neighbours, excluding the point itself.
            sigma = 0.3 * (
                distances[index][1]
                + distances[index][2]
                + distances[index][3]
            )

        elif number_of_points == 3:

            sigma = 0.3 * (
                distances[index][1]
                + distances[index][2]
            )

        else:

            sigma = 0.3 * distances[index][1]

        # Prevent extremely small or excessively
        # large Gaussian kernels.
        sigma = float(
            np.clip(sigma, 1.0, 15.0)
        )

        impulse = np.zeros(
            (height, width),
            dtype=np.float32
        )

        impulse[y, x] += 1.0

        person_density = gaussian_filter(
            impulse,
            sigma=sigma,
            mode="constant"
        )

        # Preserve the count when part of the Gaussian
        # lies outside the image boundary.
        person_sum = person_density.sum()

        if person_sum > 0:

            person_density /= person_sum

        density += person_density

    return density


# ===================================================
# GET IMAGE LIST
# ===================================================
image_files = [
    file_name
    for file_name in os.listdir(image_dir)
    if file_name.lower().endswith(".jpg")
]

image_files.sort()

print("Total Images:", len(image_files))


# ===================================================
# PROCESS ALL IMAGES
# ===================================================
for img_name in tqdm(image_files):

    img_path = os.path.join(
        image_dir,
        img_name
    )

    gt_name = (
        "GT_"
        + os.path.splitext(img_name)[0]
        + ".mat"
    )

    gt_path = os.path.join(
        gt_dir,
        gt_name
    )

    if not os.path.exists(gt_path):

        print(
            f"\nGround truth missing: {gt_path}"
        )

        continue

    # ------------------------------------------------
    # LOAD IMAGE
    # ------------------------------------------------
    img = cv2.imread(img_path)

    if img is None:

        print(
            f"\nUnable to read image: {img_path}"
        )

        continue

    height, width = img.shape[:2]

    # ------------------------------------------------
    # LOAD ANNOTATION POINTS
    # ------------------------------------------------
    mat = loadmat(gt_path)

    points = mat[
        "image_info"
    ][0, 0][0, 0][0]

    points = np.asarray(
        points,
        dtype=np.float32
    )

    # ------------------------------------------------
    # GENERATE DENSITY MAP
    # ------------------------------------------------
    density = generate_density_map(
        height,
        width,
        points
    )

    # ------------------------------------------------
    # VERIFY COUNT PRESERVATION
    # ------------------------------------------------
    expected_count = len(points)
    generated_count = float(density.sum())

    if abs(generated_count - expected_count) > 0.1:

        print(
            f"\nCount warning for {img_name}: "
            f"GT={expected_count}, "
            f"Density Sum={generated_count:.4f}"
        )

    # ------------------------------------------------
    # SAVE DENSITY MAP
    # ------------------------------------------------
    save_name = (
        os.path.splitext(img_name)[0]
        + ".npy"
    )

    save_path = os.path.join(
        output_dir,
        save_name
    )

    np.save(
        save_path,
        density.astype(np.float32)
    )


print("\nAll geometry-adaptive density maps generated.")