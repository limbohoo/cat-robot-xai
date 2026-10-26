"""Keypoint names and skeleton edges for visualization."""

from __future__ import annotations


# CatFLW paper landmark ordering, 48 points.
# The dataset does not publish semantic names for each index, so we keep them
# numeric here. The visualization layers still connect them by index groups.
CATFLW_NUM = 48

# Left/right mirror permutation for CatFLW, derived by Hungarian-matching the
# mean face shape against its mirror image (mean match error 0.3% of bbox).
# Midline points 0, 2, 16, 17 map to themselves.
CATFLW_FLIP_MAP: list[int] = [
    0, 3, 2, 1, 8, 9, 10, 11, 4, 5, 6, 7, 13, 12, 15, 14, 16, 17, 20, 21, 18, 19,
    31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 35, 34, 33, 32, 41, 40, 39, 38, 37, 36,
    43, 42, 45, 44, 47, 46,
]

# Loose face regions used only for coloring in the overlay renderer.
CATFLW_REGIONS: dict[str, list[int]] = {
    "outline": list(range(0, 48)),
}

# Animal Pose Part I / II keypoint order, category "cat".
ANIMALPOSE_KEYPOINT_NAMES: list[str] = [
    "left_eye", "right_eye", "nose",
    "left_ear", "right_ear",
    "left_front_elbow", "right_front_elbow",
    "left_back_elbow", "right_back_elbow",
    "left_front_knee", "right_front_knee",
    "left_back_knee", "right_back_knee",
    "left_front_paw", "right_front_paw",
    "left_back_paw", "right_back_paw",
    "throat", "withers", "tailbase",
]

# Skeleton edges (as index pairs, aligned with the Animal Pose "cat" category).
ANIMALPOSE_SKELETON: list[tuple[int, int]] = [
    (0, 1), (0, 2), (1, 2), (0, 3), (1, 4),
    (2, 17), (18, 19),
    (5, 9), (6, 10), (7, 11), (8, 12),
    (9, 13), (10, 14), (11, 15), (12, 16),
]

# Horizontal-flip index remap for AnimalPose (swap left <-> right pairs).
ANIMALPOSE_FLIP_MAP: list[int] = [
    1, 0, 2,                  # eyes swap, nose stays
    4, 3,                     # ears swap
    6, 5, 8, 7,               # front/back elbows
    10, 9, 12, 11,            # front/back knees
    14, 13, 16, 15,           # paws
    17, 18, 19,               # throat / withers / tailbase
]
