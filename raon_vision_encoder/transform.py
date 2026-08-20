# Originally from OpenCLIP (https://github.com/mlfoundations/open_clip)

import math


def get_image_size_for_max_num_patches(
    image_height, image_width, patch_size, max_num_patches
):
    """Find target image size preserving aspect ratio within patch budget.

    Uses binary search to find the optimal scale such that
    ceil(h*scale/ps)*ceil(w*scale/ps) <= max_num_patches.

    Args:
        image_height: Original image height.
        image_width: Original image width.
        patch_size: Patch size (int).
        max_num_patches: Maximum number of patches allowed.

    Returns:
        (target_h, target_w) both multiples of patch_size.
    """
    scale_min, scale_max = 1e-6, 100.0
    eps = 1e-5
    while (scale_max - scale_min) >= eps:
        scale = (scale_min + scale_max) / 2
        target_h = max(
            patch_size, int(math.ceil(image_height * scale / patch_size) * patch_size)
        )
        target_w = max(
            patch_size, int(math.ceil(image_width * scale / patch_size) * patch_size)
        )
        num_patches = (target_h // patch_size) * (target_w // patch_size)
        if num_patches <= max_num_patches:
            scale_min = scale
        else:
            scale_max = scale
    target_h = max(
        patch_size, int(math.ceil(image_height * scale_min / patch_size) * patch_size)
    )
    target_w = max(
        patch_size, int(math.ceil(image_width * scale_min / patch_size) * patch_size)
    )
    return target_h, target_w
