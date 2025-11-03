from typing import Tuple

import cv2
import numpy as np


def clahe(
        image: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance the contrast of a greyscale image.
    :param image: Input greyscale image as a 2D numpy array.
    :param clip_limit: The threshold for contrast limiting.
    :param tile_grid_size: Size of the grid for the histogram equalization. Input as (width, height).
    :return:
    """
    if image.ndim != 2:
        raise ValueError("CLAHE function expects a 2D greyscale image array.")

    clahe_processor = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    enhanced_image = clahe_processor.apply(image)
    return enhanced_image
