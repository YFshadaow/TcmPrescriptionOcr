import numpy as np
from skimage.morphology import skeletonize


def skeletonize_image(image: np.ndarray) -> np.ndarray:
    """
    Apply skeletonization to the given binary image.
    :param image: Input binary image as a 2D numpy array (0 and 255 values).
    :return: Skeletonized image as a 2D numpy array (0 and 255 values).
    """
    if image.ndim != 2:
        raise ValueError("Function expects a 2D binary image array.")
    if image.dtype != np.uint8:
        raise ValueError("Function expects a uint8 image array.")

    # Convert to binary format (0 and 1) for skeletonization
    binary_input = (image > 0).astype(np.uint8)

    # Apply skeletonization
    skeleton = skeletonize(binary_input)

    # Convert back to 0-255 format to match input format
    skeleton_output = np.where(skeleton, 0, 255).astype(np.uint8)

    return skeleton_output