import os

import cv2
import numpy as np

from config import ALIGNED_DATA_DIR, GREYSCALE_DATA_DIR
from utils.file_utils import find_images_recursive


def greyscale(image: np.ndarray, bgr=True) -> np.ndarray:
    """
    Convert the given image to greyscale.
    Be cautious that OpenCV uses BGR format by default.
    :param image: Input image as a numpy array.
    :param bgr: Whether the input image is in BGR format (default True). If False, assumes RGB format.
    :return: Greyscale image as a numpy array.
    """
    # Return if the image is already greyscale
    if image.ndim == 2:
        return image
    elif image.ndim == 3:
        if bgr:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("Function expects a 3D (BGR/RGB) or 2D (greyscale) image array.")

def main():
    print("Starting greyscale conversion...")
    print("Source directory:", ALIGNED_DATA_DIR)
    print("Target directory:", GREYSCALE_DATA_DIR)
    aligned_paths = find_images_recursive(ALIGNED_DATA_DIR)
    if not aligned_paths:
        print(f'No images found in {ALIGNED_DATA_DIR}. ')
        return

    for input_path in aligned_paths:
        try:
            relative_path = os.path.relpath(input_path, ALIGNED_DATA_DIR)
            output_path = os.path.join(GREYSCALE_DATA_DIR, relative_path)

            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)

            image = cv2.imread(str(input_path))
            grey_image = greyscale(image, bgr=True)
            cv2.imwrite(str(output_path), grey_image)
        except Exception as e:
            print(f"Error processing {input_path}: {e}")

    print("Greyscale conversion completed.")

if __name__ == "__main__":
    main()