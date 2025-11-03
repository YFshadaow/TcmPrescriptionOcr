import cv2
import numpy as np


def median_blur(image: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """
    Apply median blur to the given image.
    :param image: Input image as a numpy array.
    :param kernel_size: Size of the kernel (must be odd and greater than 1).
    :return: Blurred image as a numpy array.
    """
    return cv2.medianBlur(image, kernel_size)


def gaussian_blur(
        image: np.ndarray,
        kernel_size: tuple[int, int] = (9, 9),
        sigma_x: float = 0
) -> np.ndarray:
    """
    Apply Gaussian blur to the given image.
    :param image: Input image as a numpy array.
    :param kernel_size: Size of the kernel (must be odd and greater than 1).
    :param sigma_x: Standard deviation in X direction.
    :return: Blurred image as a numpy array.
    """
    return cv2.GaussianBlur(image, kernel_size, sigma_x)

def bilateral_filter(
        image: np.ndarray,
        d: int = 11,
        sigma_color: float = 90,
        sigma_space: float = 90
) -> np.ndarray:
    """
    Apply bilateral filter to the given image.
    :param image: Input image as a numpy array.
    :param d: Diameter of each pixel neighborhood.
    :param sigma_color: Filter sigma in color space.
    :param sigma_space: Filter sigma in coordinate space.
    :return: Filtered image as a numpy array.
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)