from typing import Literal

import cv2
import numpy as np
from skimage.filters import threshold_niblack, threshold_sauvola


def otsu_threshold(image: np.ndarray) -> np.ndarray:
    """
    Apply Otsu's binarization to the given greyscale image.
    :param image: Input greyscale image as a 2D numpy array.
    :return: Binary image as a 2D numpy array.
    """
    if image.ndim != 2:
        raise ValueError("Function expects a 2D greyscale image array.")
    if image.dtype != np.uint8:
        raise ValueError("Function expects a uint8 image array.")

    _, binarized = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binarized


def adaptive_threshold(
        image: np.ndarray,
        block_size: int = 31,
        c: int = 10,
        method: Literal['mean', 'gaussian'] = 'gaussian'
) -> np.ndarray:
    """
    Apply adaptive thresholding to the given greyscale image.
    :param image: Input greyscale image as a 2D numpy array.
    :param block_size: Size of a pixel neighborhood that is used to calculate a threshold value
    :param c: Constant subtracted from the mean or weighted mean.
    :param method: Method to calculate the threshold value ('mean' or 'gaussian').
    :return: Binary image as a 2D numpy array.
    """
    if image.ndim != 2:
        raise ValueError("Function expects a 2D greyscale image array.")
    if image.dtype != np.uint8:
        raise ValueError("Function expects a uint8 image array.")

    adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C if method == 'mean' else cv2.ADAPTIVE_THRESH_GAUSSIAN_C

    binarized = cv2.adaptiveThreshold(
        image,
        255,
        adaptive_method,
        cv2.THRESH_BINARY,
        block_size,
        c
    )
    return binarized


def niblack_threshold(
        image: np.ndarray,
        window_size: int = 31,
        k: float = -0.2
) -> np.ndarray:
    """
    Apply Niblack's thresholding to the given greyscale image.
    :param image: Input greyscale image as a 2D numpy array.
    :param window_size: Size of the local window to compute the threshold.
    :param k: Niblack's k value to adjust the threshold.
    :return: Binary image as a 2D numpy array.
    """
    if image.ndim != 2:
        raise ValueError("Function expects a 2D greyscale image array.")
    if image.dtype != np.uint8:
        raise ValueError("Function expects a uint8 image array.")

    thresh_niblack = threshold_niblack(image, window_size=window_size, k=k)
    binary_image = image > thresh_niblack
    uint8_image = binary_image.astype(np.uint8) * 255
    return uint8_image


def sauvola_threshold(
        image: np.ndarray,
        window_size: int = 31,
        k: float = 0.2,
        r: float = None
) -> np.ndarray:
    """
    Apply Sauvola's thresholding to the given greyscale image.
    :param image: Input greyscale image as a 2D numpy array.
    :param window_size: Size of the local window to compute the threshold.
    :param k: The positive parameter for Sauvola's formula.
    :param r: The dynamic range of standard deviation. If None, it is set to
    the maximum of the image dtype (e.g., 1.0 for float, 255 for uint8).
    :return:
    """
    if image.ndim != 2:
        raise ValueError("Function expects a 2D greyscale image array.")
    if image.dtype != np.uint8:
        raise ValueError("Function expects a uint8 image array.")

    # Assume input dtype is uint8
    image = image.astype(np.float64) / 255.0

    thresh_sauvola = threshold_sauvola(image, window_size=window_size, k=k, r=r)
    binary_image = image > thresh_sauvola
    uint8_image = binary_image.astype(np.uint8) * 255
    return uint8_image