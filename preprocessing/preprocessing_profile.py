from functools import partial
from typing import List

import numpy as np


class PreprocessingProfile:
    """
    A class representing a sequence of image preprocessing steps.
    Each step is a function that takes an image (numpy array) as input
    and returns a processed image (numpy array).
    """
    def __init__(self, steps: List[partial]):
        self.steps : List[partial]= steps

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the preprocessing steps to the given image in sequence.
        :param image: Input image as a numpy array.
        :return: Processed image as a numpy array.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy array.")

        if image.dtype != np.uint8:
            raise ValueError("Input image array must have dtype of uint8.")

        processed_image = np.copy(image)
        for step_function in self.steps:
            processed_image = step_function(processed_image)
        return processed_image

    def __str__(self):
        """
        Generate a string representation of the preprocessing profile.
        Output example: "greyscale; clahe=clip_limit=2.0, tile_grid_size=(8,8)"
        :return: String representation of the preprocessing profile.
        """
        description_parts = []

        for step in self.steps:
            line_parts = [step.func.__name__]

            for k, v in step.keywords.items():
                line_parts.append(f"{k}={v!r}")

            full_line = ", ".join(line_parts)
            description_parts.append(full_line)

        return "; ".join(description_parts)