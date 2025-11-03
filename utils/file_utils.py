from pathlib import Path
from typing import List


def find_images_recursive(root_dir: Path) -> List[Path]:
    """
    Recursively find all image files in the given directory,
    and return a sorted list of their absolute paths.
    :param root_dir: The root directory to search for images.
    :return: A sorted list of absolute paths to image files.
    """
    if not root_dir.is_absolute():
        raise ValueError("The provided path must be absolute.")
    if not root_dir.is_dir():
        raise ValueError("The provided path must be a directory.")
    print(f"Getting all image paths from {root_dir}")

    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    all_image_paths = []

    for file_path in root_dir.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            all_image_paths.append(file_path)

    all_image_paths.sort()
    return all_image_paths