from pathlib import Path
from typing import List

import cv2
import tkinter as tk

from config import ORIGINAL_DATA_DIR, ALIGNED_DATA_DIR, TARGET_IMAGE_FORMAT

# Key codes for keyboard inputs
KEY_ESC = 27
KEY_ENTER = 13
KEY_SPACE = 32


def get_screen_size():
    """
    Get the screen size using tkinter.
    :return: (width, height)
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    return width, height


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


def main():
    # Calculate maximum display size (75% of screen size)
    screen_w, screen_h = get_screen_size()
    print(f"Screen size: {screen_w}x{screen_h}")
    max_display_w = int(screen_w * 0.75)
    max_display_h = int(screen_h * 0.75)

    ALIGNED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Original images directory: {ORIGINAL_DATA_DIR}')
    print(f'Aligned images directory: {ALIGNED_DATA_DIR}')

    print("""
    Press A/D to rotate the image.
    Press Space/Enter to save the aligned image.
    Press Esc/Q to quit.
    """)

    original_images = find_images_recursive(ORIGINAL_DATA_DIR)
    if not original_images:
        print(f'No images found in {ORIGINAL_DATA_DIR}. ')

    while original_images:
        original_path = original_images.pop(0)

        relative_path = original_path.relative_to(ORIGINAL_DATA_DIR)
        aligned_path_original_ext = ALIGNED_DATA_DIR / relative_path
        aligned_path = aligned_path_original_ext.with_suffix(TARGET_IMAGE_FORMAT)

        if aligned_path.exists():
            print(f'Skipping {original_path}, already aligned as {aligned_path}')
            continue

        img = cv2.imread(str(original_path))
        if img is None:
            print(f'Failed to read image {relative_path}, skipping.')
            continue
        # Save a copy of image for further processing
        current_img = img.copy()

        window_name = f'Aligning: {relative_path}'
        print(f'Aligning image: {relative_path}')
        print('Press A/D to rotate, Space/Enter to save, Esc/Q to quit.')

        while True:
            # Resize image to fit within max display size while maintaining aspect ratio
            img_h, img_w = current_img.shape[:2]
            scale_w = max_display_w / img_w
            scale_h = max_display_h / img_h
            scale = min(scale_w, scale_h)

            if scale < 1.0:
                new_w = int(img_w * scale)
                new_h = int(img_h * scale)
                display_img = cv2.resize(current_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                display_img = current_img
            # Show the image in a window centered on the screen
            cv2.imshow(window_name, display_img)
            win_h, win_w = display_img.shape[:2]
            x = max((screen_w - win_w) // 2, 0)
            y = max((screen_h - win_h) // 2, 0)
            cv2.moveWindow(window_name, x, y)

            # Wait for a key press
            key = cv2.waitKey(0) & 0xFF

            if key in (ord('d'), ord('D')):
                current_img = cv2.rotate(current_img, cv2.ROTATE_90_CLOCKWISE)
            elif key in (ord('a'), ord('A')):
                current_img = cv2.rotate(current_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif key == KEY_ENTER or key == KEY_SPACE:
                aligned_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(aligned_path), current_img)
                print(f'Saved aligned image to {aligned_path}')
                break
            elif key == KEY_ESC or key in (ord('q'), ord('Q')):
                print('Exiting manual aligner.')
                cv2.destroyAllWindows()
                return
        cv2.destroyWindow(window_name)

    cv2.destroyAllWindows()
    print('All images processed.')


if __name__ == '__main__':
    main()
