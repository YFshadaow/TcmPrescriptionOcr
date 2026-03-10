import os
from collections import defaultdict

import cv2
import numpy as np

from preprocessing.binarization import otsu_threshold, sauvola_threshold
from utils.system_utils import get_screen_size


def _filter_candidate_lines(
        image: np.ndarray,
        min_aspect_ratio: float = 1.75,
        min_width_ratio: float = 0.4,
        min_area: int = 50,
        max_area_ratio: float = 0.02
)-> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    if image.ndim != 2:
        raise ValueError("Function expects a 2D binary image array.")
    if image.dtype != np.uint8:
        raise ValueError("Function expects a uint8 image array.")

    # Assumes input is black lines on white background
    image_inverted = cv2.bitwise_not(image)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_inverted, connectivity=8)

    image_width = image.shape[1]
    candidate_labels = []

    # Iterate from 1 because 0 is the background label
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Filter out small noises
        if area < min_area:
            continue

        image_area = image.shape[0] * image.shape[1]
        # Filter out too large areas
        if area > max_area_ratio * image_area:
            continue

        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        # Avoid division by zero
        if height == 0:
            continue

        aspect_ratio = width / height
        # Only include labels that are "horizontally long" enough
        if aspect_ratio < min_aspect_ratio:
            continue

        # Only include labels that cover enough width of the image
        if width < min_width_ratio * image_width:
            continue

        candidate_labels.append(i)

    return candidate_labels, labels, stats, centroids


def add_label_to_image(image, label_text):
    """在图像上方添加一个白色背景的标签。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]

    label_height = text_size[1] + 20
    label_img = np.full((label_height, image.shape[1]), 255, dtype=np.uint8)

    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (label_height + text_size[1]) // 2

    cv2.putText(label_img, label_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # 因为是灰度图，所以直接垂直拼接
    return cv2.vconcat([label_img, image])


def interactive_review(base_dir: str):
    """
    并排显示二值化图和筛选结果图，并居中自适应显示。
    """
    try:
        screen_width, screen_height = get_screen_size()
        max_win_width = int(screen_width * 0.9)
        max_win_height = int(screen_height * 0.9)
        print(f"检测到屏幕分辨率: {screen_width}x{screen_height}")
    except Exception as e:
        print(f"[警告] 无法获取屏幕分辨率: {e}. 将使用默认窗口大小。")
        screen_width, screen_height = None, None

    greyscale_dir = os.path.join(base_dir, 'data', 'greyscale')
    if not os.path.isdir(greyscale_dir):
        print(f"错误: 目录不存在 -> '{greyscale_dir}'")
        return

    print("--- 开始交互式审查 ---")
    print("左: 二值化图, 右: 筛选结果。按任意键继续...")

    total_files_processed = 0

    for doctor_folder in sorted(os.listdir(greyscale_dir)):
        doctor_path = os.path.join(greyscale_dir, doctor_folder)
        if not os.path.isdir(doctor_path): continue

        for filename in sorted(os.listdir(doctor_path)):
            if filename.lower().endswith('.png'):
                image_path = os.path.join(doctor_path, filename)
                relative_path = os.path.relpath(image_path, greyscale_dir)

                try:
                    grey_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if grey_image is None: continue

                    total_files_processed += 1
                    print(f"\n正在显示: '{relative_path}'")

                    binary_image_white_bg = sauvola_threshold(grey_image, window_size=401, k=0.4)
                    candidate_labels, labels, _, _ = _filter_candidate_lines(binary_image_white_bg)
                    num_candidates = len(candidate_labels)
                    print(f"找到 {num_candidates} 个候选区域。")

                    if num_candidates > 0:
                        mask = np.isin(labels, candidate_labels).astype(np.uint8)
                    else:
                        mask = np.zeros_like(binary_image_white_bg, dtype=np.uint8)

                    result_image = np.full_like(binary_image_white_bg, 255)
                    result_image[mask == 1] = binary_image_white_bg[mask == 1]

                    # --- 新增：为图像添加标签并水平拼接 ---
                    labeled_binary_img = add_label_to_image(binary_image_white_bg, "Binarized")
                    labeled_result_img = add_label_to_image(result_image, "Filtered")

                    comparison_image = np.hstack((labeled_binary_img, labeled_result_img))

                    window_title = f"Comparison for [{relative_path}] - Press any key"
                    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

                    if screen_width and screen_height:
                        img_h, img_w = comparison_image.shape[:2]
                        scale = min((max_win_width / img_w), (max_win_height / img_h))
                        win_w, win_h = int(img_w * scale), int(img_h * scale)
                        win_x, win_y = (screen_width - win_w) // 2, (screen_height - win_h) // 2

                        cv2.resizeWindow(window_title, win_w, win_h)
                        cv2.moveWindow(window_title, win_x, win_y)

                    cv2.imshow(window_title, comparison_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                except Exception as e:
                    print(f"[错误] 处理文件 {image_path} 时发生错误: {e}")
                    cv2.waitKey(0)

    print("\n--- 所有文件审查完毕 ---")
    if total_files_processed > 0:
        print(f"总共处理了 {total_files_processed} 张图片。")


if __name__ == '__main__':
    project_root_directory = r'E:\FYP\TcmPrescriptionOcr'
    interactive_review(project_root_directory)