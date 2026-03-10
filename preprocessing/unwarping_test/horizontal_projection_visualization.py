import os
from collections import defaultdict

import cv2
import numpy as np
from matplotlib import pyplot as plt

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


def interactive_review(base_dir: str):
    """
    对筛选后的结果图（白底黑字）计算并显示其水平投影波形。
    """
    greyscale_dir = os.path.join(base_dir, 'data', 'greyscale')
    if not os.path.isdir(greyscale_dir):
        print(f"错误: 目录不存在 -> '{greyscale_dir}'")
        return

    print("--- 开始交互式审查 ---")
    print("将为每个筛选结果显示水平投影图。关闭图表后继续...")

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
                    print(f"\n正在处理: '{relative_path}'")

                    # 1. 二值化，得到白底黑字图像 (前景=0, 背景=255)
                    binary_image_white_bg = sauvola_threshold(grey_image, window_size=401, k=0.4)

                    # 2. 筛选候选区域
                    candidate_labels, labels, _, _ = _filter_candidate_lines(binary_image_white_bg)
                    num_candidates = len(candidate_labels)
                    print(f"找到 {num_candidates} 个候选区域。")

                    # 3. 创建只包含候选区域的图像 (result_image)
                    # 默认创建一个全白的背景
                    result_image = np.full_like(binary_image_white_bg, 255, dtype=np.uint8)

                    # 构建掩码，找到所有属于候选标签的像素
                    if num_candidates > 0:
                        mask = np.isin(labels, candidate_labels)
                        # 将这些像素设置为前景颜色（黑色）
                        result_image[mask] = 0

                        # --- 核心部分：计算并绘制水平投影 ---

                    # 我们要统计前景像素(值为0)的数量。
                    # 将图像反转(0->255, 255->0)，然后按行求和，再除以255得到像素个数。
                    projection = np.sum(255 - result_image, axis=1) / 255

                    plt.figure(figsize=(12, 6))
                    plt.plot(projection)
                    plt.title(f"Horizontal Projection for [{relative_path}]")
                    plt.xlabel("Y-coordinate (Image Row)")
                    plt.ylabel("Number of Foreground (Black) Pixels")
                    plt.grid(True)

                    # Matplotlib的Y轴默认上小下大，符合图像坐标系，无需反转
                    # plt.gca().invert_yaxis()

                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    plt.show()  # 程序会在此暂停直到图表被关闭

                except Exception as e:
                    print(f"[错误] 处理文件 {image_path} 时发生错误: {e}")
                    import traceback
                    traceback.print_exc()

    print("\n--- 所有文件审查完毕 ---")
    if total_files_processed > 0:
        print(f"总共处理了 {total_files_processed} 张图片。")


if __name__ == '__main__':
    # **重要**：请将此路径替换为您项目的实际根目录
    project_root_directory = r'E:\FYP\TcmPrescriptionOcr'

    if not os.path.exists(project_root_directory):
        print(f"警告: 示例路径 '{project_root_directory}' 不存在。")
        print("请在代码中修改 'project_root_directory' 为您的项目路径。")
    else:
        interactive_review(project_root_directory)