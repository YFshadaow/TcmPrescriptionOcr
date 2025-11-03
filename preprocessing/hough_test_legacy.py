import cv2
import numpy as np

from preprocessing.binarization import sauvola_threshold


def canny_detect_edges(image: np.ndarray, low_threshold: int = 100, high_threshold: int = 200) -> np.ndarray:
    """
    Apply Canny edge detection to the given image.
    :param image: Input image as a numpy array.
    :param low_threshold: Lower threshold for the hysteresis procedure.
    :param high_threshold: Upper threshold for the hysteresis procedure.
    :return: Image with edges detected as a numpy array.
    """
    if image.ndim != 2:
        raise ValueError("Canny edge detection expects a 2D greyscale image array.")

    return cv2.Canny(image, low_threshold, high_threshold)


def find_reference_lines(image: np.ndarray):
    img_width = image.shape[1]
    edges = canny_detect_edges(image)
    lines = cv2.HoughLinesP(
        image=edges,
        rho=1,
        theta=np.deg2rad(1),
        threshold=50,
        minLineLength=img_width * 0.4,
        maxLineGap=75
    )
    return lines


def resize_for_display(image, max_width=1280, max_height=720):
    """
    Resizes an image to fit within a maximum width and height, preserving aspect ratio.

    Args:
        image: The input image (as a Numpy array).
        max_width (int): The maximum allowable width for the displayed image.
        max_height (int): The maximum allowable height for the displayed image.

    Returns:
        The resized image.
    """
    original_height, original_width = image.shape[:2]

    # 如果图像本身就比最大尺寸小，则无需缩放，直接返回原图
    if original_width <= max_width and original_height <= max_height:
        return image

    # 计算宽度和高度的缩放比例
    width_ratio = max_width / original_width
    height_ratio = max_height / original_height

    # 选择较小的比例作为最终的缩放比例，以保证整个图像都能被容纳
    scaling_factor = min(width_ratio, height_ratio)

    # 计算新的尺寸
    new_width = int(original_width * scaling_factor)
    new_height = int(original_height * scaling_factor)

    # 使用 cv2.resize 进行缩放
    # cv2.INTER_AREA 是一种高质量的插值方法，特别适合于缩小图像
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_image


if __name__ == "__main__":
    # --- 1. 读取和预处理图片 ---
    image_path = r"E:\FYP\TcmPrescriptionOcr\test\camera_even_lighting.png"
    image = cv2.imread(image_path)

    if image is None:
        print(f"错误：无法读取图片，请检查路径是否正确: {image_path}")
        exit()

    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- 2. 找到所有原始直线 ---
    lines = find_reference_lines(grey_image)

    # --- 3. 结果可视化 (绘制所有找到的直线) ---
    image_with_lines = image.copy()
    line_color_bgr = (0, 255, 0)
    line_thickness = 2

    print("--- 霍夫变换原始结果 ---")
    if lines is not None:
        print(f"总共找到了 {len(lines)} 条直线，正在全部绘制...")
        for line_segment in lines:
            x1, y1, x2, y2 = line_segment[0]
            cv2.line(image_with_lines, (x1, y1), (x2, y2), line_color_bgr, line_thickness)
    else:
        print("在当前参数下，没有找到任何直线。")

    # --- 4. 显示结果 ---

    #  --- 关键改动：在显示前缩放图像 ---
    # 调用我们写的函数，获取一个适合屏幕显示的缩放版本
    display_image = resize_for_display(image_with_lines, max_width=1600, max_height=900)

    # 修改窗口创建方式，让窗口可以被用户调整大小
    window_name = "Raw Hough Lines - Resized for Display"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cv2.imshow(window_name, display_image)

    print("\n图片已缩放以适应屏幕。按任意键关闭图片显示窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(r"E:\FYP\TcmPrescriptionOcr\test\camera_even_lighting_Hough.png", image_with_lines)