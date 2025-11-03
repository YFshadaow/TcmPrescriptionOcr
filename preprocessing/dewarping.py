from typing import List, Tuple

import cv2
import numpy as np
from numba import jit
import skimage as ski

from preprocessing.binarization import sauvola_threshold
from utils.system_utils import get_screen_size

# The ratio coordinates of the target area in the dewarped image
target_x_min_ratio = 0.063190
target_x_max_ratio = 0.925778
# Ensure target_y is in increasing order
target_y_ratios = [0.208799, 0.256983, 0.335894, 0.907123]

POINTS_PER_100_PIXELS = 0.7
MIN_POINTS_PER_LINE = 5
MAX_POINTS_PER_LINE = 9



target_width_ratio = target_x_max_ratio - target_x_min_ratio
target_height_ratio = target_y_ratios[-1] - target_y_ratios[0]

top_extend_ratio = target_y_ratios[0] / target_height_ratio
bottom_extend_ratio = (1.0 - target_y_ratios[-1]) / target_height_ratio
left_extend_ratio = target_x_min_ratio / target_width_ratio
right_extend_ratio = (1.0 - target_x_max_ratio) / target_width_ratio


@jit(nopython=True)
def path_finding_core(
        binary_image: np.ndarray,
        location_penalty: int,
        turn_penalty: int
) -> np.ndarray:
    """
    Core path_finding algorithm implementation using Numba for acceleration.
    :param binary_image: 2D binary image as a numpy array.
    :param location_penalty: Penalty for being in a foreground pixel.
    :param turn_penalty: Penalty for moving diagonally.
    :return: Optimal path as a list of y-coordinates for each x-coordinate.
    """
    height, width = binary_image.shape
    # Directly use a constant for INT_MAX since Numba may not support np.iinfo
    INT_MAX = 2147483647

    cost_matrix = np.full((height, width), INT_MAX, dtype=np.int32)
    path_matrix = np.zeros((height, width), dtype=np.int32)

    location_cost = (binary_image > 0) * location_penalty
    cost_matrix[:, 0] = location_cost[:, 0].astype(np.int32)

    # Main loop
    for x in range(1, width):
        prev_total_costs = cost_matrix[:, x - 1]

        # Topmost pixel
        cost_mid = prev_total_costs[0]
        cost_down = prev_total_costs[1]
        location_cost_value = location_cost[0, x]
        if cost_mid <= cost_down + turn_penalty:
            cost_matrix[0, x] = cost_mid + location_cost_value
            path_matrix[0, x] = 0
        else:
            cost_matrix[0, x] = cost_down + turn_penalty + location_cost_value
            path_matrix[0, x] = 1

        # Middle pixels
        for y_curr in range(1, height - 1):
            cost_up = prev_total_costs[y_curr - 1]
            cost_mid = prev_total_costs[y_curr]
            cost_down = prev_total_costs[y_curr + 1]
            location_cost_value = location_cost[y_curr, x]
            if cost_up < cost_down:
                if cost_mid <= cost_up + turn_penalty:
                    cost_matrix[y_curr, x] = cost_mid + location_cost_value
                    path_matrix[y_curr, x] = y_curr
                else:
                    cost_matrix[y_curr, x] = cost_up + turn_penalty + location_cost_value
                    path_matrix[y_curr, x] = y_curr - 1
            else:
                if cost_mid <= cost_down + turn_penalty:
                    cost_matrix[y_curr, x] = cost_mid + location_cost_value
                    path_matrix[y_curr, x] = y_curr
                else:
                    cost_matrix[y_curr, x] = cost_down + turn_penalty + location_cost_value
                    path_matrix[y_curr, x] = y_curr + 1

        # Bottommost pixel
        cost_up = prev_total_costs[height - 2]
        cost_mid = prev_total_costs[height - 1]
        location_cost_value = location_cost[height - 1, x]
        if cost_mid <= cost_up + turn_penalty:
            cost_matrix[height - 1, x] = cost_mid + location_cost_value
            path_matrix[height - 1, x] = height - 1
        else:
            cost_matrix[height - 1, x] = cost_up + turn_penalty + location_cost_value
            path_matrix[height - 1, x] = height - 2

    # Backtrack to find optimal path
    path = np.zeros(width, dtype=np.int32)
    final_col_costs = cost_matrix[:, width - 1]

    # Number does not support np.all, use loop instead
    all_unreachable = True
    for c in final_col_costs:
        if c != INT_MAX:
            all_unreachable = False
            break

    if all_unreachable:
        path[-1] = 0
    else:
        path[-1] = np.argmin(final_col_costs)

    for x in range(width - 2, -1, -1):
        path[x] = path_matrix[path[x + 1], x + 1]

    return path


def path_finding_find_path(binary_image: np.ndarray) -> np.ndarray:
    """
    Wrapper function to find the optimal path in a binary image using path_finding algorithm.
    :param binary_image: 2D binary image as a numpy array.
    :return: 1D numpy array of y-coordinates representing the path.
    """
    if binary_image.ndim != 2:
        raise ValueError("Function expects a 2D image array.")
    if binary_image.dtype != np.uint8:
        raise ValueError("Function expects a uint8 image array.")

    LOCATION_PENALTY = 1
    TURN_PENALTY = 2

    # Call the core path_finding function
    path = path_finding_core(binary_image, LOCATION_PENALTY, TURN_PENALTY)
    print(f'Optimal path ends at row: {path[-1]}')

    return path


def find_reference_line(
        binary_image: np.ndarray,
        path: np.ndarray,
        break_tolerance: int = 10
) -> np.ndarray:
    """
    Find the left and right endpoints of the path found by the path_finding algorithm.
    :param binary_image: 2D binary image as a numpy array.
    :param path: 1D numpy array of y-coordinates representing the path.
    :param break_tolerance: Number of consecutive background pixels to consider as a break.
    :return: List of (y, x) tuples representing the trimmed path.
    """
    width = len(path)
    if width == 0:
        return np.array([], dtype=np.int32).reshape(0, 2) # Handle empty path gracefully

    center_x = width // 2

    # Find left endpoint
    left_endpoint_x = 0  # Default to the very beginning of the image
    background_count = 0
    for x in range(center_x, -1, -1):
        if binary_image[path[x], x] == 255:  # Background pixel
            background_count += 1
        else:
            background_count = 0  # Reset on foreground pixel

        if background_count >= break_tolerance:
            left_endpoint_x = x + break_tolerance
            break

    # Find right endpoint
    right_endpoint_x = width - 1  # Default to the very end of the image
    background_count = 0
    for x in range(center_x, width):
        if binary_image[path[x], x] == 255:  # Background pixel
            background_count += 1
        else:
            background_count = 0  # Reset on foreground pixel

        if background_count >= break_tolerance:
            right_endpoint_x = x - break_tolerance
            break

    # Create path if valid
    if left_endpoint_x > right_endpoint_x:
        return np.array([], dtype=np.int32).reshape(0, 2)

    # Create the x coordinates for the trimmed path
    x_coords = np.arange(left_endpoint_x, right_endpoint_x + 1)
    # Get the corresponding y coordinates from the path array
    y_coords = path[x_coords]

    # Stack them together into an (N, 2) array
    trimmed_path = np.stack((y_coords, x_coords), axis=1)
    return trimmed_path


def erase_path(
    binary_image: np.ndarray,
    path: np.ndarray,
    y_erase_radius_ratio: float = 0.015
) -> np.ndarray:
    """
    Erase the given path from the binary image with a vertical thickness.
    :param binary_image: 2D binary image as a numpy array.
    :param path: List of y-coordinates representing the path, found by path_finding.
    :param y_erase_radius_ratio: Ratio of vertical erase radius to image height.
    :return: Modified binary image with the path erased.
    """
    if binary_image.ndim != 2:
        raise ValueError("Function expects a 2D image array.")
    if binary_image.dtype != np.uint8:
        raise ValueError("Function expects a uint8 image array.")

    height, width = binary_image.shape
    if len(path) != width:
         raise ValueError("Path length must equal image width")

    y_erase_radius = int(height * y_erase_radius_ratio)

    # Create copy of the image to modify
    modified_image = binary_image.copy()

    # Create an array of all x-coordinates
    x_coords = np.arange(width)

    # Create a 2D grid of y-coordinates for the erase region around the path.
    # This creates a (2 * y_erase_radius + 1, width) array.
    # Each column `j` contains [path[j]-r, path[j]-r+1, ..., path[j]+r]
    y_offsets = np.arange(-y_erase_radius, y_erase_radius + 1)
    y_coords_grid = path + y_offsets[:, np.newaxis]

    # Clip the y-coordinates to be within the image bounds [0, height-1]
    np.clip(y_coords_grid, 0, height - 1, out=y_coords_grid)

    # Use the calculated x and y coordinates to set pixels to white (255).
    # This advanced indexing operation is highly efficient.
    modified_image[y_coords_grid, x_coords] = 255

    return modified_image


def find_reference_lines(
        binary_image: np.ndarray,
        line_count : int = 4,
        break_tolerance: int = 10,
        y_erase_radius_ratio: float = 0.015
) -> List[np.ndarray]:
    """
    Find multiple reference lines in the binary image using path_finding algorithm.
    :param binary_image: 2D binary image as a numpy array.
    :param line_count: Number of reference lines to find.
    :param break_tolerance: Number of consecutive background pixels to consider as a break.
    :param y_erase_radius_ratio: Ratio of vertical erase radius to image height when erasing paths.
    :return: List of reference lines, each represented as a list of (y, x) tuples.
    """
    if binary_image.ndim != 2:
        raise ValueError("Function expects a 2D image array.")
    if binary_image.dtype != np.uint8:
        raise ValueError("Function expects a uint8 image array.")

    # Copy image to modify
    modified_image = binary_image.copy()
    reference_lines = []

    for _ in range(line_count):
        path = path_finding_find_path(modified_image)

        reference_line = find_reference_line(modified_image, path, break_tolerance)
        # Only add non-empty lines
        if reference_line.size > 0:
            reference_lines.append(reference_line)

        modified_image = erase_path(modified_image, path, y_erase_radius_ratio)

    # Filter out any potential empty arrays before sorting to avoid errors
    reference_lines = [line for line in reference_lines if line.shape[0] > 0]

    if not reference_lines:
        return []

    # Assuming lines do not cross each other
    # Sort by the y-coordinate of the first point in each line array
    reference_lines.sort(key=lambda line_array: line_array[0, 0])

    return reference_lines


def pad_or_crop(image: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
    """
    Pad or crop the image according to the specified distances.
    Positive values indicate padding, negative values indicate cropping.
    :param image: Input image as a numpy array.
    :param top: Top padding/cropping distance.
    :param bottom: Bottom padding/cropping distance.
    :param left: Left padding/cropping distance.
    :param right: Right padding/cropping distance.
    :return: Modified image as a numpy array.
    """

    h, w = image.shape[:2]

    crop_top = max(0, -top)
    crop_bottom = max(0, -bottom)
    crop_left = max(0, -left)
    crop_right = max(0, -right)

    # Border check
    if crop_top + crop_bottom >= h:
        raise ValueError(f"Cropping too large: top + bottom = {crop_top + crop_bottom} pixels, but image height is only {h}")
    if crop_left + crop_right >= w:
        raise ValueError(f"Cropping too large: left + right = {crop_left + crop_right} pixels, but image width is only {w}")

    # Crop first
    image = image[
            crop_top: h - crop_bottom if crop_bottom else None,
            crop_left: w - crop_right if crop_right else None
            ]

    # Then pad
    pad_top = max(0, top)
    pad_bottom = max(0, bottom)
    pad_left = max(0, left)
    pad_right = max(0, right)

    if pad_top or pad_bottom or pad_left or pad_right:
        image = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_REPLICATE
        )

    return image


def generate_control_points(
        reference_lines: List[np.ndarray],
        target_image_width: int,
        target_image_height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if not reference_lines or len(reference_lines) != len(target_y_ratios):
        raise ValueError(f'Number of reference lines ({len(reference_lines)}) does not match expected ({len(target_y_ratios)}).')

    reference_line_count = len(reference_lines)

    # --- STAGE 1, STEP 1.1: Determine the optimal number of sampling points (N) ---

    # Calculate the horizontal span (x_max - x_min) for each line.
    line_spans = [np.ptp(line[:, 1]) for line in reference_lines if line.size > 0]
    average_span = np.mean(line_spans) if line_spans else 0.0

    num_sampling_points = int(np.ceil((average_span / 100.0) * POINTS_PER_100_PIXELS))
    num_sampling_points = np.clip(num_sampling_points, MIN_POINTS_PER_LINE, MAX_POINTS_PER_LINE)

    # --- STAGE 1, STEP 1.2 & 1.3: Resample reference lines to get source points ---

    all_source_points = []
    for i, line_array in enumerate(reference_lines):
        # Strict check: Fail immediately if a line is invalid or too small.
        if len(line_array) < num_sampling_points:
            raise ValueError(
                f"Reference line {i} has only {len(line_array)} points, "
                f"but {num_sampling_points} sampling points are required."
            )

        x_coords = line_array[:, 1]
        x_min, x_max = int(x_coords.min()), int(x_coords.max())

        target_x_coords = np.linspace(x_min, x_max, num_sampling_points, dtype=int)

        # Directly calculate indices based on the continuous nature of x_coords
        indices = target_x_coords - x_min

        resampled_line = line_array[indices]
        all_source_points.append(resampled_line)

    # Vertically stack the list of arrays into a single (number of lines * number of points, 2) array
    source_points = np.vstack(all_source_points).astype(np.float32)
    # The points are in (y, x) order, so we swap them to (x, y) for transform functions.
    source_points = source_points[:, ::-1]

    # --- STAGE 2: Generate destination points ---

    # Step 2.1: Calculate target x-coordinates based on ratios.
    target_x_min = target_image_width * target_x_min_ratio
    target_x_max = target_image_width * target_x_max_ratio
    dest_x_coords = np.linspace(target_x_min, target_x_max, num_sampling_points)

    # Step 2.2: Calculate target y-coordinates based on ratios.
    dest_y_coords = np.array(target_y_ratios) * target_image_height

    # Step 2.3: Build the complete destination grid.
    # We need to pair each of the y-coordinates with the full set of N x-coordinates.
    tiled_dest_x = np.tile(dest_x_coords, reference_line_count)
    repeated_dest_y = np.repeat(dest_y_coords, num_sampling_points)

    # Combine them into a (number of lines * number of points, 2) array of (x, y) points.
    destination_points = np.vstack([tiled_dest_x, repeated_dest_y]).T.astype(np.float32)

    return source_points, destination_points


def dewarp(
        grey_image: np.ndarray,
        reference_lines: List[np.ndarray]
)-> np.ndarray:
    if grey_image.ndim != 2:
        raise ValueError("Function expects a 2D image array.")
    if grey_image.dtype != np.uint8:
        raise ValueError("Function expects a uint8 image array.")
    height, width = grey_image.shape

    # Calculate the approximate bounding box of all reference lines in the original image
    num_lines = len(reference_lines)
    assert num_lines >= 2, f"Expected at least 2 reference lines, got {num_lines}"
    # X min = average of all lines' left endpoints
    # X max = average of all lines' right endpoints
    x_mins = [line[0][1] for line in reference_lines]
    x_maxs = [line[-1][1] for line in reference_lines]
    x_min = sum(x_mins) // num_lines
    x_max = sum(x_maxs) // num_lines
    # Y min = average of topmost line's y values
    # Y max = average of bottommost line's y values
    y_min = sum(p[0] for p in reference_lines[0]) // len(reference_lines[0])
    y_max = sum(p[0] for p in reference_lines[-1]) // len(reference_lines[-1])

    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1

    print(f'Original bbox: x[{x_min}, {x_max}], y[{y_min}, {y_max}], w={bbox_width}, h={bbox_height}')

    x_min_expanded = x_min - int(left_extend_ratio * bbox_width) # Which is also -offset_x for the points
    y_min_expanded = y_min - int(top_extend_ratio * bbox_height) # Which is also -offset_y for the points
    x_max_expanded = x_max + int(right_extend_ratio * bbox_width)
    y_max_expanded = y_max + int(bottom_extend_ratio * bbox_height)

    bbox_expanded_width = x_max_expanded - x_min_expanded + 1
    bbox_expanded_height = y_max_expanded - y_min_expanded + 1

    print(f'Expanded bbox: x[{x_min_expanded}, {x_max_expanded}], y[{y_min_expanded}, {y_max_expanded}], w={bbox_expanded_width}, h={bbox_expanded_height}')
    
    pad_or_crop_left_distance = 0 - x_min_expanded
    pad_or_crop_right_distance = x_max_expanded - (width - 1)
    pad_or_crop_top_distance = 0 - y_min_expanded
    pad_or_crop_bottom_distance = y_max_expanded - (height - 1)

    padded_image = pad_or_crop(grey_image,
                               pad_or_crop_top_distance,
                               pad_or_crop_bottom_distance,
                               pad_or_crop_left_distance,
                               pad_or_crop_right_distance)

    print(f'Size of padded image: {padded_image.shape[1]} x {padded_image.shape[0]}')

    # Now offset all points in reference lines
    offset_x = pad_or_crop_left_distance
    offset_y = pad_or_crop_top_distance
    print(f'Offset to apply to reference lines: x={offset_x}, y={offset_y}')

    offset_vector = np.array([offset_y, offset_x], dtype=np.int32)
    reference_lines_with_offset = [line_array + offset_vector for line_array in reference_lines]

    # Validate that all points are within the bounds of the padded image
    image_bounds = np.array([bbox_expanded_height, bbox_expanded_width], dtype=np.int32)
    # Iterate through each line to check if its points are within bounds.
    for line_array in reference_lines_with_offset:
        # Check only non-empty line arrays to avoid errors.
        if line_array.size == 0:
            continue

        # np.all() efficiently checks if all points in the array satisfy the condition.
        # The condition checks that all coordinates are >= 0 and < the image_bounds.
        if not np.all((line_array >= 0) & (line_array < image_bounds)):
            # If any point is out of bounds, raise an error.
            raise ValueError(
                "A point in one of the reference lines is out of bounds after applying the offset."
            )

    source_points, destination_points = generate_control_points(reference_lines_with_offset, bbox_expanded_width, bbox_expanded_height)

    # Do the actual dewarp
    tps = ski.transform.ThinPlateSplineTransform()
    tps.estimate(destination_points, source_points)
    # Use cval=1 to fill out-of-bounds areas with white
    dewarped_float = ski.transform.warp(padded_image, tps, cval=1)
    dewarped_uint8 = np.clip(dewarped_float * 255, 0, 255).astype(np.uint8)
    return dewarped_uint8


def run_and_visualize(image_path: str):
    """
    加载图像，找到参考线，高质量缩放并显示原始图+参考线，
    然后调用dewarp并高质量缩放、显示校正后的图。
    """
    try:
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray_image is None:
            raise FileNotFoundError
    except Exception:
        print(f"错误：无法加载图像，请检查路径是否正确: {image_path}")
        return

    print(f"图像 '{image_path}' 加载成功，正在寻找参考线...")
    # ... (您的处理逻辑保持不变)
    binary_image = sauvola_threshold(gray_image, window_size=401, k=0.4)
    reference_lines = find_reference_lines(binary_image, line_count=4)

    print("参考线寻找完成！正在进行图像校正...")
    dewarped_image = dewarp(gray_image, reference_lines)
    print("图像校正完成！")

    # --- 第一部分：高质量显示原始图像 + 参考线 ---
    image_with_lines = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    for i, ref_line in enumerate(reference_lines):
        color = colors[i % len(colors)]
        for j in range(1, len(ref_line)):
            pt1 = (int(ref_line[j - 1][1]), int(ref_line[j - 1][0]))
            pt2 = (int(ref_line[j][1]), int(ref_line[j][0]))
            cv2.line(image_with_lines, pt1, pt2, color, thickness=4)

    win1_name = "1. Original Image with Reference Lines (High Quality Scaling)"
    cv2.namedWindow(win1_name, cv2.WINDOW_AUTOSIZE)  # 改为 AUTOSIZE，让窗口自动匹配我们提供的图像

    # 计算缩放比例
    screen_w, screen_h = get_screen_size()
    img_h, img_w, _ = image_with_lines.shape
    scale = min(screen_w / img_w, screen_h / img_h) * 0.8

    # 决定最终要显示的图像
    if scale < 1.0:
        # 如果需要缩小，就使用 cv2.resize 创建一个高质量的缩略图
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        # cv2.INTER_AREA 是缩小图像的最佳插值方法
        display_img1 = cv2.resize(image_with_lines, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        # 如果不需要缩小，就直接显示原图
        display_img1 = image_with_lines

    cv2.imshow(win1_name, display_img1)
    print("已显示原始图像和参考线。按任意键继续以显示校正后的图像...")
    cv2.waitKey(0)

    # --- 第二部分：高质量显示校正后的图像 ---
    win2_name = "2. Dewarped Image (High Quality Scaling)"
    cv2.namedWindow(win2_name, cv2.WINDOW_AUTOSIZE)  # 同样改为 AUTOSIZE

    img_h, img_w = dewarped_image.shape
    # 使用与上面相同的缩放逻辑
    scale = min(screen_w / img_w, screen_h / img_h) * 0.8

    if scale < 1.0:
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        # 再次使用 cv2.INTER_AREA 进行高质量缩放
        display_img2 = cv2.resize(dewarped_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        display_img2 = dewarped_image

    cv2.imshow(win2_name, display_img2)
    print("已显示校正结果。按任意键关闭所有窗口。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(r'E:\FYP\TcmPrescriptionOcr\test\lines.png', image_with_lines)
    cv2.imwrite(r'E:\FYP\TcmPrescriptionOcr\test\dewarped.png', dewarped_image)


# ===================================================================
#  程序入口
# ===================================================================
if __name__ == "__main__":
    # 定义要处理的图片路径
    TARGET_IMAGE_PATH = r'E:\FYP\TcmPrescriptionOcr\test\camera_uneven_lighting.png'

    # 选项 2: 运行并可视化结果
    run_and_visualize(TARGET_IMAGE_PATH)