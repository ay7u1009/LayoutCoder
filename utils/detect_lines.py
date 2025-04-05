"""
检测截图中的边框，所有水平线和垂直线
1、预处理图片（卷积），让边缘更明显
2、提取水平、垂直分割线
输出：示例图片、存储直线的json文件
"""
import os.path

import cv2
import numpy as np
import json

import time


def calculate_line_length(x1, y1, x2, y2):
    import math
    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return length


def remove_close_lines(lines, threshold=0, index=0):
    """移除距离小于threshold的直线"""
    sorted_lines = sorted(lines, key=lambda x: x[0][index])
    delete_lines = []
    for i in range(0, len(sorted_lines) - 1):
        # print(abs(sorted_lines[i][0][index] - sorted_lines[i + 1][0][index]))
        if abs(sorted_lines[i][0][index] - sorted_lines[i + 1][0][index]) < threshold:
            delete_lines.append(i + 1)

    new_lines = []
    for i, line in enumerate(sorted_lines):
        if i not in delete_lines:
            new_lines.append(line)
    return new_lines


def is_line_overlap_with_bbox(line, bbox, threshold=0.8):
    """
    判断线段与边界框的重合度是否过高。

    :param line: 线段，包含起点和终点的字典 {'start_point': (x1, y1), 'end_point': (x2, y2), ...}
    :param bbox: 边界框，包含边界的四元组 (x_min, y_min, x_max, y_max)
    :param threshold: 重合度阈值，默认值为 0.8（80%）
    :return: 如果重合度过高，则返回 False；否则返回 True
    """
    from utils.layout import is_line_intersect_bbox

    x1, y1 = line["x1"], line["y1"]  # start
    x2, y2 = line["x2"], line["y2"]  # end
    x_min, y_min, x_max, y_max = bbox["column_min"], bbox["row_min"], bbox["column_max"], bbox["row_max"]

    # 判断line是否与bbox有交叠（包含在bbox边界的情况）
    if not is_line_intersect_bbox((x1, y1), (x2, y2), bbox):
        return False

    # 计算线段总长度
    line_length = max(abs(x2 - x1), abs(y2 - y1))

    equal_threshold = 2.1  # bugfix: 2024.9.29#1405.png
    # 计算线段与 bbox 的交集部分长度
    if abs(y1 - y2) < equal_threshold:  # 水平线段
        if x1 > x2:
            x1, x2 = x2, x1
        overlap_length = max(0, min(x2, x_max) - max(x1, x_min))
    elif abs(x1 - x2) < equal_threshold:  # 垂直线段
        if y1 > y2:
            y1, y2 = y2, y1
        overlap_length = max(0, min(y2, y_max) - max(y1, y_min))
    else:
        raise ValueError(f"既不是水平线，又不是垂直线: {line}")

    # 计算重合度
    overlap_ratio = overlap_length / line_length

    # 判断重合度是否超过阈值
    if overlap_ratio >= threshold:
        return True

    return False


def remove_lines_on_bboxes(image_path, output_root, lines):
    """
    移除和UIED bboxes边界重叠程度过高的线（不是layouts，layouts依赖于line去除不合理的layouts）
    （line依赖于UIED去除不合理的line，比如wxreader在按钮附近的线）

    两个阈值：
    1）重合度80%
    2）线离bbox的距离（无论内外侧）2px以内，刚开始可以设置为0

    lines来自于传入参数, UIED bbox来自于路径定位

    测试用例：wxreader（需要被移除）、360.cn（不能被影响）
    """
    from os.path import join as pjoin
    from utils.local_json import read_json_file

    name = image_path.split('/')[-1][:-4]
    merge_root = pjoin(output_root, 'merge')

    uied_data = read_json_file(pjoin(merge_root, f"{name}.json"))
    bboxes = [bbox["position"] for bbox in uied_data["compos"]]

    remove_indexes = []
    for i, line in enumerate(lines):
        for bbox in bboxes:
            if is_line_overlap_with_bbox(line, bbox):
                remove_indexes.append(i)
                break

    new_lines = []
    for i, line in enumerate(lines):
        if i not in remove_indexes:
            new_lines.append(line)

    return new_lines


def detect_sep_lines_with_lsd(image_path, output_root):
    """
    检测图片中的水平垂直分割线
    输出：json
    """
    start = time.process_time()

    from os.path import join as pjoin

    name = image_path.split('/')[-1][:-4]
    line_root = pjoin(output_root, 'line')
    os.makedirs(line_root, exist_ok=True)

    output_json_path = pjoin(line_root, f'{name}.json')

    # 读取图像
    image = cv2.imread(image_path)  # 原图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 灰度图像
    width, height = gray.shape[::-1]
    min_line_ratio = 0.7

    # 预处理
    from utils.edge_detection import fined_edge_detection, zero_crossing_edge_detection
    # # zero_cross
    # sharpened = zero_crossing_edge_detection(gray)
    # # zero cross

    # sharpened
    sharpened = fined_edge_detection(gray)  # 边缘锐化的图像

    # 创建 LSD 检测器
    lsd = cv2.createLineSegmentDetector(
        refine=cv2.LSD_REFINE_STD,
        scale=0.8,
        sigma_scale=0.05,
        quant=2.0,
        ang_th=22.5,
        log_eps=0,
        density_th=0.6,
        n_bins=1024
    )

    # 使用 LSD 检测图像中的线段
    lines = lsd.detect(sharpened)[0]  # 返回的第一个元素是检测到的线段
    lines = [] if lines is None else lines

    # 过滤掉非水平和垂直的线段
    v_lines = []
    h_lines = []
    delta = 3
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 检查方向
        angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
        if abs(angle) <= delta or abs(angle) >= 180 - delta:  # 水平线段
            direction = "horizontal"
        elif 90 - delta <= abs(angle) <= 90 + delta:  # 垂直线段
            direction = "vertical"
        else:
            continue
        # 检查长度
        line_length = calculate_line_length(x1, y1, x2, y2)
        if direction == "horizontal" and line_length >= width * min_line_ratio:
            h_lines.append(line)
        elif direction == "vertical" and line_length >= height * min_line_ratio:
            v_lines.append(line)

    # 过滤掉相近的直线
    h_lines = remove_close_lines(h_lines, threshold=5, index=1)
    v_lines = remove_close_lines(v_lines, threshold=5, index=0)

    vertical_horizontal_lines = h_lines + v_lines

    # 处理线段数据并保存为 JSON 文件
    lines_data = []
    for line in vertical_horizontal_lines:
        x1, y1, x2, y2 = line[0]
        lines_data.append({
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2)
        })

    # 移除与UIED BBOX重叠程度较高的lines
    lines_data = remove_lines_on_bboxes(image_path, output_root, lines_data)

    with open(output_json_path, 'w') as json_file:
        json.dump(lines_data, json_file, indent=4)

    print("[SepLine Detection Completed in %.3f s] Input: %s Output: %s" % (time.process_time() - start, image_path, pjoin(line_root, name + '.json')))

    return lines_data


def draw_lines(image_path, output_root):
    """在图片上绘制分割线"""
    from os.path import join as pjoin
    from utils.local_json import read_json_file

    name = image_path.split('/')[-1][:-4]
    line_root = pjoin(output_root, 'line')  # 写
    layout_root = pjoin(output_root, 'layout')  # 读

    output_image_path = pjoin(line_root, f'{name}.png')
    blank_output_image_path = pjoin(line_root, f'{name}_blank.png')
    input_json_path = pjoin(line_root, f'{name}.json')

    lines = read_json_file(input_json_path)

    # 输出图片的背景图
    bg_image_path = pjoin(layout_root, f'{name}.png')
    bg_image = cv2.imread(bg_image_path)

    # 创建一个与原图像大小相同的空白图像
    background_color = (255, 255, 255)
    blank_image = np.full_like(bg_image, background_color, dtype=np.uint8)

    if lines:
        lsd = cv2.createLineSegmentDetector(0)
        lines = np.array([[line["x1"], line["y1"], line["x2"], line["y2"]] for line in lines], dtype=np.float32).reshape(-1, 1, 4)
        drawn_blank_image = lsd.drawSegments(blank_image, np.array(lines))
        drawn_image = lsd.drawSegments(bg_image, np.array(lines))
    else:
        drawn_blank_image = blank_image.copy()
        drawn_image = bg_image.copy()
    cv2.imwrite(blank_output_image_path, drawn_blank_image)
    cv2.imwrite(output_image_path, drawn_image)


if __name__ == '__main__':
    # detect_sep_lines_with_lsd("./data/input/real_image/360.cn.png", output_root="./data/output")
    draw_lines("./data/input/real_image/360.cn.png", output_root="./data/output")