"""
新型分割算法
对ctrip有效，对360.cn、google无效
"""

import json
import math

from PIL import Image, ImageDraw, ImageOps

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from utils.local_json import write_json_file, read_json_file


def generate_color_gradient_rgb(start_color, end_color, num_colors):
    # Convert hex colors to RGB
    start_rgb = mcolors.hex2color(start_color)
    end_rgb = mcolors.hex2color(end_color)

    # Generate linear gradient between start and end colors
    gradient = np.linspace(start_rgb, end_rgb, num_colors)

    # Convert RGB to 255 scale and then to tuples
    gradient_rgb = [tuple(int(x * 255) for x in rgb) for rgb in gradient]
    return gradient_rgb


# def read_json_file(path):
#     with open(path, 'r') as f:
#         data = json.load(f)
#     return data


def merge_intervals(boxes, direction='horizontal'):
    if direction == 'vertical':
        # 根据x_min排序
        sorted_boxes = sorted(boxes, key=lambda box: box[0])
        min_index, max_index = 0, 2
    elif direction == 'horizontal':
        # 根据y_min排序
        sorted_boxes = sorted(boxes, key=lambda box: box[1])
        min_index, max_index = 1, 3
    else:
        raise ValueError("Direction must be 'horizontal' or 'vertical'")

    # 初始化合并后的结果
    merged = []

    # 开始合并区间
    for box in sorted_boxes:
        min_val = box[min_index]
        max_val = box[max_index]

        # 如果merged为空或者当前box不与上一个合并区间重叠
        if not merged or merged[-1][1] < min_val:
            merged.append((min_val, max_val))
        else:
            # 如果重叠，合并区间
            merged[-1] = (merged[-1][0], max(merged[-1][1], max_val))

    return merged


def calculate_intervals_gaps(merged_intervals, direction):
    gaps = []
    for i in range(1, len(merged_intervals)):
        prev_max = merged_intervals[i - 1][1]
        curr_min = merged_intervals[i][0]
        gap = curr_min - prev_max
        gaps.append((gap, prev_max, curr_min, direction))
    return gaps


# def check_split_end(bounding_boxes, canvas_width, canvas_height):
#     """
#     检查是否终止分割：
#     1、高度阈值 box_height < height / 8
#     2、宽度阈值 box_width < width / 4
#
#     切图的局部区域是否小于指定的大小
#     比如切出来的块小于(w/4, h/8)就不再继续分割
#     后续终止条件可参考1）切图内的bbox数量，2）区域大小
#     """
#     if len(bounding_boxes) == 0:
#         return True
#     # 高度阈值更小一点，宽度阈值更大一点
#     filter_boxes = [box for box in bounding_boxes if box[3] - box[1] < canvas_height / 8 and box[2] - box[0] < canvas_width / 4]
#     if len(filter_boxes) == len(bounding_boxes):
#         return True
#     return False


def check_split_end(bounding_boxes, local_width, local_height, canvas_width, canvas_height):
    """
    检查是否终止分割
    """
    if len(bounding_boxes) == 0:
        return True

    threshold = 200
    w_intervals = merge_intervals(bounding_boxes, direction='vertical')
    h_intervals = merge_intervals(bounding_boxes, direction='horizontal')
    if (local_width < threshold and len(w_intervals) < 2) or (local_height < threshold and len(h_intervals) < 2):
        return True

    # 高度阈值更小一点，宽度阈值更大一点
    filter_boxes = [box for box in bounding_boxes if box[3] - box[1] < canvas_height / 8 and box[2] - box[0] < canvas_width / 4]
    if len(filter_boxes) == len(bounding_boxes):
        return True
    return False


def find_split_lines(bounding_boxes, canvas_width=None, canvas_height=None, origin_point=(0, 0), check_end=None):
    # if not bounding_boxes:
    #     return []
    if check_end(bounding_boxes, canvas_width, canvas_height):
        return []

    if canvas_width is None or canvas_height is None:
        raise ValueError("Canvas dimensions must be specified when the canvas dimensions are known.")

    # 合并bounding_box
    # 横向合并
    merged_intervals_horizontal = merge_intervals(bounding_boxes, direction='vertical')  # [1, 2]  [5, 7]  [8, 11]
    # 纵向合并
    merged_intervals_vertical = merge_intervals(bounding_boxes, direction='horizontal')
    # 计算横向区间之间的间距
    horizontal_gaps = calculate_intervals_gaps(merged_intervals_horizontal, direction='vertical')  # 5-2=3, 8-7=1
    # 计算纵向区间之间的间距
    vertical_gaps = calculate_intervals_gaps(merged_intervals_vertical, direction='horizontal')

    all_gaps = horizontal_gaps + vertical_gaps
    if not all_gaps:
        return []

    # 按照间距从大到小排序，默认间距大的地方为bbox布局被切分的地方
    max_gap = max(all_gaps, key=lambda x: x[0])

    split_lines = []

    gap, prev_max, curr_min, direction = max_gap
    # 取间距的中点为分割点
    split_position = (prev_max + curr_min) / 2  # (5+2)/2=3.5  (7+8)/2=7.5

    # 确定分割线的起点和终点
    if direction == 'vertical':
        start_point = (split_position, origin_point[1])
        end_point = (split_position, origin_point[1] + canvas_height)
    elif direction == 'horizontal':
        start_point = (origin_point[0], split_position)
        end_point = (origin_point[0] + canvas_width, split_position)
    else:
        raise ValueError("Direction must be 'horizontal' or 'vertical'")

    # 记录分割线信息
    split_lines.append({
        'split_position': split_position,
        'start_point': start_point,
        'end_point': end_point,
        'direction': direction
    })

    # 分割后的子区域需要进一步处理
    if direction == 'vertical':
        # 左侧区域
        split_lines.extend(find_split_lines(
            bounding_boxes=[box for box in bounding_boxes if box[2] <= split_position],
            canvas_width=split_position - origin_point[0],
            canvas_height=canvas_height,
            origin_point=(origin_point[0], origin_point[1]),
            check_end=check_end,
        ))

        # 右侧区域
        split_lines.extend(find_split_lines(
            bounding_boxes=[box for box in bounding_boxes if box[0] > split_position],
            canvas_width=origin_point[0] + canvas_width - split_position,
            canvas_height=canvas_height,
            origin_point=(split_position, origin_point[1]),
            check_end=check_end,
        ))

    elif direction == 'horizontal':
        # 上方区域
        split_lines.extend(find_split_lines(
            bounding_boxes=[box for box in bounding_boxes if box[3] <= split_position],
            canvas_width=canvas_width,
            canvas_height=split_position - origin_point[1],
            origin_point=(origin_point[0], origin_point[1]),
            check_end=check_end,
        ))

        # 下方区域
        split_lines.extend(find_split_lines(
            bounding_boxes=[box for box in bounding_boxes if box[1] > split_position],
            canvas_width=canvas_width,
            canvas_height=origin_point[1] + canvas_height - split_position,
            origin_point=(origin_point[0], split_position),
            check_end=check_end,
        ))

    return split_lines


def check_where_line_is(line, split_position, direction):
    """
    find_split_lines_v2专用函数
    判断line是在split_position的左侧还是右侧，上侧还是下侧
    """
    start_point = line["start_point"]
    if direction == 'vertical':
        # 左侧为True，右侧为False
        return start_point[0] < split_position
    if direction == "horizontal":
        # 上侧为True，下侧为False
        return start_point[1] < split_position


def find_split_lines_v2(bounding_boxes, canvas_width=None, canvas_height=None, origin_point=(0, 0), check_end=None, lines=None, is_auto_split=True):
    """间隔中点分割法版本二：支持导入已经存在的分割线
    is_auto_split: True 启动自动分割算法，False 不启用
    自动分割算法会根据bbox的分布进行分割，默认启用
    """
    # if not bounding_boxes:
    #     return []
    if not lines and check_end(bounding_boxes, canvas_width, canvas_height):
        return []

    if canvas_width is None or canvas_height is None:
        raise ValueError("Canvas dimensions must be specified when the canvas dimensions are known.")

    if not lines:
        # 不启用bbox自动分割，只使用分割线分割
        if not is_auto_split:
            return []
        # 合并bounding_box
        # 横向合并
        merged_intervals_horizontal = merge_intervals(bounding_boxes, direction='vertical')  # [1, 2]  [5, 7]  [8, 11]
        # 纵向合并
        merged_intervals_vertical = merge_intervals(bounding_boxes, direction='horizontal')
        # 计算横向区间之间的间距
        horizontal_gaps = calculate_intervals_gaps(merged_intervals_horizontal, direction='vertical')  # 5-2=3, 8-7=1
        # 计算纵向区间之间的间距
        vertical_gaps = calculate_intervals_gaps(merged_intervals_vertical, direction='horizontal')

        all_gaps = horizontal_gaps + vertical_gaps
        if not all_gaps:
            return []

        from ablation_config import is_gap_sort
        # 按照间距从大到小排序，默认间距大的地方为bbox布局被切分的地方
        max_gap = max(all_gaps, key=lambda x: x[0]) if is_gap_sort else all_gaps[-1]  # 消融点2: 投影间距是否排序

        gap, prev_max, curr_min, direction = max_gap
        # 取间距的中点为分割点
        split_position = (prev_max + curr_min) / 2  # (5+2)/2=3.5  (7+8)/2=7.5
    else:
        sep_line, *lines = lines
        # 直接使用split_position和direction，而不是position，所以无需对齐分割线
        split_position = sep_line["split_position"]
        direction = sep_line["direction"]

    # 确定分割线的起点和终点
    if direction == 'vertical':
        start_point = (split_position, origin_point[1])
        end_point = (split_position, origin_point[1] + canvas_height)
    elif direction == 'horizontal':
        start_point = (origin_point[0], split_position)
        end_point = (origin_point[0] + canvas_width, split_position)
    else:
        raise ValueError("Direction must be 'horizontal' or 'vertical'")

    # 记录分割线信息
    split_lines = [
        {
            'split_position': split_position,
            'start_point': start_point,
            'end_point': end_point,
            'direction': direction
        }
    ]

    # 分割后的子区域需要进一步处理
    if direction == 'vertical':
        # 左侧区域
        split_lines.extend(find_split_lines_v2(
            bounding_boxes=[box for box in bounding_boxes if box[2] <= split_position],
            canvas_width=split_position - origin_point[0],
            canvas_height=canvas_height,
            origin_point=(origin_point[0], origin_point[1]),
            check_end=check_end,
            lines=[line for line in lines if check_where_line_is(line, split_position, direction)],
            is_auto_split=is_auto_split
        ))

        # 右侧区域
        split_lines.extend(find_split_lines_v2(
            bounding_boxes=[box for box in bounding_boxes if box[0] > split_position],
            canvas_width=origin_point[0] + canvas_width - split_position,
            canvas_height=canvas_height,
            origin_point=(split_position, origin_point[1]),
            check_end=check_end,
            lines=[line for line in lines if not check_where_line_is(line, split_position, direction)],
            is_auto_split=is_auto_split
        ))

    elif direction == 'horizontal':
        # 上方区域
        split_lines.extend(find_split_lines_v2(
            bounding_boxes=[box for box in bounding_boxes if box[3] <= split_position],
            canvas_width=canvas_width,
            canvas_height=split_position - origin_point[1],
            origin_point=(origin_point[0], origin_point[1]),
            check_end=check_end,
            lines=[line for line in lines if check_where_line_is(line, split_position, direction)],
            is_auto_split=is_auto_split
        ))

        # 下方区域
        split_lines.extend(find_split_lines_v2(
            bounding_boxes=[box for box in bounding_boxes if box[1] > split_position],
            canvas_width=canvas_width,
            canvas_height=origin_point[1] + canvas_height - split_position,
            origin_point=(origin_point[0], split_position),
            check_end=check_end,
            lines=[line for line in lines if not check_where_line_is(line, split_position, direction)],
            is_auto_split=is_auto_split
        ))

    return split_lines


def draw_label_on_line(start_point, end_point, label, draw):
    """给直线打标签"""
    x = start_point[0] + (end_point[0] - start_point[0]) / 2
    y = start_point[1] + (end_point[1] - start_point[1]) / 2
    draw.text((x, y), str(label), fill="red")


def change_bg_color(img, old_color=(0, 0, 0), new_color=(147, 112, 219)):
    """更换背景色：黑色->紫色"""
    image_rgb = img.convert('RGB')
    pixels = image_rgb.load()

    # 遍历每个像素，将黑色(灰度值为0)替换为紫色
    width, height = image_rgb.size
    for y in range(height):
        for x in range(width):
            if pixels[x, y] == old_color:  # 如果灰度图像中的值是0，表示黑色
                pixels[x, y] = new_color  # 替换为紫色
    # 转换为RGB模式
    return image_rgb


def draw_split_lines_on_image(image_path, split_lines, output_path, is_show=True):
    # Define start and end colors
    start_color = '#0000FF'  # Blue
    end_color = '#00FF00'  # Green

    # Number of colors (e.g., for 5 sequential steps)
    num_colors = len(split_lines)

    # Generate the gradient in RGB format
    color_gradient_rgb = generate_color_gradient_rgb(start_color, end_color, num_colors)

    # 打开图像
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    blank_image = Image.new("L", image.size, 255)
    blank_draw = ImageDraw.Draw(blank_image)

    # 设置分割线颜色和宽度
    line_color = (255, 0, 0)  # 红色
    line_width = 3

    # 绘制分割线
    for i, line in enumerate(split_lines):
        start_point = tuple(float(num) for num in line['start_point'])
        end_point = tuple(float(num) for num in line['end_point'])
        draw.line([start_point, end_point], fill=color_gradient_rgb[i], width=line_width)
        blank_draw.line([start_point, end_point], fill="black", width=5)
        draw_label_on_line(start_point, end_point, i, draw)

    # 保存或显示图像
    image.save(output_path)
    blank_image = change_bg_color(ImageOps.invert(blank_image))
    blank_image.save(output_path.replace(".png", "_mask.png"))  # 用于下一阶段的mask图
    is_show and image.show()


"""
    bbox预处理：uied + layout
"""


def bbox_intersection(bbox1, bbox2):
    """检查两个 bbox 是否有交集"""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    intersect_x_min = max(x1_min, x2_min)
    intersect_y_min = max(y1_min, y2_min)
    intersect_x_max = min(x1_max, x2_max)
    intersect_y_max = min(y1_max, y2_max)

    return intersect_x_min < intersect_x_max and intersect_y_min < intersect_y_max


def merge_bboxes(bbox1, bbox2):
    """合并两个 bbox 到最小外接矩形"""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    merged_x_min = min(x1_min, x2_min)
    merged_y_min = min(y1_min, y2_min)
    merged_x_max = max(x1_max, x2_max)
    merged_y_max = max(y1_max, y2_max)

    return [merged_x_min, merged_y_min, merged_x_max, merged_y_max]


def merge_all_bboxes(bboxes):
    """合并所有有交集的 bbox"""
    merged = True
    while merged:
        merged = False
        new_bboxes = []
        skip_indices = set()

        for i in range(len(bboxes)):
            if i in skip_indices:
                continue

            bbox1 = bboxes[i]
            for j in range(i + 1, len(bboxes)):
                if j in skip_indices:
                    continue

                bbox2 = bboxes[j]
                if bbox_intersection(bbox1, bbox2):
                    # 合并 bbox
                    bbox1 = merge_bboxes(bbox1, bbox2)
                    skip_indices.add(j)
                    merged = True

            new_bboxes.append(bbox1)

        bboxes = new_bboxes

    return bboxes


def draw_bboxes(image_size, bboxes, bbox_color=(255, 0, 0), outline_width=2):
    """
    在指定大小的画布上绘制 bounding boxes。

    :param image_size: 画布的尺寸 (width, height)
    :param bboxes: 包含所有 bbox 的列表，每个 bbox 以 (x_min, y_min, x_max, y_max) 表示
    :param bbox_color: bbox 的颜色，默认为红色
    :param outline_width: bbox 的边框宽度，默认为2
    :return: 绘制了 bbox 的图像对象
    """
    from PIL import Image, ImageDraw

    # 创建白色背景的空白图像
    image = Image.new("RGB", image_size, (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # 绘制每个 bbox
    for bbox in bboxes:
        draw.rectangle(bbox, outline=bbox_color, width=outline_width)
    image.show("")
    return image


def merge_uied_and_layouts(image_path, output_root, is_show=True):
    """融合UIED merge和layouts bbox，统一bbox格式"""
    from os.path import join as pjoin

    name = image_path.split('/')[-1][:-4]
    merge_root = pjoin(output_root, "merge")
    layout_root = pjoin(output_root, "layout")

    uied_path = pjoin(merge_root, f"{name}.json")
    layout_path = pjoin(layout_root, f"{name}.json")

    uied_data = read_json_file(uied_path)
    from ablation_config import is_ui_group
    layouts_data = read_json_file(layout_path) if is_ui_group else []  # 消融点1：是否进行UI分组
    image_size = uied_data["img_shape"][::-1][1:]
    bboxes = [
        [box["position"]["column_min"], box["position"]["row_min"], box["position"]["column_max"], box["position"]["row_max"], ]
        for box in uied_data["compos"] + layouts_data
    ]
    bboxes = merge_all_bboxes(bboxes)
    is_show and draw_bboxes(image_size, bboxes)
    return bboxes, image_size


"""
    分割线预处理
"""


def format_sep_lines(image_path, output_root, is_show=True):
    from os.path import join as pjoin

    # 格式化数据，确保和分割算法连接上
    name = image_path.split('/')[-1][:-4]

    line_root = pjoin(output_root, "line")
    lines = read_json_file(pjoin(line_root, f"{name}.json"))

    merge_root = pjoin(output_root, "merge")
    uied_data = read_json_file(pjoin(merge_root, f"{name}.json"))
    image_size = uied_data["img_shape"][::-1][1:]

    split_lines = []
    threshold = 1
    for line in lines:
        start_point = line["x1"], line["y1"]
        end_point = line["x2"], line["y2"]
        if abs(start_point[0] - end_point[0]) < threshold:
            direction = "vertical"
            split_position = start_point[0]
        elif abs(start_point[1] - end_point[1]) < threshold:
            direction = "horizontal"
            split_position = start_point[1]
        else:
            continue  # bugfix: 2024.9.29
            # raise ValueError(f"Invalid line: Point {start_point} and {end_point}")
        split_lines.append({
            "start_point": start_point,
            "end_point": end_point,
            "direction": direction,
            "split_position": split_position
        })
    # 按照规则排序
    sorted_split_lines = sort_lines(split_lines)
    is_show and draw_lines_with_order(sorted_split_lines, image_size=image_size)


    return sorted_split_lines


def normalize_line(line):
    """
    规范化线段的起始点和终止点，使水平线从左到右，垂直线从上到下。

    :param line: 线段字典，包含 "start_point", "end_point", "direction"
    :return: 规范化后的线段
    """
    start_point = line["start_point"]
    end_point = line["end_point"]
    direction = line["direction"]

    # 对于水平线，确保从左到右
    if direction == "horizontal" and start_point[0] > end_point[0]:
        line["start_point"], line["end_point"] = end_point, start_point

    # 对于垂直线，确保从上到下
    if direction == "vertical" and start_point[1] > end_point[1]:
        line["start_point"], line["end_point"] = end_point, start_point

    return line


def line_position(line):
    """
    判断线段的类型和位置。

    :param line: 包含线段信息的字典，包含 'start_point', 'end_point', 'direction', 'split_position'
    :return: 线段类型 ('horizontal' or 'vertical') 和 位置 ((min_x, min_y), (max_x, max_y))
    """
    start_point = line["start_point"]
    end_point = line["end_point"]
    direction = line["direction"]

    if direction == "horizontal":
        return 'horizontal', ((min(start_point[0], end_point[0]), start_point[1]),
                              (max(start_point[0], end_point[0]), end_point[1]))
    elif direction == "vertical":
        return 'vertical', ((start_point[0], min(start_point[1], end_point[1])),
                            (end_point[0], max(start_point[1], end_point[1])))
    else:
        raise ValueError("Line direction is not strictly horizontal or vertical")


def compare_lines(line1, line2):
    """
    比较两条线段以确定它们的排序顺序。

    :param line1: 第一个线段的信息字典
    :param line2: 第二个线段的信息字典
    :return: 比较结果 (-1 if line1 < line2, 1 if line1 > line2, 0 if equal)
    """
    type1, pos1 = line_position(line1)
    type2, pos2 = line_position(line2)

    start_1, end_1 = pos1  # 线1的起始点、终止点
    start_2, end_2 = pos2  # 线2的起始点、终止点

    if type1 != type2:
        # 选择左边是-1，选择右边是1
        if type1 == 'horizontal' and type2 == 'vertical':
            # 规则：如果垂直线严格在水平线的上侧或下侧，则使用水平线
            if start_1[0] < start_2[0] < end_1[0]:
                return -1
            # 规则：如果水平线严格在垂直线的左侧或右侧，则使用垂直线
            if start_2[1] < end_1[1] < end_2[1]:
                return 1
            # 规则：如果垂直线不严格在水平线的下侧，则优先使用水平线
            return -1
        if type1 == 'vertical' and type2 == 'horizontal':  # ！可优化，缩减代码
            # 规则：如果垂直线严格在水平线的上侧或下侧，则使用水平线
            if start_2[0] < start_1[0] < end_2[0]:
                return 1
            # 规则：如果水平线严格在垂直线的左侧或右侧，则使用垂直线
            if start_1[1] < end_2[1] < end_1[1]:
                return -1
            # 规则：如果垂直线不严格在水平线的下侧，则优先使用水平线
            return 1
    else:
        # 相同类型的线段，根据位置排序
        if type1 == 'horizontal':
            return start_1[1] - start_2[1]
        else:
            return start_1[0] - start_2[0]


def sort_lines(lines):
    """
    根据规则对线段进行排序。

    :param lines: 包含所有线段的列表，每个线段是一个字典
    :return: 排序后的线段列表
    """
    import functools

    lines = [normalize_line(line) for line in lines]
    # 使用sorted函数和自定义的compare_lines进行排序
    sorted_lines = sorted(lines, key=functools.cmp_to_key(compare_lines))
    return sorted_lines


def draw_lines_with_order(lines, image_size=(400, 400)):
    """
    在空白图片上绘制线段并标记顺序（基于新的数据结构）。

    :param lines: 包含所有线段的列表，每个线段为字典，包含 "start_point", "end_point", "direction", "split_position"
    :param image_size: 图像的宽高 (width, height)
    :return: PIL Image 对象
    """
    from PIL import Image, ImageDraw, ImageFont

    # 创建空白图片
    img = Image.new('RGB', image_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # 尝试加载一个默认字体
    font = ImageFont.load_default(15)

    # 绘制线段并标记顺序
    for i, line in enumerate(lines):
        start_point = line["start_point"]
        end_point = line["end_point"]
        direction = line["direction"]
        split_position = line["split_position"]

        # 画线
        draw.line([start_point, end_point], fill=(0, 0, 0), width=2)

        # 标记顺序，显示方向和分割点
        label = f"{i + 1} ({direction}) @ {split_position}"
        draw.text(start_point, label, fill=(255, 0, 0), font=font)

    img.show()


def divide_layout(image_path, output_root):
    """布局分割入口"""
    import os
    from os.path import join as pjoin
    from functools import partial

    name = image_path.split('/')[-1][:-4]

    bounding_boxes, image_size = merge_uied_and_layouts(image_path, output_root, is_show=False)
    canvas_width, canvas_height = image_size
    sorted_split_lines = format_sep_lines(image_path, output_root, is_show=False)
    check_end = partial(check_split_end, canvas_width=canvas_width, canvas_height=canvas_height)
    split_lines = find_split_lines_v2(bounding_boxes, canvas_width=canvas_width, canvas_height=canvas_height, check_end=check_end, lines=sorted_split_lines)

    # 在图像上绘制分割线
    sep_root = pjoin(output_root, "sep")
    os.makedirs(sep_root, exist_ok=True)
    draw_split_lines_on_image(image_path, split_lines, output_path=pjoin(sep_root, f"{name}.png"), is_show=False)

    write_json_file(pjoin(sep_root, f"{name}.json"), {
        "split_lines": split_lines,
        "canvas_size": (canvas_width, canvas_height)
    })  # 暂时用不到


def manual_divide_layout(image_path, output_root):
    """手动布局分割入口"""
    import os
    from os.path import join as pjoin
    from functools import partial

    split_lines_data = read_json_file(image_path.replace(".png", ".json"))
    split_lines = split_lines_data["split_lines"]
    canvas_width, canvas_height = split_lines_data["canvas_size"]
    name = image_path.split('/')[-1][:-4]

    # 在图像上绘制分割线
    sep_root = pjoin(output_root, "sep")
    os.makedirs(sep_root, exist_ok=True)
    sorted_split_lines = sort_lines(split_lines)
    check_end = partial(check_split_end, canvas_width=canvas_width, canvas_height=canvas_height)
    sorted_split_lines = find_split_lines_v2([], canvas_width=canvas_width, canvas_height=canvas_height, check_end=check_end, lines=sorted_split_lines)
    draw_split_lines_on_image(image_path, sorted_split_lines, output_path=pjoin(sep_root, f"{name}.png"), is_show=False)

    # write_json_file(pjoin(sep_root, f"{name}.json"), {
    #     "split_lines": split_lines,
    #     "canvas_size": (canvas_width, canvas_height)
    # })  # 暂时用不到


if __name__ == '__main__':
    divide_layout("./data/input/real_image/bilibili.png", "./data/output/")

    # from os.path import join as pjoin
    #
    # fn = "taobao"
    # image_path = f"./data/input/real_image/{fn}.png"
    # output_root = "./data/output/"
    # output_image_path = pjoin(output_root, "sep", f"{fn}.png")
    #
    # bounding_boxes, image_size = merge_uied_and_layouts(image_path,output_root, is_show=False)
    # canvas_width, canvas_height = image_size
    # sorted_split_lines = format_sep_lines(image_path, output_root, is_show=False)
    #
    # # data = read_json_file(f'./data/output/ip/{fn}.json')
    # # boxes = data['compos']
    # # canvas_width, canvas_height = data['img_shape'][1], data['img_shape'][0]
    # # bounding_boxes = [(box["column_min"], box["row_min"], box["column_max"], box["row_max"]) for box in boxes]
    #
    # # # 横向合并
    # # merged_intervals_horizontal = merge_intervals(bounding_boxes, direction='horizontal')
    # # print("Merged Intervals (Horizontal):", merged_intervals_horizontal)
    # # # 纵向合并
    # # merged_intervals_vertical = merge_intervals(bounding_boxes, direction='vertical')
    # # print("Merged Intervals (Vertical):", merged_intervals_vertical)
    #
    # from functools import partial
    #
    # check_end = partial(check_split_end, canvas_width=canvas_width, canvas_height=canvas_height)
    #
    # # 横向分割线，使用四舍五入
    # split_lines = find_split_lines_v2(bounding_boxes, canvas_width=canvas_width, canvas_height=canvas_height, check_end=check_end, lines=sorted_split_lines)
    # print("Horizontal Split Lines (with positions):", split_lines)
    #
    # # # 指定图像路径和输出路径
    # # image_path = f"./data/input/real_image/{fn}.png"
    # # # image_path = "./data/output/ip/ctrip_blank.jpg"
    # # output_path = f"./data/output/sep/{fn}.png"
    #
    # # 在图像上绘制分割线
    # draw_split_lines_on_image(image_path, split_lines, output_image_path, is_show=False)


