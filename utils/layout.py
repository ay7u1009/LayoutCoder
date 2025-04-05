"""
对bbox的空间关系进行建模
在此基础上分析bbox所构成的布局
"""
import os.path

from utils.local_json import read_json_file, write_json_file
from utils.draw import draw_bounding_boxes

import time


"""
    bbox建模
"""


class Box(object):
    def __init__(self, data):
        self.box_id = data.get("id", -1)
        self.cls = data.get("class", 'Compo')

        # box的position坐标
        self.column_min = data["position"]["column_min"]
        self.row_min = data["position"]["row_min"]
        self.column_max = data["position"]["column_max"]
        self.row_max = data["position"]["row_max"]

        # box的宽高
        self.width = self.column_max - self.column_min
        self.height = self.row_max - self.row_min

        # === 严格意义只检查右侧👉和下侧⬇️ ===

        # 相邻的box，建议存储id，避免循环引用
        # 左右：从上往下；上下：从左往右
        self.left_children = []
        self.right_children = []
        self.top_children = []
        self.bottom_children = []

        # 相邻的box是否唯一，是否对齐
        # 左上角对齐，右上角
        self.is_left_aligned = False
        self.is_right_aligned = False  #
        self.is_top_aligned = False
        self.is_bottom_aligned = False  #

        # 相邻的box的间距, {A_id: self_pos - A_pos}
        self.left_gap = {}
        self.right_gap = {}  #
        self.top_gap = {}
        self.bottom_gap = {}  #

    def __repr__(self):
        return f"Box_{self.box_id}"

    def info(self):
        return {
            "id": self.box_id,
            "size": {
                "width": self.width,
                "height": self.height,
            },
            "position": {
                "top_left": (self.column_min, self.row_min),
                "bottom_right": (self.column_max, self.row_max),
            },
            "children": {
                "left": self.left_children,
                "right": self.right_children,
                "top": self.top_children,
                "bottom": self.bottom_children
            },
            "align": {
                "left": self.is_left_aligned,
                "right": self.is_right_aligned,
                "top": self.is_top_aligned,
                "bottom": self.is_bottom_aligned
            },
            "gap": {
                "left": self.left_gap,
                "right": self.right_gap,
                "top": self.top_gap,
                "bottom": self.bottom_gap
            }
        }
"""
    bbox 初始化建模
"""


def is_adjacent(bbox_a, bbox_b, direction):
    if direction == 'left':
        return bbox_a.column_max <= bbox_b.column_min and \
            not (bbox_a.row_max <= bbox_b.row_min or bbox_a.row_min >= bbox_b.row_max)
    elif direction == 'right':
        return bbox_a.column_min >= bbox_b.column_max and \
            not (bbox_a.row_max <= bbox_b.row_min or bbox_a.row_min >= bbox_b.row_max)
    elif direction == 'top':
        return bbox_a.row_max <= bbox_b.row_min and \
            not (bbox_a.column_max <= bbox_b.column_min or bbox_a.column_min >= bbox_b.column_max)
    elif direction == 'bottom':
        return bbox_a.row_min >= bbox_b.row_max and \
            not (bbox_a.column_max <= bbox_b.column_min or bbox_a.column_min >= bbox_b.column_max)


def is_blocked(bbox_a, bbox_b, bboxes, direction):
    """检查在bbox_a和bbox_b之间是否有其他bbox阻挡"""
    for bbox in bboxes:
        if bbox.box_id in [bbox_a.box_id, bbox_b.box_id]:
            continue

        if direction == 'left' or direction == 'right':
            if max(bbox_a.row_min, bbox_b.row_min) <= bbox.row_max and \
                    min(bbox_a.row_max, bbox_b.row_max) >= bbox.row_min:
                if (direction == 'left' and bbox_a.column_max <= bbox.column_min < bbox_b.column_min) or \
                        (direction == 'right' and bbox_a.column_min >= bbox.column_max > bbox_b.column_max):
                    return True
        elif direction == 'top' or direction == 'bottom':
            if max(bbox_a.column_min, bbox_b.column_min) <= bbox.column_max and \
                    min(bbox_a.column_max, bbox_b.column_max) >= bbox.column_min:
                if (direction == 'top' and bbox_a.row_max <= bbox.row_min < bbox_b.row_min) or \
                        (direction == 'bottom' and bbox_a.row_min >= bbox.row_max > bbox_b.row_max):
                    return True
    return False


def filter_adjacent_boxes(bbox_a, candidates, bboxes, direction):
    filtered = []
    for bbox_b in candidates:
        if not is_blocked(bbox_a, bbox_b, bboxes, direction):
            filtered.append(bbox_b.box_id)
    return filtered


def find_adjacent_boxes(bboxes):
    """bbox的邻居bbox，搜索bbox的空间关系"""
    adjacent_boxes = {bbox.box_id: bbox for bbox in bboxes}

    for i, bbox_a in enumerate(bboxes):
        candidates = {'left': [], 'right': [], 'top': [], 'bottom': []}

        for j, bbox_b in enumerate(bboxes):
            if i != j:
                if is_adjacent(bbox_a, bbox_b, 'left'):
                    candidates['left'].append(bbox_b)
                if is_adjacent(bbox_a, bbox_b, 'right'):
                    candidates['right'].append(bbox_b)
                if is_adjacent(bbox_a, bbox_b, 'top'):
                    candidates['top'].append(bbox_b)
                if is_adjacent(bbox_a, bbox_b, 'bottom'):
                    candidates['bottom'].append(bbox_b)

        adjacent_boxes[bbox_a.box_id].right_children += filter_adjacent_boxes(bbox_a, candidates['left'], bboxes, 'left')
        adjacent_boxes[bbox_a.box_id].left_children += filter_adjacent_boxes(bbox_a, candidates['right'], bboxes, 'right')
        adjacent_boxes[bbox_a.box_id].bottom_children += filter_adjacent_boxes(bbox_a, candidates['top'], bboxes, 'top')
        adjacent_boxes[bbox_a.box_id].top_children += filter_adjacent_boxes(bbox_a, candidates['bottom'], bboxes, 'bottom')

    return adjacent_boxes


def check_box_aligned(box_a, box_b, direction):
    # 对齐的阈值应当略小一点，避免跨行合并情况
    # 真正对齐的box一定是0误差的
    threshold_ratio = 0.03
    threshold = max(box_a.height, box_a.width) * threshold_ratio
    if direction == 'right' or direction == 'left':
        return abs(box_a.row_min - box_b.row_min) <= threshold and abs(box_a.row_max - box_b.row_max) <= threshold
    if direction == 'bottom' or direction == 'top':
        return abs(box_a.column_min - box_b.column_min) <= threshold and abs(box_a.column_max - box_b.column_max) <= threshold


def compute_box_alignment(bboxes):
    """计算中心box与周边box的对齐情况，如果左侧与中心box不对齐，则False"""
    for box_id, center_bbox in bboxes.items():
        bbox_right_children = center_bbox.right_children
        if len(bbox_right_children) == 1 and check_box_aligned(bboxes[bbox_right_children[0]], center_bbox, "right"):
            center_bbox.is_right_aligned = True

        bbox_left_children = center_bbox.left_children
        if len(bbox_left_children) == 1 and check_box_aligned(bboxes[bbox_left_children[0]], center_bbox, "left"):
            center_bbox.is_left_aligned = True

        bbox_bottom_children = center_bbox.bottom_children
        if len(bbox_bottom_children) == 1 and check_box_aligned(bboxes[bbox_bottom_children[0]], center_bbox, "bottom"):
            center_bbox.is_bottom_aligned = True

        bbox_top_children = center_bbox.top_children
        if len(bbox_top_children) == 1 and check_box_aligned(bboxes[bbox_top_children[0]], center_bbox, "top"):
            center_bbox.is_top_aligned = True


def compute_box_gap(bboxes):
    """计算中心box与周边box的间距"""
    def compute_box_gap_helper(box_children, center_box, direction):
        if direction == "right":
            center_box.right_gap = {child_id: bboxes[child_id].column_min - center_box.column_max for child_id in box_children}
        if direction == "left":
            center_box.left_gap = {child_id: center_box.column_min - bboxes[child_id].column_max for child_id in box_children}
        if direction == "bottom":
            center_box.bottom_gap = {child_id: bboxes[child_id].row_min - center_box.row_max for child_id in box_children}
        if direction == "top":
            center_box.top_gap = {child_id: center_box.row_min - bboxes[child_id].row_max for child_id in box_children}

    for box_id, center_bbox in bboxes.items():
        bbox_right_children = center_bbox.right_children
        compute_box_gap_helper(bbox_right_children, center_bbox, direction="right")

        bbox_left_children = center_bbox.left_children
        compute_box_gap_helper(bbox_left_children, center_bbox, direction="left")

        bbox_bottom_children = center_bbox.bottom_children
        compute_box_gap_helper(bbox_bottom_children, center_bbox, direction="bottom")

        bbox_top_children = center_bbox.top_children
        compute_box_gap_helper(bbox_top_children, center_bbox, direction="top")


"""
    Layout 搜索
"""


def expand_group(start_bbox, bboxes, visited):
    """以start_bbox为起点，搜索满足条件的bbox加入当前group"""
    group = [start_bbox]
    queue = [start_bbox]

    while queue:
        # remove first element
        current_bbox = queue.pop(0)
        # 获取上下左右四个方向的邻居bbox
        neighbors = get_neighbors(current_bbox, bboxes)

        # 根据上一个状态的group判断要不要检查下一个状态的group的[间隔是否相等]
        can_check_gap = can_check_box_gap(group)

        for direction, neighbor in neighbors.items():
            if neighbor is None:
                continue
            if neighbor.box_id not in visited:
                if can_add_to_group(current_bbox, neighbor, direction, can_check_gap[direction]):
                    group.append(neighbor)
                    queue.append(neighbor)
                    visited.add(neighbor.box_id)
    return group


def get_neighbors(bbox, bboxes):
    # 获取上下左右方向的相邻bbox
    # 示例返回格式: {'top': bbox_top, 'bottom': bbox_bottom, 'left': bbox_left, 'right': bbox_right}
    neighbors = {'top': None, 'bottom': None, 'left': None, 'right': None}
    if bbox.is_left_aligned:
        neighbors["left"] = bboxes[bbox.left_children[0]]
    if bbox.is_right_aligned:
        neighbors["right"] = bboxes[bbox.right_children[0]]
    if bbox.is_top_aligned:
        neighbors["top"] = bboxes[bbox.top_children[0]]
    if bbox.is_bottom_aligned:
        neighbors["bottom"] = bboxes[bbox.bottom_children[0]]
    return neighbors


def can_add_to_group(current_bbox, neighbor, direction, can_check_gap):
    """根据对齐和间隔规则，判断邻居box是否可以加入当前group"""
    can_check_gap, gap = can_check_gap
    # gap的阈值根据情况微调
    threshold_ratio = 0.05
    threshold = max(current_bbox.height, current_bbox.width) * threshold_ratio
    if direction == "left":
        return abs(current_bbox.left_gap[neighbor.box_id] - gap) < threshold if can_check_gap else True
    if direction == "right":
        return abs(current_bbox.right_gap[neighbor.box_id] - gap) < threshold if can_check_gap else True
    if direction == "top":
        return abs(current_bbox.top_gap[neighbor.box_id] - gap) < threshold if can_check_gap else True
    if direction == "bottom":
        return abs(current_bbox.bottom_gap[neighbor.box_id] - gap) < threshold if can_check_gap else True


def compute_bbox_distribution(group):
    if len(group) == 0 or len(group) == 1:
        return None, []

    threshold_ratio = 0.05
    threshold = max(group[0].height, group[0].width) * threshold_ratio

    # 从上往下，从左往右
    group.sort(key=lambda bbox: (bbox.row_min, bbox.column_min))

    group_v = []
    changed = -1

    for i in range(0, len(group) - 1):
        if i <= changed:
            continue
        else:
            changed = -1

        group_h = [group[i]]

        for j in range(i + 1, len(group)):
            if abs(group[i].row_min - group[j].row_min) < threshold:
                group_h.append(group[j])
                # 右侧指针先达到终点
                if j == len(group) - 1:
                    changed = len(group)
            else:
                changed = j - 1
                break

        group_v.append(group_h)

    # bugfix: 修复group长度为2无法检查最后一个元素，导致布局判断错误
    if len(group) == 2 and len(group_v[0]) == 1:
        group_v = [[group[0]], [group[1]]]

    grid_size = [len(group_h) for group_h in group_v]
    return group_v, grid_size


def can_check_box_gap(group):
    """
    粗略判断布局类型，根据上一步的group情况来决定下一步的group是否需要检查间距是否相等

    判断因素：
    1）布局类型row、col、grid；2）元素数量（横向或纵向必须超过2个）
    """
    if len(group) == 0 or len(group) == 1:
        return {
            'top': (False, -1), 'bottom': (False, -1), 'left': (False, -1), 'right': (False, -1)
        }

    group_v, grid_size = compute_bbox_distribution(group)  # grid布局 (3, 4) [4, 4, 4] 3行4列

    try:
        can_check_h_gap, can_check_v_gap = False, False

        # 1横向检查
        row = 0
        for i, s in enumerate(grid_size):
            # 横向必须超过2个，才能检查间隔是否相等
            if s >= 2:
                row = i
                can_check_h_gap = True

        # 获取横向的间隔作为[判断间隔相等]的依据
        if group_v[row][0].right_gap:
            h_gap = list(group_v[row][0].right_gap.values())[0] if can_check_h_gap else -1
        elif group_v[row][0].right_gap:
            h_gap = list(group_v[row][0].left_gap.values())[0] if can_check_h_gap else -1
        else:  # bugfix: 2024.9.29
            h_gap = -1
            can_check_h_gap = False

        # 2纵向检查
        # 纵向必须超过2个，才能检查间隔是否相等
        if len(grid_size) >= 2:
            can_check_v_gap = True

        # 获取纵向的间隔作为[判断间隔相等]的依据
        if group_v[0][0].bottom_gap:
            v_gap = list(group_v[0][0].bottom_gap.values())[0] if can_check_v_gap else -1
        elif group_v[0][0].top_gap:
            v_gap = list(group_v[0][0].top_gap.values())[0] if can_check_v_gap else -1
        else:  # bugfix: 2024.9.29
            v_gap = -1
            can_check_v_gap = False

        value = {
            "left": (can_check_h_gap, h_gap),
            "right": (can_check_h_gap, h_gap),
            "top": (can_check_v_gap, v_gap),
            "bottom": (can_check_v_gap, v_gap),
        }

    except IndexError as e:
        print(group_v)
        raise e

    return value


def analyze_group(group):
    """假设group中仅有一种布局"""
    # 根据给定的规则分析group的布局类型（行、列、网格、不符合）
    group_v, grid_size = compute_bbox_distribution(group)
    if len(grid_size) == 0:
        return None, None

    rows, cols = len(grid_size), grid_size[0]
    # if rows == 1 and cols == 1:
    #     print("debug")

    # 1行布局
    if rows == 1 and cols > 1:  # [a]
        return "row", (1, cols)
    # 2列布局
    if all([s == 1 for s in grid_size]):  # 连续的1 [1, 1, ...]
        box = group_v[0][0]
        # bugfix#20241011: list index out of range
        if box.bottom_gap:
            gap = list(box.bottom_gap.values())[0]
        else:
            return None, None
        gap_threshold_ratio = 5.0
        # bugfix: 布局过滤-远距离跨越的列分组不计入列布局
        # 纵向间隔超过box高度的多倍
        if gap / box.height > gap_threshold_ratio:
            return None, None
        return "col", (rows, 1)
    # 3网格布局
    if all([s == cols for s in grid_size]):  # 连续的a [a, a, ...]
        return "grid", (rows, cols)

    return 'Complex', grid_size


def calculate_min_cell_of_layout(group, layout_position, layout_info, cls):
    """
    计算布局中的最小单元

    # 拆分最小单元的条件：
    1）布局内的元素大小一致
    2）Non-Text类型，不支持对文本提取最小单元
    3）布局内的元素大小超过(100px, 100px)
    """
    cell_min_size = (100, 100)
    row, col = layout_info  # 行数，列数
    cell_position = None  # layout中的最小单元
    std_box = group[0]  # 存储bbox的layout分组

    if is_boxes_similar_size(group) and cls == "Compo" and std_box.width >= cell_min_size[0] and std_box.height >= cell_min_size[1]:
        cell_position = {
            "column_min": layout_position["column_min"],
            "row_min": layout_position["row_min"],
            "column_max": (layout_position["column_max"] - layout_position["column_min"]) / col + layout_position[
                "column_min"],
            "row_max": (layout_position["row_max"] - layout_position["row_min"]) / row + layout_position["row_min"],
        }
    return cell_position


def search_layout(boxes, cls=None, is_detail_print=True):
    """
    将满足[对齐和间隔、大小]的bbox聚集在各自的group中

    基于已进行空间关系建模的boxes，进行布局类型的分析
    :param cls: Text-Compo-Block
    :param is_detail_print: 是否打印详细输出layout信息

    layout分为两种：
    1）不规则的，元素可以大小不一，但满足对齐和间隔
    2）规则的，元素大小一致，且满足对齐和间隔 => 可以提取layout的最小单元cell
    """
    import random

    visited = set()
    groups = []

    while len(visited) < len(boxes):
        start_bbox = random.choice([bbox for bbox in list(boxes.values()) if bbox.box_id not in visited])
        visited.add(start_bbox.box_id)
        # 以start_bbox为起点，搜索满足条件的bbox加入当前group
        group = expand_group(start_bbox, boxes, visited)
        groups.append(group)

    groups = [group for group in groups if len(group) > 1]
    layouts = []

    for group in groups:
        # 1分析布局类型
        layout_type, layout_info = analyze_group(group)

        # 2过滤不符合要求的布局
        if layout_type == "col":
            pass

        # 3计算布局相关信息
        if layout_type == 'row' or layout_type == 'col' or layout_type == 'grid':
            # 1-计算layout bbox坐标
            layout_position = calculate_layout_position(group, layout_type, cls)

            # 2-计算layout中最小单元（row、col、grid）
            cell_position = calculate_min_cell_of_layout(group, layout_position, layout_info, cls)

            layouts.append({
                "layout_id": -1,
                "layout_type": layout_type,
                "size": layout_info,  # 不完全准确，debug中
                "children": group,
                'position': layout_position,
                "class": cls,
                "cell_position": cell_position,  # layout中的最小单元，列表中的子元素，代码复用的基础
            })

        group = sorted([box.box_id for box in group])
        is_detail_print and print(f"Group with bboxes {group} is a {layout_type}-{layout_info} layout")

    return layouts


"""
    group绘制
"""


def calculate_similarity(bbox1, bbox2):
    """计算两个bbox的大小相似性"""
    threshold_ratio = 0.1
    width_similarity = abs(bbox1.width - bbox2.width) / max(bbox1.width, bbox2.width)
    height_similarity = abs(bbox1.height - bbox2.height) / max(bbox1.height, bbox2.height)
    return width_similarity < threshold_ratio and height_similarity < threshold_ratio


def is_boxes_similar_size(boxes):
    """判断box list中的box是否大小一致"""
    standard_box = boxes[0]
    for i, box in enumerate(boxes):
        similarity = calculate_similarity(standard_box, box)
        if not similarity:
            return False
    return True


def calculate_layout_position(bboxes, layout_type, cls):
    # 初始化边界
    min_column = float('inf')
    max_column = float('-inf')
    min_row = float('inf')
    max_row = float('-inf')

    # 遍历所有 bbox 以确定边界
    for bbox in bboxes:
        min_column = min(min_column, bbox.column_min)
        max_column = max(max_column, bbox.column_max)
        min_row = min(min_row, bbox.row_min)
        max_row = max(max_row, bbox.row_max)

    if cls == "Compo" and is_boxes_similar_size(bboxes):
        try:
            # 1）non-text ✅ ｜ text ❎
            # 2）group中的box大小一致
            # 网格布局向下扩充一个bottom_gap的距离
            if layout_type == "grid":
                bboxes.sort(key=lambda box: box.row_min)
                max_row += list(bboxes[0].bottom_gap.values())[0]
            # 行布局向右扩充一个right_gap的距离
            # elif layout_type == "row":
            #     bboxes.sort(key=lambda box: box.column_min)
            #     # 360.cn box#18 布局识别有问题，col->row
            #     if bboxes[0].right_gap:
            #         max_column += list(bboxes[0].right_gap.values())[0]
        except IndexError as e:
            print(bboxes[0].info())
            raise e

    # 返回最小矩形框
    return {
        'column_min': min_column,
        'row_min': min_row,
        'column_max': max_column,
        'row_max': max_row
    }


def draw_layouts(bg_img_path, layouts, output_path=None):
    from PIL import Image, ImageDraw
    # 读取图片
    image = Image.open(bg_img_path)
    draw = ImageDraw.Draw(image)

    # 绘制每个 bounding box
    for layout in layouts:
        top_left = (layout['position']['column_min'], layout['position']['row_min'])
        bottom_right = (layout['position']['column_max'], layout['position']['row_max'])
        color = "orange" if layout["class"] == "Text" else "blue"
        draw.rectangle([top_left, bottom_right], outline=color, width=2)  # 边框
        # draw.rectangle([top_left, bottom_right], outline=None, fill=color)  # 填充，实心框

        if layout["cell_position"]:
            unit_top_left = (layout['cell_position']['column_min'], layout['cell_position']['row_min'])
            unit_bottom_right = (layout['cell_position']['column_max'], layout['cell_position']['row_max'])
            draw.rectangle([unit_top_left, unit_bottom_right], outline="black", width=2)

    # 如果指定了输出路径，则保存图片，否则显示图片
    if output_path:
        image.save(output_path)
    else:
        image.show()


def reassign_ids(elements):
    """重新分配ID"""
    new_elements = []
    for i, element in enumerate(elements):
        element["layout_id"] = i
        new_elements.append(element)
    return new_elements


"""
    layout处理主函数
"""


def process_layout(input_img_path, output_root=None, use_uied_img=True, is_detail_print=True, use_sep_line=True):
    start = time.process_time()

    from os.path import join as pjoin

    name = input_img_path.split('/')[-1][:-4]
    # 读取
    merge_root = pjoin(output_root, 'merge')
    line_root = pjoin(output_root, 'line')
    cluster_root = pjoin(output_root, 'cluster')
    # 写入
    layout_root = pjoin(output_root, 'layout')
    os.makedirs(layout_root, exist_ok=True)

    output_json_path = pjoin(layout_root, f'{name}.json')

    data = read_json_file(merge_root + f"/{name}.json")
    bbox_list = data["compos"]
    # image_size = (data["img_shape"][1], data["img_shape"][0])  # w, h

    # 1-bbox基础建模
    non_text_bboxes = [Box(bbox) for bbox in bbox_list if bbox["class"] == 'Compo' or bbox["class"] == 'Block']  # [Box()]
    text_bboxes = [Box(bbox) for bbox in bbox_list if bbox["class"] == 'Text']  # [Box()]

    non_text_bboxes = find_adjacent_boxes(non_text_bboxes)  # {box_id: Box()}
    text_bboxes = find_adjacent_boxes(text_bboxes)

    compute_box_alignment(non_text_bboxes)
    compute_box_gap(non_text_bboxes)

    compute_box_alignment(text_bboxes)
    compute_box_gap(text_bboxes)

    # input_image_path = f"../data/output/cluster/{filename}_id.png"  # 原图
    # output_image_path = f"../data/output/layout/{filename}.png"  # 原图
    # if use_org_img:
    #     bg_img_path = input_file
    # else:
    #     bg_img_path = pjoin()

    # 2-搜索layout
    is_detail_print and print("non-text:")
    non_text_layouts = search_layout(non_text_bboxes, cls="Compo", is_detail_print=is_detail_print)
    is_detail_print and print("\ntext:")
    text_layouts = search_layout(text_bboxes, cls="Text", is_detail_print=is_detail_print)

    # bugfix: text_layouts 在+左边，确保text layouts先被绘制，non-text后被绘制，刚好覆盖text
    layouts = text_layouts + non_text_layouts

    # 3-去除被分割线穿过的layout bbox
    if use_sep_line:
        sep_line_path = pjoin(line_root, f"{name}.json")
        if os.path.exists(sep_line_path):
            sep_lines = read_json_file(sep_line_path)
            layouts = remove_layouts_intersected_by_lines(layouts, sep_lines)

    bg_img_path = pjoin(merge_root, f"{name}.jpg") if use_uied_img else input_img_path
    # 4-绘制layout bbox
    draw_layouts(bg_img_path, layouts, output_path=pjoin(layout_root, f'{name}.png'))
    # 5-写入layouts bbox信息
    write_json_file(output_json_path, layouts, is_box=True)

    print("[Layout Detection Completed in %.3f s] Input: %s Output: %s" % (
    time.process_time() - start, input_img_path, pjoin(layout_root, name + '.png')))
    return layouts


"""
    Layout与其他数据的交互
    1、layouts与sep_lines
"""


def remove_layouts_intersected_by_lines(layouts, lines):
    """移除被分割线穿过的layout bbox"""
    delete_indexes = []
    for i, layout in enumerate(layouts):
        bbox = layout["position"]

        for line in lines:
            start_point = line["x1"], line["y1"]
            end_point = line["x2"], line["y2"]

            if is_line_intersect_bbox(start_point, end_point, bbox):
                delete_indexes.append(i)
                break

    new_layouts = []
    for i, layout in enumerate(layouts):
        if i in delete_indexes:
            continue
        new_layouts.append(layout)

    return new_layouts


def is_line_intersecting_bbox_horizontal(start_point, end_point, bbox):
    y = start_point[1]

    # 检查是否在 bbox 的纵向范围内
    if bbox['row_min'] <= y <= bbox['row_max']:
        # 检查水平线的 x 范围是否与 bbox 有重叠
        x_min = min(start_point[0], end_point[0])
        x_max = max(start_point[0], end_point[0])

        if x_min <= bbox['column_max'] and x_max >= bbox['column_min']:
            return True
    return False


def is_line_intersecting_bbox_vertical(start_point, end_point, bbox):
    x = start_point[0]

    # 检查是否在 bbox 的横向范围内
    if bbox['column_min'] <= x <= bbox['column_max']:
        # 检查垂直线的 y 范围是否与 bbox 有重叠
        y_min = min(start_point[1], end_point[1])
        y_max = max(start_point[1], end_point[1])

        if y_min <= bbox['row_max'] and y_max >= bbox['row_min']:
            return True
    return False


def is_line_intersect_bbox(start_point, end_point, bbox):
    """判断直线是否与bbox相交（可定制阈值）"""
    threshold = 2.1  # bugfix: 2024.9.29, Design2Code#1405.png
    if start_point[0] - end_point[0] < threshold:  # 垂直线
        return is_line_intersecting_bbox_vertical(start_point, end_point, bbox)
    elif start_point[1] - end_point[1] < threshold:  # 水平线
        return is_line_intersecting_bbox_horizontal(start_point, end_point, bbox)
    else:
        print(start_point, end_point)
        # 处理异常情况，非水平或垂直线
        raise ValueError("The line must be either horizontal or vertical.")


def is_line_intersect_bbox_shapely(line_start, line_end, bbox_min, bbox_max):
    """判断直线是否与bbox相交（不可定制阈值）"""
    from shapely.geometry import LineString, Polygon

    # 创建直线对象
    line = LineString([line_start, line_end])
    # 创建边界框对象
    bbox = Polygon([(bbox_min[0], bbox_min[1]), (bbox_min[0], bbox_max[1]),
                    (bbox_max[0], bbox_max[1]), (bbox_max[0], bbox_min[1])])

    # 检查直线是否与边界框相交
    return line.intersects(bbox)


if __name__ == "__main__":
    process_layout('../data/input/real_image/360.cn.png', "../data/output/")

    # # WARNING: 注释🙅❌不可删除
    # filename = "360.cn"
    # input_json_path = f"../data/output/merge/{filename}.json"
    # data = read_json_file(input_json_path)
    # bbox_list = data["compos"]
    # image_size = (data["img_shape"][1], data["img_shape"][0])  # w, h
    #
    # non_text_bboxes = [Box(bbox) for bbox in bbox_list if bbox["class"] == 'Compo' or bbox["class"] == 'Block']  # [Box()]
    # text_bboxes = [Box(bbox) for bbox in bbox_list if bbox["class"] == 'Text']  # [Box()]
    #
    # non_text_bboxes = find_adjacent_boxes(non_text_bboxes)  # {box_id: Box()}
    # text_bboxes = find_adjacent_boxes(text_bboxes)
    #
    # # adjacent_boxes = {
    # #     box_id: {
    # #         "right": box.right_children,
    # #         "left": box.left_children,
    # #         "bottom": box.bottom_children,
    # #         "top": box.top_children
    # #     } for box_id, box in bboxes.items()
    # # }
    # # print(adjacent_boxes)
    #
    # compute_box_alignment(non_text_bboxes)
    # compute_box_gap(non_text_bboxes)
    #
    # compute_box_alignment(text_bboxes)
    # compute_box_gap(text_bboxes)
    #
    # # aligned_gap_boxes = {
    # #     box_id: {
    # #         "aligned": {
    # #             "left": box.is_left_aligned,
    # #             "right": box.is_right_aligned,
    # #             "top": box.is_top_aligned,
    # #             "bottom": box.is_bottom_aligned
    # #         },
    # #         "gap": {
    # #             "left": box.left_gap,
    # #             "right": box.right_gap,
    # #             "top": box.top_gap,
    # #             "bottom": box.bottom_gap
    # #         }
    # #     }
    # #     for box_id, box in bboxes.items()
    # # }
    # # print(aligned_gap_boxes)
    #
    # # input_image_path = f"../data/output/cluster/{filename}_id.png"  # bbox图
    # input_image_path = f"../data/output/cluster/{filename}_id.png"  # 原图
    # output_image_path = f"../data/output/layout/{filename}.png"  # 原图
    # print("non-text:")
    # non_text_layouts = search_layout(non_text_bboxes, cls="Compo")
    # print("\ntext:")
    # text_layouts = search_layout(text_bboxes, cls="Text")
    #
    # layouts = non_text_layouts + text_layouts
    # draw_layouts(input_image_path, layouts, output_path=None)
    #
    # print("Finished!!!")