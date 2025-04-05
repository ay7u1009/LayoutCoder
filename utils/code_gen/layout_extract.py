

"""
mask => mask with sep => json

增加保存mask区域在全图中的位置
"""

import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.common import proj_path, nested_dict, set_value, get_value
from utils.local_json import write_json_file, read_json_file

page_path = os.path.join(proj_path, "data", "page")
page_path_1 = os.path.join(proj_path, "data", "page_top1000")
geck_path = os.path.join(proj_path, "data", "geckodriver")
test_path = os.path.join(proj_path, "data", "test")
sep_path = os.path.join(proj_path, "data", "sep_detect")


"""
TODO:
❎1、分割记录结构
"""


def numbers_to_portions(numbers: Iterable[any]) -> Iterable[any]:
    import math
    from functools import reduce

    def gcd_multiple(numbers):
        return reduce(math.gcd, numbers)

    result = gcd_multiple(list(map(lambda d: d["portion"], numbers)))

    # return list(map(lambda x: x / result, numbers))
    return list(map(lambda d: {**d, "portion": d["portion"] / result}, numbers))


@dataclass
class Point:
    x: int
    y: int

    def move(self, point):
        return Point(self.x + point.x, self.y + point.y)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __hash__(self):
        return hash((self.x, self.y))


@dataclass
class Line:
    """分割线"""
    start: Point
    end: Point

    def val(self):
        return self.start.x, self.start.y, self.end.x, self.end.y

    def direction(self):
        return "y" if self.start.x == self.end.x else "x"

    def color(self):
        return (0, 255, 0) if self.direction() == "x" else (255, 0, 0)

    def __hash__(self):
        return hash((self.start, self.end))


@dataclass
class AbsImage:
    """
    image: PIL.Image
    absolute point: 相对原图的绝对坐标
    """
    img: Image
    abs_p: Point
    path: List


def transform2line(line, line_direct, img_size):
    """
    x: 不变
    y: 旋转90度
    [(x1, y1), (x2, y2)] ->
    [(y1, w-x1), (y2, w-x2)]
    img_size = (w, h)
    """
    return Line(Point(0, line), Point(img_size[0], line)) if line_direct == "x" else Line(Point(line, 0),
                                                                                          Point(line, img_size[1]))
    # return Line(Point(0, line), Point(img_size[0], line)) \
    #     if line_direct == "x" else Line(Point(line, img_size[0] - 0), Point(line, 0))


def transform2absolute(line, absolute_point: Point):
    """
    变换line的相对坐标为绝对坐标
    absolute_point: 绝对坐标点
    line：需要平行移动的线
    """
    return Line(line.start.move(absolute_point), line.end.move(absolute_point))


def soft_separation_lines(img, bbox=None, var_thresh=60, diff_thresh=5, diff_portion=0.3, sliding_window=30):
    img_array = np.array(img.convert("L"))
    img_array = img_array if bbox is None else img_array[bbox[1]:bbox[3] + 1, bbox[0]:bbox[2] + 1]
    offset = 0 if bbox is None else bbox[1]
    lines = []
    for i in range(1 + sliding_window, len(img_array) - 1):
        upper = img_array[i - sliding_window - 1]
        window = img_array[i - sliding_window: i]
        lower = img_array[i]
        is_blank = np.var(window) < var_thresh
        # 对于mask这种标准化的图，diff_portion设置接近1.0；diff_portion表示横纵分割线占图片宽高的比例
        is_border_top = np.mean(np.abs(upper - window[0]) > diff_thresh) > diff_portion
        is_border_bottom = np.mean(np.abs(lower - window[-1]) > diff_thresh) > diff_portion
        if is_blank and (is_border_top or is_border_bottom):
            line = i if is_border_bottom else i - sliding_window
            lines.append(line + offset)
    return sorted(lines)


def hard_separation_lines(img, bbox=None, var_thresh=60, diff_thresh=5, diff_portion=0.3):
    img_array = np.array(img.convert("L"))
    img_array = img_array if bbox is None else img_array[bbox[1]:bbox[3] + 1, bbox[0]:bbox[2] + 1]
    offset = 0 if bbox is None else bbox[1]
    prev_row = None
    lines = []
    for i in range(len(img_array)):
        row = img_array[i]
        if np.var(img_array[i]) < var_thresh:
            if prev_row is not None:
                if np.mean(np.abs(row - prev_row) > diff_thresh) > diff_portion:
                    lines.append(i + offset)
            prev_row = row
    return lines


"""
    旋转图像使用rotate()方法，此方法按逆时针旋转，并返回一个新的Image对象：

    # [逆时针]旋转90度
    im.rotate(90)
    im.rotate(180)
    im.rotate(20, expand=True)
    旋转的时候，会将图片超出边界的边角裁剪掉。如果加入expand=True参数，就可以将图片边角保存住。
"""


# 3.0 变换坐标
def cut_img(image, var_thresh=60, diff_thresh=5, diff_portion=0.3, line_direct="x", verbose=False):
    assert line_direct in ["x", "y"], "line_direct must be 'x' or 'y'"

    abs_img, abs_p = image.img, image.abs_p
    # 顺时针90度 img.rotate(-90)
    img = abs_img if line_direct == "x" else abs_img.rotate(-90, expand=True)
    img_array = np.array(img.convert("L"))
    lines = soft_separation_lines(img, None, var_thresh, diff_thresh, diff_portion, sliding_window=5)
    # lines += hard_separation_lines(img, None, var_thresh, diff_thresh, diff_portion)
    # lines = hard_separation_lines(img, None, var_thresh, diff_thresh, diff_portion)
    if lines == []:
        return [], []
    lines = sorted(list(set([0, ] + lines + [img_array.shape[0], ])))

    cut_imgs = []
    for i in range(1, len(lines)):
        cut = img.crop((0, lines[i - 1], img_array.shape[1], lines[i]))
        # if cut.size[1] < 40:
        # if np.array(cut.convert("L")).mean() == 255:  # old: 300
        #     continue
        if cut.size[1] <= 10 or cut.size[0] <= 10:
            continue
        elif np.array(cut.convert("L")).mean() >= 200:  # old: 300
            continue
        # 逆时针90度
        cut = cut if line_direct == "x" else cut.rotate(90, expand=True)
        # cut_imgs.append(cut)
        # 保存图片对应的绝对点
        x = 0 if line_direct == "x" else lines[i - 1]
        y = 0 if line_direct == "y" else lines[i - 1]
        cut_imgs.append(AbsImage(cut, Point(x, y) + abs_p, deepcopy(image.path)))

    for i, cut in enumerate(cut_imgs):
        cut_imgs[i].path += ["value", i]

    if len(cut_imgs) == 1:
        return [], []

    if verbose:
        draw = ImageDraw.Draw(img)
        for line in lines:
            draw.line((0, line, img_array.shape[1], line), fill=(0, 255, 0), width=1)
        img = img if line_direct == "x" else img.rotate(90, expand=True)
        img.show()

    # 调整为(start, end)的line
    # 1、[x, x] -> [(x, x), (x, x)] 点变线
    # 2、[(x, x), (x, x)] -> [(x + d, x + d), (x + d, x + d)] 相对坐标 -> 绝对坐标
    adjusted_lines = [transform2absolute(transform2line(line, line_direct, abs_img.size), abs_p) for line in lines]
    return cut_imgs, adjusted_lines


def flatten(items, only_list=False):
    """Yield items from any nested iterable; see Reference."""
    for x in items:
        instance = List if only_list else Iterable
        if isinstance(x, instance) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x


def var(img):
    return np.var(np.array(img.convert("L")))


def draw_sep_lines(img, lines, verbose=True):
    draw = ImageDraw.Draw(img)
    for line in lines:
        # print(line.color())
        # draw.line(line.val(), fill=(0, 255, 0), width=5)
        draw.line(line.val(), fill=line.color(), width=1)
    verbose and img.show()
    return img


def draw_bbox(img, data: Iterable[any], verbose=True):
    draw = ImageDraw.Draw(img)
    for ele in data:
        position = ele["position"]
        x1, y1, x2, y2 = position["column_min"], position["row_min"], position["column_max"], position["row_max"],
        draw.rectangle(((x1, y1), (x2, y2)), outline="black", fill=None, width=3)
        _id, depth = ele["id"], ele["depth"]
        content = f"id: {_id}, depth: {depth}"
        font = ImageFont.load_default(img.size[0] / 70)  # 按图片尺寸调整字体大小
        draw.text((x1 + 5, y1 + 5), content, "black", font)
    verbose and img.show()
    return img


def tag_and_get_atomic_components(structure_data):
    """对structure中的每个atomic组件按顺序标记序号，并将所有atomic组件转化为平行的列表数据"""

    def process_structure(structure, atomic_id_list, atomic_id=1, depth=0):
        """
        递归遍历结构体，并给每个 atomic 组件标记序号，同时记录每个 atomic id、position、depth信息。

        :param structure: 当前结构（字典格式）
        :param atomic_id_list: 用于存储 atomic 组件 id 和 position 信息的列表
        :param atomic_id: 当前 atomic 组件的 id 序号
        :param depth: 当前结构的深度（position）
        :return: 更新后的 atomic_id
        """
        # 如果当前结构是一个 atomic 组件，给它标记 id 并记录它的信息
        if structure['type'] == 'atomic':
            atomic_info = {
                'id': atomic_id,
                'position': structure['position'],
                'depth': depth,
            }
            atomic_id_list.append(atomic_info)
            atomic_id += 1  # 序号自增
        else:
            # 递归处理嵌套的子结构
            if 'value' in structure:
                for item in structure['value']:
                    atomic_id = process_structure(item, atomic_id_list, atomic_id, depth + 1)

        return atomic_id

    # 创建一个用于存储 atomic 组件信息的列表
    atomic_components = []

    # 调用处理函数
    process_structure(structure_data, atomic_components)
    return atomic_components


def recursive_cut_draw(image_path, depth=3):
    """
    递归切分、绘制完整布局图
    depth：向下切分深度，2到3为佳
    """
    # img.size = (w, h)
    origin_img = Image.open(image_path)
    # print("========\n", [calculate_edge_density(origin_img)], "===========\n")
    abs_img = AbsImage(origin_img, Point(0, 0), [])

    total_lines = []
    inverse_count = 0  # 反转次数
    line_direct = "x"  # 初始方向为横向

    # ====struct start====
    structure = nested_dict()
    # ====struct end  ====

    img_list = []
    next_list = []
    img_list.append(abs_img)
    # 横向切完，纵向切；横向切不出来，换纵向切。连续两次切不出来，切下一个
    while inverse_count < depth:
        next_list = []
        for i, img in enumerate(img_list):
            cut_imgs, lines = cut_img(img, verbose=False, line_direct=line_direct, diff_portion=0.9)
            if not cut_imgs:
                line_direct = "x" if line_direct == "y" else "y"
                cut_imgs, lines = cut_img(img, verbose=False, line_direct=line_direct, diff_portion=0.9)
                if not cut_imgs:
                    # print(len(cut_imgs))
                    continue
            # print(len(cut_imgs))

            # ====struct start====
            if len(cut_imgs) > 0:
                current_type = "column"
                h_w = 1
                if line_direct == "x":
                    current_type = "column"
                    h_w = 1
                elif line_direct == "y":
                    current_type = "row"
                    h_w = 0

                portions = []
                for cut in cut_imgs:
                    portion_part = cut.img.size[h_w]
                    portions.append({
                        "abs_pos": cut.abs_p,
                        "portion": portion_part,
                        "size": cut.img.size
                    })
                portions = numbers_to_portions(portions)

                child_structure = {
                    "type": current_type,
                    "value": [
                        {
                            "type": "atomic",
                            "portion": p["portion"],
                            # "value": "block",
                            "value": "     ",
                            "position": {
                                "column_min": p["abs_pos"].x,
                                "row_min": p["abs_pos"].y,
                                "column_max": p["abs_pos"].x + p["size"][0],
                                "row_max": p["abs_pos"].y + p["size"][1],
                            }
                        }
                        for p in portions
                    ]
                }
                # list中只有一个元素，上升原子组件的位置
                if len(child_structure["value"]) == 1:
                    child_structure = child_structure["value"]

                if len(cut_imgs[0].path) <= 2:
                    structure = child_structure
                else:
                    # current_type = get_value(structure, cut_imgs[0].path[:-2] + ["type"])
                    portion = get_value(structure, cut_imgs[0].path[:-2] + ["portion"])
                    set_value(structure, cut_imgs[0].path[:-2], child_structure)
                    set_value(structure, cut_imgs[0].path[:-2] + ["portion"], portion)
                    # set_value(structure, cut_imgs[0].path[:-2] + ["type"], current_type)


            # ====struct end  ====
            # next_list.append(cut_imgs)
            next_list += cut_imgs
            total_lines += lines
            # ====
            # low_info_imgs += [(img.img, calculate_color_space_complexity(img.img)) for img in cut_imgs]
        # from pprint import pprint
        # pprint(structure)
        line_direct = "x" if line_direct == "y" else "y"
        # print(len(next_list))
        # img_list = list(flatten(deepcopy(next_list), only_list=True))
        img_list = deepcopy(next_list)
        # img_list = [img for img in img_list if calculate_color_space_complexity(img.img) >= 200]
        inverse_count += 1

    # from pprint import pprint
    # pprint(structure)
    atomic_components = tag_and_get_atomic_components(structure)
    result_img = draw_sep_lines(origin_img, list(set(flatten(total_lines))), verbose=False)
    result_img = draw_bbox(result_img, atomic_components, verbose=False)

    return result_img, {
        "structure": structure,
        "page_size": origin_img.size
    }


def mask2json(input_root, output_root, name):
    from os.path import join as pjoin

    input_path = pjoin(input_root, f"{name}_mask.png")
    sep_path = pjoin(output_root, f"{name}_sep.png")
    os.makedirs(output_root, exist_ok=True)
    # json_path = pjoin(output_root, f"{name}_sep.json")  # 布局结构信息

    result_img, data = recursive_cut_draw(input_path, depth=5)
    result_img.save(sep_path)
    # write_json_file(json_path, data)
    return data


def json2html(output_root, name, data):
    from os.path import join as pjoin
    from utils.code_gen.struct2code2mask_utils import json_to_html_css, add_html_template, prettify_html

    html_path = pjoin(output_root, f"{name}_sep.html")
    structure = data["structure"]
    page_width, page_height = data["page_size"]

    html_output = json_to_html_css(structure)
    html_template = add_html_template(html_output, ratio=page_width / page_height)

    with open(html_path, "w") as f:
        f.write(prettify_html(html_template))


def make_local_code(image_path, output_root, struct_data):
    """生成局部代码
    绑定<mask bbox, 切图, 局部代码>
    """
    from .partial_code import PartialCodeMaker

    maker = PartialCodeMaker(image_path, output_root, is_debug=False)
    struct_data_with_code = maker.code(struct_data)
    task_usage = maker.openai.usage
    # name = image_path.split("/")[-1][:-4]
    # full_image = Image.open(image_path)
    # cropped_image =
    return struct_data_with_code, task_usage


def make_layout_code(image_path, output_root, force=True):
    """全局布局代码生成"""
    from os.path import join as pjoin

    name = image_path.split('/')[-1][:-4]
    sep_root = pjoin(output_root, "sep")
    struct_root = pjoin(output_root, "struct")
    # bugfix#20241026: 避免重复调用GPT-4
    struct_json_path = pjoin(struct_root, f"{name}_sep.json")
    if not force and os.path.exists(struct_json_path):
        layout_struct_data = read_json_file(struct_json_path)
    else:
        layout_struct_data = mask2json(sep_root, struct_root, name)
        # # 根据bbox找到切图，生成局部代码，存储到json中，在json2html时，插入到布局代码中
        # # 在布局结构信息中添加局部代码块
        code, usage = make_local_code(image_path, output_root, layout_struct_data["structure"])
        layout_struct_data = {
            **layout_struct_data,
            "structure": code,
            "token_usage": usage,
        }
        write_json_file(pjoin(struct_root, f"{name}_sep.json"), layout_struct_data)
    json2html(struct_root, name, layout_struct_data)


if __name__ == "__main__":
    from os.path import join as pjoin

    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    make_layout_code(pjoin(base_path, "data/input/real_image/bilibili.png"), pjoin(base_path, "data/output/"))