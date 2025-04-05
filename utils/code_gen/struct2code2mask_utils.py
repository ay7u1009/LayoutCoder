import json
import os
import random
import time
from bs4 import BeautifulSoup

# from geckodriver import WebDriver


"""
批量生成flex layout数据
json struct | html | mask pic
"""


"""
1、随机生成结构
✅1）避免出现连续多层都是row，或column；交错出现
✅2）如果depth=1，且只有一个元素，确保外层套一层row；list中不允许只出现一个元素
⚠️3）atomic组件要限制最小宽度和高度，如果不设置div content；=> 不能设置最小高度和宽度，应当修改外层容器视口宽高
✅4）外层容器设置视口宽高 width: 100vw; height: 100vh;
  5) 当current_depth越深，list中的元素应当越少4-3-2-1，且不同元素的portion差距越小1:9->8:2->6:4
"""


count = 50
max_depth = 3
fn_time_suffix = time.strftime("%Y%m%d-%H%M%S")
data_path = os.path.join(os.getcwd(), "data")
data_path += "/"
# config = {
#     "background-color": "mediumpurple",
#     ""
# }


block_content = "     "

def add_html_template(data, border=True, margin="2px", bg_color="white", ratio=None):
# def add_html_template(data, border=False, margin="0", bg_color="transparent", ratio=None):
    """补全html, 后续做成根据class_name配置attribute
    old_args: border=True, margin="2px", bg_color="white", ratio=None
    new_args: border=False, margin="0", bg_color="transparent", ratio=None
    """
    border = "border: 1px solid white;" if border else "border: none;"
    margin = f"margin: {margin};"
    bg_color = f"background-color: {bg_color};" if bg_color else ""  # mediumpurple
    root_ratio = f"width: 100%; aspect-ratio: {ratio};" if ratio else "width: 100vw; height: 100vh;"  # 页面长宽比例
    # Q3、Q4: 
    # - root: 控制视口宽高
    # - row、col: 容器组件
    # - atomic: 原子组件
    return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;700&display=swap" rel="stylesheet">
            <script src="https://cdn.tailwindcss.com"></script>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
            <title>VAN UI2Code</title>
            <style>
            body {{
              font-family: 'Noto Sans SC', sans-serif;
            }}
            </style>
            <title>Random Structure</title>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    font-family: Arial, sans-serif;
                }}
                .root {{
                    margin: 0;
                    padding: 0;
                    {root_ratio}
                }}
                .row {{
                    margin: 0;
                    padding: 0;
                    display: flex;
                    flex-direction: row;
                    width: 100%;
                }}
                .column {{
                    margin: 0;
                    padding: 0;
                    display: flex;
                    flex-direction: column;
                    width: 100%;
                }}
                .atomic {{
                    padding: 0;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    {border}
                    {margin}
                    {bg_color}
                }}
            </style>
        </head>
        <body>
            {data}
        </body>
        </html>
        """


def prettify_html(data):
    """美化html"""
    return BeautifulSoup(data, 'html.parser').prettify()


def generate_atomic():
    types = ["Text", "Image", "Icon"]
    return {
        "type": "atomic",
        "portion": random.randint(1, 6),
        # "value": random.choice(types),
        "value": block_content,
    }


def generate_structure(depth, current_depth=1, parent_type=random.choice(["row", "column"])):
    if current_depth >= depth:
        return generate_atomic()

    # current_type = random.choice(["row", "column"])
    # Q1: row和column嵌套时，交错出现
    current_type = "row" if parent_type == "column" else "column"
    # Q2: 确保每个list至少有2个元素
    num_elements = random.randint(2, 3)  # Number of elements in the row/column

    structure = {
        "type": current_type,
        "value": []
    }

    for _ in range(num_elements):
        # 嵌套组件
        if random.random() > 0.3 and current_depth + 1 < depth:  # 70% chance to go deeper
            child_structure = generate_structure(depth, current_depth + 1, parent_type=current_type)
            child_structure["portion"] = random.randint(1, 6)
            structure["value"].append(child_structure)
        # 原子组件
        else:
            atomic = generate_atomic()
            atomic["portion"] = random.randint(1, 6)
            structure["value"].append(atomic)

    return structure


def json_to_html_css(structure):
    """将布局结构转化为前端代码"""
    def process_node(node, parent_type='row', root=False):
        """
        process_node params
        - root, 添加root节点的视口宽高（外层容器）
        """
        html = ''
        if node['type'] == 'row':
            html += f'<div class="row{" root" if root else ""}" style="flex: {node.get("portion", 1)};">\n'
            for child in node['value']:
                html += process_node(child, parent_type='row')
            html += '</div>\n'
        elif node['type'] == 'column':
            html += f'<div class="column{" root" if root else ""}" style="flex: {node.get("portion", 1)};">\n'
            for child in node['value']:
                html += process_node(child, parent_type='column')
            html += '</div>\n'
        elif node['type'] == 'atomic':
            # 如果局部代码code字段存在，则 html += f'<div class="atomic" style="flex: {node["portion"]};">{node["code"]}</div>\n'
            # html += f'<div class="atomic" style="flex: {node["portion"]};">{node["value"]}</div>\n'
            # html += f'<div class="atomic" style="flex: {node["portion"]};">{node["id"]}</div>\n'
            # html += f'<div class="atomic" style="flex: {node["portion"]}; width: {node["size"][0]}px; height: {node["size"][1]}px;">{node["code"]}</div>\n'
            html += f'<div class="atomic" style="flex: {node["portion"]};">{node["code"]}</div>\n'
        return html

    return process_node(structure, root=True)


if __name__ == '__main__':

    for index in range(count):
        # json_path = data_path + f"{fn_time_suffix}.json"
        # html_path = data_path + f"{fn_time_suffix}.html"
        json_path = data_path + f"{fn_time_suffix}-{index}.json"
        html_path = data_path + f"{fn_time_suffix}-{index}.html"

        # 1、随机生成结构
        # random_structure = generate_random_structure()
        max_depth = 4
        random_structure = generate_structure(depth=max_depth)
        # 2、根据结构生成flex代码
        random_html_output = json_to_html_css(random_structure)
        html_template = add_html_template(random_html_output)

        with open(json_path, "w") as f:
            json.dump(random_structure, f, indent=4)

        with open(html_path, "w") as f:
            f.write(prettify_html(html_template))

        # # 3、网页全截图
        # driver = WebDriver(save_path=data_path)
        # driver.get(html_path)
        # driver.save_full_page_screenshot()
        # driver.quit()
