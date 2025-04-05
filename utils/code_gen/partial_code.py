"""
局部代码生成
"""
from PIL import Image

import base64
import re
import requests

import os, time
from os.path import join as pjoin

from openai import OpenAI as RealOpenAI


HTML_TAILWINDCSS_SIMPLE_PROMPT = """
You are an expert Tailwind developer
You take screenshots of a reference web page from the user, and then build single page apps
using TailwindCSS, HTML and JS.
Return only the full code in <html></html> tags.
Do not include markdown "```" or "```html" at the start or end.
"""

# direct prompt，全局代码生成
HTML_TAILWINDCSS_GLOBAL_PROMPT = """
You are an expert Tailwind developer
You take screenshots of a reference web page from the user, and then build single page apps
using Tailwind, HTML and JS.
You might also be given a screenshot(The second image) of a web page that you have already built, and asked to
update it to look more like the reference image(The first image).

- Make sure the app looks exactly like the screenshot.
- Pay close attention to background color, text color, font size, font family,
padding, margin, border, etc. Match the colors and sizes exactly.
- Use the exact text from the screenshot.
- Do not add comments in the code such as "<!-- Add other navigation links as needed -->" and "<!-- ... other news items ... -->" in place of writing the full code. WRITE THE FULL CODE.
- Repeat elements as needed to match the screenshot. For example, if there are 15 items, the code should have 15 items. DO NOT LEAVE comments like "<!-- Repeat for each news item -->" or bad things will happen.
- For images, use placeholder images from https://placehold.co and include a detailed description of the image in the alt text so that an image generation AI can generate the image later.

In terms of libraries,

- Use this script to include Tailwind: <script src="https://cdn.tailwindcss.com"></script>
- You can use Google Fonts
- Font Awesome for icons: <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"></link>

Return only the full code in <html></html> tags.
Do not include markdown "```" or "```html" at the start or end.
"""

HTML_CSS_PROMPT = """
Generate the corresponding HTML code based on the input webpage image.
If you cannot generate specific HTML code, please generate code based on your best understanding of the webpage description.
Return only the full code in <html></html> tags.
Do not include markdown "```" or "```html" at the start or end.
"""

# If you cannot generate specific HTML code, please generate code based on your best understanding of the webpage description.


# ours, 局部代码生成
HTML_TAILWINDCSS_LOCAL_PROMPT = """
You are an expert Tailwind developer
You take screenshots of a reference web page from the user, and then build single page apps 
using Tailwind, HTML and JS.

- Make sure the app looks exactly like the screenshot.
- Pay close attention to background color, text color, font size, font family, 
padding, margin, border, etc. Match the colors and sizes exactly.
- Use the exact text from the screenshot.
- Do not add comments in the code such as "<!-- Add other navigation links as needed -->" and "<!-- ... other news items ... -->" in place of writing the full code. WRITE THE FULL CODE.
- Repeat elements as needed to match the screenshot. For example, if there are 15 items, the code should have 15 items. DO NOT LEAVE comments like "<!-- Repeat for each news item -->" or bad things will happen.
- For images, use placeholder images from https://placehold.co and include a detailed description of the image in the alt text so that an image generation AI can generate the image later.

In terms of libraries,

- Use this script to include Tailwind: <script src="https://cdn.tailwindcss.com"></script>
- You can use Google Fonts
- Font Awesome for icons: <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"></link>

Get the full code in <html></html> tags.

Extract the body of the full html not including <body> tag
- Make sure the aspect ratio of the div and the image are identical
- Ensure the code can be nested within other tags, extend to fill the entire container and adapt to varying container.
- Use flex layout and relative units from Tailwind CSS.
- Apply w-full and h-full classes to the outermost div.
- Don't use max-width or max-height, and set margin and padding to 0

Return only the code in <div></div> tags.
Do not include markdown "```" or "```html" at the start or end.
"""

# 简单局部代码生成，消融点3：是否使用定制局部代码生成提示词
HTML_TAILWINDCSS_LOCAL_PROMPT_SIMPLE = """
You are an expert Tailwind developer
You take screenshots of a reference web page from the user, and then build single page apps 
using Tailwind, HTML and JS.
Return only the code in <div></div> tags.
Do not include markdown "```" or "```html" at the start or end.
"""

# os.environ["OPENAI_API_KEY"] = "xxxx"
# os.environ["OPENAI_API_KEY"] = "xxxx"
# os.environ["OPENAI_BASE_URL"] = "http://xx.xx.xx.xx:xxxx/"


class OpenAI:
    def __init__(self, is_debug=False):
        self.is_debug = is_debug
        self.client = RealOpenAI(
            api_key="xx",
            base_url="https://xxx.xxx"
        )
        self.model = "chatgpt-4o-xxx"
        self.usage = {
            "token": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        }

    def partial_code(self, image_path):
        """局部代码生成, div结尾，ours"""
        from ablation_config import is_custom_prompt
        # 消融点3: 是否定制局部代码生成提示词
        prompt = HTML_TAILWINDCSS_LOCAL_PROMPT if is_custom_prompt else HTML_TAILWINDCSS_LOCAL_PROMPT_SIMPLE
        return self.ui2code(image_path, prompt)

    def global_code(self, image_path):
        """全局代码生成，html结尾，cot prompt"""
        # return self.ui2code(image_path, HTML_CSS_PROMPT)
        return self.ui2code(image_path, HTML_TAILWINDCSS_GLOBAL_PROMPT)

    def ui2code(self, image_path, prompt):
        if self.is_debug:
            name = image_path.split('/')[-1][:-4]
            return name

        print(f"正在调用{self.model}, {image_path}")
        base64_image = OpenAI.encode_image(image_path)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4096,
            temperature=0,
            seed=42
        )
        result = response.choices[0].message.content
        self.usage["token"]["prompt_tokens"] += response.usage.prompt_tokens
        self.usage["token"]["completion_tokens"] += response.usage.completion_tokens
        self.usage["token"]["total_tokens"] += response.usage.total_tokens

        # # bugfix: gpt-4o更新后，不遵循“移除```html```”指令
        # if result.startswith("```html"):
        #     result = result[len("```html"):].strip()
        # if result.endswith("```"):
        #     result = result[:-len("```")].strip()

        # matched = re.findall(r"```html([^`]+)```", result)
        html_pattern = re.compile(r'<div[^>]*>.*</div>', re.DOTALL)
        matched = html_pattern.findall(result)
        if matched:
            result = matched[0]
        return result

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @staticmethod
    def is_white_page(image_path, threadhold=0.08):
        '''
        大律法+直方图： 判断是否白屏
        感谢vivo Blog提供的白屏检测方案
        https://quickapp.vivo.com.cn/how-to-use-picture-contrast-algorithm-to-deal-with-white-screen-detection/
        '''
        import cv2
        import numpy as np

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        white_img = np.ones_like(img, dtype=img.dtype) * 255
        dst = cv2.cvtColor(white_img, cv2.COLOR_BGR2GRAY)
        dst1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, th = cv2.threshold(dst1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        hist_base = cv2.calcHist([dst], [0], None, [256], (0, 256), accumulate=False)
        hist_test1 = cv2.calcHist([th], [0], None, [256], (0, 256), accumulate=False)

        cv2.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_test1, hist_test1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        base_test1 = cv2.compareHist(hist_base, hist_test1, 3)
        print(f"空白页评估值：{base_test1}")
        return base_test1 <= threadhold


class PartialCodeMaker:
    def __init__(self, full_img_path, output_root, is_debug=False):
        self.openai = OpenAI(is_debug)
        self.full_img_path = full_img_path
        self.name = full_img_path.split('/')[-1][:-4]
        self.full_img = Image.open(full_img_path)
        self.partial_img_root = pjoin(output_root, "struct", "partial")
        os.makedirs(self.partial_img_root, exist_ok=True)

    def crop(self, bbox, path):
        """根据bbox切分图片，并保存"""
        crop_area = (max(0, bbox[0]), max(0, bbox[1]), min(self.full_img.width, bbox[2]), min(self.full_img.height, bbox[3]))
        cropped_img = self.full_img.crop(crop_area)
        cropped_img.save(path)

    def code(self, structure):
        """生成structure中所有的局部代码片段"""
        self.add_ids_and_codes(structure)
        return structure

    def add_ids_and_codes(self, structure, current_id=1, depth=0):
        """
        递归遍历结构，对每个 atomic 组件增加 id 和 code 字段。

        :param structure: 当前结构（字典格式）
        :param current_id: 当前 atomic 组件的 id 序号
        :param depth: 当前结构的深度
        :return: 下一个 atomic 组件的 id
        """
        # 检查是否是 atomic 组件
        if structure['type'] == 'atomic':
            cropped_path = pjoin(self.partial_img_root, f'{self.name}_part_{current_id}.png')
            position = structure["position"]
            bbox = (position["column_min"], position["row_min"], position["column_max"], position["row_max"])
            self.crop(bbox, cropped_path)  # 根据bbox切图
            # outer_class, inner_code = self.openai.partial_code(cropped_path)  # 切图->code
            # structure["outer_class"] = outer_class
            # structure["inner_code"] = inner_code
            structure['id'] = current_id
            if OpenAI.is_white_page(cropped_path):
                structure['code'] = "    "  # 空白区域不消耗token
            else:
                structure['code'] = self.openai.partial_code(cropped_path)  # 切图->code
            structure["size"] = (bbox[2] - bbox[0], bbox[3] - bbox[1])
            return current_id + 1  # id 递增

        # 如果不是 atomic，递归处理其嵌套的 value
        if 'value' in structure:
            for item in structure['value']:
                current_id = self.add_ids_and_codes(item, current_id, depth + 1)

        return current_id


if __name__ == '__main__':
    openai = OpenAI(is_debug=False)
    # result = openai.partial_code("/data/username/UIED/InteractiveLayoutEditor/output/struct/partial/bilibili_part_8.png")
    # print(result)
    result = openai.global_code("/data/username/UIED/data/ours_dataset/4shared.com.png")
    with open("test.html", "w") as f:
        f.write(result)
    print(openai.usage)
