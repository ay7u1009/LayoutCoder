"""
提取截图中的组件bbox
"""
import datetime
import json
import shutil
import sys
import time
import traceback
from os.path import join as pjoin
import cv2
import os
import numpy as np

proj_path = os.path.dirname(__file__)


def resize_height_by_longest_edge(img_path, resize_length=800):
    org = cv2.imread(img_path)
    try:
        height, width = org.shape[:2]
    except Exception as e:
        print(img_path)
        raise e
    return height
    # if height > width:
    #     return resize_length
    # else:
    #     return int(resize_length * (height / width))


def color_tips():
    color_map = {'Text': (0, 0, 255), 'Compo': (0, 255, 0), 'Block': (0, 255, 255), 'Text Content': (255, 0, 255)}
    board = np.zeros((200, 200, 3), dtype=np.uint8)

    board[:50, :, :] = (0, 0, 255)
    board[50:100, :, :] = (0, 255, 0)
    board[100:150, :, :] = (255, 0, 255)
    board[150:200, :, :] = (0, 255, 255)
    cv2.putText(board, 'Text', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(board, 'Non-text Compo', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(board, "Compo's Text Content", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(board, "Block", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.imshow('colors', board)


def uied(input_path_img=None, output_root=None):
    '''
        ele:min-grad: gradient threshold to produce binary map
        ele:ffl-block: fill-flood threshold
        ele:min-ele-area: minimum area for selected elements
        ele:merge-contained-ele: if True, merge elements contained in others
        text:max-word-inline-gap: words with smaller distance than the gap are counted as a line
        text:max-line-gap: lines with smaller distance than the gap are counted as a paragraph

        Tips:
        1. Larger *min-grad* produces fine-grained binary-map while prone to over-segment element to small pieces
        2. Smaller *min-ele-area* leaves tiny elements while prone to produce noises
        3. If not *merge-contained-ele*, the elements inside others will be recognized, while prone to produce noises
        4. The *max-word-inline-gap* and *max-line-gap* should be dependent on the input image size and resolution

        mobile: {'min-grad':4, 'ffl-block':5, 'min-ele-area':50, 'max-word-inline-gap':6, 'max-line-gap':1}
        web   : {'min-grad':3, 'ffl-block':5, 'min-ele-area':25, 'max-word-inline-gap':4, 'max-line-gap':4}
    '''
    key_params = {'min-grad': 10, 'ffl-block':5, 'min-ele-area':50,
                  'merge-contained-ele':True, 'merge-line-to-paragraph':True, 'remove-bar':False}

    # set input image path
    # input_path_img = 'data/input/497.jpg'
    """
    weak: bilibili.png 不能出现悬浮的弹窗 对比bilibili2.png
    weak: yandex.ru.png  背景图干扰挺严重的
    weak: google2.png 颜色相近误差增大，可能需要调整超参数
    """
    input_path_img = input_path_img or './data/input/real_image/360.cn_6.png'
    output_root = output_root or './data/output'

    resized_height = resize_height_by_longest_edge(input_path_img, resize_length=800)  # author限制高度的目的是什么？提高计算速度？
    # color_tips()

    is_ip = True
    is_clf = False  # 吃性能
    is_ocr = True
    is_merge = True

    if is_ocr:
        print("====enter====[ocr]")
        import UIED.detect_text.text_detection as text
        os.makedirs(pjoin(output_root, 'ocr'), exist_ok=True)
        text.text_detection(input_path_img, output_root, show=False)
        print("====exit====[ocr]")

    if is_ip:
        print("====enter====[ip]")
        import UIED.detect_compo.ip_region_proposal as ip
        os.makedirs(pjoin(output_root, 'ip'), exist_ok=True)
        # switch of the classification func
        classifier = None
        if is_clf:
            classifier = {}
            from UIED.cnn.CNN import CNN
            # classifier['Image'] = CNN('Image')
            classifier['Elements'] = CNN('Elements')
            # classifier['Noise'] = CNN('Noise')
        ip.compo_detection(input_path_img, output_root, key_params,
                           classifier=classifier, resize_by_height=resized_height, show=False)
        print("====exit====[ip]")

    if is_merge:
        print("====enter====[merge]")
        import UIED.detect_merge.merge as merge
        os.makedirs(pjoin(output_root, 'merge'), exist_ok=True)
        name = input_path_img.split('/')[-1][:-4]
        compo_path = pjoin(output_root, 'ip', str(name) + '.json')
        ocr_path = pjoin(output_root, 'ocr', str(name) + '.json')
        # max_line_gap=70 确保跨两行, 30 跨一行
        merge.merge(input_path_img, compo_path, ocr_path, pjoin(output_root, 'merge'),
                    is_remove_bar=key_params['remove-bar'], is_paragraph=key_params['merge-line-to-paragraph'], show=False, max_line_gap=30)
        print("====exit====[merge]")


def ui2code_pipeline(input_path_img, output_root, **key_params):
    from utils import layout, detect_lines, page_layout_divider, code_gen
    from utils.eval import geckodriver
    from utils.experiment import baseline

    is_uied = key_params['is_uied']
    is_layout = key_params['is_layout']
    is_lines = key_params['is_lines']
    is_divide = key_params['is_divide']
    is_global_gen = key_params['is_global_gen']
    # is_screenshot = key_params["is_screenshot"]
    # is_direct_prompt = key_params["is_direct_prompt"]

    start = time.process_time()

    # 识别text、non-text
    is_uied and uied(input_path_img, output_root)
    # 识别分割线
    is_lines and detect_lines.detect_sep_lines_with_lsd(input_path_img, output_root)
    # 识别layout（依赖分割线去除错误的layout)
    is_layout and layout.process_layout(input_path_img, output_root, use_uied_img=True, is_detail_print=False,
                                        use_sep_line=True)
    # 绘制分割线
    is_lines and detect_lines.draw_lines(input_path_img, output_root)
    # 布局分割
    is_divide and page_layout_divider.divide_layout(input_path_img, output_root)
    # 布局分割线=>mask=>json=>layout html
    is_global_gen and code_gen.make_layout_code(input_path_img, output_root, force=False)  # force不强制更新本地json数据
    # # 对生成的html截图（ours）
    # is_screenshot and geckodriver.web_screenshot(input_path_img, output_root)
    # # cot prompt生成html，并截图❎
    # is_direct_prompt and baseline.direct_prompt(input_path_img, output_root)

    print("[UI2Code Pipeline Completed in %.3f s]\n" % (time.process_time() - start))


def single_process():
    """单处理"""
    start = time.process_time()
    input_path_img = './data_20250103/snap2code_seen/image/huffpost.com.png'
    output_root = './data_20250103/snap2code_seen/html_layoutcoder_temp/'

    os.makedirs(output_root, exist_ok=True)

    # input_path_img = './utils/prepare_dataset/test_2/360.cn/www.360.cn__0.png'
    # output_root = './data/output2/'

    ui2code_pipeline(
        input_path_img=input_path_img,
        output_root=output_root,
        # 流水线阶段
        is_uied=False,
        is_layout=False,
        is_lines=False,
        is_divide=False,
        is_global_gen=True,
        # is_screenshot=False,
        # is_direct_prompt=False
    )

    print("[Total Completed in %.3f]\n" % (time.process_time() - start))


def batch_sub_process(i, img, output_root):
    try:
        print(f"[UI2Code Sub][Processing 第{i}张: {img}]")
        input_path_img = img

        start_time = time.perf_counter()  # 使用高精度计时器

        ui2code_pipeline(
            input_path_img=input_path_img,
            output_root=output_root,
            # 流水线阶段
            is_uied=False,         # UI检测
            is_lines=False,
            is_layout=False,       # UI分组
            is_divide=False,       # 图片分割，解析布局
            is_global_gen=True,    # 代码合成
            # is_screenshot=False,
            # is_direct_prompt=False
        )

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        print(f"[UI2Code Sub][Finished 第{i}张: {img}]")
    except Exception as e:
        return img, traceback.format_exc(), None
    return img, None, elapsed_time


def batch_process_async(dirname=None):
    """批处理:多进程版本"""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing

    """消融实验👇"""
    from ablation_config import is_ui_group, is_gap_sort, is_custom_prompt
    ablation_suffix = ""
    if not is_ui_group:
        ablation_suffix += "_ab1_group"
    if not is_gap_sort:
        ablation_suffix += "_ab2_gap"
    if not is_custom_prompt:
        ablation_suffix += "_ab3_prompt"
    """消融实验s👆️"""

    start = time.process_time()

    data_path = os.path.join(proj_path, "data_20250103")
    input_root = os.path.join(data_path, dirname, "image")
    output_root = os.path.join(data_path, dirname, f"html_layoutcoder_temp{ablation_suffix}")
    # input_root = f'./data/{dirname}/'
    # output_root = f'./data/{dirname}_output{ablation_suffix}/'
    os.makedirs(output_root, exist_ok=True)
    print(output_root)

    image_path_list = [os.path.join(input_root, file) for file in os.listdir(input_root) if file.endswith(".png")]
    # image_list = [pjoin(input_root, url) for url in ["360.cn.png", "taobao.png", "bilibili.png", "ctrip.png"]]
    results = []
    # int(multiprocessing.cpu_count() / 5)
    with ProcessPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(batch_sub_process, i, img, output_root): os.path.splitext(os.path.basename(img))[0] for i, img in enumerate(image_path_list)}

        time_usage = []
        for future in as_completed(futures):
            i = futures[future]
            img, error_text, elapsed_time = future.result()
            if error_text:
                results.append((i, img, error_text))
            else:
                time_usage.append({
                    "url": os.path.splitext(os.path.basename(img))[0],
                    "time": elapsed_time
                })

    # 要分开计算两部分时间，再将分割和生成部分合并在一起
    with open(os.path.join(output_root, "time_usage_gen.json"), "w") as f:
        json.dump(time_usage, f, ensure_ascii=False, indent=4)

    with open(f"./logs/{dirname}_error_image_async.txt", "a") as f:
        print("\n\n", file=f)
        print("\n".join([img.split("/")[-1] for i, img, error_text in results]), file=f)
    with open(f"./logs/{dirname}_error_async.txt", "a") as f:
        print(f"\n\n========================================={datetime.datetime.now()}\n", file=f)
        print("\n".join([error_text for i, img, error_text in results]), file=f)

    print("[Total Completed in %.3f]\n" % (time.process_time() - start))


def extract_html_from_tmp(dataset, ab=""):
    """从LayoutCoder暂存区抽取最终的HTML生成代码"""
    data_path = os.path.join(proj_path, "data_20250103")
    # image_path = os.path.join(data_path, dataset, "image")
    temp_path = os.path.join(data_path, dataset, f"html_layoutcoder_temp{ab}")
    html_path = os.path.join(data_path, dataset, f"html_layoutcoder{ab}")
    os.makedirs(html_path, exist_ok=True)
    # html_list = [os.path.join(temp_path, "") for file in os.listdir(image_path)]
    html_list = [
        (
            os.path.join(html_path, file[:-9] + ".html"),
            os.path.join(temp_path, "struct", file)
        )
        for file in os.listdir(os.path.join(temp_path, "struct"))
        if file.endswith("_sep.html")
    ]

    for dst, src in html_list:
        shutil.copy(src, dst)

    print(len(html_list))


def check_miss_url():
    data_path = os.path.join(proj_path, "data_20250103")
    datasets = ["design2code", "snap2code_seen", "snap2code_unseen"]

    # 定位目录
    for data in datasets:

        image_dir = os.path.join(data_path, data, "image")
        # html_dir = os.path.join(data_path, data, f"html_claude_layoutcoder_tem", "struct")
        html_ab = os.path.join(data_path, data, f"html_layoutcoder")
        # html_ab1_dir = os.path.join(data_path, data, f"html_layoutcoder_ab1_group")
        # html_ab2_dir = os.path.join(data_path, data, f"html_layoutcoder_ab2_gap")
        # html_ab3_dir = os.path.join(data_path, data, f"html_layoutcoder_ab3_prompt")

        image_len = len([file for file in os.listdir(image_dir) if file.endswith(".png")])
        # html_len = len([file for file in os.listdir(html_dir) if file.endswith(".html")])
        html_ab_len = len([file for file in os.listdir(html_ab) if file.endswith(".html")])
        # html_ab1_len = len([file for file in os.listdir(html_ab1_dir) if file.endswith(".html")])
        # html_ab2_len = len([file for file in os.listdir(html_ab2_dir) if file.endswith(".html")])
        # html_ab3_len = len([file for file in os.listdir(html_ab3_dir) if file.endswith(".html")])

        # print(image_len, html_ab1_len, html_ab2_len, html_ab3_len)
        print(html_ab_len)

    # 定位URL
    image_dir = os.path.join(data_path, "snap2code_seen", "image")
    html_dir = os.path.join(data_path, "snap2code_seen", "html_layoutcoder_temp", "struct")

    urls = set([os.path.splitext(file)[0] for file in os.listdir(image_dir) if file.endswith(".png")])
    un_urls = set([os.path.splitext(file)[0][:-4] for file in os.listdir(html_dir) if file.endswith(".html")])
    print(urls - un_urls)


if __name__ == '__main__':
    # 从temp中抽取html
    # datasets = ["design2code", "snap2code_seen", "snap2code_unseen"]
    # abs = ["_ab1_group", "_ab2_gap", "_ab3_prompt"]
    # for dataset in datasets:
    #     # extract_html_from_tmp(dataset, "")
    #
    #     for ab in abs:
    #         extract_html_from_tmp(dataset, ab)

    # check_miss_url()

    # snap2code_seen: 'amazon.ca'
    # snap2code_unseen: 'nattglans.se'

    import argparse
    parser = argparse.ArgumentParser(description="ui2code pipeline")
    parser.add_argument("--dataset", help="the directory you need to handle",nargs="?")
    parser.add_argument("--ab1", help="the directory you need to handle", action="store_true")
    parser.add_argument("--ab2", help="the directory you need to handle", action="store_true")
    parser.add_argument("--ab3", help="the directory you need to handle", action="store_true")
    args = parser.parse_args()

    import ablation_config
    ablation_config.is_ui_group = not args.ab1
    ablation_config.is_gap_sort = not args.ab2
    ablation_config.is_custom_prompt = not args.ab3

    # # 单处理
    # single_process()

    # 批处理：多进程
    batch_process_async(dirname=args.dataset)

