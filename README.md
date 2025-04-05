# LayoutCoder

## Dataset

All experimental data is published in `https://drive.google.com/drive/folders/1YJHXaZhOps34C1fd2rAtflM5-cKRuk6h?usp=sharing` and categorized by RQ.

Note: Download the dataset and place it in the `data` directory.

1、`Design2Code`
Access from `https://huggingface.co/datasets/SALT-NLP/Design2Code?row=3`

2、`Snap2Code` Construction Method

-  Obtain the top 500 website list from `https://moz.com/top500`.

- Use `Selenium` and `geckodriver` to capture webpage source code and screenshots.


## Code Implementation

`run_single.py`
```python
    # UI Element Detection
    is_uied and uied(input_path_img, output_root)
    # Module A: Element Relation Constuction
    is_layout and layout.process_layout(input_path_img, output_root)
    # Module B: UI Layout Parsing
    is_divide and page_layout_divider.divide_layout(input_path_img, output_root)
    # Module C: Layout-Guided Code Fusion
    is_global_gen and code_gen.make_layout_code(input_path_img, output_root)
```

```
       
├── data/
├── logs/
├── sdk/
├── UIED/
├── README.md  
├── requirements_lc.txt
├── requirements_uied.txt
├── enviroment.yml
├── run_single.py                      < entry point >
└──utils/
    ├──code_gen/                       < Module C: Layout-Guided Code Fusion >
    │   ├── __init__.py
    │   ├── layout_extract.py
    │   ├── partial_code.py
    │   └── struct2code2mask_utils.py
    ├── layout.py                      < Module A: Element Relation Constuction >
    ├── local_json.py
    └── page_layout_divider.py         < Module B: UI Layout Parsing >
    
```

## Prompts
We present the implementation of three different prompts for MLLMs based on code: Direct, CoT, and Self-Refine.


```python
PROMPT_DIRECT = """Here is a screenshot of a webpage. Return a single piece of HTML and tail-wind CSS code to reproduce exactly the website. Use a placeholder image to replace the images. Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout. Respond with the content of the HTML+tail-wind CSS code."""
PROMPT_COT = """Here is a screenshot of a webpage. Return a single piece of HTML and tail-wind CSS code to reproduce exactly the website. Please think step by step by dividing the screenshot into multiple parts, write the code for each part, and combine them to form the final code. Use a placeholder image to replace the images. Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout. Respond with the content of the HTML+tail-wind CSS code."""
PROMPT_MULTI = """Here is a screenshot of a webpage. I have an HTML file for implementing a webpage but it has some missing or wrong elements that are different from the original webpage. Please compare the two webpages and revise the original HTML implementation. Return a single piece of HTML and tail-wind CSS code to reproduce exactly the website. Use a placeholder image to replace the images. Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout. Respond with the content of the HTML+tail-wind CSS code. The current implementation I have is: \n\n [CODE]"""

@timer
@debug_wrapper
def direct_prompt(bot, image_path, html_path, force_update=False):
    if not force_update:
        if os.path.exists(html_path):
            print(f"{html_path} existed! ")
            return None
    bot = bot_map[bot]()
    single_turn(bot, PROMPT_DIRECT, image_path, html_path)
    return bot.usage


@timer
@debug_wrapper
def cot_prompt(bot, image_path, html_path, force_update=False):
    if not force_update:
        if os.path.exists(html_path):
            print(f"{html_path} existed! ")
            return None
    bot = bot_map[bot]()
    single_turn(bot, PROMPT_COT, image_path, html_path)
    return bot.usage


@timer
@debug_wrapper
def self_refine_prompt(bot, image_path, html_path, n_turns=2, force_update=False):
    if not force_update:
        if os.path.exists(html_path):
            print(f"{html_path} existed! ")
            return None

    bot = bot_map[bot]()
    next_html = single_turn(bot, PROMPT_DIRECT, image_path)
    for i in range(n_turns - 1):
        prompt_self_refine = PROMPT_MULTI.replace("[CODE]", next_html)
        next_html = single_turn(bot, prompt_self_refine, image_path)
    html_write(next_html, html_path)
    return bot.usage
```


## Environment Setup
Python 3.8.18

```bash
conda env create -f environment.yml
pip install -r requirements.txt
```

## Execution
```bash
python ./run_single.py --dirname ours_dataset
```

## Acknowledgements
This project is built upon the following outstanding open-source projects. We extend our gratitude to their contributors:

**[UIED](https://github.com/MulongXie/UIED)**
The UI element detection algorithm from UIED served as the preprocessing foundation for LayoutCoder.

**[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)**
We integrated PaddleOCR for precise text extraction across complex UI components. Its high accuracy in multilingual scenarios ensured reliable OCR results for downstream tasks.

**[DCGen](https://github.com/WebPAI/DCGen)**
The line segmentation detection algorithm from DCGen inspired key optimizations in our layout parsing logic.

All projects above are used under their respective open-source licenses (e.g., MIT, Apache-2.0). We respect and adhere to the principles of open-source collaboration.