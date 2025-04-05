def draw_bounding_boxes(image_path, bboxes, output_path=None, color="blue"):
    """读取图片绘制bbox"""
    from PIL import Image, ImageDraw

    # 读取图片
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # 绘制每个 bounding box
    for bbox in bboxes:
        top_left = (bbox['column_min'], bbox['row_min'])
        bottom_right = (bbox['column_max'], bbox['row_max'])
        draw.rectangle([top_left, bottom_right], outline=color, width=2)  # 边框
        # draw.rectangle([top_left, bottom_right], outline=None, fill=color)  # 填充，实心框

    # 如果指定了输出路径，则保存图片，否则显示图片
    if output_path:
        image.save(output_path)
    else:
        image.show()