
import cv2
import numpy as np


def zero_crossing_edge_detection(img):
    # Apply Gaussian Blur
    # img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    # Apply Laplacian to find zero crossings
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    # Detect zero-crossings
    _, edges = cv2.threshold(np.abs(laplacian), 0, 255, cv2.THRESH_BINARY)
    return edges


def fined_edge_detection(image, kernel_type="extend"):
    import time
    start = time.process_time()

    from edge.utils.ndarray_operator import ndarray_add_num, ndarray_add_arr

    # 定义卷积核（以锐化为例）
    kernel_extend = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]])

    kernel_simple = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]])

    kernel_extend_5x5 = np.array([[-1, -1, -1, -1, -1],
                                  [-1, -1, -1, -1, -1],
                                  [-1, -1, 24, -1, -1],
                                  [-1, -1, -1, -1, -1],
                                  [-1, -1, -1, -1, -1]])

    kernel_scharr_x = np.array([[-3, 0, 3],
                                [-10, 0, 10],
                                [-3, 0, 3]])

    kernel_scharr_y = np.array([[-3, -10, -3],
                                [0, 0, 0],
                                [3, 10, 3]])

    kernel_choice = {
        'simple': kernel_simple,
        'extend': kernel_extend,
        'extend_5x5': kernel_extend_5x5
    }

    # 应用卷积核
    sharpened = cv2.filter2D(image, -1, kernel_choice[kernel_type])

    _, binary_image = cv2.threshold(sharpened, 10, 255, cv2.THRESH_BINARY)

    # 使用 connectedComponentsWithStats 来移除小噪点区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    # 遍历所有的连通区域，移除面积小于某个阈值的区域
    min_size = 50  # 根据需要调整此值
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_size:
            binary_image[labels == i] = 0

    # 查找轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个掩膜用于漫水填充，尺寸比图像大2个像素
    h, w = binary_image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    for cnt in contours:
        # 获取左上角的点作为种子点
        x, y, w, h = cv2.boundingRect(cnt)
        seed_point = (x, y)

        # 执行漫水填充
        cv2.floodFill(binary_image, mask, seed_point, 255)

    # 取反图像
    binary_image_inv = cv2.bitwise_not(binary_image)
    enhanced_image = ndarray_add_num(sharpened, 255, 10)
    final_image = ndarray_add_arr(ndarray_add_num(enhanced_image, 255, 10), binary_image_inv)
    # final_image = cv2.bitwise_not(final_image)

    end = time.process_time()
    print("==== detect_sep_lines_with_lsd Completed in %.3f s " % (end - start))
    return final_image

