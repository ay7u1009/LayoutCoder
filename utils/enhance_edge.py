"""
使用图片的曲线功能，增强浅色组件的背景色，方便UIED进行non-text检测
"""
import cv2
import numpy as np


def apply_curve(value, curve):
    """Applies the curve adjustment to a single channel value."""
    curve_value = np.interp(value, np.linspace(0, 255, len(curve)), curve)
    return np.clip(curve_value, 0, 255).astype(np.uint8)


def adjust_npimage_with_curve(org, curve=None):
    """Applies curve adjustment to an image and saves the result."""
    if curve is None:
        # Define the custom curve
        curve = np.zeros(256)

        # 0 to 1/2 (0 to 127) - increasing to the midpoint
        curve[:128] = np.linspace(0, 127, 128)

        # 1/2 to 3/4 (128 to 191) - decreasing to the lowest point
        curve[128:192] = np.linspace(127, 0, 64)

        # 3/4 to 1 (192 to 255) - increasing back to the highest point
        curve[192:] = np.linspace(0, 255, 64)

    # Convert image to numpy array
    img_array = np.array(org)

    # If the image is grayscale, apply the curve to the single channel
    if len(img_array.shape) == 2:
        adjusted_img = apply_curve(img_array, curve)
    else:
        # If the image has multiple channels (e.g., RGB), apply the curve to each channel
        adjusted_img = np.zeros_like(img_array)
        for i in range(img_array.shape[2]):
            adjusted_img[:, :, i] = apply_curve(img_array[:, :, i], curve)

    return adjusted_img


def adjust_image_with_curve(input_path, output_path, curve=None):
    """Applies curve adjustment to an image and saves the result."""
    if curve is None:
        # Define the custom curve
        curve = np.zeros(256)

        # 0 to 1/2 (0 to 127) - increasing to the midpoint
        curve[:128] = np.linspace(0, 127, 128)

        # 1/2 to 3/4 (128 to 191) - decreasing to the lowest point
        curve[128:192] = np.linspace(127, 0, 64)

        # 3/4 to 1 (192 to 255) - increasing back to the highest point
        curve[192:] = np.linspace(0, 255, 64)

    # Load an image
    # image = Image.open(input_path)
    image = cv2.imread(input_path)

    # Convert image to numpy array
    img_array = np.array(image)

    # If the image is grayscale, apply the curve to the single channel
    if len(img_array.shape) == 2:
        adjusted_img = apply_curve(img_array, curve)
    else:
        # If the image has multiple channels (e.g., RGB), apply the curve to each channel
        adjusted_img = np.zeros_like(img_array)
        for i in range(img_array.shape[2]):
            adjusted_img[:, :, i] = apply_curve(img_array[:, :, i], curve)

    # Convert back to an image and save the result
    # adjusted_image = Image.fromarray(adjusted_img)
    # adjusted_image.save(output_path)
    cv2.imwrite(output_path, adjusted_img)
    print(f"Image saved to {output_path}")


if __name__ == '__main__':
    adjust_image_with_curve("../data/input/real_image/ctrip.png", "../data/input/real_image/ctrip_curve2.png")