import cv2
import numpy as np

def negative_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    neg_img = 255-gray_image
    return np.clip(neg_img, 0, 255).astype(np.uint8)

def log_transform(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    image_float = np.float32(gray_image)
    c = 255 / np.log(1 + np.max(image_float))
    log_image = c * (np.log(image_float + 1))
    return np.clip(log_image, 0, 255).astype(np.uint8)

def power_law_transform(image, gamma):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    normalized_image = gray_image / 255.0
    gamma_corrected = np.array(255 * (normalized_image ** gamma), dtype=np.uint8)
    return np.clip(gamma_corrected, 0, 255).astype(np.uint8)
def bright_spot(image, low_limit):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    bright_spot_vec = np.vectorize(lambda pix: 255 if pix >= low_limit else 0)
    bright_spot_img = bright_spot_vec(gray_image)
    return np.clip(bright_spot_img, 0, 255).astype(np.uint8)

def gray_level_slicing(image, low_limit, high_limit):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray_level_vec = np.vectorize(lambda pix: 255 if (pix >= low_limit and pix<=high_limit) else 0)
    gray_level_img = gray_level_vec(gray_image)
    return np.clip(gray_level_img, 0, 255).astype(np.uint8)

def contrast_stretching(image, r1, s1, r2, s2):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    contrast_stretching_vec = np.vectorize(lambda pix: (s1 / r1) * pix if 0 <= pix <= r1
                                           else ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
                                           if r1 < pix <= r2
                                           else ((255 - s2) / (255 - r2)) * (pix - r2) + s2)
    stretched_image = contrast_stretching_vec(gray_image)
    return np.clip(stretched_image, 0, 255).astype(np.uint8)


def histogram_equalization(image):
    clip_limit = 2.0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    hist, bins = np.histogram(gray_image.flatten(), bins=256, range = [0, 256])
    bin_clip_limit = clip_limit * (gray_image.size / 256.0)
    hist = np.clip(hist, 0, bin_clip_limit)

    cdf = hist.cumsum()
    cdf_normalized = cdf * (255 / cdf[-1])

    lookup_table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lookup_table[i] = cdf_normalized[i]

    equalized_image = lookup_table[gray_image]
    return equalized_image
def get_processing_functions():
    return {
        'Negative Image': negative_image,
        'Log Transform': log_transform,
        'Power Law Transform': power_law_transform,
        'Contrast Stretching': contrast_stretching,
        'Gray Level Slicing': gray_level_slicing,
        'Bright Spot Detection': bright_spot,
        'Histogram Equalization': histogram_equalization
    }
def process_image(image, function_name, **params):
    processing_functions = get_processing_functions()
    if function_name in processing_functions:
        return processing_functions[function_name](image, **params)
    else:
        raise ValueError(f"Function '{function_name}' not found.")
