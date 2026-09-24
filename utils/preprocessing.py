"""
utils/preprocessing.py
──────────────────────
OpenCV-based image preprocessing pipeline for hand-sign images.
"""

import cv2
import numpy as np

IMG_SIZE = 64   # final feature image size

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    return img

def resize(img, size=IMG_SIZE):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize_hist(gray):
    """CLAHE histogram equalization for contrast normalization."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def gaussian_blur(gray, k=3):
    return cv2.GaussianBlur(gray, (k, k), 0)

def canny_edges(gray, lo=40, hi=120):
    return cv2.Canny(gray, lo, hi)

def normalize(img):
    """Float32 [0,1] normalization."""
    return img.astype(np.float32) / 255.0

def preprocess(img_or_path, size=IMG_SIZE):
    """
    Full pipeline:
      BGR → resize → grayscale → CLAHE → Gaussian blur → normalize
    Returns flat 1-D numpy array suitable for scikit-learn.
    """
    if isinstance(img_or_path, str):
        img = load_image(img_or_path)
    else:
        img = img_or_path

    img  = resize(img, size)
    gray = to_gray(img)
    gray = equalize_hist(gray)
    gray = gaussian_blur(gray)
    flat = normalize(gray).flatten()
    return flat

def preprocess_visual(img_or_path, size=IMG_SIZE):
    """
    Returns intermediate stages as dict of uint8 images for UI display.
    """
    if isinstance(img_or_path, str):
        img = load_image(img_or_path)
    else:
        img = img_or_path.copy()

    resized = resize(img, size)
    gray    = to_gray(resized)
    clahe   = equalize_hist(gray)
    blurred = gaussian_blur(clahe)
    edges   = canny_edges(blurred)

    return {
        "Original (resized)": cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
        "Grayscale":          cv2.cvtColor(gray,    cv2.COLOR_GRAY2RGB),
        "CLAHE":              cv2.cvtColor(clahe,   cv2.COLOR_GRAY2RGB),
        "Gaussian Blur":      cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB),
        "Canny Edges":        cv2.cvtColor(edges,   cv2.COLOR_GRAY2RGB),
    }
