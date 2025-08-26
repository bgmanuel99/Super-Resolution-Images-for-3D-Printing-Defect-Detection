import time
import tracemalloc
from math import sqrt

import cv2
import numpy as np

DEF_EPS = 1e-9

def time_algorithm(func, *args, **kwargs):
    """Return (result, elapsed_seconds) for the function.

    Only the body execution time is measured.
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed

def memory_algorithm(func, *args, **kwargs):
    """Return result measuring peak memory for the function body.

    The peak in bytes is computed via tracemalloc (discarded here).
    """
    tracemalloc.start()
    result = func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak

def mae(a, b):
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))

def rmse(a, b):
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(sqrt(np.mean(diff * diff) + DEF_EPS))

def _ensure_gray_f32(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img.dtype != np.float32:
        img = img.astype(np.float32, copy=False)
    if img.max() > 1.5:
        img = img / 255.0
    return img

def sobel_mag(img):
    g = _ensure_gray_f32(img)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)

def gradient_mse(hr, sr):
    hr_m = sobel_mag(hr)
    sr_m = sobel_mag(sr)
    return float(np.mean((hr_m - sr_m) ** 2))

def epi(hr, sr):
    hr_m = sobel_mag(hr)
    sr_m = sobel_mag(sr)
    return float((sr_m.sum() + DEF_EPS) / (hr_m.sum() + DEF_EPS))

def hf_energy_ratio(hr, sr, radius_frac=0.6):
    # Inputs grayscale float or uint8.
    hr_f = hr.astype(np.float32)
    sr_f = sr.astype(np.float32)
    F_hr = np.fft.fftshift(np.fft.fft2(hr_f))
    F_sr = np.fft.fftshift(np.fft.fft2(sr_f))
    h, w = hr_f.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    r_max = r.max() + DEF_EPS
    mask = r > (radius_frac * r_max)
    num = np.abs(F_sr)[mask].sum() + DEF_EPS
    den = np.abs(F_hr)[mask].sum() + DEF_EPS
    return float(num / den)

def kl_divergence(p_img, q_img, bins=256):
    # p_img (HR), q_img (SR); assume grayscale 0-255 or float 0-1.
    if p_img.dtype != np.uint8:
        p_arr = np.clip(p_img, 0, 1) * 255.0
    else:
        p_arr = p_img.astype(np.float32)
    if q_img.dtype != np.uint8:
        q_arr = np.clip(q_img, 0, 1) * 255.0
    else:
        q_arr = q_img.astype(np.float32)
    p_hist, _ = np.histogram(p_arr, bins=bins, range=(0, 255), density=True)
    q_hist, _ = np.histogram(q_arr, bins=bins, range=(0, 255), density=True)
    eps = 1e-12
    P = p_hist + eps
    Q = q_hist + eps
    return float(np.sum(P * np.log(P / Q)))

def kl_divergence_color(p_rgb, q_rgb, bins=64):
    # Expect RGB float 0-255 or 0-1.
    if p_rgb.dtype != np.uint8:
        p = np.clip(p_rgb, 0, 1) * 255.0
    else:
        p = p_rgb.astype(np.float32)
    if q_rgb.dtype != np.uint8:
        q = np.clip(q_rgb, 0, 1) * 255.0
    else:
        q = q_rgb.astype(np.float32)
    eps = 1e-12
    total = 0.0
    for c in range(p.shape[2]):
        p_hist, _ = np.histogram(p[..., c], bins=bins, range=(0, 255), density=True)
        q_hist, _ = np.histogram(q[..., c], bins=bins, range=(0, 255), density=True)
        P = p_hist + eps
        Q = q_hist + eps
        total += np.sum(P * np.log(P / Q))
    return float(total / max(1, p.shape[2]))