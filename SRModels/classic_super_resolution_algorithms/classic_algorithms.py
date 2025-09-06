import cv2
import numpy as np
from typing import Tuple
from skimage import img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma

def interpolate_bilinear(lr_img, target_shape: Tuple[int, int]):
    """Bilinear upscaling."""
    return cv2.resize(lr_img, target_shape, interpolation=cv2.INTER_LINEAR)

def interpolate_bicubic(lr_img, target_shape: Tuple[int, int]):
    """Bicubic upscaling."""
    return cv2.resize(lr_img, target_shape, interpolation=cv2.INTER_CUBIC)

def interpolate_area(lr_img, target_shape: Tuple[int, int]):
    """Area (resampling) upscaling."""
    return cv2.resize(lr_img, target_shape, interpolation=cv2.INTER_AREA)

def interpolate_lanczos(lr_img, target_shape: Tuple[int, int]):
    """Lanczos-4 upscaling."""
    return cv2.resize(lr_img, target_shape, interpolation=cv2.INTER_LANCZOS4)

def back_projection(hr_image, lr_image, iterations=10):
    """Iterative Back-Projection on grayscale images."""
    
    hr = hr_image.astype(np.float32).copy()
    
    for _ in range(iterations):
        down = cv2.resize(
            hr,
            (lr_image.shape[1], lr_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        diff = lr_image.astype(np.float32) - down
        diff_up = cv2.resize(
            diff,
            (hr.shape[1], hr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        hr += diff_up
        
    return np.clip(hr, 0, 255).astype(np.uint8)

def non_local_means(hr_g, lr_g):
    sigma_est = np.mean(estimate_sigma(lr_g, channel_axis=None))

    denoised = denoise_nl_means(
        img_as_float(lr_g),
        h=1.15 * sigma_est,
        patch_size=5,
        patch_distance=6,
        fast_mode=True,
    )

    return cv2.resize(
        denoised, 
        (hr_g.shape[1], hr_g.shape[0]), 
        interpolation=cv2.INTER_LANCZOS4
    )

def edge_guided_interpolation(ground_truth, image):
    """Edge-guided interpolation using Sobel magnitude as sharpening prior."""
    
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    edges = np.hypot(grad_x, grad_y)

    upscaled = cv2.resize(
        image,
        (ground_truth.shape[1], ground_truth.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    up_edges = cv2.resize(edges, (upscaled.shape[1], upscaled.shape[0]))
    sharpened = cv2.addWeighted(
        upscaled.astype(np.float32),
        1.0,
        up_edges.astype(np.float32),
        0.3,
        0,
    )
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def frequency_extrapolation(ground_truth, image):
    """Frequency-domain zero padding / extrapolation of LR spectrum
    to HR size.
    """
    
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    
    rows, cols = image.shape
    pad_rows, pad_cols = ground_truth.shape
    f_padded = np.zeros((pad_rows, pad_cols), dtype=complex)
    center_row = pad_rows // 2
    center_col = pad_cols // 2
    half_rows = rows // 2
    half_cols = cols // 2
    
    row_start = center_row - half_rows
    row_end = row_start + rows
    col_start = center_col - half_cols
    col_end = col_start + cols

    f_padded[row_start:row_end, col_start:col_end] = fshift
    f_ishift = np.fft.ifftshift(f_padded)
    img_upscaled = np.fft.ifft2(f_ishift)
    
    return np.abs(img_upscaled)