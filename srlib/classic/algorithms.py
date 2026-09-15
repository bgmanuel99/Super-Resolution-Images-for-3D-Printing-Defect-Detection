import cv2
import numpy as np
from typing import Tuple
from skimage import img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma

def apply_per_channel(algorithm, image, **kwargs):
    """Run a single-channel algorithm independently on every channel.

    Parameters
    ----------
    algorithm : callable
        Function taking a 2-D array as its first argument.
    image : np.ndarray
        (H, W, C) image.
    **kwargs
        Forwarded to ``algorithm`` on every channel.

    Returns
    -------
    np.ndarray
        (H', W', C) result, stacked back along the channel axis.
    """

    channels = [
        algorithm(image[..., c], **kwargs) for c in range(image.shape[2])
    ]

    return np.stack(channels, axis=-1)

def interpolate_nearest(lr_img, target_shape: Tuple[int, int]):
    """Nearest-neighbour upscaling.

    Replicates pixels instead of blending them, so it changes the frame
    size without inventing any detail. That is what makes it the resampling
    of the low-resolution baseline of the detection pipeline rather than a
    competing reconstruction, and it is why it is not one of the classic
    algorithms the benchmark ranks.
    """
    return cv2.resize(lr_img, target_shape, interpolation=cv2.INTER_NEAREST)

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

def back_projection(lr_image, target_shape: Tuple[int, int], iterations=10):
    """Iterative Back-Projection on a single channel.

    Parameters
    ----------
    lr_image : np.ndarray
        Low-resolution single-channel image.
    target_shape : tuple of int
        Target ``(height, width)`` of the reconstruction.
    iterations : int
        Number of back-projection refinement steps.

    Returns
    -------
    np.ndarray
        uint8 single-channel reconstruction at ``target_shape``.
    """

    target_h, target_w = target_shape
    lr = lr_image.astype(np.float32)

    # Initial estimate built only from the low-resolution input.
    hr = cv2.resize(
        lr,
        (target_w, target_h),
        interpolation=cv2.INTER_CUBIC,
    )

    for _ in range(iterations):
        down = cv2.resize(
            hr,
            (lr.shape[1], lr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        diff = lr - down
        diff_up = cv2.resize(
            diff,
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR,
        )
        hr += diff_up
        
    return np.clip(hr, 0, 255).astype(np.uint8)

def non_local_means(lr_gray, target_shape: Tuple[int, int]):
    """Non-local means denoising followed by Lanczos-4 upscaling.

    Parameters
    ----------
    lr_gray : np.ndarray
        Low-resolution single-channel image.
    target_shape : tuple of int
        Target ``(height, width)`` of the reconstruction.

    Returns
    -------
    np.ndarray
        uint8 image at ``target_shape``, on the same scale as every other
        algorithm of this module, so metrics against the reference stay
        comparable.
    """

    target_h, target_w = target_shape

    # The noise has to be estimated on the same scale the filter runs on.
    # estimate_sigma does not convert dtypes, so measuring it on the uint8
    # input yields a sigma in 0-255 units; feeding that as h to an image
    # normalised to [0, 1] makes every weight exp(-d^2/h^2) collapse to 1
    # and turns the filter into a plain box average over the search window.
    normalised = img_as_float(lr_gray)
    sigma_est = float(np.mean(estimate_sigma(normalised, channel_axis=None)))

    # img_as_float normalises to [0, 1], so the result is rescaled below.
    denoised = denoise_nl_means(
        normalised,
        h=1.15 * sigma_est,
        patch_size=5,
        patch_distance=6,
        fast_mode=True,
    )

    upscaled = cv2.resize(
        denoised,
        (target_w, target_h),
        interpolation=cv2.INTER_LANCZOS4,
    )

    return np.clip(upscaled * 255.0, 0, 255).astype(np.uint8)

def edge_guided_interpolation(lr_gray, target_shape: Tuple[int, int]):
    """Edge-guided interpolation using Sobel magnitude as sharpening prior.

    Parameters
    ----------
    lr_gray : np.ndarray
        Low-resolution single-channel image.
    target_shape : tuple of int
        Target ``(height, width)`` of the reconstruction.

    Returns
    -------
    np.ndarray
        uint8 single-channel image at ``target_shape``.
    """

    target_h, target_w = target_shape

    grad_x = cv2.Sobel(lr_gray, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(lr_gray, cv2.CV_64F, 0, 1)
    edges = np.hypot(grad_x, grad_y)

    upscaled = cv2.resize(
        lr_gray,
        (target_w, target_h),
        interpolation=cv2.INTER_LINEAR,
    )
    up_edges = cv2.resize(edges, (target_w, target_h))
    sharpened = cv2.addWeighted(
        upscaled.astype(np.float32),
        1.0,
        up_edges.astype(np.float32),
        0.3,
        0,
    )
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def frequency_extrapolation(lr_gray, target_shape: Tuple[int, int]):
    """Frequency-domain zero padding / extrapolation of the LR spectrum.

    Parameters
    ----------
    lr_gray : np.ndarray
        Low-resolution single-channel image.
    target_shape : tuple of int
        Target ``(height, width)`` of the reconstruction.

    Returns
    -------
    np.ndarray
        uint8 image at ``target_shape``, photometrically aligned with the
        input: the mean intensity is preserved.
    """

    rows, cols = lr_gray.shape
    pad_rows, pad_cols = target_shape

    fshift = np.fft.fftshift(np.fft.fft2(lr_gray.astype(np.float32)))

    f_padded = np.zeros((pad_rows, pad_cols), dtype=complex)
    row_start = pad_rows // 2 - rows // 2
    col_start = pad_cols // 2 - cols // 2
    f_padded[row_start:row_start + rows, col_start:col_start + cols] = fshift

    # np.fft.ifft2 divides by the transform size, which is now larger than
    # the one the spectrum was measured on. The ratio of sample counts
    # restores the original amplitude.
    scale = (pad_rows * pad_cols) / float(rows * cols)
    upscaled = np.abs(np.fft.ifft2(np.fft.ifftshift(f_padded))) * scale

    return np.clip(upscaled, 0, 255).astype(np.uint8)