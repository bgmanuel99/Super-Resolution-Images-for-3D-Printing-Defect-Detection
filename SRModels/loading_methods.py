import os
import cv2
import numpy as np

def add_padding(image, patch_size, stride):
    """Add padding to ensure full coverage."""

    h, w, _ = image.shape

    # Compute required padding so patches cover entire image
    pad_h = ((patch_size - (h % stride)) % stride if h % stride != 0 else 0)
    pad_w = ((patch_size - (w % stride)) % stride if w % stride != 0 else 0)

    # Add extra padding to guarantee full coverage
    pad_h = max(pad_h, patch_size - stride)
    pad_w = max(pad_w, patch_size - stride)

    # Use reflected padding (mirror) to preserve edge continuity
    padded_img = np.pad(
        image, 
        ((0, pad_h), (0, pad_w), (0, 0)), 
        mode='reflect'
    )

    return padded_img

def get_all_image_paths(root):
    image_paths = []
    
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
            ):
                image_paths.append(os.path.join(dirpath, filename))
    
    return sorted(image_paths)

def load_dataset_as_patches(
        hr_root,
        lr_root,
        mode='srcnn',
        patch_size=33,
        stride=14,
        scale_factor=2,
        upsample_interp=cv2.INTER_CUBIC):
    """
    Dataset loader producing aligned LR/HR patch pairs.

    Modes:
        'srcnn': Upscale LR to HR size first; both patches are
            (patch_size x patch_size).
        'scale': Treat patch_size as LR patch size; HR patch size =
            patch_size * scale_factor.

    Parameters
    ----------
    hr_root : str
        Path to HR images.
    lr_root : str
        Path to LR images.
    mode : {'srcnn', 'scale'}
        Loader behavior.
    patch_size : int
        Patch size (SRCNN) or LR patch size (scale mode).
    stride : int
        Stride for sliding window over LR (and HR in SRCNN).
    scale_factor : int
        Scaling factor (only used in scale mode).
    upsample_interp :
        OpenCV interpolation flag (SRCNN mode).

    Returns
    -------
    X : np.ndarray
        LR patches.
    Y : np.ndarray
        HR patches.
    meta (optional) : dict
        {'hr_h': int, 'hr_w': int} for SRCNN mode.
    """
    
    if mode not in ('srcnn', 'scale'):
        raise ValueError("mode must be 'srcnn' or 'scale'")
    if not os.path.exists(hr_root) or not os.path.exists(lr_root):
        raise ValueError("Both HR and LR root directories must exist.")
    if not os.path.isdir(hr_root) or not os.path.isdir(lr_root):
        raise ValueError("Both HR and LR root paths must be directories.")
    if not isinstance(patch_size, int) or patch_size <= 0:
        raise ValueError("patch_size must be positive int.")
    if not isinstance(stride, int) or stride <= 0:
        raise ValueError("stride must be positive int.")
    if (mode == 'scale' 
        and (not isinstance(scale_factor, int) or scale_factor <= 0)):
        raise ValueError("scale_factor must be positive int.")

    X, Y = [], []
    
    hr_paths = get_all_image_paths(hr_root)
    lr_paths = get_all_image_paths(lr_root)
    
    if not hr_paths or not lr_paths:
        raise ValueError("No images found in provided directories.")

    hr_dict = {os.path.basename(p): p for p in hr_paths}
    lr_dict = {os.path.basename(p): p for p in lr_paths}
    common_filenames = sorted(set(hr_dict) & set(lr_dict))

    for fname in common_filenames:
        hr_img = cv2.imread(hr_dict[fname], cv2.IMREAD_COLOR)
        lr_img = cv2.imread(lr_dict[fname], cv2.IMREAD_COLOR)

        # Normalize
        hr_img = (
            cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
            .astype(np.float32) / 255.0
        )
        lr_img = (
            cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
            .astype(np.float32) / 255.0
        )

        hr_h, hr_w, _ = hr_img.shape
        lr_h, lr_w, _ = lr_img.shape

        if mode == 'srcnn':
            lr_up = cv2.resize(
                lr_img, (hr_w, hr_h), interpolation=upsample_interp
            )
            
            # Add padding to ensure full coverage
            hr_proc = add_padding(hr_img, patch_size, stride)
            lr_proc = add_padding(lr_up, patch_size, stride)
            
            for i in range(0, hr_h - patch_size + 1, stride):
                for j in range(0, hr_w - patch_size + 1, stride):
                    hr_patch = hr_proc[i:i+patch_size, j:j+patch_size, :]
                    lr_patch = lr_proc[i:i+patch_size, j:j+patch_size, :]
                    
                    X.append(lr_patch)
                    Y.append(hr_patch)
        else: # mode == 'scale'
            patch_size_hr = patch_size * scale_factor
            
            # Add padding to ensure full coverage
            hr_proc = add_padding(hr_img, patch_size_hr, stride)
            lr_proc = add_padding(lr_img, patch_size, stride)
            
            for i in range(0, lr_h - patch_size + 1, stride):
                for j in range(0, lr_w - patch_size + 1, stride):
                    # Extract LR patch
                    lr_patch = lr_proc[i:i+patch_size, j:j+patch_size]
                    
                    # Extract corresponding HR patch
                    hr_i = i * scale_factor
                    hr_j = j * scale_factor
                    
                    if ((hr_i + patch_size_hr <= hr_h)
                        and (hr_j + patch_size_hr <= hr_w)):
                        hr_patch = hr_proc[
                            hr_i:hr_i + patch_size_hr,
                            hr_j:hr_j + patch_size_hr,
                        ]
                        
                        if (lr_patch.shape[:2] == (patch_size, patch_size)
                            and hr_patch.shape[:2] 
                                == (patch_size_hr, patch_size_hr)):
                            X.append(lr_patch)
                            Y.append(hr_patch)
    
    X_arr = np.array(X)
    Y_arr = np.array(Y)
    
    if mode == 'srcnn':
        return X_arr, Y_arr, hr_h, hr_w
    return X_arr, Y_arr