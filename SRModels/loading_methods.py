import os
import cv2
import pickle
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
    interpolation_map_path=None):
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
    interpolation_map_path : str
        Path to interpolation map (SRCNN mode).

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

    if mode == 'srcnn' and interpolation_map_path is not None:
        interpolation_map = None
        with open(interpolation_map_path, 'rb') as f:
            interpolation_map = pickle.load(f)

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
            interp_method = cv2.INTER_CUBIC
            if interpolation_map is not None:
                interp_method = interpolation_map.get(fname, cv2.INTER_CUBIC)
                
                name_to_code = {
                    'INTER_LINEAR': cv2.INTER_LINEAR,
                    'INTER_CUBIC': cv2.INTER_CUBIC,
                    'INTER_AREA': cv2.INTER_AREA,
                    'INTER_LANCZOS4': cv2.INTER_LANCZOS4,
                }
                
                if interp_method in name_to_code:
                    interp_code = name_to_code[interp_method]
            
            lr_up = cv2.resize(
                lr_img, (hr_w, hr_h), interpolation=interp_code
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


def load_defects_dataset_as_patches(
    hr_root,
    patch_size=33,
    stride=14,
    class_map_path=None):
    """
    Build a classification dataset from HR images organized in subfolders.

    Behavior:
      - Recursively reads all images under `hr_root` (expects subfolders per class).
      - Normalizes images to [0,1] in RGB order.
      - Adds padding (using add_padding) and extracts HR patches of size `patch_size`
        with a sliding window of `stride` (like SRCNN HR path).
            - Reads a provided pickle mapping of HR image basename -> class id using
                the absolute `class_map_path` and assigns that label to all patches from
                that image. No labels are inferred or persisted by this function.

    Parameters
    ----------
    hr_root : str
        Path to HR images root that contains subfolders per defect type.
    patch_size : int
        HR patch size (height == width).
    stride : int
        Stride for the sliding window over HR images.
    class_map_path : str
        Absolute path to a pickle file containing a dict mapping
        { basename.png: class_id }. Required.

    Returns
    -------
    X : np.ndarray
        HR patches (N, patch_size, patch_size, 3) in float32 [0,1].
    y : np.ndarray
        Class labels for each patch as integers, aligned with X.
    """
    
    if not os.path.exists(hr_root):
        raise ValueError("HR root directory must exist.")
    if not os.path.isdir(hr_root):
        raise ValueError("HR root path must be a directory.")
    if not isinstance(patch_size, int) or patch_size <= 0:
        raise ValueError("patch_size must be positive int.")
    if not isinstance(stride, int) or stride <= 0:
        raise ValueError("stride must be positive int.")
    if not class_map_path or not isinstance(class_map_path, str):
        raise ValueError("class_map_path must be a non-empty string.")
    if not os.path.exists(class_map_path):
        raise FileNotFoundError(f"Class labels map not found: {class_map_path}")

    hr_paths = get_all_image_paths(hr_root)
    if not hr_paths:
        raise ValueError("No images found under HR root directory.")
    
    with open(class_map_path, 'rb') as f:
        class_labels_map = pickle.load(f)
    if not isinstance(class_labels_map, dict):
        raise ValueError("class_labels_map pickle must contain a dict of {basename: class_id}.")

    # Sort deterministically by basename
    hr_paths = sorted(hr_paths, key=lambda p: os.path.basename(p))

    X, y = [], []

    for img_path in hr_paths:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        hr_img = (
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            .astype(np.float32) / 255.0
        )

        hr_h, hr_w, _ = hr_img.shape

        # Determine class label from provided map using basename
        base = os.path.basename(img_path)
        if base not in class_labels_map:
            raise KeyError(f"Missing class id for image basename in class_labels_map: {base}")
        class_id = int(class_labels_map[base])

        # Padding and patch extraction (like SRCNN HR path)
        hr_proc = add_padding(hr_img, patch_size, stride)
        for i in range(0, hr_h - patch_size + 1, stride):
            for j in range(0, hr_w - patch_size + 1, stride):
                hr_patch = hr_proc[i:i+patch_size, j:j+patch_size, :]
                X.append(hr_patch)
                y.append(class_id)

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int64)
    
    return X_arr, y_arr


def load_predictions_dataset(
    lr_root,
    class_map_path):
    """
    Load a predictions dataset of full LR images (no patching).

    Behavior:
      - Recursively discovers all images under `lr_root` using `get_all_image_paths`.
      - Normalizes images to float32 in [0,1], RGB order.
      - Associates a class label per image using a provided pickle mapping
        of { basename.png: class_id }.
            - Returns numpy arrays X, y. Assumes images share the same spatial size to
                allow stacking without padding.

    Parameters
    ----------
    lr_root : str
        Root folder containing LR images in any subfolder structure.
    class_map_path : str
        Path to a pickle file containing a dict mapping LR basenames to class ids.

    Returns
    -------
    X : np.ndarray
        Array of LR images as float32 in [0,1], shape (N, H, W, 3). H and W are the
        maximum height/width among images after optional padding.
    y : np.ndarray
        Array of class ids (N,) as integers.
    """

    if not lr_root or not isinstance(lr_root, str) or not os.path.exists(lr_root):
        raise ValueError("lr_root must be an existing directory path.")
    if not os.path.isdir(lr_root):
        raise ValueError("lr_root must be a directory.")
    if not class_map_path or not isinstance(class_map_path, str):
        raise ValueError("class_map_path must be a non-empty string.")
    if not os.path.exists(class_map_path):
        raise FileNotFoundError(f"Class labels map not found: {class_map_path}")

    lr_paths = get_all_image_paths(lr_root)
    if not lr_paths:
        raise ValueError("No images found under LR root directory.")

    with open(class_map_path, 'rb') as f:
        class_labels_map = pickle.load(f)
    if not isinstance(class_labels_map, dict):
        raise ValueError("class_labels_map pickle must contain a dict of {basename: class_id}.")

    X = []
    y = []

    for img_path in lr_paths:
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Failed to read image: {img_path}")

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        base = os.path.basename(img_path)
        if base not in class_labels_map:
            raise KeyError(f"Missing class id for LR basename in class_labels_map: {base}")
        class_id = int(class_labels_map[base])

        X.append(img)
        y.append(class_id)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return X, y