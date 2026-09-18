import os
import cv2
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from srlib.progress import stage
from srlib.constants import (
    DATASET_FRACTION,
    EDSR_PATCH_SIZE,
    EDSR_SCALE_FACTOR,
    EDSR_STRIDE,
    ESRGAN_PATCH_SIZE,
    ESRGAN_SCALE_FACTOR,
    ESRGAN_STRIDE,
    RANDOM_SEED,
    SRCNN_PATCH_SIZE,
    SRCNN_STRIDE,
    SRCNN_UPSCALE_INTERPOLATION,
    TEST_SIZE,
    VAL_SIZE,
    VGG_PATCH_SIZE,
    VGG_STRIDE,
)

def add_padding(image, patch_size, stride):
    """
    Reflect-pad bottom and right so the patch grid covers the whole image.

    Shared by the training loaders and every inference path, so the two
    grids cannot drift apart.

    Parameters
    ----------
    image : np.ndarray
        (H, W, C) image.
    patch_size : int
        Patch size (height == width).
    stride : int
        Sliding window stride the extraction loop will use.

    Returns
    -------
    np.ndarray
        Padded image, or the input untouched when it already fits the grid.
    """

    height, width = image.shape[:2]

    # Smallest pad p such that (dim + p - patch_size) is a multiple of the
    # stride. When the image is smaller than one patch, grow it to exactly
    # one patch instead.
    pad_h = (-(height - patch_size)) % stride if height > patch_size else patch_size - height
    pad_w = (-(width - patch_size)) % stride if width > patch_size else patch_size - width

    if pad_h == 0 and pad_w == 0:
        return image

    # Reflected padding (mirror) preserves edge continuity.
    return np.pad(
        image,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode='reflect',
    )

def scale_padding_to_hr(hr_image, lr_image, lr_padded, scale_factor):
    """
    Build the HR counterpart of an already padded LR image.

    Each padded LR position is resolved to the LR pixel it came from and
    the HR block of that pixel is copied whole, which keeps
    ``hr_index = lr_index * scale_factor`` valid inside the reflected band.
    Padding each image against its own border would not, since the two
    mirror axes are not related by the scale factor.

    Parameters
    ----------
    hr_image : np.ndarray
        (H, W, C) high-resolution image.
    lr_image : np.ndarray
        (h, w, C) low-resolution image, with ``H == h * scale_factor``.
    lr_padded : np.ndarray
        Result of ``add_padding`` over ``lr_image``.
    scale_factor : int
        Upscaling factor between the two.

    Returns
    -------
    np.ndarray
        HR image padded to ``lr_padded.shape * scale_factor``.

    Raises
    ------
    ValueError
        If the HR size is not exactly ``scale_factor`` times the LR one,
        which would make the index correspondence undefined.
    """

    lr_h, lr_w = lr_image.shape[:2]
    hr_h, hr_w = hr_image.shape[:2]

    if (hr_h, hr_w) != (lr_h * scale_factor, lr_w * scale_factor):
        raise ValueError(
            f"HR size {(hr_h, hr_w)} is not {scale_factor}x the LR size "
            f"{(lr_h, lr_w)}, so LR and HR patches cannot be aligned."
        )

    pad_h = lr_padded.shape[0] - lr_h
    pad_w = lr_padded.shape[1] - lr_w
    if pad_h == 0 and pad_w == 0:
        return hr_image

    rows = _hr_index_map(lr_h, pad_h, scale_factor)
    cols = _hr_index_map(lr_w, pad_w, scale_factor)

    return hr_image[np.ix_(rows, cols, np.arange(hr_image.shape[2]))]

def _hr_index_map(lr_size, pad, scale_factor):
    """
    HR indices to read along one axis of a padded LR axis.

    Mirrors the source index the same way ``np.pad(mode='reflect')`` does,
    then expands each LR index into its ``scale_factor`` HR indices.
    """

    lr_index = np.arange(lr_size + pad)
    lr_index[lr_size:] = 2 * (lr_size - 1) - lr_index[lr_size:]

    return (
        lr_index[:, None] * scale_factor + np.arange(scale_factor)[None, :]
    ).ravel()

def get_all_image_paths(root):
    image_paths = []
    
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
            ):
                image_paths.append(os.path.join(dirpath, filename))
    
    return sorted(image_paths)

def read_image_as_rgb(path):
    """
    Read a single image from disk as float32 RGB in [0, 1].

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    np.ndarray
        Image of shape (H, W, 3), float32, RGB order, values in [0, 1].
    """

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

def read_class_labels_map(class_map_path):
    """
    Read the pickled { basename: class_id } mapping used to label images.

    Parameters
    ----------
    class_map_path : str
        Absolute path to the pickle file.

    Returns
    -------
    dict
        Mapping of image basename to class id.
    """

    if not class_map_path or not isinstance(class_map_path, str):
        raise ValueError("class_map_path must be a non-empty string.")
    if not os.path.exists(class_map_path):
        raise FileNotFoundError(f"Class labels map not found: {class_map_path}")

    with open(class_map_path, 'rb') as f:
        class_labels_map = pickle.load(f)

    if not isinstance(class_labels_map, dict):
        raise ValueError(
            "class_labels_map pickle must contain a dict of {basename: class_id}."
        )

    return class_labels_map

def index_image_pairs(hr_root, lr_root, class_map_path=None):
    """
    Index aligned LR/HR image pairs by basename without reading any pixel.

    Subsampling and partitioning are decided here, on paths and labels, so
    patch extraction can then be streamed image by image.

    Parameters
    ----------
    hr_root : str
        Root folder containing HR images in any subfolder structure.
    lr_root : str
        Root folder containing LR images in any subfolder structure.
    class_map_path : str, optional
        Path to the pickled { basename: class_id } mapping. When provided,
        a label array aligned with ``basenames`` is returned.

    Returns
    -------
    basenames : np.ndarray
        Basenames present in both roots, sorted deterministically.
    pairs : dict
        Mapping of basename to the ``(hr_path, lr_path)`` tuple.
    labels : np.ndarray or None
        Class id per basename, or None when ``class_map_path`` is omitted.
    """

    if not hr_root or not isinstance(hr_root, str) or not os.path.isdir(hr_root):
        raise ValueError("hr_root must be an existing directory path.")
    if not lr_root or not isinstance(lr_root, str) or not os.path.isdir(lr_root):
        raise ValueError("lr_root must be an existing directory path.")

    hr_paths = get_all_image_paths(hr_root)
    lr_paths = get_all_image_paths(lr_root)
    if not hr_paths:
        raise ValueError("No images found under HR root directory.")
    if not lr_paths:
        raise ValueError("No images found under LR root directory.")

    hr_dict = {os.path.basename(p): p for p in hr_paths}
    lr_dict = {os.path.basename(p): p for p in lr_paths}

    # Deterministic order: every loader must index the dataset identically,
    # otherwise the shared seed no longer yields the same partition.
    common = sorted(set(hr_dict) & set(lr_dict))
    if not common:
        raise ValueError("No matching basenames found between LR and HR roots.")

    pairs = {base: (hr_dict[base], lr_dict[base]) for base in common}
    basenames = np.array(common, dtype=object)

    labels = None
    if class_map_path is not None:
        class_labels_map = read_class_labels_map(class_map_path)
        missing = [base for base in common if base not in class_labels_map]
        if missing:
            raise KeyError(
                f"Missing class id for {len(missing)} basenames in "
                f"class_labels_map, first one: {missing[0]}"
            )
        labels = np.array(
            [int(class_labels_map[base]) for base in common], dtype=np.int64
        )

    return basenames, pairs, labels

def subsample_stratified(
    basenames,
    labels,
    fraction=DATASET_FRACTION,
    seed=RANDOM_SEED):
    """
    Take a random, class-stratified fraction of the image list.

    Stratification is required because the list is ordered by basename and
    basenames start with the defect type, so any contiguous slice is
    correlated with the class.

    Parameters
    ----------
    basenames : np.ndarray
        Image basenames to subsample.
    labels : np.ndarray
        Class id per basename, used as the stratification target.
    fraction : float or None
        Fraction of images to keep, in (0, 1]. None or >= 1.0 keeps them all.
    seed : int
        Random seed.

    Returns
    -------
    basenames : np.ndarray
    labels : np.ndarray
    """

    if fraction is None or fraction >= 1.0:
        return basenames, labels
    if fraction <= 0.0:
        raise ValueError("fraction must be in (0, 1] or None.")

    keep_idx, _ = train_test_split(
        np.arange(len(basenames)),
        train_size=fraction,
        shuffle=True,
        random_state=seed,
        stratify=labels,
    )

    # Keep the original basename order so the partition stays reproducible.
    keep_idx = np.sort(keep_idx)

    return basenames[keep_idx], labels[keep_idx]

def select_dataset_basenames(
    hr_root,
    lr_root,
    class_map_path,
    fraction=DATASET_FRACTION,
    seed=RANDOM_SEED):
    """
    Index the pairs and keep the stratified fraction the models are fed.

    No train/val/test split happens here, so the callers that only describe
    the dataset read the whole selected fraction.

    Parameters
    ----------
    hr_root, lr_root : str
        Roots of the HR and LR image trees.
    class_map_path : str
        Path to the pickled ``{basename: class_id}`` mapping, needed as the
        stratification target.
    fraction : float or None
        Fraction of images to keep. None or >= 1.0 keeps them all.
    seed : int
        Random seed, shared with the model loaders.

    Returns
    -------
    basenames : np.ndarray
        Selected basenames, in their original sorted order.
    pairs : dict
        Mapping of basename to ``(hr_path, lr_path)`` for every indexed pair.
    labels : np.ndarray
        Class id per selected basename.
    """

    basenames, pairs, labels = index_image_pairs(hr_root, lr_root, class_map_path)
    basenames, labels = subsample_stratified(basenames, labels, fraction, seed)

    return basenames, pairs, labels

def split_stratified(
    basenames,
    labels,
    test_size=TEST_SIZE,
    val_size=VAL_SIZE,
    seed=RANDOM_SEED):
    """
    Split the image list into train/val/test, stratified by class.

    The split is at IMAGE level, so no patch of a test image can reach the
    training set.

    Parameters
    ----------
    basenames : np.ndarray
        Image basenames to partition.
    labels : np.ndarray
        Class id per basename, used as the stratification target.
    test_size : float
        Fraction of the whole list held out as test.
    val_size : float
        Fraction of the remaining train list held out as validation.
    seed : int
        Random seed.

    Returns
    -------
    train : tuple of (np.ndarray, np.ndarray)
        ``(basenames, labels)`` of the train partition.
    val : tuple of (np.ndarray, np.ndarray)
        ``(basenames, labels)`` of the validation partition.
    test : tuple of (np.ndarray, np.ndarray)
        ``(basenames, labels)`` of the test partition.
    """

    train_idx, test_idx = train_test_split(
        np.arange(len(basenames)),
        test_size=test_size,
        shuffle=True,
        random_state=seed,
        stratify=labels,
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=val_size,
        shuffle=True,
        random_state=seed,
        stratify=labels[train_idx],
    )

    train = (basenames[train_idx], labels[train_idx])
    val = (basenames[val_idx], labels[val_idx])
    test = (basenames[test_idx], labels[test_idx])

    return train, val, test

def read_image_pairs(pairs, basenames, desc="images"):
    """
    Read the full LR/HR images of the given basenames into stacked arrays.

    Parameters
    ----------
    pairs : dict
        Mapping produced by ``index_image_pairs``.
    basenames : sequence of str
        Basenames to read, in the desired output order.
    desc : str
        Label of the progress bar, normally the name of the partition.

    Returns
    -------
    X_LR : np.ndarray
        (N, H_lr, W_lr, 3) float32 in [0, 1].
    X_HR : np.ndarray
        (N, H_hr, W_hr, 3) float32 in [0, 1].
    """

    X_LR, X_HR = [], []

    for base in tqdm(basenames, desc=f"    {desc:<5}", unit="img", leave=False):
        hr_path, lr_path = pairs[base]
        X_HR.append(read_image_as_rgb(hr_path))
        X_LR.append(read_image_as_rgb(lr_path))

    return (
        np.array(X_LR, dtype=np.float32),
        np.array(X_HR, dtype=np.float32),
    )

def patch_origins(height, width, patch_size, stride):
    """
    Yield the top-left corner of every patch of the sliding-window grid.

    ``height`` and ``width`` must be the dimensions AFTER ``add_padding``,
    or the right and bottom bands are left uncovered.

    Parameters
    ----------
    height : int
        Padded image height.
    width : int
        Padded image width.
    patch_size : int
        Patch size (height == width).
    stride : int
        Sliding window stride.

    Yields
    ------
    tuple of int
        ``(row, col)`` origin of each patch.
    """

    for i in range(0, height - patch_size + 1, stride):
        for j in range(0, width - patch_size + 1, stride):
            yield i, j

def extract_upscaled_patch_pairs(
    pairs,
    basenames,
    patch_size,
    stride,
    upscale_interpolation=SRCNN_UPSCALE_INTERPOLATION,
    desc="patches"):
    """
    Extract LR/HR patch pairs after upscaling LR to HR size (SRCNN layout).

    Both patches are ``patch_size x patch_size`` because the LR image is
    brought up to the HR frame before the grid is applied.

    Parameters
    ----------
    pairs : dict
        Mapping produced by ``index_image_pairs``.
    basenames : sequence of str
        Basenames of the partition to extract.
    patch_size : int
        Patch size (height == width).
    stride : int
        Sliding window stride.
    upscale_interpolation : int
        OpenCV interpolation used to bring LR up to the HR frame size.
    desc : str
        Label of the progress bar, normally the name of the partition.

    Returns
    -------
    X : np.ndarray
        (N, patch_size, patch_size, 3) LR patches.
    Y : np.ndarray
        (N, patch_size, patch_size, 3) HR patches.
    hr_sizes : set of tuple of int
        Distinct ``(height, width)`` HR frame sizes found in the partition,
        so callers can validate that the dataset is homogeneous.
    """

    X, Y = [], []
    hr_sizes = set()

    for base in tqdm(basenames, desc=f"    {desc:<5}", unit="img", leave=False):
        hr_path, lr_path = pairs[base]
        hr_img = read_image_as_rgb(hr_path)
        lr_img = read_image_as_rgb(lr_path)

        hr_h, hr_w = hr_img.shape[:2]
        hr_sizes.add((hr_h, hr_w))

        lr_up = cv2.resize(
            lr_img, (hr_w, hr_h), interpolation=upscale_interpolation
        )
        lr_up = np.clip(lr_up, 0.0, 1.0)

        hr_proc = add_padding(hr_img, patch_size, stride)
        lr_proc = add_padding(lr_up, patch_size, stride)

        height, width = hr_proc.shape[:2]
        for i, j in patch_origins(height, width, patch_size, stride):
            X.append(lr_proc[i:i+patch_size, j:j+patch_size, :])
            Y.append(hr_proc[i:i+patch_size, j:j+patch_size, :])

    return (
        np.array(X, dtype=np.float32),
        np.array(Y, dtype=np.float32),
        hr_sizes,
    )

def extract_scaled_patch_pairs(
    pairs,
    basenames,
    patch_size,
    stride,
    scale_factor,
    desc="patches"):
    """
    Extract LR/HR patch pairs keeping the LR resolution (EDSR/ESRGAN layout).

    ``patch_size`` is the LR patch size; the paired HR patch is
    ``patch_size * scale_factor``.

    Parameters
    ----------
    pairs : dict
        Mapping produced by ``index_image_pairs``.
    basenames : sequence of str
        Basenames of the partition to extract.
    patch_size : int
        LR patch size (height == width).
    stride : int
        Sliding window stride over the LR image.
    scale_factor : int
        Upscaling factor between LR and HR.
    desc : str
        Label of the progress bar, normally the name of the partition.

    Returns
    -------
    X : np.ndarray
        (N, patch_size, patch_size, 3) LR patches.
    Y : np.ndarray
        (N, patch_size * scale_factor, patch_size * scale_factor, 3) HR patches.
    """

    X, Y = [], []
    patch_size_hr = patch_size * scale_factor

    for base in tqdm(basenames, desc=f"    {desc:<5}", unit="img", leave=False):
        hr_path, lr_path = pairs[base]
        hr_img = read_image_as_rgb(hr_path)
        lr_img = read_image_as_rgb(lr_path)

        lr_proc = add_padding(lr_img, patch_size, stride)
        hr_proc = scale_padding_to_hr(hr_img, lr_img, lr_proc, scale_factor)

        # Iterate over padded LR, then index into padded HR
        height, width = lr_proc.shape[:2]
        for i, j in patch_origins(height, width, patch_size, stride):
            lr_patch = lr_proc[i:i+patch_size, j:j+patch_size, :]

            hr_i = i * scale_factor
            hr_j = j * scale_factor
            hr_patch = hr_proc[hr_i:hr_i+patch_size_hr, hr_j:hr_j+patch_size_hr, :]

            # Shapes should already match thanks to padding; keep a guard
            if (lr_patch.shape[:2] == (patch_size, patch_size)
                and hr_patch.shape[:2] == (patch_size_hr, patch_size_hr)):
                X.append(lr_patch)
                Y.append(hr_patch)

    return (
        np.array(X, dtype=np.float32),
        np.array(Y, dtype=np.float32),
    )

def extract_classification_patches(
    pairs,
    basenames,
    labels,
    patch_size,
    stride,
    desc="patches"):
    """
    Extract labelled patches from the HR side of the LR/HR pair.

    Every patch inherits the class id of its image, so the partition stays
    disjoint at image level. Only the HR side is extracted because the
    frame size the classifier is trained at is the one every reconstruction
    is produced at.

    Parameters
    ----------
    pairs : dict
        Mapping produced by ``index_image_pairs``.
    basenames : sequence of str
        Basenames of the partition to extract.
    labels : sequence of int
        Class id per basename, aligned with ``basenames``.
    patch_size : int
        Patch size (height == width).
    stride : int
        Sliding window stride.
    desc : str
        Label of the progress bar, normally the name of the partition.

    Returns
    -------
    X : np.ndarray
        (N, patch_size, patch_size, 3) patches in float32 [0, 1].
    y : np.ndarray
        (N,) class ids as int64, aligned with X.
    """

    X, y = [], []

    for base, label in tqdm(
            list(zip(basenames, labels)),
            desc=f"    {desc:<5}", unit="img", leave=False):
        hr_path, _ = pairs[base]
        image = read_image_as_rgb(hr_path)

        proc = add_padding(image, patch_size, stride)

        height, width = proc.shape[:2]
        for i, j in patch_origins(height, width, patch_size, stride):
            X.append(proc[i:i+patch_size, j:j+patch_size, :])
            y.append(label)

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int64),
    )

def report_patch_counts(step, X_train, X_val, X_test, Y_train=None):
    """Report the shape, weight and sanity of the extracted partitions.

    Parameters
    ----------
    step : callable
        The reporting helper yielded by ``srlib.progress.stage``.
    X_train, X_val, X_test : np.ndarray
        Input arrays of each partition.
    Y_train : np.ndarray, optional
        Target array, reported alongside the input when it holds images.

    Raises
    ------
    ValueError
        If the training partition holds NaN or infinite values, which would
        silently stop the loss converging.
    """

    arrays = [x for x in (X_train, X_val, X_test, Y_train) if x is not None]
    total_gb = sum(x.nbytes for x in arrays) / 2 ** 30
    shape = " x ".join(str(d) for d in X_train.shape[1:])

    step(
        f"patches   {len(X_train):,} train / {len(X_val):,} val / "
        f"{len(X_test):,} test   of {shape}"
    )
    step(f"memory    {total_gb:.2f} GB")

    ranges = [f"X [{X_train.min():.4f}, {X_train.max():.4f}]"]
    if Y_train is not None:
        ranges.append(f"Y [{Y_train.min():.4f}, {Y_train.max():.4f}]")
    step("range     " + "   ".join(ranges))

    invalid = [name for name, x in (("X", X_train), ("Y", Y_train))
               if x is not None and not np.isfinite(x).all()]
    if invalid:
        raise ValueError(
            f"{', '.join(invalid)} of the training partition holds NaN or "
            "infinite values, which would silently stop the loss converging."
        )

def validate_patch_params(patch_size, stride, scale_factor=None):
    """Validate the sliding window parameters shared by every loader."""

    if not isinstance(patch_size, int) or patch_size <= 0:
        raise ValueError("patch_size must be positive int.")
    if not isinstance(stride, int) or stride <= 0:
        raise ValueError("stride must be positive int.")
    if (scale_factor is not None
        and (not isinstance(scale_factor, int) or scale_factor <= 0)):
        raise ValueError("scale_factor must be positive int.")

def partition_dataset_images(
    hr_root,
    lr_root,
    class_map_path,
    subsample_fraction=DATASET_FRACTION,
    test_size=TEST_SIZE,
    val_size=VAL_SIZE,
    seed=RANDOM_SEED):
    """
    Index, subsample and partition the dataset at IMAGE level.

    Every loader starts here, so any two of them called with the same
    roots, class map, fraction, split sizes and seed yield the same
    train/val/test images.

    Parameters
    ----------
    hr_root : str
        Root folder containing HR images.
    lr_root : str
        Root folder containing LR images.
    class_map_path : str
        Path to the pickled { basename: class_id } mapping. Required, because
        both the subsample and the split are stratified by class.
    subsample_fraction : float or None
        Fraction of images to keep before splitting. None keeps them all.
    test_size : float
        Fraction of the kept images held out as test.
    val_size : float
        Fraction of the remaining train images held out as validation.
    seed : int
        Random seed.

    Returns
    -------
    pairs : dict
        Mapping of basename to the ``(hr_path, lr_path)`` tuple.
    train : tuple of (np.ndarray, np.ndarray)
    val : tuple of (np.ndarray, np.ndarray)
    test : tuple of (np.ndarray, np.ndarray)
    """

    basenames, pairs, labels = index_image_pairs(
        hr_root, lr_root, class_map_path
    )
    indexed = len(basenames)

    basenames, labels = subsample_stratified(
        basenames, labels, subsample_fraction, seed
    )
    train, val, test = split_stratified(
        basenames, labels, test_size, val_size, seed
    )

    # Seeing the same four numbers in two notebooks is what confirms they
    # derive the same partition.
    print(
        f"  images    {indexed} indexed -> {len(basenames)} kept -> "
        f"{len(train[0])} train / {len(val[0])} val / {len(test[0])} test",
        flush=True,
    )

    return pairs, train, val, test

def load_srcnn_dataset(
    hr_root,
    lr_root,
    class_map_path,
    patch_size=SRCNN_PATCH_SIZE,
    stride=SRCNN_STRIDE,
    upscale_interpolation=SRCNN_UPSCALE_INTERPOLATION,
    subsample_fraction=DATASET_FRACTION,
    test_size=TEST_SIZE,
    val_size=VAL_SIZE,
    seed=RANDOM_SEED):
    """
    Build the SRCNN dataset: LR patches upscaled to HR size -> HR patches.

    Images are split first and patches extracted afterwards, so no patch of
    a test image reaches the training set.

    Parameters
    ----------
    hr_root : str
        Root folder containing HR images.
    lr_root : str
        Root folder containing LR images.
    class_map_path : str
        Path to the pickled { basename: class_id } mapping, used only to
        stratify the subsample and the split.
    patch_size : int
        Patch size (height == width) on the HR frame.
    stride : int
        Sliding window stride.
    upscale_interpolation : int
        OpenCV interpolation used to bring LR up to the HR frame size.
    subsample_fraction : float or None
        Fraction of images to keep before splitting. None keeps them all.
    test_size : float
        Fraction of the kept images held out as test.
    val_size : float
        Fraction of the remaining train images held out as validation.
    seed : int
        Random seed.

    Returns
    -------
    X_train, Y_train, X_val, Y_val, X_test, Y_test : np.ndarray
        LR and HR patches of each partition.
    hr_h : int
        HR frame height, needed to reassemble full images at inference time.
    hr_w : int
        HR frame width.
    """

    validate_patch_params(patch_size, stride)

    with stage(f"SRCNN dataset | patch {patch_size}, stride {stride}") as step:
        pairs, train, val, test = partition_dataset_images(
            hr_root,
            lr_root,
            class_map_path,
            subsample_fraction=subsample_fraction,
            test_size=test_size,
            val_size=val_size,
            seed=seed,
        )

        X_train, Y_train, sizes_train = extract_upscaled_patch_pairs(
            pairs, train[0], patch_size, stride, upscale_interpolation,
            desc="train",
        )
        X_val, Y_val, sizes_val = extract_upscaled_patch_pairs(
            pairs, val[0], patch_size, stride, upscale_interpolation,
            desc="val",
        )
        X_test, Y_test, sizes_test = extract_upscaled_patch_pairs(
            pairs, test[0], patch_size, stride, upscale_interpolation,
            desc="test",
        )
        report_patch_counts(step, X_train, X_val, X_test, Y_train)

        # The returned frame size becomes the target shape of every SR
        # method, so it has to be a property of the whole dataset.
        hr_sizes = sizes_train | sizes_val | sizes_test
        if len(hr_sizes) != 1:
            raise ValueError(
                "HR images must all share the same frame size, found "
                f"{sorted(hr_sizes)}"
            )

        hr_h, hr_w = hr_sizes.pop()
        step(f"frame     {hr_w} x {hr_h}")

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, hr_h, hr_w

def load_edsr_dataset(
    hr_root,
    lr_root,
    class_map_path,
    patch_size=EDSR_PATCH_SIZE,
    stride=EDSR_STRIDE,
    scale_factor=EDSR_SCALE_FACTOR,
    subsample_fraction=DATASET_FRACTION,
    test_size=TEST_SIZE,
    val_size=VAL_SIZE,
    seed=RANDOM_SEED):
    """
    Build the EDSR dataset: LR patches -> HR patches at ``scale_factor``.

    Images are split first and patches extracted afterwards, so no patch of
    a test image reaches the training set.

    Parameters
    ----------
    hr_root : str
        Root folder containing HR images.
    lr_root : str
        Root folder containing LR images.
    class_map_path : str
        Path to the pickled { basename: class_id } mapping, used only to
        stratify the subsample and the split.
    patch_size : int
        LR patch size (height == width).
    stride : int
        Sliding window stride over the LR image.
    scale_factor : int
        Upscaling factor between LR and HR.
    subsample_fraction : float or None
        Fraction of images to keep before splitting. None keeps them all.
    test_size : float
        Fraction of the kept images held out as test.
    val_size : float
        Fraction of the remaining train images held out as validation.
    seed : int
        Random seed.

    Returns
    -------
    X_train, Y_train, X_val, Y_val, X_test, Y_test : np.ndarray
        LR and HR patches of each partition.
    """

    validate_patch_params(patch_size, stride, scale_factor)

    with stage(
            f"EDSR dataset | patch {patch_size}, stride {stride}, "
            f"x{scale_factor}") as step:
        pairs, train, val, test = partition_dataset_images(
            hr_root,
            lr_root,
            class_map_path,
            subsample_fraction=subsample_fraction,
            test_size=test_size,
            val_size=val_size,
            seed=seed,
        )

        X_train, Y_train = extract_scaled_patch_pairs(
            pairs, train[0], patch_size, stride, scale_factor, desc="train"
        )
        X_val, Y_val = extract_scaled_patch_pairs(
            pairs, val[0], patch_size, stride, scale_factor, desc="val"
        )
        X_test, Y_test = extract_scaled_patch_pairs(
            pairs, test[0], patch_size, stride, scale_factor, desc="test"
        )
        report_patch_counts(step, X_train, X_val, X_test, Y_train)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def load_esrgan_dataset(
    hr_root,
    lr_root,
    class_map_path,
    patch_size=ESRGAN_PATCH_SIZE,
    stride=ESRGAN_STRIDE,
    scale_factor=ESRGAN_SCALE_FACTOR,
    subsample_fraction=DATASET_FRACTION,
    test_size=TEST_SIZE,
    val_size=VAL_SIZE,
    seed=RANDOM_SEED):
    """
    Build the ESRGAN dataset: LR patches -> HR patches at ``scale_factor``.

    Images are split first and patches extracted afterwards, so no patch of
    a test image reaches the training set.

    Parameters
    ----------
    hr_root : str
        Root folder containing HR images.
    lr_root : str
        Root folder containing LR images.
    class_map_path : str
        Path to the pickled { basename: class_id } mapping, used only to
        stratify the subsample and the split.
    patch_size : int
        LR patch size (height == width).
    stride : int
        Sliding window stride over the LR image.
    scale_factor : int
        Upscaling factor between LR and HR.
    subsample_fraction : float or None
        Fraction of images to keep before splitting. None keeps them all.
    test_size : float
        Fraction of the kept images held out as test.
    val_size : float
        Fraction of the remaining train images held out as validation.
    seed : int
        Random seed.

    Returns
    -------
    X_train, Y_train, X_val, Y_val, X_test, Y_test : np.ndarray
        LR and HR patches of each partition.
    """

    validate_patch_params(patch_size, stride, scale_factor)

    with stage(
            f"ESRGAN dataset | patch {patch_size}, stride {stride}, "
            f"x{scale_factor}") as step:
        pairs, train, val, test = partition_dataset_images(
            hr_root,
            lr_root,
            class_map_path,
            subsample_fraction=subsample_fraction,
            test_size=test_size,
            val_size=val_size,
            seed=seed,
        )

        X_train, Y_train = extract_scaled_patch_pairs(
            pairs, train[0], patch_size, stride, scale_factor, desc="train"
        )
        X_val, Y_val = extract_scaled_patch_pairs(
            pairs, val[0], patch_size, stride, scale_factor, desc="val"
        )
        X_test, Y_test = extract_scaled_patch_pairs(
            pairs, test[0], patch_size, stride, scale_factor, desc="test"
        )
        report_patch_counts(step, X_train, X_val, X_test, Y_train)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def load_vgg16_dataset(
    hr_root,
    lr_root,
    class_map_path,
    patch_size=VGG_PATCH_SIZE,
    stride=VGG_STRIDE,
    subsample_fraction=DATASET_FRACTION,
    test_size=TEST_SIZE,
    val_size=VAL_SIZE,
    seed=RANDOM_SEED):
    """
    Build the VGG16 classification dataset: HR patches -> class labels.

    The classifier is trained on the HR side only, which is the frame size
    every reconstruction is produced at. ``lr_root`` is still required
    because the partition is derived from the basenames present in BOTH
    roots, which is what keeps this split identical to every other loader's.

    Parameters
    ----------
    hr_root : str
        Root folder containing HR images.
    lr_root : str
        Root folder containing LR images, used to resolve the partition.
    class_map_path : str
        Path to the pickled { basename: class_id } mapping.
    patch_size : int
        Patch size (height == width).
    stride : int
        Sliding window stride.
    subsample_fraction : float or None
        Fraction of images to keep before splitting. None keeps them all.
    test_size : float
        Fraction of the kept images held out as test.
    val_size : float
        Fraction of the remaining train images held out as validation.
    seed : int
        Random seed.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test : np.ndarray
        Patches and class labels of each partition.
    """

    validate_patch_params(patch_size, stride)

    with stage(
            f"VGG16 dataset | patch {patch_size}, stride {stride}") as step:
        pairs, train, val, test = partition_dataset_images(
            hr_root,
            lr_root,
            class_map_path,
            subsample_fraction=subsample_fraction,
            test_size=test_size,
            val_size=val_size,
            seed=seed,
        )

        X_train, y_train = extract_classification_patches(
            pairs, train[0], train[1], patch_size, stride, desc="train"
        )
        X_val, y_val = extract_classification_patches(
            pairs, val[0], val[1], patch_size, stride, desc="val"
        )
        X_test, y_test = extract_classification_patches(
            pairs, test[0], test[1], patch_size, stride, desc="test"
        )
        report_patch_counts(step, X_train, X_val, X_test)

        classes, counts = np.unique(
            np.concatenate([y_train, y_val, y_test]), return_counts=True
        )
        step(f"classes   {dict(zip(classes.tolist(), counts.tolist()))}")

    return X_train, y_train, X_val, y_val, X_test, y_test

def load_defect_detection_pipeline_dataset(
    hr_root,
    lr_root,
    class_map_path,
    subsample_fraction=DATASET_FRACTION,
    test_size=TEST_SIZE,
    val_size=VAL_SIZE,
    seed=RANDOM_SEED):
    """
    Build the defect detection pipeline dataset: full LR/HR images + labels.

    No patching happens here, since the pipeline works on whole frames.
    The partition is computed as in every other loader, so calling this
    with the same fraction, split sizes and seed as ``load_vgg16_dataset``
    guarantees the test images were never seen during training.

    Parameters
    ----------
    hr_root : str
        Root folder containing HR images.
    lr_root : str
        Root folder containing LR images.
    class_map_path : str
        Path to the pickled { basename: class_id } mapping.
    subsample_fraction : float or None
        Fraction of images to keep before splitting. None keeps them all.
    test_size : float
        Fraction of the kept images held out as test.
    val_size : float
        Fraction of the remaining train images held out as validation.
    seed : int
        Random seed.

    Returns
    -------
    X_lr_train, X_hr_train, y_train : np.ndarray
    X_lr_val, X_hr_val, y_val : np.ndarray
    X_lr_test, X_hr_test, y_test : np.ndarray
        Full LR/HR images in float32 [0, 1] and their class labels.
    """

    with stage("Detection pipeline dataset | full frames") as step:
        pairs, train, val, test = partition_dataset_images(
            hr_root,
            lr_root,
            class_map_path,
            subsample_fraction=subsample_fraction,
            test_size=test_size,
            val_size=val_size,
            seed=seed,
        )

        X_lr_train, X_hr_train = read_image_pairs(pairs, train[0], desc="train")
        X_lr_val, X_hr_val = read_image_pairs(pairs, val[0], desc="val")
        X_lr_test, X_hr_test = read_image_pairs(pairs, test[0], desc="test")

        step(
            f"frames    LR {X_lr_test.shape[2]}x{X_lr_test.shape[1]} | "
            f"HR {X_hr_test.shape[2]}x{X_hr_test.shape[1]}"
        )
        step(f"test      {len(X_lr_test)} images, the split the pipeline scores")

    return (
        X_lr_train, X_hr_train, train[1],
        X_lr_val, X_hr_val, val[1],
        X_lr_test, X_hr_test, test[1],
    )