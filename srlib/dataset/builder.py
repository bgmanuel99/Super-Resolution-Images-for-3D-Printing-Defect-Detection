import hashlib
import json
import os
import pickle
import time
from dataclasses import dataclass

import cv2
import numpy as np

from srlib.progress import format_duration
from srlib.constants import (
    BLANK_FRAME_MIN_STD,
    CLASS_LABELS_PATH,
    DEFAULT_DEGRADATION_CONFIG,
    DEFAULT_DEGRADATION_SEED,
    DEGRADATION_LOG_PATH,
    DEGRADATION_LOG_VERSION,
    DEGRADATION_SCALE_FACTOR,
    HR_ROOT,
    INTERPOLATION_NAME_TO_CODE,
    LR_ROOT,
    REPLAY_DEGRADATION_FROM_LOG,
    VIDEO_CLASS_ID_PER_FOLDER,
    VIDEO_FRAME_INTERVAL_PER_FOLDER,
    VIDEO_MAX_VIDEOS_PER_FOLDER,
    VIDEOS_ROOT,
)

def smart_square_crop(img):
    """
    Crops the image to a square (width x width) region containing the main object.
    The crop is centered on the largest contour (assumed to be the object).
    If no contour is found, crops the center square.

    The side is forced to be even so that the LR frame is exactly half of
    it. An odd side would make the HR/LR ratio differ from the scale factor
    and break the index correspondence between paired patches.
    """
    
    h, w = img.shape[:2]
    crop_size = min(w, h) // 2 * 2
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold to find object (assume object is not background)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        x, y, ww, hh = cv2.boundingRect(largest)
        
        # Center crop on the object
        cx = x + ww // 2
        cy = y + hh // 2
        
        # Calculate crop box
        half = crop_size // 2
        left = max(0, cx - half)
        top = max(0, cy - half)
        
        # Ensure crop is within image
        if left + crop_size > w:
            left = w - crop_size
            
        if top + crop_size > h:
            top = h - crop_size
            
        left = max(0, left)
        top = max(0, top)
        crop = img[top:top+crop_size, left:left+crop_size]
    else:
        # Fallback: center crop
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        crop = img[top:top+crop_size, left:left+crop_size]
    
    return crop

def derive_image_rng(basename, master_seed=DEFAULT_DEGRADATION_SEED):
    """
    Builds the random generator used to degrade one image.

    The basename is mixed into the seed only to decorrelate images from each
    other. Without it a single generator would be consumed in processing
    order, so adding a video or changing frame_interval would silently alter
    the degradation of every image that comes after it. With it, an image
    keeps its degradation no matter when or in how many runs it is produced.

    The master seed is what selects the dataset: changing it redraws all of it.

    Parameters:
        basename (str): LR image file name, e.g. "low_z_offset42.png".
        master_seed (int): Seed of the whole LR dataset.

    Returns:
        np.random.Generator: Generator for this image.
    """

    name_digest = hashlib.sha256(basename.encode("utf-8")).digest()[:8]

    return np.random.default_rng(
        [int(master_seed), int.from_bytes(name_digest, "big")]
    )

def sample_degradation_params(rng, config=None):
    """
    Draws the degradation decisions for a single image.

    Parameters:
        rng (np.random.Generator): Generator, normally from derive_image_rng.
        config (dict): Degradation hyperparameters. Defaults to
            DEFAULT_DEGRADATION_CONFIG.

    Returns:
        dict: JSON-serialisable decisions, where a None entry means the step
            is skipped for this image.
    """

    cfg = config if config is not None else DEFAULT_DEGRADATION_CONFIG
    params = {}

    blur_cfg = cfg["gaussian_blur"]
    if rng.random() < blur_cfg["probability"]:
        sigma_low, sigma_high = blur_cfg["sigma_range"]
        params["gaussian_blur"] = {
            "ksize": int(rng.choice(blur_cfg["ksize_choices"])),
            "sigma": float(rng.uniform(sigma_low, sigma_high)),
        }
    else:
        params["gaussian_blur"] = None

    motion_cfg = cfg["motion_blur"]
    if rng.random() < motion_cfg["probability"]:
        params["motion_blur"] = {
            "size": int(rng.choice(motion_cfg["size_choices"])),
        }
    else:
        params["motion_blur"] = None

    params["interpolation"] = str(
        rng.choice(cfg["downscale"]["interpolation_choices"])
    )

    noise_cfg = cfg["gaussian_noise"]
    if rng.random() < noise_cfg["probability"]:
        std_low, std_high = noise_cfg["std_range"]
        params["gaussian_noise"] = {
            "std": float(rng.uniform(std_low, std_high)),
            # The noise field is a full array and cannot be written to the
            # log, so it gets its own seed and is regenerated from it.
            "seed": int(rng.integers(0, 2 ** 32)),
        }
    else:
        params["gaussian_noise"] = None

    jpeg_cfg = cfg["jpeg"]
    if rng.random() < jpeg_cfg["probability"]:
        quality_low, quality_high = jpeg_cfg["quality_range"]
        params["jpeg"] = {
            "quality": int(rng.integers(quality_low, quality_high)),
        }
    else:
        params["jpeg"] = None

    return params

def apply_degradation(hr_image, params, scale_factor=DEGRADATION_SCALE_FACTOR):
    """
    Applies an already sampled degradation to an HR image.

    Fully deterministic: the same params always yield the same LR image, and
    nothing is sampled here. This is what makes a degradation log replayable
    even after the sampling distributions in the config have been retuned.

    Parameters:
        hr_image (np.ndarray): HR image in BGR uint8.
        params (dict): Decisions from sample_degradation_params.
        scale_factor (float): Downscaling factor applied to reach LR size.

    Returns:
        np.ndarray: LR image in BGR uint8.
    """

    image = hr_image

    blur = params.get("gaussian_blur")
    if blur is not None:
        ksize = int(blur["ksize"])
        image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=float(blur["sigma"]))

    motion = params.get("motion_blur")
    if motion is not None:
        size = int(motion["size"])
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        image = cv2.filter2D(image, -1, kernel_motion_blur)

    interp_name = params["interpolation"]
    if interp_name not in INTERPOLATION_NAME_TO_CODE:
        raise ValueError(
            f"Unknown interpolation in degradation params: {interp_name}"
        )

    h, w = image.shape[:2]
    lr_image = cv2.resize(
        image,
        (int(w * scale_factor), int(h * scale_factor)),
        interpolation=INTERPOLATION_NAME_TO_CODE[interp_name],
    )

    noise = params.get("gaussian_noise")
    if noise is not None:
        noise_rng = np.random.default_rng(int(noise["seed"]))
        noise_field = noise_rng.normal(
            0, float(noise["std"]), lr_image.shape
        ).astype(np.float32)
        lr_image = np.clip(
            lr_image.astype(np.float32) + noise_field, 0, 255
        ).astype(np.uint8)

    jpeg = params.get("jpeg")
    if jpeg is not None:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg["quality"])]
        ok, encimg = cv2.imencode('.jpeg', lr_image, encode_param)
        if not ok:
            raise RuntimeError("JPEG encoding failed while degrading image.")
        lr_image = cv2.imdecode(encimg, 1)

    return lr_image

def degrade_image(
        hr_image,
        scale_factor=DEGRADATION_SCALE_FACTOR,
        rng=None,
        config=None,
        params=None):
    """
    Applies a combination of realistic degradations to an HR image to generate an LR image.

    Parameters:
        hr_image (np.ndarray): HR image in BGR uint8.
        scale_factor (float): Downscaling factor applied to reach LR size.
        rng (np.random.Generator): Generator used to sample the degradation.
            Ignored when params is given. Defaults to an unseeded generator,
            which is NOT reproducible.
        config (dict): Degradation hyperparameters. Ignored when params is given.
        params (dict): Degradation read back from a degradation log. When
            given, it is replayed exactly and nothing is sampled.

    Returns:
        tuple: (lr_image, params), where params is JSON-serialisable and is
            what must be stored to regenerate this exact LR image.
    """

    if params is None:
        rng = rng if rng is not None else np.random.default_rng()
        params = sample_degradation_params(rng, config=config)

    lr_image = apply_degradation(hr_image, params, scale_factor=scale_factor)

    return lr_image, params

def load_degradation_log(path):
    """
    Reads a degradation log from disk.

    Parameters:
        path (str): Path to the JSON log.

    Returns:
        dict: The log, with the per-image degradations under "images".
    """

    with open(path, 'r', encoding="utf-8") as f:
        return json.load(f)

def save_degradation_log(path, log):
    """
    Writes a degradation log as indented JSON so that it diffs cleanly in git.

    Parameters:
        path (str): Path to the JSON log.
        log (dict): Log to persist.
    """

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, 'w', encoding="utf-8") as f:
        json.dump(log, f, indent=2, sort_keys=True)
        f.write("\n")

def init_degradation_log(
        path,
        master_seed=DEFAULT_DEGRADATION_SEED,
        scale_factor=DEGRADATION_SCALE_FACTOR,
        config=None,
        metadata=None):
    """
    Loads the existing degradation log or starts a new one.

    Refuses to append to a log written with a different seed, config or scale
    factor. The dataset is built video by video, so without this check half of
    it could end up degraded under one set of hyperparameters and half under
    another, and no single configuration would describe it. When the settings
    are meant to change, delete the LR images and the log and start a cycle.

    Parameters:
        path (str): Path to the JSON log.
        master_seed (int): Seed of the whole LR dataset.
        scale_factor (float): Downscaling factor applied to reach LR size.
        config (dict): Degradation hyperparameters. Defaults to
            DEFAULT_DEGRADATION_CONFIG.
        metadata (dict): Free-form extraction settings worth recording, such
            as the frame interval or the class label of each video folder.

    Returns:
        dict: Log ready to receive per-image degradations.
    """

    cfg = config if config is not None else DEFAULT_DEGRADATION_CONFIG

    if os.path.exists(path):
        log = load_degradation_log(path)

        if log.get("images"):
            mismatches = []
            # Checked first: a log of another version may not even mean what
            # the fields below are read as, so comparing them would report
            # the wrong reason.
            if log.get("version") != DEGRADATION_LOG_VERSION:
                mismatches.append(
                    f"version {log.get('version')} != {DEGRADATION_LOG_VERSION}"
                )
            if log.get("master_seed") != master_seed:
                mismatches.append(
                    f"master_seed {log.get('master_seed')} != {master_seed}"
                )
            if log.get("scale_factor") != scale_factor:
                mismatches.append(
                    f"scale_factor {log.get('scale_factor')} != {scale_factor}"
                )
            if log.get("config") != cfg:
                mismatches.append("config differs")

            if mismatches:
                raise ValueError(
                    "Refusing to extend the existing degradation log at "
                    f"{path}: {'; '.join(mismatches)}. Delete the LR images "
                    "and this log before generating a dataset with new "
                    "degradation settings."
                )

        log.setdefault("images", {})
        if metadata is not None:
            log.setdefault("metadata", {}).update(metadata)

        return log

    return {
        "version": DEGRADATION_LOG_VERSION,
        "master_seed": master_seed,
        "scale_factor": scale_factor,
        "config": cfg,
        "metadata": metadata if metadata is not None else {},
        "images": {},
    }

@dataclass
class VideoExtractionStats:
    """
    What one video contributed to the dataset.

    Returned instead of printed so that the caller decides whether to report
    it, and so the numbers can be asserted in a test.

    Attributes
    ----------
    images_saved : int
        Pairs written by this call.
    images_in_directory : int
        Pairs in the output directory afterwards, including earlier runs.
    hr_frame_size : tuple of int or None
        ``(width, height)`` of the square HR crop, None when nothing was
        saved.
    """

    video_path: str
    total_frames: int
    images_saved: int
    images_in_directory: int
    source_frame_size: tuple
    hr_frame_size: tuple
    lr_frame_size: tuple
    skipped_blank: int = 0

def format_extraction_report(stats):
    """Render a VideoExtractionStats as one line.

    One video per line keeps the whole build readable inside a notebook
    cell. The frame sizes are constant across a folder, so they are
    reported once per folder instead of once per video.
    """

    name = os.path.basename(stats.video_path)
    blank = f" ({stats.skipped_blank} blank)" if stats.skipped_blank else ""

    return (
        f"    {name:<28} {stats.total_frames:>6} frames -> "
        f"{stats.images_saved:>3} pairs{blank}"
    )

def _validate_extraction_args(
        video_path,
        output_name,
        class_label,
        frame_interval,
        scale_factor):
    """Reject argument combinations that cannot produce a valid dataset."""

    if not video_path or not isinstance(video_path, str):
        raise ValueError("video_path must be a non-empty string.")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not isinstance(output_name, str) or not output_name:
        raise ValueError("output_name must be a non-empty string.")
    if not isinstance(class_label, int) or class_label < 0:
        raise ValueError("class_label must be a non-negative integer.")
    if not isinstance(frame_interval, int) or frame_interval < 0:
        raise ValueError("frame_interval must be a non-negative integer.")
    if not isinstance(scale_factor, (int, float)) or scale_factor <= 0:
        raise ValueError("scale_factor must be a positive number.")

def _resolve_degradation_settings(
        replay_log, master_seed, degradation_config, scale_factor):
    """
    Decide which degradation settings apply to this call.

    When replaying, the log on disk is the source of truth, so its seed and
    config win over the arguments and a contradicting scale factor is an
    error rather than something to reconcile silently.

    Returns
    -------
    tuple
        ``(master_seed, degradation_config)`` to use.
    """

    if replay_log is None:
        return master_seed, degradation_config

    if replay_log.get("scale_factor") != scale_factor:
        raise ValueError(
            f"scale_factor {scale_factor} does not match the replayed "
            f"degradation log ({replay_log.get('scale_factor')})."
        )

    return (
        replay_log.get("master_seed", master_seed),
        replay_log.get("config", degradation_config),
    )

def _next_image_index(hr_dir, output_name):
    """
    Find the index the next image of this video should take.

    Numbering continues from what is already on disk so that several videos
    of the same defect type accumulate into one sequence instead of
    overwriting each other.
    """

    if not os.path.isdir(hr_dir):
        return 0

    indices = []
    for name in os.listdir(hr_dir):
        if not (name.startswith(output_name) and name.endswith(".png")):
            continue
        suffix = name[len(output_name):-len(".png")]
        if suffix.isdigit():
            indices.append(int(suffix))

    return max(indices) + 1 if indices else 0

def _read_class_map(path):
    """Read the { HR basename: class id } mapping, or start an empty one."""

    if not os.path.exists(path):
        return {}

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Warning: could not read class labels map, starting empty: {e}")
        return {}

def _write_class_map(path, class_map):
    """Persist the { HR basename: class id } mapping."""

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    try:
        with open(path, "wb") as f:
            pickle.dump(class_map, f)
    except Exception as e:
        print(f"Warning: failed to save class labels map: {e}")

def _degrade_frame(
        hr_frame,
        basename,
        scale_factor,
        master_seed,
        degradation_config,
        replay_log):
    """
    Produce the LR counterpart of one HR crop.

    The generator is derived from the basename rather than consumed in
    processing order, so an image keeps its degradation no matter how many
    frames were produced before it.

    Returns
    -------
    tuple
        ``(lr_image, params)``, where params is what must be logged to
        regenerate this exact LR image.
    """

    if replay_log is None:
        return degrade_image(
            hr_frame,
            scale_factor=scale_factor,
            rng=derive_image_rng(basename, master_seed=master_seed),
            config=degradation_config,
        )

    recorded = replay_log.get("images", {}).get(basename)
    if recorded is None:
        raise KeyError(
            f"{basename} is missing from the replayed degradation log."
        )

    return degrade_image(hr_frame, scale_factor=scale_factor, params=recorded)

def _is_blank(image, min_std=BLANK_FRAME_MIN_STD):
    """Report whether a crop carries no usable content.

    Parameters
    ----------
    image : np.ndarray
        BGR crop as written to disk.
    min_std : float
        Grayscale standard deviation below which the crop counts as blank.

    Returns
    -------
    bool
        True when the crop is flat enough to be a fade frame.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return float(gray.std()) < min_std

def _iter_selected_frames(cap, frame_interval):
    """Yield the frames that land on the sampling interval."""

    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_interval == 0 or index % frame_interval == 0:
            yield frame

        index += 1

def create_hr_lr_images_from_video(
        video_path,
        output_name,
        class_label,
        frame_interval=10,
        scale_factor=DEGRADATION_SCALE_FACTOR,
        master_seed=DEFAULT_DEGRADATION_SEED,
        degradation_config=None,
        replay_log=None):
    """
    Turn one video into HR/LR image pairs on disk.

    Frames are sampled at a fixed interval, cropped to the square region
    holding the main object, and written as an HR image plus its degraded LR
    counterpart. Numbering continues from whatever is already in the output
    directory.

    Every degradation is recorded in the degradation log, which is the
    artefact that makes the LR dataset reproducible: it is small, diffable
    and versioned, so the dataset can be rebuilt from the videos on any
    machine.

    The log is written for reproducibility only. No model is allowed to read
    the per-image degradation back: the kernel an image was degraded with is
    not knowable for a real low-resolution capture, so using it at training
    time would be privileged information.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    output_name : str
        Defect type, used as both the subfolder name and the filename prefix.
    class_label : int
        Class id assigned to every image extracted from this video.
    frame_interval : int
        Sample one frame every this many. 0 keeps all of them.
    scale_factor : float
        Downscaling factor applied to reach LR size.
    master_seed : int
        Seed of the LR dataset. Change it to draw a new dataset from the same
        degradation hyperparameters.
    degradation_config : dict, optional
        Degradation hyperparameters. These are what to tune to make the LR
        images more or less degraded. Defaults to DEFAULT_DEGRADATION_CONFIG.
    replay_log : dict, optional
        Degradation log read with ``load_degradation_log``. When given, the
        recorded degradations are replayed instead of sampled, and the seed
        and config are taken from it.

    Returns
    -------
    VideoExtractionStats
        What this video contributed.
    """

    _validate_extraction_args(
        video_path, output_name, class_label, frame_interval, scale_factor,
    )
    master_seed, degradation_config = _resolve_degradation_settings(
        replay_log, master_seed, degradation_config, scale_factor
    )

    hr_dir = os.path.join(HR_ROOT, output_name)
    lr_dir = os.path.join(LR_ROOT, output_name)
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)

    class_map = _read_class_map(CLASS_LABELS_PATH)
    degradation_log = init_degradation_log(
        DEGRADATION_LOG_PATH,
        master_seed=master_seed,
        scale_factor=scale_factor,
        config=degradation_config,
        metadata={
            output_name: {
                "frame_interval": frame_interval,
                "class_label": class_label,
            }
        },
    )

    cap = cv2.VideoCapture(video_path)
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        source_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        start_index = _next_image_index(hr_dir, output_name)
        next_index = start_index
        hr_size, lr_size = None, None
        skipped_blank = 0

        for frame in _iter_selected_frames(cap, frame_interval):
            hr_crop = smart_square_crop(frame)

            # Fades at the start and end of a recording yield frames with no
            # object at all. They cannot be degraded, so the pair ends up
            # identical and reports a PSNR of 80 dB or more, which is not a
            # reconstruction result but a black square scoring against itself.
            if _is_blank(hr_crop):
                skipped_blank += 1
                continue

            basename = f"{output_name}{next_index}.png"
            cv2.imwrite(os.path.join(hr_dir, basename), hr_crop)
            class_map[basename] = class_label

            lr_image, params = _degrade_frame(
                hr_crop, basename, scale_factor,
                master_seed, degradation_config, replay_log,
            )
            cv2.imwrite(os.path.join(lr_dir, basename), lr_image)
            degradation_log["images"][basename] = params

            # Measured on the crop, so the reported size is the one written
            # to disk. Stays None when the video yields no frame.
            hr_size = (hr_crop.shape[1], hr_crop.shape[0])
            lr_size = (lr_image.shape[1], lr_image.shape[0])

            next_index += 1
    finally:
        cap.release()

    _write_class_map(CLASS_LABELS_PATH, class_map)

    # Not allowed to fail silently: without the log the LR images on disk
    # cannot be regenerated.
    save_degradation_log(DEGRADATION_LOG_PATH, degradation_log)

    return VideoExtractionStats(
        video_path=video_path,
        total_frames=total_frames,
        images_saved=next_index - start_index,
        images_in_directory=next_index,
        source_frame_size=source_size,
        hr_frame_size=hr_size,
        lr_frame_size=lr_size,
        skipped_blank=skipped_blank,
    )

def _defect_type_from_filename(video_file):
    """
    Derive the defect type from a video filename.

    ``low_z_offset_3.mp4`` becomes ``low_z_offset``, so every video of the
    same defect feeds one image sequence. A name without a trailing index is
    used as is.
    """

    name = os.path.splitext(video_file)[0]
    parts = name.rsplit("_", 1)

    return parts[0] if len(parts) == 2 and parts[1].isdigit() else name

def _list_videos(directory):
    """List the mp4 files of a folder in a deterministic order."""

    return sorted(
        f for f in os.listdir(directory) if f.lower().endswith(".mp4")
    )

def build_dataset_from_videos(
        videos_root=VIDEOS_ROOT,
        max_videos_per_folder=None,
        frame_interval_per_folder=None,
        class_id_per_folder=None,
        scale_factor=DEGRADATION_SCALE_FACTOR,
        master_seed=DEFAULT_DEGRADATION_SEED,
        degradation_config=None,
        replay_from_log=REPLAY_DEGRADATION_FROM_LOG,
        verbose=True):
    """
    Build the whole HR/LR dataset by walking the video folders.

    One subfolder per defect class, each contributing up to a configured
    number of videos at its own frame interval. Videos are processed in
    sorted order so that a rerun assigns the same image numbers.

    A video that fails does not abort the run, but the failures are returned
    and reported at the end instead of being left in the scrollback: a
    half-built dataset that looks complete is worse than a loud error.

    Parameters
    ----------
    videos_root : str
        Root holding one subfolder per defect class.
    max_videos_per_folder : dict, optional
        Videos to take from each subfolder. Defaults to the value in
        ``constants``, which also defines which subfolders are used.
    frame_interval_per_folder : dict, optional
        Frame sampling interval per subfolder.
    class_id_per_folder : dict, optional
        Class id per subfolder.
    replay_from_log : bool
        Rebuild the exact dataset recorded in the degradation log instead of
        sampling a new one.
    verbose : bool
        Print the per-video analysis block.

    Returns
    -------
    dict
        ``{'processed': {folder: count}, 'images': {folder: count},
        'failures': [(video_path, message)]}``.
    """

    max_videos_per_folder = (
        max_videos_per_folder
        if max_videos_per_folder is not None
        else VIDEO_MAX_VIDEOS_PER_FOLDER
    )
    frame_interval_per_folder = (
        frame_interval_per_folder
        if frame_interval_per_folder is not None
        else VIDEO_FRAME_INTERVAL_PER_FOLDER
    )
    class_id_per_folder = (
        class_id_per_folder
        if class_id_per_folder is not None
        else VIDEO_CLASS_ID_PER_FOLDER
    )

    replay_log = (
        load_degradation_log(DEGRADATION_LOG_PATH) if replay_from_log else None
    )

    processed, images, failures = {}, {}, []
    mode = "replaying the log" if replay_from_log else f"seed {master_seed}"
    started = time.perf_counter()
    print(f"\n{'=' * 64}\n Dataset build | {mode}, scale {scale_factor}\n"
          f"{'=' * 64}", flush=True)

    for folder, max_videos in max_videos_per_folder.items():
        directory = os.path.join(videos_root, folder)
        if not os.path.isdir(directory):
            print(f"Skipping missing folder: {directory}")
            continue

        class_id = class_id_per_folder.get(folder)
        if class_id is None:
            print(f"Skipping {folder}: no class id defined for it.")
            continue

        frame_interval = frame_interval_per_folder.get(folder, 40)
        count, saved, blank = 0, 0, 0
        frame_sizes = set()

        candidates = _list_videos(directory)[:max_videos]
        print(
            f"\n  {folder}  (class {class_id}, {len(candidates)} videos, "
            f"1 frame every {frame_interval})",
            flush=True,
        )

        for video_file in candidates:
            if count >= max_videos:
                break

            video_path = os.path.join(directory, video_file)
            try:
                stats = create_hr_lr_images_from_video(
                    video_path,
                    output_name=_defect_type_from_filename(video_file),
                    class_label=class_id,
                    frame_interval=frame_interval,
                    scale_factor=scale_factor,
                    master_seed=master_seed,
                    degradation_config=degradation_config,
                    replay_log=replay_log,
                )
            except Exception as e:
                failures.append((video_path, str(e)))
                continue

            if verbose:
                print(format_extraction_report(stats), flush=True)

            if stats.hr_frame_size:
                frame_sizes.add((stats.hr_frame_size, stats.lr_frame_size))
            count += 1
            saved += stats.images_saved
            blank += stats.skipped_blank

        for hr_size, lr_size in sorted(frame_sizes):
            print(
                f"    -> HR {hr_size[0]}x{hr_size[1]}  "
                f"LR {lr_size[0]}x{lr_size[1]}",
                flush=True,
            )
        summary = f"    -> {saved} pairs from {count} videos"
        if blank:
            summary += f", {blank} blank frames discarded"
        print(summary, flush=True)

        processed[folder] = count
        images[folder] = saved

    print(f"\n  TOTAL     {sum(images.values())} HR/LR pairs from "
          f"{sum(processed.values())} videos", flush=True)
    if failures:
        print(f"  FAILED    {len(failures)} video(s):", flush=True)
        for video_path, message in failures:
            print(f"    {os.path.basename(video_path)}: {message}", flush=True)
    print(f"  done in {format_duration(time.perf_counter() - started)}\n",
          flush=True)

    return {"processed": processed, "images": images, "failures": failures}


