import cv2
import numpy as np

def smart_square_crop(img):
    """
    Crops the image to a square (width x width) region containing the main object.
    The crop is centered on the largest contour (assumed to be the object).
    If no contour is found, crops the center square.
    """
    
    h, w = img.shape[:2]
    crop_size = min(w, h)
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

def degrade_image(hr_image, scale_factor=0.5):
    """
    Applies a combination of realistic degradations to an HR image to generate an LR image.
    Returns (lr_image, interp_name) where interp_name is the OpenCV interpolation
    method name used, so later we can upscale with the same method.
    """
    
    if np.random.rand() < 0.7:
        ksize = np.random.choice([3, 5, 7])
        sigma = np.random.uniform(0.8, 2.0)
        hr_image = cv2.GaussianBlur(hr_image, (ksize, ksize), sigmaX=sigma)
    
    if np.random.rand() < 0.3:
        size = np.random.choice([5, 7, 9])
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        hr_image = cv2.filter2D(hr_image, -1, kernel_motion_blur)
    
    interp_code_to_name = {
        cv2.INTER_LINEAR: "INTER_LINEAR",
        cv2.INTER_CUBIC: "INTER_CUBIC",
        cv2.INTER_AREA: "INTER_AREA",
        cv2.INTER_LANCZOS4: "INTER_LANCZOS4",
    }
    interp_method = np.random.choice([
        cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4
    ])
    interp_name = interp_code_to_name.get(interp_method, str(interp_method))
    h, w = hr_image.shape[:2]
    lr_image = cv2.resize(
        hr_image,
        (int(w*scale_factor), int(h*scale_factor)),
        interpolation=interp_method
    )
    
    if np.random.rand() < 0.7:
        noise_std = np.random.uniform(2, 10)
        noise = np.random.normal(0, noise_std, lr_image.shape).astype(np.float32)
        lr_image = np.clip(
            lr_image.astype(np.float32) + noise, 0, 255
        ).astype(np.uint8)
    
    if np.random.rand() < 0.7:
        encode_param = [
            int(cv2.IMWRITE_JPEG_QUALITY), np.random.randint(20, 60)
        ]
        _, encimg = cv2.imencode('.jpeg', lr_image, encode_param)
        lr_image = cv2.imdecode(encimg, 1)
        
    return lr_image, interp_name