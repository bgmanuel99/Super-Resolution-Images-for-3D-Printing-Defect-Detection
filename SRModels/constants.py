import os
import cv2

PATCH_SIZE = 24
STRIDE = 12
INTERPOLATION = cv2.INTER_CUBIC
SCALE_FACTOR = 2
RANDOM_SEED = 42
HR_ROOT = os.path.abspath(os.path.join(os.getcwd(), "../data/images/HR"))
LR_ROOT = os.path.abspath(os.path.join(os.getcwd(), "../data/images/LR"))