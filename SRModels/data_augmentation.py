import cv2
import numpy as np
import tensorflow as tf

class AdvancedAugmentGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, Y, batch_size=16, shuffle=True, use_mix=True):
        """Advanced paired augmentation generator.

        Parameters
        ----------
        X, Y : np.ndarray
            Input (LR) and target (HR) image tensors in range [0,1]. Shape: (N, H, W, C)
        batch_size : int
            Batch size.
        shuffle : bool
            Whether to shuffle indices each epoch.
        use_mix : bool, default True
            If True applies mixup/cutmix (heavy mode). If False, creates a "light" version
            disabling mixup/cutmix while keeping per-image geometric/color transforms.
        """
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_mix = use_mix  # controls heavy mix strategies
        self.indices = np.arange(len(X))
        self.on_epoch_end()
        
    def _mixup(self, X, Y, alpha=0.2):
        lam = np.random.beta(alpha, alpha)
        idx = np.random.permutation(len(X))
        
        X_mix = lam * X + (1 - lam) * X[idx]
        Y_mix = lam * Y + (1 - lam) * Y[idx]
        
        return X_mix, Y_mix

    def _cutmix(self, X, Y, alpha=1.0):
        lam = np.random.beta(alpha, alpha)
        idx = np.random.permutation(len(X))
        h, w = X.shape[1:3]
        
        cut_rat = np.sqrt(1. - lam)
        cut_w, cut_h = int(w * cut_rat), int(h * cut_rat)
        cx, cy = np.random.randint(w), np.random.randint(h)
        
        x1, y1 = np.clip(cx - cut_w // 2, 0, w), np.clip(cy - cut_h // 2, 0, h)
        x2, y2 = np.clip(cx + cut_w // 2, 0, w), np.clip(cy + cut_h // 2, 0, h)
        
        X_cut, Y_cut = X.copy(), Y.copy()
        
        X_cut[:, y1:y2, x1:x2, :] = X[idx, y1:y2, x1:x2, :]
        Y_cut[:, y1:y2, x1:x2, :] = Y[idx, y1:y2, x1:x2, :]
        
        return X_cut, Y_cut

    def _elastic_transform(self, image, alpha, sigma):
        random_state = np.random.RandomState(None)
        shape = image.shape[:2]
        
        dx = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (17, 17), sigma) * alpha
        dy = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (17, 17), sigma) * alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def _random_perspective(self, image):
        h, w = image.shape[:2]
        margin = 0.1
        
        pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
        pts2 = pts1 + np.random.uniform(-margin*w, margin*w, pts1.shape).astype(np.float32)
        
        M = cv2.getPerspectiveTransform(pts1, pts2)
        
        return cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    def _color_augment(self, image):
        if np.random.rand() < 0.5:
            img = cv2.cvtColor((image*255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            img = img.astype(np.float32)
            img[...,0] = (img[...,0] + np.random.uniform(-10,10)) % 180
            img[...,1] = np.clip(img[...,1] * np.random.uniform(0.8,1.2), 0, 255)
            img[...,2] = np.clip(img[...,2] * np.random.uniform(0.8,1.2), 0, 255)
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
            return img.astype(np.float32)/255.0
        else:
            img = cv2.cvtColor((image*255).astype(np.uint8), cv2.COLOR_RGB2LAB)
            img = img.astype(np.float32)
            img[...,0] = np.clip(img[...,0] + np.random.uniform(-10,10), 0, 255)
            img[...,1] = np.clip(img[...,1] + np.random.uniform(-10,10), 0, 255)
            img[...,2] = np.clip(img[...,2] + np.random.uniform(-10,10), 0, 255)
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_LAB2RGB)
            
            return img.astype(np.float32)/255.0
        
    def _random_rotate(self, image, angle_range=15):
        angle = np.random.uniform(-angle_range, angle_range)
        h, w = image.shape[:2]
        
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    def _random_zoom(self, image, zoom_range=(0.1, 0.3)):
        h, w = image.shape[:2]
        zx, zy = np.random.uniform(*zoom_range, 2)
        
        M = cv2.getRotationMatrix2D((w/2, h/2), 0, zx)
        
        zoomed = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        if zx < 1.0 or zy < 1.0:
            pad_h = int((h - h*zx) // 2)
            pad_w = int((w - w*zy) // 2)
            zoomed = cv2.copyMakeBorder(zoomed, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT)
            zoomed = zoomed[:h, :w]
            
        return zoomed

    def _random_flip(self, image):
        if np.random.rand() < 0.5:
            return np.fliplr(image)
        
        return image

    def _random_shift(self, image, width_shift_range=0.1, height_shift_range=0.1):
        h, w = image.shape[:2]
        
        tx = np.random.uniform(-width_shift_range, width_shift_range) * w
        ty = np.random.uniform(-height_shift_range, height_shift_range) * h
        
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    def _advanced_augment(self, X, Y):
        """Apply advanced augmentations (optionally skipping mix strategies)."""
        X_aug, Y_aug = X.copy(), Y.copy()

        # Heavy mixing strategies (can be disabled for light mode)
        if self.use_mix:
            if np.random.rand() < 0.5:
                X_aug, Y_aug = self._mixup(X_aug, Y_aug)
            else:
                X_aug, Y_aug = self._cutmix(X_aug, Y_aug)

        for i in range(len(X_aug)):
            # Elastic transform
            if np.random.rand() < 0.3:
                X_aug[i] = self._elastic_transform(X_aug[i], alpha=1, sigma=8)
                Y_aug[i] = self._elastic_transform(Y_aug[i], alpha=1, sigma=8)
            # Random perspective
            if np.random.rand() < 0.3:
                X_aug[i] = self._random_perspective(X_aug[i])
                Y_aug[i] = self._random_perspective(Y_aug[i])
            # Color augment
            if np.random.rand() < 0.5:
                X_aug[i] = self._color_augment(X_aug[i])
                Y_aug[i] = self._color_augment(Y_aug[i])
            # Random rotate
            if np.random.rand() < 0.5:
                X_aug[i] = self._random_rotate(X_aug[i])
                Y_aug[i] = self._random_rotate(Y_aug[i])
            # Random zoom
            if np.random.rand() < 0.5:
                X_aug[i] = self._random_zoom(X_aug[i])
                Y_aug[i] = self._random_zoom(Y_aug[i])
            # Random flip
            if np.random.rand() < 0.5:
                X_aug[i] = self._random_flip(X_aug[i])
                Y_aug[i] = self._random_flip(Y_aug[i])
            # Random shift
            if np.random.rand() < 0.5:
                X_aug[i] = self._random_shift(X_aug[i])
                Y_aug[i] = self._random_shift(Y_aug[i])
                
        return X_aug, Y_aug
        
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        inds = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        
        X_batch = self.X[inds]
        Y_batch = self.Y[inds]
        
        X_aug, Y_aug = self._advanced_augment(X_batch, Y_batch)
        
        return X_aug, Y_aug
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)