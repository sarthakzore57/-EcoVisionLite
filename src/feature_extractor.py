"""
EcoVision Lite - Enhanced Feature Extraction Module (v2).

Key Accuracy Improvements:
1. High-Fidelity 96x96 Resolution: Preserves fine structural edges, text, and rims.
2. Dual-Zone Color Profiling:
   - Global HSV Histogram (12 bins per channel = 36 dims)
   - Center 50% ROI HSV Histogram (12 bins per channel = 36 dims) -> isolates foreground object from white/studio background!
3. HOG Descriptors:
   - 9 orientations over (12, 12) cells with (2, 2) blocks and L2-Hys normalization.
   - Dimension: 7 x 7 x 4 x 9 = 1,764 spatial gradient features.
4. Total Feature Dimension: 1,764 (HOG) + 72 (Dual-Zone Color) = 1,836 dimensions.
5. Memory Footprint: Only ~90 MB for all 12,259 images.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union
from skimage.feature import hog

class WasteFeatureExtractor:
    """
    OpenCV + High-Fidelity HOG & Dual-Zone Color Extractor.
    """
    def __init__(
        self,
        image_size: Tuple[int, int] = (96, 96),
        orientations: int = 9,
        pixels_per_cell: Tuple[int, int] = (12, 12),
        cells_per_block: Tuple[int, int] = (2, 2),
        block_norm: str = "L2-Hys",
        include_color_hist: bool = True,
        color_bins: int = 12
    ):
        self.image_size = image_size
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        self.include_color_hist = include_color_hist
        self.color_bins = color_bins

        # Calculate exact HOG dimension
        bx = (self.image_size[0] // self.pixels_per_cell[0]) - self.cells_per_block[0] + 1
        by = (self.image_size[1] // self.pixels_per_cell[1]) - self.cells_per_block[1] + 1
        self.hog_dim = bx * by * (self.cells_per_block[0] * self.cells_per_block[1] * self.orientations)
        
        # Dual-Zone color dimension (Global + Center 50% ROI)
        self.color_dim = (2 * 3 * self.color_bins) if self.include_color_hist else 0
        self.total_dim = self.hog_dim + self.color_dim

    def preprocess_image(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resize, apply Gaussian smoothing, and contrast-enhance grayscale image.
        """
        # Resize to standard canonical window
        resized_bgr = cv2.resize(img_bgr, self.image_size, interpolation=cv2.INTER_AREA)

        # Mild blur to prevent sensor noise from dominating fine gradients
        blurred = cv2.GaussianBlur(resized_bgr, (3, 3), 0)

        # Grayscale conversion
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        # Contrast Limited Adaptive Histogram Equalization (CLAHE) for illumination invariance
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)

        return enhanced_gray, resized_bgr

    def extract_features(
        self, 
        img_bgr: np.ndarray, 
        return_visualization: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Extract 1D feature vector from a single BGR image.
        Optionally returns normalized HOG visual map for educational UI display.
        """
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Input image is empty or None")

        enhanced_gray, resized_bgr = self.preprocess_image(img_bgr)

        # Compute HOG descriptors
        if return_visualization:
            hog_features, hog_vis = hog(
                enhanced_gray,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm=self.block_norm,
                visualize=True
            )
            # Normalize visualization image to 0-255 uint8 for easy UI rendering
            vis_norm = cv2.normalize(hog_vis, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            hog_features = hog(
                enhanced_gray,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm=self.block_norm,
                visualize=False
            )
            vis_norm = None

        hog_vector = hog_features.astype(np.float32)

        if not self.include_color_hist:
            if return_visualization:
                return hog_vector, vis_norm
            return hog_vector

        # Compute Dual-Zone HSV color histogram:
        # Zone 1: Global image color
        hsv = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [self.color_bins], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [self.color_bins], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [self.color_bins], [0, 256])
        global_color = np.concatenate([h_hist, s_hist, v_hist]).flatten().astype(np.float32)
        norm_g = np.linalg.norm(global_color)
        if norm_g > 1e-6:
            global_color /= norm_g

        # Zone 2: Center 50% ROI color (isolates foreground waste object from backdrop)
        h, w = hsv.shape[:2]
        center_hsv = hsv[h//4: 3*h//4, w//4: 3*w//4]
        h_c = cv2.calcHist([center_hsv], [0], None, [self.color_bins], [0, 180])
        s_c = cv2.calcHist([center_hsv], [1], None, [self.color_bins], [0, 256])
        v_c = cv2.calcHist([center_hsv], [2], None, [self.color_bins], [0, 256])
        center_color = np.concatenate([h_c, s_c, v_c]).flatten().astype(np.float32)
        norm_c = np.linalg.norm(center_color)
        if norm_c > 1e-6:
            center_color /= norm_c

        dual_color = np.concatenate([global_color, center_color])

        combined_vector = np.concatenate([hog_vector, dual_color]).astype(np.float32)

        if return_visualization:
            return combined_vector, vis_norm
        return combined_vector

    def extract_from_path(
        self, 
        image_path: str | Path, 
        return_visualization: bool = False
    ) -> Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """Load image from file path and extract features."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            return self.extract_features(img, return_visualization=return_visualization)
        except Exception:
            return None

    def get_config(self) -> dict:
        """Return extractor configuration dictionary."""
        return {
            "image_size": list(self.image_size),
            "orientations": self.orientations,
            "pixels_per_cell": list(self.pixels_per_cell),
            "cells_per_block": list(self.cells_per_block),
            "block_norm": self.block_norm,
            "include_color_hist": self.include_color_hist,
            "color_bins": self.color_bins,
            "hog_dim": self.hog_dim,
            "color_dim": self.color_dim,
            "total_dim": self.total_dim
        }
