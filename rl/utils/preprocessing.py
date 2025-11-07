from typing import Tuple

import cv2
import numpy as np


def preprocess_frame(frame: np.ndarray, output_size: Tuple[int, int] = (84, 84)) -> np.ndarray:
    """
    Convert an RGB frame (H, W, C) to grayscale 84x84 uint8.
    """
    if frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame
    resized = cv2.resize(gray, output_size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)


class FrameStack:
    """Numpy-based frame stacker to return (num_frames, H, W)."""

    def __init__(self, num_frames: int = 4, height: int = 84, width: int = 84):
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self._frames = np.zeros((num_frames, height, width), dtype=np.uint8)

    def reset(self, initial_frame: np.ndarray) -> np.ndarray:
        processed = preprocess_frame(initial_frame, (self.width, self.height))
        for i in range(self.num_frames):
            self._frames[i] = processed
        return self._frames.copy()

    def append(self, frame: np.ndarray) -> np.ndarray:
        processed = preprocess_frame(frame, (self.width, self.height))
        self._frames[:-1] = self._frames[1:]
        self._frames[-1] = processed
        return self._frames.copy()



