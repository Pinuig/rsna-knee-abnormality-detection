import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple

class KneeMRIDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        dicom_root: str,
        target_cols: List[str],
        target_slices: int = 16,
        img_size: int = 256,
        is_train: bool = True
    ):
        self.df = df
        self.dicom_root = dicom_root
        self.target_cols = target_cols
        self.target_slices = target_slices
        self.img_size = img_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def _sample_and_stack_25d(self, volume: np.ndarray) -> torch.Tensor:
        """
        Input volume: (Depth, H, W)
        Output tensor: (target_slices, 3, H, W)
        """
        depth, h, w = volume.shape
        idx = np.linspace(0, depth - 1, self.target_slices).astype(int)
        sampled = volume[idx]

        stacked_25d = np.zeros((self.target_slices, 3, h, w), dtype=np.float32)
        for i in range(self.target_slices):
            p = max(0, i - 1)
            n = min(self.target_slices - 1, i + 1)
            stacked_25d[i, 0] = sampled[p]
            stacked_25d[i, 1] = sampled[i]
            stacked_25d[i, 2] = sampled[n]

        return torch.from_numpy(stacked_25d)

    def _load_plane_volume(self, study_id: str, plane: str) -> np.ndarray:
        """Loads plane volume. Generates mock array if DICOM path does not exist."""
        plane_dir = os.path.join(self.dicom_root, str(study_id), plane)
        

        if os.path.exists(plane_dir) and os.listdir(plane_dir):
            import pydicom
            files = sorted(os.listdir(plane_dir))
            slices = []
            for f in files:
                dcm = pydicom.dcmread(os.path.join(plane_dir, f))
                arr = dcm.pixel_array.astype(np.float32)
                # Rescale and normalize [0, 1]
                arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-6)
                arr = cv2.resize(arr, (self.img_size, self.img_size))
                slices.append(arr)
            return np.stack(slices, axis=0)

        return np.random.rand(24, self.img_size, self.img_size).astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        study_id = row["study_id"]

        sag = self._sample_and_stack_25d(self._load_plane_volume(study_id, "sagittal"))
        cor = self._sample_and_stack_25d(self._load_plane_volume(study_id, "coronal"))
        axi = self._sample_and_stack_25d(self._load_plane_volume(study_id, "axial"))

        labels = torch.tensor(row[self.target_cols].values.astype(np.float32), dtype=torch.float32)
        return sag, cor, axi, labels