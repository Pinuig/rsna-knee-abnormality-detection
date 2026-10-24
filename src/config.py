import os
import torch
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:

    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_DIR: str = os.path.join(BASE_DIR, "data", "raw")
    PROCESSED_DATA_DIR: str = os.path.join(BASE_DIR, "data", "processed")
    TRAIN_CSV: str = os.path.join(BASE_DIR, "data", "train_labels.csv")
    CHECKPOINT_DIR: str = os.path.join(BASE_DIR, "checkpoints")

    IMG_SIZE: int = 256
    TARGET_SLICES: int = 16   # Uniform slice depth per plane
    CHANNELS_PER_SLICE: int = 3  # 2.5D slice stacking (i-1, i, i+1)

    BACKBONE: str = "convnext_tiny"  # Options: resnet34, efficientnet_b0, convnext_tiny
    EMBEDDING_DIM: int = 384
    DROPOUT: float = 0.3

    BATCH_SIZE: int = 4
    NUM_WORKERS: int = 2
    LR: float = 2e-4
    WEIGHT_DECAY: float = 1e-4
    EPOCHS: int = 10
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    TARGETS: List[str] = field(default_factory=lambda: [
        "acl_tear",
        "pcl_tear",
        "mcl_tear",
        "lcl_tear",
        "medial_meniscal_tear",
        "lateral_meniscal_tear",
        "patellofemoral_oa",
        "tibiofemoral_oa",
        "joint_effusion",
        "bone_contusion",
        "fracture",
        "bakers_cyst"
    ])