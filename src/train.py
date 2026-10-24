import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from config import Config
from loss import AsymmetricLoss
from dataset import KneeMRIDataset
from models.full_model import RSNAKneeModel

def create_mock_dataframe(cfg: Config, num_samples: int = 16) -> pd.DataFrame:
    """It Generates synthetic dataset if real CSV is missing, ensuring immediate runnability."""
    os.makedirs(os.path.dirname(cfg.TRAIN_CSV), exist_ok=True)
    data = {"study_id": [f"study_{i:04d}" for i in range(num_samples)]}
    for col in cfg.TARGETS:
        data[col] = np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2])
    df = pd.DataFrame(data)
    df.to_csv(cfg.TRAIN_CSV, index=False)
    return df

def calculate_macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) > 1:
            scores.append(roc_auc_score(y_true[:, c], y_pred[:, c]))
    return float(np.mean(scores)) if scores else 0.5

def main():
    cfg = Config()
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    print(f" Initializing RSNA Knee Abnormality Detection on Device: {cfg.DEVICE}")

    # Load or initialize CSV metadata
    if not os.path.exists(cfg.TRAIN_CSV):
        print(" Warning: train_labels.csv not found. Generating synthetic mock data for verification...")
        df = create_mock_dataframe(cfg)
    else:
        df = pd.read_csv(cfg.TRAIN_CSV)

    # 80/20 Train-Val Split
    val_size = max(2, int(len(df) * 0.2))
    train_df = df.iloc[:-val_size].reset_index(drop=True)
    val_df = df.iloc[-val_size:].reset_index(drop=True)

    train_ds = KneeMRIDataset(train_df, cfg.RAW_DATA_DIR, cfg.TARGETS, cfg.TARGET_SLICES, cfg.IMG_SIZE)
    val_ds = KneeMRIDataset(val_df, cfg.RAW_DATA_DIR, cfg.TARGETS, cfg.TARGET_SLICES, cfg.IMG_SIZE, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)

    # Initialize Network, Loss, Optimizer, Scaler
    model = RSNAKneeModel(
        backbone_name=cfg.BACKBONE,
        num_classes=len(cfg.TARGETS),
        embedding_dim=cfg.EMBEDDING_DIM,
        dropout=cfg.DROPOUT
    ).to(cfg.DEVICE)

    criterion = AsymmetricLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.DEVICE == 'cuda'))

    best_auc = 0.0

    for epoch in range(1, cfg.EPOCHS + 1):
        # 1. Train Loop
        model.train()
        train_loss = 0.0

        for sag, cor, axi, targets in train_loader:
            sag, cor, axi = sag.to(cfg.DEVICE), cor.to(cfg.DEVICE), axi.to(cfg.DEVICE)
            targets = targets.to(cfg.DEVICE)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(cfg.DEVICE == 'cuda')):
                logits = model(sag, cor, axi)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        # 2. Validation Loop
        model.eval()
        val_preds, val_targets = [], []

        with torch.no_grad():
            for sag, cor, axi, targets in val_loader:
                sag, cor, axi = sag.to(cfg.DEVICE), cor.to(cfg.DEVICE), axi.to(cfg.DEVICE)
                with torch.amp.autocast('cuda', enabled=(cfg.DEVICE == 'cuda')):
                    logits = model(sag, cor, axi)
                    probs = torch.sigmoid(logits)

                val_preds.append(probs.cpu().numpy())
                val_targets.append(targets.numpy())

        y_true = np.vstack(val_targets)
        y_pred = np.vstack(val_preds)
        val_auc = calculate_macro_auc(y_true, y_pred)
        avg_loss = train_loss / len(train_loader)

        print(f"Epoch [{epoch:02d}/{cfg.EPOCHS:02d}] - Loss: {avg_loss:.4f} - Val Macro ROC-AUC: {val_auc:.4f}")

        # Save Best Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "best_rsna_knee_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"   Checkpoint saved to: {ckpt_path}")

    print(f"\n Training complete! Best Macro ROC-AUC: {best_auc:.4f}")

if __name__ == "__main__":
    main()