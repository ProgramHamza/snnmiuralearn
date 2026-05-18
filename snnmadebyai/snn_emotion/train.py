from __future__ import annotations

import argparse
import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional

from config import BATCH_SIZE, BEST_CHECKPOINT_PATH, CHECKPOINT_DIR, EPOCHS, LR, RANDOM_SEED
from data.case_dataset import CASEDataset, build_emotion_target
from models.snn_classifier import ConvSNN


def _prepare_batch(batch, device: torch.device):
    spikes, valence, arousal = batch
    spikes = spikes.to(device)
    valence = valence.to(device)
    arousal = arousal.to(device)
    spikes = spikes.permute(1, 0, 2, 3).contiguous()
    affect_target = torch.stack([valence, arousal], dim=1)
    emotion_target = torch.stack([build_emotion_target(v, a) for v, a in zip(valence, arousal)], dim=0).to(device)
    return spikes, affect_target, emotion_target


def _evaluate(model, loader, device):
    model.eval()
    mse = nn.MSELoss()
    total_loss = 0.0
    batch_count = 0
    with torch.no_grad():
        for batch in loader:
            spikes, affect_target, emotion_target = _prepare_batch(batch, device)
            functional.reset_net(model)
            affect_pred, emotion_pred = model(spikes)
            loss = mse(affect_pred, affect_target) + 0.01 * mse(emotion_pred, emotion_target)
            total_loss += float(loss.item())
            batch_count += 1
    return total_loss / max(1, batch_count)


def main():
    parser = argparse.ArgumentParser(description="Train the CASE convolutional SNN emotion classifier.")
    parser.add_argument("--data-root", required=True, help="Path to the CASE dataset root.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    device = torch.device(args.device)

    train_dataset = CASEDataset(args.data_root, split="train")
    val_dataset = CASEDataset(args.data_root, split="val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    sample_spikes, _, _ = train_dataset[0]
    input_channels = sample_spikes.shape[1]
    feature_length = sample_spikes.shape[2]

    model = ConvSNN(input_channels=input_channels, feature_length=feature_length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    mse = nn.MSELoss()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            spikes, affect_target, emotion_target = _prepare_batch(batch, device)
            functional.reset_net(model)

            optimizer.zero_grad(set_to_none=True)
            affect_pred, emotion_pred = model(spikes)
            loss = mse(affect_pred, affect_target) + 0.01 * mse(emotion_pred, emotion_target)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            batch_count += 1

        scheduler.step()
        train_loss = running_loss / max(1, batch_count)
        val_loss = _evaluate(model, val_loader, device)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_channels": input_channels,
                    "feature_length": feature_length,
                    "val_loss": best_val_loss,
                    "epoch": epoch,
                },
                BEST_CHECKPOINT_PATH,
            )

    print(json.dumps({"best_checkpoint": str(BEST_CHECKPOINT_PATH), "best_val_loss": best_val_loss}, indent=2))


if __name__ == "__main__":
    main()
