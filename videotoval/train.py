from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

try:
    from .model import Net
    from .model import save_model
    from .plot_graphs import (
        plot_beta_values,
        plot_gradient_norms,
        plot_learning_rate,
        plot_loss_curves,
        plot_membrane_trace,
        plot_prediction_timeseries,
        plot_scatter,
        plot_spike_rates,
        plot_timeseries,
        plot_total_loss,
    )
    from .training_data_load import VideoFrameDataset, build_dataloaders, build_inference_dataset
except ImportError:
    from model import Net
    from model import save_model
    from plot_graphs import (
        plot_beta_values,
        plot_gradient_norms,
        plot_learning_rate,
        plot_loss_curves,
        plot_membrane_trace,
        plot_prediction_timeseries,
        plot_scatter,
        plot_spike_rates,
        plot_timeseries,
        plot_total_loss,
    )
    from training_data_load import VideoFrameDataset, build_dataloaders, build_inference_dataset


def ccc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    pred_mean = pred.mean()
    target_mean = target.mean()
    covariance = ((pred - pred_mean) * (target - target_mean)).mean()
    pred_var = pred.var(unbiased=False)
    target_var = target.var(unbiased=False)
    denominator = pred_var + target_var + (pred_mean - target_mean) ** 2 + 1e-8
    ccc = (2.0 * covariance) / denominator
    return 1.0 - ccc


def layer_gradient_norms(model: Net) -> Dict[str, float]:
    norms: Dict[str, float] = {}
    for layer_name in ("fc1", "fc2", "fc3"):
        layer = getattr(model, layer_name)
        grads = [parameter.grad.detach().norm(2) for parameter in layer.parameters() if parameter.grad is not None]
        if grads:
            norms[layer_name] = float(torch.norm(torch.stack(grads), p=2).item())
        else:
            norms[layer_name] = 0.0
    return norms


def batch_loss_components(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    mse_valence = nn.functional.mse_loss(pred[:, 0], target[:, 0])
    mse_arousal = nn.functional.mse_loss(pred[:, 1], target[:, 1])
    ccc_valence = ccc_loss(pred[:, 0], target[:, 0])
    ccc_arousal = ccc_loss(pred[:, 1], target[:, 1])
    total = mse_valence + mse_arousal + 0.5 * (ccc_valence + ccc_arousal)
    return {
        "total": total,
        "mse_valence": mse_valence,
        "mse_arousal": mse_arousal,
        "ccc_valence": ccc_valence,
        "ccc_arousal": ccc_arousal,
    }


def initialize_history() -> Dict[str, list]:
    return {
        "train_total_loss": [],
        "val_total_loss": [],
        "train_mse_valence": [],
        "val_mse_valence": [],
        "train_mse_arousal": [],
        "val_mse_arousal": [],
        "train_ccc_valence": [],
        "val_ccc_valence": [],
        "train_ccc_arousal": [],
        "val_ccc_arousal": [],
        "learning_rates": [],
        "gradient_norms": {"fc1": [], "fc2": [], "fc3": []},
        "spike_rates": {"layer1": [], "layer2": [], "layer3": []},
        "beta_values": {"layer1": [], "layer2": [], "layer3": []},
    }


def mean_batch(values: List[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def run_epoch(model: Net, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    totals = {
        "total": [],
        "mse_valence": [],
        "mse_arousal": [],
        "ccc_valence": [],
        "ccc_arousal": [],
    }
    predictions = []
    targets = []
    gradient_norms = {"fc1": [], "fc2": [], "fc3": []}
    spike_rates = {"layer1": [], "layer2": [], "layer3": []}

    for images, target in loader:
        images = images.to(device)
        target = target.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        pred, state = model(images)
        loss_parts = batch_loss_components(pred, target)

        if training:
            loss_parts["total"].backward()
            gradient_norms_batch = layer_gradient_norms(model)
            for layer_name, value in gradient_norms_batch.items():
                gradient_norms[layer_name].append(value)
            optimizer.step()

        for key in totals:
            totals[key].append(float(loss_parts[key].detach().cpu().item()))

        predictions.append(pred.detach().cpu())
        targets.append(target.detach().cpu())

        for layer_name, value in state["spike_rates"].items():
            spike_rates[layer_name].append(float(value))

    predictions_tensor = torch.cat(predictions, dim=0) if predictions else torch.empty(0, 2)
    targets_tensor = torch.cat(targets, dim=0) if targets else torch.empty(0, 2)
    epoch_metrics = {key: mean_batch(values) for key, values in totals.items()}
    epoch_metrics["predictions"] = predictions_tensor.numpy()
    epoch_metrics["targets"] = targets_tensor.numpy()
    epoch_metrics["gradient_norms"] = {key: mean_batch(values) for key, values in gradient_norms.items()}
    epoch_metrics["spike_rates"] = {key: mean_batch(values) for key, values in spike_rates.items()}
    return epoch_metrics


def collect_sequence_predictions(model: Net, dataset: VideoFrameDataset, video_path: str, device):
    samples = list(dataset.iter_video_samples(video_path))
    times = []
    predictions = []
    targets = []
    membrane_trace = []

    model.eval()
    state = None
    with torch.no_grad():
        for sample in samples:
            frame = dataset._read_frame(sample.video_path, sample.frame_idx)
            image = dataset.transform(torchvision_to_pil(frame)).unsqueeze(0).to(device)
            pred, state = model(image, state=state)
            predictions.append(pred.squeeze(0).cpu().numpy())
            targets.append([sample.valence, sample.arousal])
            times.append(sample.time_sec)
            membrane_trace.append(state["membrane"]["output"][-1].squeeze(0).cpu().numpy())

    return np.asarray(times), np.asarray(predictions), np.asarray(targets), np.asarray(membrane_trace)


def collect_inference_predictions(model: Net, dataset, video_path: str, device):
    samples = list(dataset.iter_video_samples(video_path))
    times = []
    predictions = []
    membrane_trace = []

    model.eval()
    state = None
    with torch.no_grad():
        for sample_video_path, frame_idx, time_sec in samples:
            frame = dataset._read_frame(sample_video_path, frame_idx)
            image = dataset.transform(torchvision_to_pil(frame)).unsqueeze(0).to(device)
            pred, state = model(image, state=state)
            times.append(time_sec)
            predictions.append(pred.squeeze(0).cpu().numpy())
            membrane_trace.append(state["membrane"]["output"][-1].squeeze(0).cpu().numpy())

    return np.asarray(times), np.asarray(predictions), np.asarray(membrane_trace)


def torchvision_to_pil(frame: np.ndarray):
    from torchvision import transforms

    return transforms.ToPILImage()(frame)


def save_history(history: Dict[str, list], output_path: Path) -> None:
    serializable = {}
    for key, value in history.items():
        if isinstance(value, dict):
            serializable[key] = {sub_key: list(sub_value) for sub_key, sub_value in value.items()}
        else:
            serializable[key] = list(value)
    output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the video-to-valence/arousal SNN and save diagnostic plots.")
    parser.add_argument("--data-root", type=Path, default=None, help="Root directory containing Youtube_*_chunk_* folders")
    parser.add_argument("--plot-dir", type=Path, default=Path("plots"), help="Directory to save plots")
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/video_va_net.pt"), help="Model checkpoint path")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--frame-step", type=int, default=3, help="Frame sampling step; 3 maps 30 fps to 10 fps")
    parser.add_argument("--image-size", type=int, nargs=2, default=(360, 640), metavar=("H", "W"), help="Frame resize target")
    parser.add_argument("--num-steps", type=int, default=8, help="Internal SNN timesteps per frame")
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--train-video-limit", type=int, default=None, help="Optional cap on number of training videos")
    parser.add_argument("--val-video-limit", type=int, default=None, help="Optional cap on number of validation videos")
    parser.add_argument("--train-sample-limit", type=int, default=None, help="Optional cap on number of training samples")
    parser.add_argument("--val-sample-limit", type=int, default=None, help="Optional cap on number of validation samples")
    parser.add_argument("--test-video-limit", type=int, default=None, help="Optional cap on number of unlabeled test videos")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        frame_step=args.frame_step,
        image_size=tuple(args.image_size),
        train_video_limit=args.train_video_limit,
        val_video_limit=args.val_video_limit,
        train_sample_limit=args.train_sample_limit,
        val_sample_limit=args.val_sample_limit,
    )

    model = Net(input_shape=(1, args.image_size[0], args.image_size[1]), num_steps=args.num_steps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    history = initialize_history()
    plot_dir = args.plot_dir
    plot_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if len(train_dataset) == 0:
        raise RuntimeError("No training samples were discovered. Check the data root and parquet labels.")
    if len(val_dataset) == 0:
        raise RuntimeError("No validation samples were discovered. Check the data root and parquet labels.")

    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, device, optimizer=None)

        scheduler.step()

        history["train_total_loss"].append(train_metrics["total"])
        history["val_total_loss"].append(val_metrics["total"])
        history["train_mse_valence"].append(train_metrics["mse_valence"])
        history["val_mse_valence"].append(val_metrics["mse_valence"])
        history["train_mse_arousal"].append(train_metrics["mse_arousal"])
        history["val_mse_arousal"].append(val_metrics["mse_arousal"])
        history["train_ccc_valence"].append(train_metrics["ccc_valence"])
        history["val_ccc_valence"].append(val_metrics["ccc_valence"])
        history["train_ccc_arousal"].append(train_metrics["ccc_arousal"])
        history["val_ccc_arousal"].append(val_metrics["ccc_arousal"])
        history["learning_rates"].append(float(scheduler.get_last_lr()[0]))
        for layer_name, values in train_metrics["gradient_norms"].items():
            history["gradient_norms"][layer_name].append(values)
        for layer_name, values in train_metrics["spike_rates"].items():
            history["spike_rates"][layer_name].append(values)
        for layer_name, value in model.beta_values.items():
            history["beta_values"][layer_name].append(value)

        epoch_tag = f"epoch_{epoch + 1:03d}"
        plot_loss_curves(history, plot_dir)
        plot_total_loss(history, plot_dir)
        plot_learning_rate(history, plot_dir)
        plot_gradient_norms(history, plot_dir)
        plot_spike_rates(history, plot_dir)
        plot_beta_values(history, plot_dir)
        plot_scatter(val_metrics["predictions"], val_metrics["targets"], plot_dir, split_name=epoch_tag)

        first_video_path = val_dataset.video_paths[0]
        times, seq_predictions, seq_targets, seq_membrane = collect_sequence_predictions(model, val_dataset, first_video_path, device)
        sample_name = f"{epoch_tag}_{Path(first_video_path).parent.name}"
        plot_timeseries(times, seq_predictions, seq_targets, plot_dir, sample_name=sample_name)
        plot_membrane_trace(times, seq_membrane, plot_dir, sample_name=sample_name)

        save_model(
            model,
            args.checkpoint_path,
            optimizer=optimizer,
            history=history,
            extra={"epoch": epoch + 1},
        )

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs:03d} | "
            f"train_total={train_metrics['total']:.4f} | val_total={val_metrics['total']:.4f} | "
            f"lr={history['learning_rates'][-1]:.6f}"
        )

    save_history(history, plot_dir / "history.json")

    test_dataset = build_inference_dataset(
        data_root=args.data_root,
        split="test",
        frame_step=args.frame_step,
        image_size=tuple(args.image_size),
    )
    if args.test_video_limit is not None:
        test_dataset = build_inference_dataset(
            data_root=args.data_root,
            split="test",
            frame_step=args.frame_step,
            image_size=tuple(args.image_size),
            video_paths=test_dataset.video_paths[: args.test_video_limit],
        )

    test_output_dir = plot_dir / "test_predictions"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    for video_path in test_dataset.video_paths:
        times, predictions_array, membrane_array = collect_inference_predictions(model, test_dataset, video_path, device)
        csv_path = test_output_dir / f"{Path(video_path).stem}_predictions.csv"
        np.savetxt(
            csv_path,
            np.column_stack([times, predictions_array]),
            delimiter=",",
            header="time_sec,pred_valence,pred_arousal",
            comments="",
        )
        sample_name = f"test_{Path(video_path).stem}"
        plot_prediction_timeseries(times, predictions_array, test_output_dir, sample_name=sample_name)
        plot_membrane_trace(times, membrane_array, test_output_dir, sample_name=sample_name)
    print(f"Saved plots to {plot_dir.resolve()}")
    print(f"Saved checkpoint to {args.checkpoint_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())