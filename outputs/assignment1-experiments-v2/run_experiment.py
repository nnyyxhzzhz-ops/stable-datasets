"""Train controlled LeJEPA extensions using the revised local dataset builders.

This command never fits a probe or reads test images. Development checkpoint
selection and final evaluation are separate commands.
"""

import argparse
import copy
import json
import math
import platform
import random
import resource
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from common import (
    ROOT,
    environment,
    file_sha256,
    initialize_runtime,
    load_splits,
    split_manifest,
    write_json,
)


class Views:
    """Create label-free global and local image views with the B0 augmentations."""

    def __init__(self, config):
        def augment(size, scale):
            return T.Compose(
                [
                    T.RandomResizedCrop(size, scale=scale),
                    T.RandomHorizontalFlip(),
                    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                    T.RandomGrayscale(p=0.2),
                    T.ToTensor(),
                    T.Normalize([0.5] * 3, [0.5] * 3),
                ]
            )

        self.global_transform = augment(config["image_size"], (0.3, 1.0))
        self.local_transform = augment(config["local_size"], (0.05, 0.3))

    def __call__(self, row):
        return {
            "global_views": [self.global_transform(row["image"]) for _ in range(2)],
            "local_views": [self.local_transform(row["image"]) for _ in range(2)],
        }


def seed_worker(_):
    """Seed each augmentation worker from its DataLoader-provided seed."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def initialize_model(config, reference_path):
    """Match the B0 constructor order and assert its exact initial encoder."""
    from stable_pretraining.methods.lejepa import LeJEPA

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    # The projector was constructed before the encoder in all six B0 runs.
    projector = nn.Sequential(
        nn.Linear(192, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 128)
    )
    model = LeJEPA(
        encoder_name="vit_tiny_patch16_224",
        projector=projector,
        n_slices=config["n_slices"],
        n_points=17,
        lamb=0.02,
        pretrained=False,
        drop_path_rate=0.0,
    )
    assert model.embed_dim == 192
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    actual = model.backbone.state_dict()
    if set(actual) != set(reference) or any(
        not torch.equal(value, reference[key]) for key, value in actual.items()
    ):
        raise RuntimeError(
            "Encoder initialization differs from the preserved B0 random encoder."
        )
    return model


def make_loader(train, config):
    """Shuffle the complete training split and expose image views only."""
    return DataLoader(
        train.with_transform(Views(config)),
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=config["num_workers"],
        persistent_workers=config["num_workers"] > 0,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(config["seed"]),
    )


def learning_rate_factor(step, total_steps, warmup_steps):
    """Apply the B0 warmup/cosine rule over the declared run budget."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    return 0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2


def synchronize(device):
    """Finish asynchronous work before collecting timing measurements."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def parent_peak_rss_bytes():
    """Return parent-process peak RSS with platform-specific units normalized."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def save_checkpoint(path, model, optimizer, epoch, step, config, loader):
    """Save weights, optimization state and parent/loader random states."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "steps": step,
            "config": config,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "loader_rng": loader.generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "worker_rng_states_saved": False,
            "exact_training_resume_implemented": False,
        },
        path,
    )


def run_training(
    train,
    config,
    out_dir,
    reference,
    mode="train",
    warmup_batches=5,
    measured_batches=30,
):
    """Train on image-only batches, or measure a separate non-result run."""
    device = config["device"]
    model = initialize_model(config, reference).to(device)
    initial = copy.deepcopy(model.backbone).cpu().state_dict()
    torch.save(initial, out_dir / "random-encoder.pt")
    loader = make_loader(train, config)
    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise ValueError("Training split is smaller than one full batch.")
    if mode == "benchmark" and warmup_batches + measured_batches > steps_per_epoch:
        raise ValueError("Timing batches must fit in one epoch.")
    epochs = (
        min(config["epochs"], config.get("stop_after") or config["epochs"])
        if mode == "train"
        else 1
    )
    total_steps = config["epochs"] * steps_per_epoch
    warmup = min(config["warmup_epochs"] * steps_per_epoch, max(1, total_steps // 2))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=0.05
    )
    history, timings = [], []
    step = 0
    run_start = time.perf_counter()
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        totals = np.zeros(3)
        iterator = iter(loader)
        limit = (
            steps_per_epoch if mode == "train" else warmup_batches + measured_batches
        )
        for batch_index in range(limit):
            synchronize(device)
            batch_start = time.perf_counter()
            batch = next(iterator)
            if set(batch) != {"global_views", "local_views"}:
                raise RuntimeError("SSL batch must contain only augmented images.")
            factor = learning_rate_factor(step, total_steps, warmup)
            for group in optimizer.param_groups:
                group["lr"] = config["learning_rate"] * factor
            output = model(
                global_views=[view.to(device) for view in batch["global_views"]],
                local_views=[view.to(device) for view in batch["local_views"]],
            )
            if not torch.isfinite(output.loss):
                raise RuntimeError("Non-finite LeJEPA loss.")
            optimizer.zero_grad(set_to_none=True)
            output.loss.backward()
            if not all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise RuntimeError("Non-finite LeJEPA gradient.")
            optimizer.step()
            totals += [
                float(value.detach())
                for value in (output.loss, output.inv_loss, output.sigreg_loss)
            ]
            step += 1
            synchronize(device)
            if mode == "benchmark" and batch_index >= warmup_batches:
                timings.append(time.perf_counter() - batch_start)
        record = dict(zip(("loss", "invariance", "sigreg"), (totals / limit).tolist()))
        record.update(
            epoch=epoch,
            steps=step,
            seconds=time.perf_counter() - epoch_start,
            learning_rate=optimizer.param_groups[0]["lr"],
        )
        history.append(record)
        print(json.dumps(record), flush=True)
        write_json(out_dir / "training.json", history)
        if mode == "train" and (epoch in {20, 50, 100} or epoch == epochs):
            save_checkpoint(
                out_dir / f"checkpoint-epoch{epoch:03d}.pt",
                model,
                optimizer,
                epoch,
                step,
                config,
                loader,
            )
    current = model.backbone.state_dict()
    changed = sum(
        not torch.equal(initial[key], value.cpu()) for key, value in current.items()
    )
    if not changed or not all(
        torch.isfinite(value).all() for value in model.state_dict().values()
    ):
        raise RuntimeError("Encoder did not change or its final state is non-finite.")
    assert model.sigreg.global_step.item() == step
    summary = {
        "mode": mode,
        "epochs_completed": epochs,
        "declared_schedule_epochs": config["epochs"],
        "steps": step,
        "steps_per_epoch": steps_per_epoch,
        "train_examples": len(train),
        "dropped_tail_per_epoch": len(train) % config["batch_size"],
        "changed_encoder_tensors": changed,
        "initial_encoder_equal_b0": True,
        "initial_b0_checkpoint_sha256": file_sha256(reference),
        "ssl_seconds": time.perf_counter() - run_start,
        "parent_peak_rss_bytes": parent_peak_rss_bytes(),
        "test_images_accessed": False,
        "probe_fitted": False,
        "status": "completed",
    }
    if mode == "benchmark":
        summary["benchmark"] = {
            "warmup_batches": warmup_batches,
            "measured_batches": len(timings),
            "mean_batch_seconds": float(np.mean(timings)),
            "median_batch_seconds": float(np.median(timings)),
            "estimated_100_epoch_training_seconds": float(
                np.mean(timings) * steps_per_epoch * 100
            ),
            "includes_batch_loading_and_finite_checks": True,
        }
    write_json(out_dir / "training-summary.json", summary)
    return summary


def main():
    """Parse a fixed candidate config and write a new isolated run directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("eth80", "freiburg"), required=True)
    parser.add_argument("--candidate", choices=("B1", "B2"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("train", "benchmark"), default="train")
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override only for explicitly identified smoke/development runs.",
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        help="Stop at this checkpoint while retaining the declared schedule.",
    )
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--measured-batches", type=int, default=30)
    args = parser.parse_args()
    config_path = Path(__file__).parent / "configs" / f"{args.candidate}.json"
    config = json.loads(config_path.read_text())
    config.update(
        dataset=args.dataset,
        candidate=args.candidate,
        seed=args.seed,
        device=args.device,
    )
    if args.epochs is not None:
        if args.epochs < 1:
            parser.error("epochs must be positive")
        config["epochs"] = args.epochs
    if args.stop_after is not None:
        if not 1 <= args.stop_after <= config["epochs"]:
            parser.error("stop-after must lie within the declared epoch budget")
        config["stop_after"] = args.stop_after
    if args.warmup_batches < 0 or args.measured_batches < 1:
        parser.error(
            "timing requires non-negative warmup and positive measured batches"
        )
    initialize_runtime(config["threads"])
    torch.use_deterministic_algorithms(True)
    splits = load_splits(args.dataset, config)
    args.output.mkdir(parents=True, exist_ok=False)
    reference = (
        ROOT
        / "outputs/experiments"
        / f"{args.dataset}-seed{args.seed}"
        / "random-encoder.pt"
    )
    write_json(args.output / "config.json", config)
    write_json(args.output / "splits.json", split_manifest(splits))
    metadata = environment()
    metadata.update(
        script_sha256=file_sha256(Path(__file__)),
        config_source_sha256=file_sha256(config_path),
        common_sha256=file_sha256(Path(__file__).parent / "common.py"),
        initial_reference_sha256=file_sha256(reference),
    )
    write_json(args.output / "environment.json", metadata)
    summary = run_training(
        splits["train"],
        config,
        args.output,
        reference,
        args.mode,
        args.warmup_batches,
        args.measured_batches,
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
