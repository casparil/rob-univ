import argparse
from typing import Any
import torch
import torch.utils.data
import timm
import numpy as np
import datetime
import os.path
import pandas as pd
from typing import Optional, Tuple, Dict
import univ.utils.datasets
import univ.utils.similarity
from loguru import logger
from torch.utils.data import DataLoader
from univ.rob import datasets
from robustness import data_augmentation as da
from univ.utils import ImageFolderLMDB
from univ.utils import model_import as mi
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import os


class RepresentationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        representations: torch.Tensor,
        original_dataset: Optional[ImageFolderLMDB] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        self.representations = representations

        if original_dataset is None and labels is None:
            raise ValueError("Either original_dataset or labels must be provided")
        self.original_dataset = original_dataset
        self.labels = labels

        self.get_label_fn = self.__get_tensor_label if labels is not None else self.__get_lmdb_label

    def __get_lmdb_label(self, idx):
        return self.original_dataset[idx][1]  # type: ignore

    def __get_tensor_label(self, idx):
        return self.labels[idx]  # type: ignore

    def __len__(self):
        return len(self.representations)

    def __getitem__(self, idx):
        return self.representations[idx], self.get_label_fn(idx)


@torch.inference_mode()
def get_activations(
    model: torch.nn.Module,
    layer: str,
    device: torch.device,
    dataloader: DataLoader,
):
    """
    path_to_input_data (str): A path to an lmdb file or a directory of images.
    """
    with torch.autocast(device_type=device.type):  # type: ignore
        activations = univ.utils.similarity.get_activations([model], [layer], [dataloader], device)
    activations = activations[0]
    activations = torch.concat(activations, dim=0)

    # reps are in (n,f) or (n,c,h,w)?
    if activations.dim() > 2:
        logger.debug(f"Average pooling over spatial dims: {activations.shape}")
        activations = torch.mean(activations, dim=(2, 3))
        logger.debug(f"Shape after pooling: {activations.shape}")

    return activations


def get_data(
    lmdb_file: str, batch_size: int, workers: int, pin_memory_device: str = ""
) -> Tuple[ImageFolderLMDB, DataLoader]:
    dataset = ImageFolderLMDB(lmdb_file, da.TEST_TRANSFORMS_IMAGENET)
    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        pin_memory_device=pin_memory_device,
        prefetch_factor=2,
    )
    return dataset, loader


def get_representation_layer(model_arch: str) -> str:
    # Check which layer to use for the representation
    attrs = univ.utils.datasets.get_dataset_attr(dataset="imagenet/", cnns="imagenet/")
    model_to_rep_layer = {}
    for model_name, layer in zip(attrs["model_names"], attrs["layers"]):
        model_to_rep_layer[model_name] = layer

    return model_to_rep_layer[model_arch]


def get_model(
    arch: str,
    ckpt_path: str,
    device: torch.device,
    dataset_name: str,
) -> torch.nn.Module:
    assert ckpt_path is not None, "Cannot load model without checkpoint path"

    if dataset_name == "imagenet1k":
        dataset = datasets.ImageNet("")
    elif dataset_name == "imagenet50":
        dataset = datasets.ImageNet50("")
    elif dataset_name == "imagenet100":
        dataset = datasets.ImageNet100("")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    vit = True if arch == "tiny_vit_5m" else False
    if vit:
        arch = timm.create_model("tiny_vit_5m_224.in1k", pretrained=False)
    model, _ = mi.make_and_restore_model(arch=arch, dataset=dataset, device=device, resume_path=ckpt_path, vit=vit)
    model.eval()
    return model


def get_representation_dataset(
    reps_file: Optional[str],
    lmdb_file: Optional[str],
    model_arch: str,
    model_ckpt_path: str,
    dataset_name: str,
    batch_size: int,
    workers: int,
    device: torch.device,
    labels_file: Optional[str] = None,
    path_to_save_reps: Optional[str] = None,
) -> RepresentationDataset:
    if reps_file is None:
        assert lmdb_file is not None, "Must provide lmdb file if reps file is not provided"
        model = get_model(model_arch, model_ckpt_path, device, dataset_name)

        logger.info(f"Loading dataset from {lmdb_file}")
        dataset, loader = get_data(lmdb_file, batch_size, workers, device.type)

        logger.debug(f"Getting activations from {model_arch} layer {get_representation_layer(model_arch)}")
        layer = get_representation_layer(model_arch)
        reps = get_activations(model, layer, device, loader)
        if path_to_save_reps is not None:
            logger.info(f"Saving representations to {path_to_save_reps}")
            torch.save(reps, path_to_save_reps)
    else:
        logger.info(f"Loading representations from {reps_file}")
        reps = torch.load(reps_file)
        logger.debug(f"Loaded representations of shape {reps.shape}")
    reps = reps.float()

    if labels_file is not None:
        logger.info(f"Loading labels from {labels_file}")
        labels = torch.load(labels_file)
        train_rep_dataset = RepresentationDataset(reps, labels=labels)
    else:
        assert lmdb_file is not None, "Must provide lmdb file if labels file is not provided"
        logger.info(f"Loading dataset from {lmdb_file}")
        dataset, loader = get_data(lmdb_file, batch_size, workers, device.type)
        train_rep_dataset = RepresentationDataset(reps, original_dataset=dataset)

    return train_rep_dataset


def normalize_representations(
    train_rep_dataset: RepresentationDataset, normalization_stats: Optional[Dict[str, torch.Tensor]] = None
) -> Tuple[RepresentationDataset, Dict[str, torch.Tensor]]:
    if normalization_stats is None:
        mean = train_rep_dataset.representations.mean(dim=0, keepdim=True)
        std = train_rep_dataset.representations.std(dim=0, keepdim=True)
    else:
        mean = normalization_stats["mean"]
        std = normalization_stats["std"]
    non_zero_std_mask = (std > 0).squeeze()
    train_rep_dataset.representations.sub_(mean)
    train_rep_dataset.representations[:, non_zero_std_mask].div_(std[:, non_zero_std_mask])

    return train_rep_dataset, {"mean": mean, "std": std}


def train_epoch(
    probe: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,  # type: ignore
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss_fn: torch.nn.Module,
    writer: SummaryWriter,
    global_step: int,
    log_loss_every_n_batches: int = 30,
) -> int:
    """Train a model for one epoch

    Args:
        model: Model to calibrate
        loader: DataLoader with validation data
        device: Device to run on
        temperature: Temperature parameter to optimize
        optimizer: Optimizer for temperature parameter
        loss_fn: Loss function (typically CrossEntropyLoss)
    """
    probe.train()
    for i, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)

        with torch.autocast(device_type=device.type):  # type: ignore
            logits = probe(x)
            loss = loss_fn(logits, y)
        scaler.scale(loss).backward()  # type: ignore
        # torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if i % log_loss_every_n_batches == 0:
            # Calculate total gradient norm
            total_norm = 0.0
            for p in probe.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm()
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5

            logger.info(f"Batch {i}/{len(loader)}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            writer.add_scalar("train/batch_loss", loss.item(), global_step)
            writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], global_step)
            writer.add_scalar("train/gradient_norm", total_norm, global_step)
        global_step += 1

        if i == 0:
            logger.debug("First batch")
            logger.debug(f"{x=}")
            logger.debug(f"{y=}")

    scheduler.step()
    return global_step


@torch.inference_mode()
def validate(
    probe: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: torch.nn.Module,
) -> Tuple[float, float]:
    """Validate probe performance

    Returns:
        Tuple of (average loss, accuracy)
    """
    probe.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = probe(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * x.size(0)

        predicted = logits.argmax(1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


def train_probe(
    n_epochs: int,
    batch_size: int,
    learning_rate: float,
    num_features: int,
    num_classes: int,
    device: torch.device,
    log_dir: str,
    train_rep_loader: DataLoader,
    train_rep_loader_for_eval: DataLoader,
    val_rep_loader: Optional[DataLoader] = None,
    log_every_n_batches: int = 30,
    dataset_name: str = "",
    model_arch: str = "",
):
    probe = torch.nn.Linear(num_features, num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()  # type: ignore

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir)

    # Log hyperparameters
    hparams = {
        "model_arch": model_arch,
        "dataset": dataset_name,
        "batch_size": batch_size,
        "num_epochs": n_epochs,
        "optimizer": optimizer.__class__.__name__,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "weight_decay": optimizer.param_groups[0].get("weight_decay", 0.0),
        "num_features": num_features,
        "num_classes": num_classes,
        "probe_architecture": str(probe),
        "scheduler": scheduler.__class__.__name__,
        "scheduler_min_lr": scheduler.eta_min,
    }
    global_step = 0

    best_val_acc = 0.0  # Track best validation accuracy
    best_val_loss = float("inf")  # Track best validation loss
    best_train_acc = 0.0
    best_train_loss = float("inf")

    for epoch in range(n_epochs):
        global_step = train_epoch(
            probe,
            train_rep_loader,
            device,
            optimizer,
            scaler,
            scheduler,
            loss_fn,
            writer,
            global_step,
            log_loss_every_n_batches=log_every_n_batches,
        )

        if epoch % 3 == 0:
            # Validate on training data
            train_loss, train_acc = validate(probe, train_rep_loader_for_eval, device, loss_fn)
            logger.info(f"Epoch {epoch + 1}/{n_epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            writer.add_scalar("train/epoch_loss", train_loss, epoch)
            writer.add_scalar("train/epoch_accuracy", train_acc, epoch)

            # Update best train metrics
            if train_acc > best_train_acc:
                best_train_acc = train_acc
            if train_loss < best_train_loss:
                best_train_loss = train_loss

        # Validate on separate validation set if available
        if val_rep_loader is not None:
            val_loss, val_acc = validate(probe, val_rep_loader, device, loss_fn)
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            writer.add_scalar("val/epoch_loss", val_loss, epoch)
            writer.add_scalar("val/epoch_accuracy", val_acc, epoch)

            # Update best validation metrics
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(probe, log_dir, f"best_acc_checkpoint.pt")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(probe, log_dir, f"best_loss_checkpoint.pt")
        writer.flush()

    # If no validation set, save final model
    if val_rep_loader is None:
        save_checkpoint(probe, log_dir, "final_checkpoint.pt")

    writer.add_hparams(
        hparams,
        metric_dict={
            "val/best_acc": best_val_acc,
            "val/best_loss": best_val_loss,
            "train/best_acc": best_train_acc,
            "train/best_loss": best_train_loss,
        },
    )
    writer.close()


def save_checkpoint(probe: torch.nn.Module, log_dir: str, filename: str):
    """Save model checkpoint to the log directory"""
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_path = os.path.join(log_dir, filename)
    torch.save(probe.state_dict(), checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a linear probe for ImageNet1k or ImageNet100 given representations as input."
    )
    parser.add_argument("-m", "--model-ckpt-path", type=str)
    parser.add_argument("-a", "--model-arch", type=str, default=None)
    parser.add_argument("-d", "--dataset-name", type=str, default="imagenet1k", choices=["imagenet1k", "imagenet50"])
    parser.add_argument("--lmdb-file", type=str)
    parser.add_argument("--labels-file", type=str)
    parser.add_argument("--reps-file", type=str)
    parser.add_argument("--val-lmdb-file", type=str)
    parser.add_argument("--val-labels-file", type=str)
    parser.add_argument("--val-reps-file", type=str)
    parser.add_argument("-b", "--batch-size", help="Batch size", type=int, default=1024)
    parser.add_argument("-w", "--workers", help="Number of workers", type=int, default=2)
    parser.add_argument("--lr", help="Learning rate", type=float, default=5e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-probes", type=int, default=1)
    parser.add_argument("--log-every-n-batches", type=int, default=30)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    # For faster inference. Not tested that it helps.
    torch.backends.cudnn.benchmark = True  # type: ignore

    device = torch.device(args.device)
    if args.model_arch is None:
        args.model_arch = os.path.split(args.model_ckpt_path)[-1].split(".")[0]
        logger.info(f"Model architecture not specified, using {args.model_arch} from checkpoint path")

    train_rep_dataset = get_representation_dataset(
        reps_file=args.reps_file,
        lmdb_file=args.lmdb_file,
        model_arch=args.model_arch,
        model_ckpt_path=args.model_ckpt_path,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
        labels_file=args.labels_file,
        path_to_save_reps="reps.pt",
    )
    # train_rep_dataset, normalization_stats = normalize_representations(train_rep_dataset)

    train_rep_loader = DataLoader(
        train_rep_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True
    )
    train_rep_loader_for_eval = DataLoader(
        train_rep_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    if args.val_reps_file is not None or args.val_lmdb_file is not None:
        val_rep_dataset = get_representation_dataset(
            reps_file=args.val_reps_file,
            lmdb_file=args.val_lmdb_file,
            model_arch=args.model_arch,
            model_ckpt_path=args.model_ckpt_path,
            dataset_name=args.dataset_name,
            batch_size=args.batch_size,
            workers=args.workers,
            device=device,
            labels_file=args.val_labels_file,
            path_to_save_reps="val_reps.pt",
        )
        # val_rep_dataset, _ = normalize_representations(val_rep_dataset, normalization_stats)
        val_rep_loader = DataLoader(
            val_rep_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
        )
    else:
        val_rep_loader = None

    num_features = train_rep_dataset.representations.size(1)
    if args.dataset_name == "imagenet1k":
        num_classes = 1000
    elif args.dataset_name == "imagenet50":
        num_classes = 50
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    for i in range(args.num_probes):
        if args.run_name is None:
            log_dir = os.path.join("runs", current_time, str(i))
        else:
            log_dir = os.path.join("runs", args.run_name, current_time, str(i))
        train_probe(
            args.epochs,
            args.batch_size,
            args.lr,
            num_features,
            num_classes,
            device,
            log_dir,
            train_rep_loader,
            train_rep_loader_for_eval,
            val_rep_loader,
            args.log_every_n_batches,
            args.dataset_name,
            args.model_arch,
        )


if __name__ == "__main__":
    main()
