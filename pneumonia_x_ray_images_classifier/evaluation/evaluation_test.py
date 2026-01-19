# evaluate_test.py
"""
Evaluate a trained PneumoniaClassifierModel on the held-out TEST split.

Outputs:
- Accuracy
- Precision/Recall/F1 for PNEUMONIA (class=1)
- Confusion matrix
- (Optional) ROC-AUC if sklearn is installed

Usage:
  python evaluate_test.py --checkpoint models/checkpoints/<your_best>.pth

"""

import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from pneumonia_x_ray_images_classifier.models.model import PneumoniaClassifierModel
from pneumonia_x_ray_images_classifier.dataset import PneumoniaDataset
from pneumonia_x_ray_images_classifier.config import PROCESSED_DATA_DIR
from sklearn.metrics import roc_auc_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint (state_dict).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--dropout", type=float, default=0.5, help="Dropout used in the saved model (match training).")
    p.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="If you trained with frozen backbone, pass this. (For fine-tuned models, omit.)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    return p.parse_args()


def compute_metrics_from_confusion(tp, tn, fp, fn):
    # Positive class = pneumonia (1)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return acc, precision, recall, f1


@torch.no_grad()
def main():
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device(args.device)
    test_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    data_root = Path(PROCESSED_DATA_DIR)

    test_ds = PneumoniaDataset(data_root, split="test", transform=test_transforms)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = PneumoniaClassifierModel(
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    model.eval()

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)

    # Confusion matrix counts for PNEUMONIA
    tp = tn = fp = fn = 0

    all_probs = []
    all_labels = []

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probs = F.softmax(logits, dim=1)[:, 1]  # P(pneumonia)
        preds = logits.argmax(dim=1)

        tp += ((preds == 1) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()

        all_probs.append(probs.detach().cpu())
        all_labels.append(labels.detach().cpu())

    acc, precision, recall, f1 = compute_metrics_from_confusion(tp, tn, fp, fn)

    print("\n=== Test Evaluation ===")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Data root : {data_root}")
    print(f"Device    : {device}")
    print()
    print("Confusion matrix (rows=true, cols=pred):")
    print("                Pred 0     Pred 1")
    print(f"True 0 (NORMAL)   {tn:6d}    {fp:6d}")
    print(f"True 1 (PNEU)     {fn:6d}    {tp:6d}")
    print()
    print(f"Accuracy              : {acc:.4f}")
    print(f"Precision (PNEU=1)    : {precision:.4f}")
    print(f"Recall (PNEU=1)       : {recall:.4f}")
    print(f"F1 (PNEU=1)           : {f1:.4f}")

    y_true = torch.cat(all_labels).numpy()
    y_score = torch.cat(all_probs).numpy()
    auc = roc_auc_score(y_true, y_score)
    print(f"ROC-AUC (PNEU=1)      : {auc:.4f}")

    print("======================\n")


if __name__ == "__main__":
    main()
