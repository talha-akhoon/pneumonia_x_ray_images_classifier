import os
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from torchvision import transforms

from pneumonia_x_ray_images_classifier.models.model import PneumoniaClassifierModel

APP_TITLE = "Pneumonia X-ray Classifier"
DEFAULT_CHECKPOINT = "models/checkpoints/mobilenet_v2_lr0.0001_drop0.5_Unfrozen_layers_3_best_05_recall0.994845.pth"

CHECKPOINT_PATH = Path(os.getenv("MODEL_CHECKPOINT", DEFAULT_CHECKPOINT))
DEVICE = torch.device(os.getenv("DEVICE", "cpu"))

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

app = FastAPI(title=APP_TITLE, version="1.0.0")

model: PneumoniaClassifierModel | None = None


def load_model():
    global model
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    model = PneumoniaClassifierModel(num_classes=2, dropout=0.5, freeze_backbone=False, unfreeze_last_n=3)
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()


@app.on_event("startup")
def _startup():
    load_model()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "checkpoint": str(CHECKPOINT_PATH),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Basic content-type check (best effort)
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg", "image/webp"}:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.content_type}")

    try:
        img = Image.open(file.file).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    x = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        p_pneumonia = float(probs[1].item())
        pred_idx = int(torch.argmax(probs).item())

    pred_label: Literal["NORMAL", "PNEUMONIA"] = "PNEUMONIA" if pred_idx == 1 else "NORMAL"

    return {
        "prediction": pred_label,
        "p_pneumonia": p_pneumonia,
        "note": "High-recall screening model; not a medical device."
    }
