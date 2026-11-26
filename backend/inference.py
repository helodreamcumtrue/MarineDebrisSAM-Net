"""
PyTorch Inference & Foreground Segmentation Engine for MarineDebrisSAM-Net.
Imports the exact model architectures from ../models.py and aligns preprocessing
with MarineDebrisSAM.ipynb (224x224 RGB, ImageNet mean/std).
"""

import io
import os
import sys
import time
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.data import TARGET_CLASSES

CHECKPOINTS_DIR = Path(os.environ.get("CHECKPOINTS_DIR", ROOT_DIR / "checkpoints"))

MODEL_CHECKPOINT_FILES = {
    "a1": "best_model_a1.pth",
    "a2": "best_model_a2.pth",
    "a3": "best_model_a3.pth",
    "a5": "best_model_a5.pth",
    "a5_opt": "best_model_optimized_a5.pth",
}

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms
    from models import SimpleConcatFusionNet, MDMAM_Net

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def build_single_stream_mobilenet(num_classes: int = 6):
    """Replicates Baseline A1 / A2 MobileNetV3-Small from MarineDebrisSAM.ipynb."""
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


class ModelRegistry:
    def __init__(self):
        self.device = torch.device("cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu") if TORCH_AVAILABLE else None
        self.loaded_models: Dict[str, object] = {}
        self.weights_status: Dict[str, bool] = {}
        if TORCH_AVAILABLE:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = None
        self.scan_checkpoints()

    def scan_checkpoints(self) -> Dict[str, bool]:
        for model_id, filename in MODEL_CHECKPOINT_FILES.items():
            self.weights_status[model_id] = (CHECKPOINTS_DIR / filename).is_file()
        return self.weights_status

    def _instantiate_model(self, model_id: str):
        if model_id in ("a1", "a2"):
            return build_single_stream_mobilenet(num_classes=len(TARGET_CLASSES))
        if model_id == "a3":
            return SimpleConcatFusionNet(num_classes=len(TARGET_CLASSES))
        if model_id in ("a5", "a5_opt"):
            return MDMAM_Net(num_classes=len(TARGET_CLASSES))
        raise ValueError(f"Unknown model_id: {model_id}")

    def get_model(self, model_id: str) -> Tuple[object, bool]:
        """Returns (model, weights_loaded). Loads .pth checkpoint if available."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed in this Python environment.")

        ckpt_path = CHECKPOINTS_DIR / MODEL_CHECKPOINT_FILES[model_id]
        has_weights = ckpt_path.is_file()
        self.weights_status[model_id] = has_weights

        if model_id in self.loaded_models:
            return self.loaded_models[model_id], has_weights

        model = self._instantiate_model(model_id)
        if has_weights:
            state_dict = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.loaded_models[model_id] = model
        return model, has_weights

    def predict(
        self,
        model_id: str,
        orig_img: Image.Image,
        seg_img: Image.Image,
    ) -> dict:
        if model_id not in MODEL_CHECKPOINT_FILES:
            raise ValueError(f"Unsupported model_id '{model_id}'. Choose from {list(MODEL_CHECKPOINT_FILES.keys())}.")

        model, weights_loaded = self.get_model(model_id)
        t0 = time.perf_counter()

        x_orig = self.transform(orig_img.convert("RGB")).unsqueeze(0).to(self.device)
        x_seg = self.transform(seg_img.convert("RGB")).unsqueeze(0).to(self.device)

        attention_map: Optional[List[List[float]]] = None

        with torch.no_grad():
            if model_id == "a1":
                logits = model(x_orig)
            elif model_id == "a2":
                logits = model(x_seg)
            elif model_id == "a3":
                logits = model(x_orig, x_seg)
            else:
                # MDMAM_Net forward with spatial cross-attention map extraction
                F_ctx = model.features_orig(x_orig)
                F_obj = model.features_seg(x_seg)
                B, C, H, W = F_ctx.size()
                N = H * W
                proj_query = model.query_conv(F_ctx).view(B, C, N).permute(0, 2, 1)
                proj_key = model.key_conv(F_obj).view(B, C, N)
                proj_value = model.value_conv(F_obj).view(B, C, N)
                energy = torch.bmm(proj_query, proj_key) / (C ** 0.5)
                attention = F.softmax(energy, dim=-1)  # [1, N, N]
                out = torch.bmm(proj_value, attention.permute(0, 2, 1))
                F_att = out.view(B, C, H, W)
                F_fusion = F_ctx + F_att
                pooled = model.gap(F_fusion).view(B, C)
                logits = model.classifier(pooled)

                # Average attention received per spatial key location -> [H, W] normalized 0..1
                attn_grid = attention.mean(dim=1).view(H, W).cpu().numpy()
                a_min, a_max = float(attn_grid.min()), float(attn_grid.max())
                if a_max > a_min:
                    attn_grid = (attn_grid - a_min) / (a_max - a_min)
                attention_map = np.round(attn_grid, 4).tolist()

            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        pred_idx = int(np.argmax(probs))

        return {
            "model_id": model_id,
            "weights_loaded": weights_loaded,
            "checkpoint_expected": MODEL_CHECKPOINT_FILES[model_id],
            "device": str(self.device),
            "predicted_class": TARGET_CLASSES[pred_idx],
            "confidence": round(float(probs[pred_idx]), 4),
            "probabilities": [
                {"class_name": TARGET_CLASSES[i], "probability": round(float(probs[i]), 4)}
                for i in range(len(TARGET_CLASSES))
            ],
            "latency_ms": latency_ms,
            "attention_map": attention_map,
        }


def generate_foreground_mask(
    orig_img: Image.Image,
    bbox: Optional[Tuple[int, int, int, int]] = None,
) -> dict:
    """
    Generates a binary foreground mask + background-zeroed RGB image (224x224).
    Uses OpenCV GrabCut inside the user bounding box if OpenCV is installed,
    otherwise uses a contrast/saliency mask inside the bounding box.
    """
    rgb = orig_img.convert("RGB")
    w, h = rgb.size
    if bbox is None:
        x1, y1, x2, y2 = int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85)
    else:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(w - 2, x1))
        y1 = max(0, min(h - 2, y1))
        x2 = max(x1 + 2, min(w, x2))
        y2 = max(y1 + 2, min(h, y2))

    arr = np.array(rgb)
    method = "saliency_bbox_fallback"

    if CV2_AVAILABLE:
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        rect = (x1, y1, max(2, x2 - x1), max(2, y2 - y1))
        try:
            cv2.grabCut(arr, mask, rect, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_RECT)
            binary_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
            if binary_mask.sum() > 16:
                method = "opencv_grabcut_bbox"
            else:
                binary_mask = np.zeros((h, w), np.uint8)
                binary_mask[y1:y2, x1:x2] = 1
        except Exception:
            binary_mask = np.zeros((h, w), np.uint8)
            binary_mask[y1:y2, x1:x2] = 1
    else:
        # Pure-NumPy color-distance saliency inside the bounding box
        binary_mask = np.zeros((h, w), np.uint8)
        crop = arr[y1:y2, x1:x2].astype(np.float32)
        border = np.concatenate([
            arr[max(0, y1 - 8):y1 + 2, x1:x2].reshape(-1, 3),
            arr[y2 - 2:min(h, y2 + 8), x1:x2].reshape(-1, 3),
        ], axis=0)
        bg_med = np.median(border, axis=0) if len(border) > 0 else np.median(crop.reshape(-1, 3), axis=0)
        dist = np.linalg.norm(crop - bg_med, axis=2)
        thresh = max(18.0, float(np.percentile(dist, 42)))
        binary_mask[y1:y2, x1:x2] = (dist >= thresh).astype(np.uint8)

    seg_arr = arr.copy()
    seg_arr[binary_mask == 0] = [0, 0, 0]

    mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST)
    seg_pil = Image.fromarray(seg_arr).resize((224, 224), Image.LANCZOS)

    def to_data_url(im: Image.Image) -> str:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "method": method,
        "bbox": [x1, y1, x2, y2],
        "mask_data_url": to_data_url(mask_pil),
        "segmented_data_url": to_data_url(seg_pil),
    }
