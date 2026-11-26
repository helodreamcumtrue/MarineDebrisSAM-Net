# Trained Model Checkpoints Directory

Place your trained PyTorch checkpoint files exported from Google Colab (`MarineDebrisSAM.ipynb`) in this directory so the FastAPI inference server (`backend/main.py`) automatically loads them on startup:

- `best_model_a1.pth` — Baseline A1 (MobileNetV3-Small, Original Images Only)
- `best_model_a2.pth` — Baseline A2 (MobileNetV3-Small, SAM-Segmented Only)
- `best_model_a3.pth` — Baseline A3 (`SimpleConcatFusionNet`)
- `best_model_a5.pth` — Proposed `MDMAM_Net` (Unweighted Cross-Entropy)
- `best_model_optimized_a5.pth` — Optimized `MDMAM_Net` (Class-Weighted + CosineAnnealingLR)

Verify loaded weights at `GET /api/health`.
