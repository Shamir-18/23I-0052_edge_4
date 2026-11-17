"""export_onnx.py - exports the best NAS model to ONNX and checks it against PyTorch.
Run:  python q2_nas/export_onnx.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import config
import common as C
from search_space import build_model

if __name__ == "__main__":
    cfg = C.load_json(os.path.join(C.RESULTS_DIR, "nas_best_config.json"))["config"]
    model = build_model(cfg, config.Q2_NUM_CLASSES)
    C.load_ckpt(model, os.path.join(C.CKPT_DIR, "q2_nas_best.pth")).eval()
    pre = C.Preprocess(config.Q2_MEAN, config.Q2_STD, cfg["resolution"])
    path = os.path.join(C.ONNX_DIR, "q2_nas_best.onnx")
    wrapper = C.export_onnx(model, pre, path)

    _, _, xte, _, _ = C.load_cifar("cifar10")
    sample = xte[:64]
    with torch.no_grad():
        ref = wrapper(sample.float() / 255.0).numpy()
    out = C.ort_logits(C.ort_session(path), sample)
    row = {"model": "nas_best", "onnx_file": os.path.relpath(path, C.ROOT), "size_mb": round(C.file_mb(path), 3),
           "max_abs_diff_vs_pytorch": f"{float(np.abs(ref - out).max()):.2e}",
           "prediction_agreement": float((ref.argmax(1) == out.argmax(1)).mean())}
    print(row)
    C.save_csv([row], os.path.join(C.RESULTS_DIR, "q2_onnx_export_check.csv"))
