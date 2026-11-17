"""export_onnx.py - exports the trained students (KD + baseline) to ONNX and checks them against PyTorch.
The ONNX models take a raw image batch (batch,3,32,32) with pixel values in 0..1 and return the 100 class scores.
Run:  python q1_distillation/export_onnx.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import config
import common as C
from students import StudentNet

if __name__ == "__main__":
    _, _, xte, _, _ = C.load_cifar("cifar100")
    sample = xte[:64]
    rows = []
    for n in config.STUDENTS:
        for v in ("kd", "baseline"):
            ckpt = os.path.join(C.CKPT_DIR, f"q1_{v}_{n}.pth")
            if not os.path.exists(ckpt):
                print(f"missing {ckpt}, skipped")
                continue
            model = StudentNet(n, config.Q1_NUM_CLASSES, pretrained=False)
            C.load_ckpt(model, ckpt).eval()
            pre = C.Preprocess(config.Q1_MEAN, config.Q1_STD, config.Q1_IMG_SIZE)
            path = os.path.join(C.ONNX_DIR, f"q1_{n}_{v}.onnx")
            wrapper = C.export_onnx(model, pre, path)
            # compare PyTorch (CPU) and ONNX Runtime (CPU) on 64 test images
            x = sample.float() / 255.0
            with torch.no_grad():
                ref = wrapper(x).numpy()
            out = C.ort_logits(C.ort_session(path), sample)
            diff = float(np.abs(ref - out).max())
            agree = float((ref.argmax(1) == out.argmax(1)).mean())
            rows.append({"model": n, "variant": v, "onnx_file": os.path.relpath(path, C.ROOT),
                         "size_mb": round(C.file_mb(path), 2), "max_abs_diff_vs_pytorch": f"{diff:.2e}",
                         "prediction_agreement": agree})
            print(rows[-1])
    C.save_csv(rows, os.path.join(C.RESULTS_DIR, "q1_onnx_export_check.csv"))
