"""inference.py - inference pipeline for the best NAS model (PyTorch).
  python q2_nas/inference.py                  # accuracy on the CIFAR-10 test set
  python q2_nas/inference.py --image dog.jpg  # predict one picture"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import config
import common as C
from search_space import build_model


def load_nas_model():
    cfg = C.load_json(os.path.join(C.RESULTS_DIR, "nas_best_config.json"))["config"]
    model = build_model(cfg, config.Q2_NUM_CLASSES)
    C.load_ckpt(model, os.path.join(C.CKPT_DIR, "q2_nas_best.pth"))
    pre = C.Preprocess(config.Q2_MEAN, config.Q2_STD, cfg["resolution"])
    return C.Deployable(model, pre).to(C.DEVICE).eval()


@torch.no_grad()
def predict(net, x_uint8, bs=500):
    probs = []
    for i in range(0, len(x_uint8), bs):
        xb = x_uint8[i:i + bs].to(C.DEVICE).float() / 255.0
        probs.append(torch.softmax(net(xb).float(), dim=1).cpu())
    probs = torch.cat(probs)
    return probs.argmax(1), probs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    a = ap.parse_args()
    _, _, xte, yte, classes = C.load_cifar("cifar10")
    net = load_nas_model()
    if a.image:
        from PIL import Image
        img = Image.open(a.image).convert("RGB").resize((32, 32), Image.BILINEAR)
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).contiguous()
        _, probs = predict(net, x)
        top = probs[0].topk(3)
        print("NAS model top-3:", [(classes[i], round(p.item(), 3)) for p, i in zip(top.values, top.indices)])
    else:
        pred, _ = predict(net, xte)
        acc = (pred == yte).float().mean().item()
        print(f"NAS best model | CIFAR-10 test accuracy: {acc:.4f}")
        C.save_csv([{"model": "nas_best", "test_acc": round(acc, 4)}], os.path.join(C.RESULTS_DIR, "q2_inference_summary.csv"))
