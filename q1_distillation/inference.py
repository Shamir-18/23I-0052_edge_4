"""inference.py - ONE unified inference pipeline for all trained students (KD or baseline).

Examples
  python q1_distillation/inference.py --all                                   # test accuracy of all 6 students
  python q1_distillation/inference.py --model mobilenetv2 --variant kd        # one student on the CIFAR-100 test set
  python q1_distillation/inference.py --model squeezenet --variant kd --image cat.jpg   # predict one picture
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import config
import common as C
from students import StudentNet


def load_student(name, variant):
    """Builds the student, loads its checkpoint and wraps it with the preprocessing (raw 32x32 in -> logits out)."""
    ckpt = os.path.join(C.CKPT_DIR, f"q1_{variant}_{name}.pth")
    model = StudentNet(name, config.Q1_NUM_CLASSES, pretrained=False)
    C.load_ckpt(model, ckpt)
    pre = C.Preprocess(config.Q1_MEAN, config.Q1_STD, config.Q1_IMG_SIZE)
    return C.Deployable(model, pre).to(C.DEVICE).eval()


@torch.no_grad()
def predict(net, x_uint8, bs=250):
    """x_uint8: (N,3,32,32) uint8 tensor -> predicted class ids (N,) and probabilities (N,100)."""
    probs = []
    for i in range(0, len(x_uint8), bs):
        xb = x_uint8[i:i + bs].to(C.DEVICE).float() / 255.0
        probs.append(torch.softmax(net(xb).float(), dim=1).cpu())
    probs = torch.cat(probs)
    return probs.argmax(1), probs


def load_image(path):
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((32, 32), Image.BILINEAR)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).contiguous()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=config.STUDENTS)
    ap.add_argument("--variant", choices=["kd", "baseline"], default="kd")
    ap.add_argument("--image")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    _, _, xte, yte, classes = C.load_cifar("cifar100")
    todo = [(n, v) for n in config.STUDENTS for v in ("baseline", "kd")] if a.all else [(a.model, a.variant)]
    rows = []
    for n, v in todo:
        if not os.path.exists(os.path.join(C.CKPT_DIR, f"q1_{v}_{n}.pth")):
            print(f"missing checkpoint for {n}/{v}, skipped")
            continue
        net = load_student(n, v)
        if a.image:
            pred, probs = predict(net, load_image(a.image))
            top5 = probs[0].topk(5)
            print(f"{n} ({v}) top-5:", [(classes[i], round(p.item(), 3)) for p, i in zip(top5.values, top5.indices)])
        else:
            pred, _ = predict(net, xte)
            acc = (pred == yte).float().mean().item()
            print(f"{n:<20} {v:<9} CIFAR-100 test accuracy: {acc:.4f}")
            rows.append({"model": n, "variant": v, "test_acc": round(acc, 4)})
    if rows:
        C.save_csv(rows, os.path.join(C.RESULTS_DIR, "q1_inference_summary.csv"))
