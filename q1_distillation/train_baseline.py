"""train_baseline.py - trains the 3 students ALONE (no teacher, plain cross-entropy). This is the comparison group.
Run:  python q1_distillation/train_baseline.py [--student mobilenetv2] [--force]"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

import config
import common as C
from students import StudentNet


def train_one(name, data, force):
    ckpt = os.path.join(C.CKPT_DIR, f"q1_baseline_{name}.pth")
    if os.path.exists(ckpt) and not force:
        print(f"[{name}] baseline checkpoint exists - skipping.")
        return
    C.set_seed()
    train, test = data
    model = StudentNet(name, config.Q1_NUM_CLASSES, config.PRETRAINED).to(C.DEVICE).to(memory_format=torch.channels_last)
    print(f"\n=== BASELINE {name} (no distillation) | {C.count_params_m(model):.2f} M params ===")

    def step(m, x, y):
        logits = m(x).float()
        return F.cross_entropy(logits, y), logits

    logs, secs = C.fit(f"baseline-{name}", model, step, train, test, config.STUDENT_EPOCHS,
                       config.LR[name], config.WEIGHT_DECAY,
                       log_path=os.path.join(C.RESULTS_DIR, f"q1_baseline_{name}_log.csv"))
    C.save_ckpt(model, ckpt)
    C.save_json({"model": name, "variant": "baseline", "dataset": "cifar100", "epochs": config.STUDENT_EPOCHS,
                 "final_test_acc": logs[-1]["test_acc"], "params_M": round(C.count_params_m(model), 2),
                 "train_time_min": round(secs / 60, 2), "checkpoint": os.path.relpath(ckpt, C.ROOT)},
                os.path.join(C.RESULTS_DIR, f"q1_baseline_{name}_metrics.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="all")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    xtr, ytr, xte, yte, _ = C.load_cifar("cifar100")
    pre = C.Preprocess(config.Q1_MEAN, config.Q1_STD, config.Q1_IMG_SIZE)
    data = (C.GPUData(xtr, ytr, pre, config.Q1_BATCH, True), C.GPUData(xte, yte, pre, config.Q1_BATCH, False))
    for n in (config.STUDENTS if a.student == "all" else [a.student]):
        train_one(n, data, a.force)
