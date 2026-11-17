"""train_teacher.py - trains the ResNet-50 teacher on CIFAR-100.   Run:  python q1_distillation/train_teacher.py [--force]"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

import config
import common as C
from teacher import TeacherNet


def main(force=False):
    ckpt = os.path.join(C.CKPT_DIR, "q1_teacher_resnet50.pth")
    if os.path.exists(ckpt) and not force:
        print(f"Teacher checkpoint already exists ({ckpt}) - skipping. Use --force to retrain.")
        return
    C.set_seed()
    xtr, ytr, xte, yte, classes = C.load_cifar("cifar100")
    pre = C.Preprocess(config.Q1_MEAN, config.Q1_STD, config.Q1_IMG_SIZE)
    train = C.GPUData(xtr, ytr, pre, config.Q1_BATCH, train=True)
    test = C.GPUData(xte, yte, pre, config.Q1_BATCH, train=False)

    model = TeacherNet(config.Q1_NUM_CLASSES, config.PRETRAINED).to(C.DEVICE).to(memory_format=torch.channels_last)
    print(f"Teacher ResNet-50: {C.count_params_m(model):.1f} M parameters | device: {C.DEVICE}")

    def step(m, x, y):
        logits = m(x).float()
        return F.cross_entropy(logits, y), logits

    logs, secs = C.fit("teacher", model, step, train, test, config.TEACHER_EPOCHS,
                       config.LR["resnet50"], config.WEIGHT_DECAY,
                       log_path=os.path.join(C.RESULTS_DIR, "q1_teacher_training_log.csv"))
    C.save_ckpt(model, ckpt, half=True)      # fp16 weights -> ~47 MB, safe for GitHub's 100 MB limit
    metrics = {"model": "resnet50_teacher", "dataset": "cifar100", "epochs": config.TEACHER_EPOCHS,
               "final_test_acc": logs[-1]["test_acc"], "params_M": round(C.count_params_m(model), 2),
               "train_time_min": round(secs / 60, 2), "checkpoint": os.path.relpath(ckpt, C.ROOT)}
    C.save_json(metrics, os.path.join(C.RESULTS_DIR, "q1_teacher_metrics.json"))
    print("TEACHER DONE:", metrics)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
