"""train_kd.py - trains the 3 students WITH the teacher (knowledge distillation).
Run:  python q1_distillation/train_kd.py [--student mobilenetv2] [--force]"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import config
import common as C
from teacher import TeacherNet
from students import StudentNet
from distillation import DistillationLoss


def train_one(name, teacher, data, force):
    ckpt = os.path.join(C.CKPT_DIR, f"q1_kd_{name}.pth")
    if os.path.exists(ckpt) and not force:
        print(f"[{name}] KD checkpoint exists - skipping.")
        return
    C.set_seed()
    train, test = data
    model = StudentNet(name, config.Q1_NUM_CLASSES, config.PRETRAINED).to(C.DEVICE).to(memory_format=torch.channels_last)
    criterion = DistillationLoss(model.feat_channels, teacher.feat_channels, config.KD_TEMPERATURE,
                                 config.KD_ALPHA, config.KD_FEATURE_BETA, use_feature=True).to(C.DEVICE)
    print(f"\n=== KNOWLEDGE DISTILLATION -> {name} | T={config.KD_TEMPERATURE} alpha={config.KD_ALPHA} "
          f"beta(feature)={config.KD_FEATURE_BETA} ===")

    def step(m, x, y):
        with torch.no_grad():                                   # the teacher only gives advice, it is not trained
            t_logits, t_feat = teacher(x, return_feat=True)
        s_logits, s_feat = m(x, return_feat=True)
        loss, _ = criterion(s_logits, t_logits, y, s_feat, t_feat)
        return loss, s_logits.float()

    logs, secs = C.fit(f"kd-{name}", model, step, train, test, config.STUDENT_EPOCHS,
                       config.LR[name], config.WEIGHT_DECAY, extra_modules=[criterion],
                       log_path=os.path.join(C.RESULTS_DIR, f"q1_kd_{name}_log.csv"))
    C.save_ckpt(model, ckpt)
    C.save_json({"model": name, "variant": "kd", "dataset": "cifar100", "epochs": config.STUDENT_EPOCHS,
                 "temperature": config.KD_TEMPERATURE, "alpha": config.KD_ALPHA, "feature_beta": config.KD_FEATURE_BETA,
                 "final_test_acc": logs[-1]["test_acc"], "params_M": round(C.count_params_m(model), 2),
                 "train_time_min": round(secs / 60, 2), "checkpoint": os.path.relpath(ckpt, C.ROOT)},
                os.path.join(C.RESULTS_DIR, f"q1_kd_{name}_metrics.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="all")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    teacher_ckpt = os.path.join(C.CKPT_DIR, "q1_teacher_resnet50.pth")
    assert os.path.exists(teacher_ckpt), "Train the teacher first (train_teacher.py)."
    teacher = TeacherNet(config.Q1_NUM_CLASSES, pretrained=False)
    C.load_ckpt(teacher, teacher_ckpt)
    teacher = teacher.to(C.DEVICE).to(memory_format=torch.channels_last).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    xtr, ytr, xte, yte, _ = C.load_cifar("cifar100")
    pre = C.Preprocess(config.Q1_MEAN, config.Q1_STD, config.Q1_IMG_SIZE)
    data = (C.GPUData(xtr, ytr, pre, config.Q1_BATCH, True), C.GPUData(xte, yte, pre, config.Q1_BATCH, False))
    print(f"Teacher test accuracy: {C.evaluate(teacher, data[1]):.4f}")
    for n in (config.STUDENTS if a.student == "all" else [a.student]):
        train_one(n, teacher, data, a.force)
