"""train_best.py - takes the best architecture found by the search and trains it FROM SCRATCH (random start, no pretrained
weights) on the full CIFAR-10 training split. Reports validation accuracy per epoch and the final TEST accuracy.
Run:  python q2_nas/train_best.py [--force]"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

import config
import common as C
from search_space import build_model
from hardware import measure_hardware

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    ckpt = os.path.join(C.CKPT_DIR, "q2_nas_best.pth")
    if os.path.exists(ckpt) and not a.force:
        print("NAS-best checkpoint exists - skipping (use --force to retrain).")
        sys.exit(0)

    best = C.load_json(os.path.join(C.RESULTS_DIR, "nas_best_config.json"))
    cfg = best["config"]
    print("Training best architecture from scratch:", cfg)
    C.set_seed()
    xtr, ytr, xte, yte, _ = C.load_cifar("cifar10")
    xt, yt, xv, yv = C.split_train_val(xtr, ytr, config.NAS_VAL_SIZE)
    pre = C.Preprocess(config.Q2_MEAN, config.Q2_STD, cfg["resolution"])
    train = C.GPUData(xt, yt, pre, config.Q2_BATCH, True)
    val = C.GPUData(xv, yv, pre, config.Q2_BATCH, False)
    test = C.GPUData(xte, yte, pre, config.Q2_BATCH, False)

    model = build_model(cfg, config.Q2_NUM_CLASSES).to(C.DEVICE).to(memory_format=torch.channels_last)

    def step(m, x, y):
        logits = m(x).float()
        return F.cross_entropy(logits, y), logits

    logs, secs = C.fit("nas-best", model, step, train, val, config.NAS_FINAL_EPOCHS, config.NAS_LR, config.NAS_WD,
                       optimizer="adamw", log_path=os.path.join(C.RESULTS_DIR, "q2_nas_best_training_log.csv"))
    test_acc = C.evaluate(model, test)
    C.save_ckpt(model, ckpt)
    hw = measure_hardware(cfg)
    metrics = {"model": "nas_best", "dataset": "cifar10", "epochs": config.NAS_FINAL_EPOCHS,
               "final_val_acc": logs[-1]["test_acc"], "final_test_acc": round(test_acc, 4),
               "train_time_min": round(secs / 60, 2), **hw, "checkpoint": os.path.relpath(ckpt, C.ROOT)}
    C.save_json(metrics, os.path.join(C.RESULTS_DIR, "q2_nas_best_metrics.json"))
    print("NAS BEST DONE:", metrics)
