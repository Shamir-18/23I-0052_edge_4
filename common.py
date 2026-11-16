"""
common.py - helpers used by every script:
  * folders, seeds, device
  * CIFAR loading + a fast GPU data pipeline (augmentation, resizing, normalising)
  * the generic training loop `fit`
  * checkpoint, ONNX, ONNX-Runtime and timing helpers
"""
import copy
import csv
import io
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config

# ----------------------------------------------------------------------------- folders
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "_dev_outputs") if config.FAST_DEV else ROOT
CKPT_DIR = os.path.join(OUT, "checkpoints")
ONNX_DIR = os.path.join(OUT, "onnx_models")
RESULTS_DIR = os.path.join(OUT, "results")
for _d in (CKPT_DIR, ONNX_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)


def _pick_data_dir():
    for base in ("/kaggle/temp", "/tmp"):
        try:
            os.makedirs(base + "/data", exist_ok=True)
            return base + "/data"
        except OSError:
            pass
    return os.path.join(ROOT, "data")


DATA_DIR = _pick_data_dir()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- data
def load_cifar(name):
    """Returns uint8 tensors (N,3,32,32) + labels + class names. name = 'cifar10' | 'cifar100'."""
    import torchvision
    cls = {"cifar10": torchvision.datasets.CIFAR10, "cifar100": torchvision.datasets.CIFAR100}[name]
    tr = cls(DATA_DIR, train=True, download=True)
    te = cls(DATA_DIR, train=False, download=True)

    def to_tensor(a):
        return torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2).contiguous()

    xtr, ytr = to_tensor(tr.data), torch.tensor(tr.targets, dtype=torch.long)
    xte, yte = to_tensor(te.data), torch.tensor(te.targets, dtype=torch.long)
    if config.FAST_DEV:
        g = torch.Generator().manual_seed(0)
        i = torch.randperm(len(xtr), generator=g)[:config.DEV_TRAIN_SUBSET]
        j = torch.randperm(len(xte), generator=g)[:config.DEV_TEST_SUBSET]
        xtr, ytr, xte, yte = xtr[i], ytr[i], xte[j], yte[j]
    return xtr, ytr, xte, yte, list(tr.classes)


def split_train_val(x, y, val_size, seed=config.SEED):
    g = torch.Generator().manual_seed(seed)
    p = torch.randperm(len(x), generator=g)
    v, t = p[:val_size], p[val_size:]
    return x[t], y[t], x[v], y[v]


class Preprocess(nn.Module):
    def __init__(self, mean, std, size):
        super().__init__()
        self.size = int(size)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x):
        if self.size != x.shape[-1]:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std


class Deployable(nn.Module):
    def __init__(self, model, pre):
        super().__init__()
        self.pre, self.model = pre, model

    def forward(self, x):
        return self.model(self.pre(x))


def augment_batch(x, pad=4):
    b, c, h, w = x.shape
    dev = x.device
    flip = torch.rand(b, device=dev) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    i = torch.randint(0, 2 * pad + 1, (b,), device=dev)
    j = torch.randint(0, 2 * pad + 1, (b,), device=dev)
    rows = (i[:, None] + torch.arange(h, device=dev))[:, None, :, None]
    cols = (j[:, None] + torch.arange(w, device=dev))[:, None, None, :]
    bidx = torch.arange(b, device=dev)[:, None, None, None]
    cidx = torch.arange(c, device=dev)[None, :, None, None]
    return xp[bidx, cidx, rows, cols]


class GPUData:
    def __init__(self, x, y, pre, batch_size, train):
        self.x, self.y = x.to(DEVICE), y.to(DEVICE)
        self.pre = pre.to(DEVICE)
        self.bs, self.train = batch_size, train

    def __len__(self):
        n = len(self.x)
        return max(1, n // self.bs) if self.train else math.ceil(n / self.bs)

    def __iter__(self):
        n = len(self.x)
        idx = torch.randperm(n, device=DEVICE) if self.train else torch.arange(n, device=DEVICE)
        for k in range(len(self)):
            b = idx[k * self.bs:(k + 1) * self.bs]
            xb = self.x[b].float().div_(255.0)
            if self.train:
                xb = augment_batch(xb)
            xb = self.pre(xb).contiguous(memory_format=torch.channels_last)
            yield xb, self.y[b]


# ----------------------------------------------------------------------------- training loop
def _make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def evaluate(model, data, use_amp=None):
    use_amp = (DEVICE.type == "cuda") if use_amp is None else use_amp
    model.eval()
    correct = total = 0
    for xb, yb in data:
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
            out = model(xb)
        correct += (out.argmax(1) == yb).sum().item()
        total += len(yb)
    return correct / max(1, total)


def fit(name, model, step_fn, train_data, test_data, epochs, lr, wd,
        extra_modules=(), log_path=None, optimizer="sgd", verbose=True):
    use_amp = DEVICE.type == "cuda"
    params = list(model.parameters()) + [p for m in extra_modules for p in m.parameters()]
    if optimizer == "sgd":
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    else:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    # ---- FIX: guard against zero-length warmup phase (hits newer PyTorch with small FAST_DEV data) ----
    _total = max(epochs * len(train_data), 1)
    _pct   = min(max(0.2, 2.0 / _total), 0.9)   # warmup gets >=2 steps -> no ZeroDivisionError
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=_total, pct_start=_pct)
    # ----------------------------------------------------------------------------------------------------
    scaler = _make_scaler(use_amp)
    logs, t_start = [], time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for m in extra_modules:
            m.train()
        t0 = time.time()
        loss_sum = torch.zeros((), device=DEVICE)
        correct = torch.zeros((), device=DEVICE)
        seen = 0
        for xb, yb in train_data:
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
                loss, logits = step_fn(model, xb, yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.detach() * len(yb)
            correct += (logits.detach().argmax(1) == yb).sum()
            seen += len(yb)
        test_acc = evaluate(model, test_data, use_amp)
        row = {"epoch": ep, "train_loss": round(loss_sum.item() / seen, 4),
               "train_acc": round(correct.item() / seen, 4), "test_acc": round(test_acc, 4),
               "epoch_time_s": round(time.time() - t0, 1)}
        logs.append(row)
        if verbose:
            print(f"[{name}] epoch {ep:>2}/{epochs}  loss {row['train_loss']:.3f}  "
                  f"train_acc {row['train_acc']:.3f}  eval_acc {row['test_acc']:.3f}  ({row['epoch_time_s']}s)", flush=True)
    if log_path:
        save_csv(logs, log_path)
    return logs, time.time() - t_start


# ----------------------------------------------------------------------------- files
def save_csv(rows, path):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_ckpt(model, path, half=False):
    sd = {k: (v.half() if (half and v.is_floating_point()) else v).cpu() for k, v in model.state_dict().items()}
    torch.save(sd, path)


def load_ckpt(model, path):
    sd = torch.load(path, map_location="cpu")
    sd = {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}
    model.load_state_dict(sd)
    return model


def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def file_mb(path):
    return os.path.getsize(path) / 1e6


def state_dict_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


# ----------------------------------------------------------------------------- ONNX / ONNX Runtime
def export_onnx(model, pre, path, opset=17):
    wrapper = Deployable(copy.deepcopy(model), copy.deepcopy(pre)).cpu().eval()
    wrapper.to(memory_format=torch.contiguous_format)
    try:
        import timm
        timm.layers.set_exportable(True)
    except Exception:
        pass
    dummy = torch.rand(1, 3, 32, 32)
    common = dict(input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=opset)
    last_err = None
    for extra in ({"dynamo": False}, {}, {"dynamo": True}):
        try:
            torch.onnx.export(wrapper, dummy, path, **common, **extra)
            return wrapper
        except Exception as e:
            last_err = e
    raise RuntimeError(f"ONNX export failed: {last_err}")


def ort_session(path, threads=1):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def ort_logits(sess, x_uint8, bs=200):
    outs = []
    name = sess.get_inputs()[0].name
    for i in range(0, len(x_uint8), bs):
        xb = (x_uint8[i:i + bs].numpy().astype(np.float32)) / 255.0
        outs.append(sess.run(None, {name: xb})[0])
    return np.concatenate(outs)


def timeit_ms(fn, warmup=20, runs=100):
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000)
    t = np.array(t)
    return {"mean_ms": float(t.mean()), "p50_ms": float(np.percentile(t, 50)),
            "p95_ms": float(np.percentile(t, 95)), "std_ms": float(t.std())}
