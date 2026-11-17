"""run_onnx_cpu.py - CPU inference with ONNX Runtime for ALL exported models (Q1 students + Q2 NAS model).
  python onnx_inference/run_onnx_cpu.py                                   # test accuracy + latency of every .onnx file
  python onnx_inference/run_onnx_cpu.py --model onnx_models/q1_mobilenetv2_kd.onnx --image cat.jpg
File names start with q1_ (CIFAR-100 models) or q2_ (CIFAR-10 model)."""
import os, sys, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import common as C

DATASET_OF = {"q1": "cifar100", "q2": "cifar10"}


def dataset_for(path):
    return DATASET_OF[os.path.basename(path)[:2]]


def evaluate_model(path, cache):
    ds = dataset_for(path)
    if ds not in cache:
        _, _, xte, yte, classes = C.load_cifar(ds)
        cache[ds] = (xte, yte, classes)
    xte, yte, _ = cache[ds]
    sess = C.ort_session(path, threads=1)
    logits = C.ort_logits(sess, xte)
    acc = float((logits.argmax(1) == yte.numpy()).mean())
    one = (xte[:1].numpy().astype(np.float32)) / 255.0
    name = sess.get_inputs()[0].name
    lat = C.timeit_ms(lambda: sess.run(None, {name: one}), warmup=20, runs=100)
    return {"model": os.path.basename(path), "dataset": ds, "test_acc": round(acc, 4),
            "latency_ms_p50_1thread": round(lat["p50_ms"], 3), "latency_ms_p95_1thread": round(lat["p95_ms"], 3)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--image")
    a = ap.parse_args()
    cache = {}
    if a.image:
        from PIL import Image
        assert a.model, "give --model path.onnx together with --image"
        ds = dataset_for(a.model)
        _, _, _, _, classes = C.load_cifar(ds)
        img = Image.open(a.image).convert("RGB").resize((32, 32), Image.BILINEAR)
        x = np.array(img).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        sess = C.ort_session(a.model)
        p = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
        p = np.exp(p - p.max()); p /= p.sum()
        top = p.argsort()[::-1][:5]
        print("ONNX Runtime CPU prediction:", [(classes[i], round(float(p[i]), 3)) for i in top])
    else:
        files = [a.model] if a.model else sorted(glob.glob(os.path.join(C.ONNX_DIR, "q*.onnx")))
        rows = []
        for f in files:
            r = evaluate_model(f, cache)
            rows.append(r)
            print(r)
        C.save_csv(rows, os.path.join(C.RESULTS_DIR, "onnx_cpu_inference_summary.csv"))
