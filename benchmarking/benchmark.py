"""benchmark.py - automatic benchmark of every exported ONNX model on the CPU (ONNX Runtime).
Measures: file size, parameters, test accuracy, latency (batch 1, 1 thread: mean/p50/p95), throughput (batch 64),
and PEAK MEMORY (each model runs in its own fresh process; peak RSS minus the empty-process baseline).
Run:  python benchmarking/benchmark.py        -> results/benchmark_results.csv"""
import os, sys, json, subprocess

# ---- child mode: runs in a separate light process so the memory number is clean (no torch imported here)
if __name__ == "__main__" and ("--single" in sys.argv or "--baseline" in sys.argv):
    import resource
    import numpy as np
    import onnxruntime as ort
    if "--single" in sys.argv:
        path = sys.argv[sys.argv.index("--single") + 1]
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = 1, 1
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        x = np.random.rand(1, 3, 32, 32).astype(np.float32)
        name = sess.get_inputs()[0].name
        for _ in range(30):
            sess.run(None, {name: x})
    print(json.dumps({"maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}))
    sys.exit(0)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import numpy as np
import config
import common as C

THIS = os.path.abspath(__file__)


def child_rss(*args):
    out = subprocess.run([sys.executable, THIS, *args], capture_output=True, text=True).stdout.strip().splitlines()
    return json.loads(out[-1])["maxrss_mb"]


def count_params_m(path):
    try:
        import onnx
        m = onnx.load(path)
        return round(sum(int(np.prod(t.dims)) for t in m.graph.initializer) / 1e6, 3)
    except Exception:
        return ""


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(C.ONNX_DIR, "q*.onnx")))
    assert files, "No ONNX files found - run the export scripts first."
    baseline = min(child_rss("--baseline") for _ in range(3))
    data, rows = {}, []
    for f in files:
        base = os.path.basename(f)
        ds = {"q1": "cifar100", "q2": "cifar10"}[base[:2]]
        if ds not in data:
            _, _, xte, yte, _ = C.load_cifar(ds)
            data[ds] = (xte, yte)
        xte, yte = data[ds]
        sess1 = C.ort_session(f, threads=1)
        name = sess1.get_inputs()[0].name
        acc = float((C.ort_logits(sess1, xte).argmax(1) == yte.numpy()).mean())
        one = xte[:1].numpy().astype(np.float32) / 255.0
        lat = C.timeit_ms(lambda: sess1.run(None, {name: one}), warmup=30, runs=200)
        b64 = xte[:64].numpy().astype(np.float32) / 255.0
        sessn = C.ort_session(f, threads=os.cpu_count() or 1)
        tb = C.timeit_ms(lambda: sessn.run(None, {name: b64}), warmup=3, runs=10)
        peak = max(0.0, child_rss("--single", f) - baseline)
        rows.append({"model": base, "dataset": ds, "size_mb": round(C.file_mb(f), 3), "params_M": count_params_m(f),
                     "test_acc": round(acc, 4), "latency_mean_ms": round(lat["mean_ms"], 3),
                     "latency_p50_ms": round(lat["p50_ms"], 3), "latency_p95_ms": round(lat["p95_ms"], 3),
                     "throughput_img_per_s_bs64": round(64 / (tb["p50_ms"] / 1000), 1),
                     "peak_inference_mem_mb": round(peak, 1)})
        print(rows[-1], flush=True)
    C.save_csv(rows, os.path.join(C.RESULTS_DIR, "benchmark_results.csv"))
    print("\nSaved benchmark_results.csv | CPU threads for throughput test:", os.cpu_count())
