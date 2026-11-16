"""report_figures.py - makes the pictures for the PDF report (saved in results/). Run after everything else."""
import os, json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import common as C

R = C.RESULTS_DIR


def p(name):
    return os.path.join(R, name)


# 1) Q1: accuracy without / with distillation
if os.path.exists(p("q1_baseline_vs_kd.csv")):
    df = pd.read_csv(p("q1_baseline_vs_kd.csv"))
    df = df[df["baseline_acc"].notna()]
    if len(df):
        ax = df.set_index("model")[["baseline_acc", "kd_acc"]].mul(100).plot(kind="bar", figsize=(7, 4))
        ax.set_ylabel("CIFAR-100 test accuracy (%)")
        ax.set_title("Students: without vs with knowledge distillation")
        plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(p("fig_q1_kd_vs_baseline.png"), dpi=150); plt.close()

# 2) Q1: training curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4)); any_curve = False
for ax, n in zip(axes, config.STUDENTS):
    for v in ("baseline", "kd"):
        f = p(f"q1_{v}_{n}_log.csv")
        if os.path.exists(f):
            d = pd.read_csv(f); ax.plot(d["epoch"], d["test_acc"] * 100, label=v); any_curve = True
    ax.set_title(n); ax.set_xlabel("epoch"); ax.set_ylabel("test acc (%)"); ax.legend()
if any_curve:
    plt.tight_layout(); plt.savefig(p("fig_q1_training_curves.png"), dpi=150)
plt.close()

# 3) Q2: search results
if os.path.exists(p("nas_search_history.csv")):
    h = pd.read_csv(p("nas_search_history.csv"))
    t, rj = h[h["status"] == "trained"], h[h["status"] == "rejected"]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    sc = ax[0].scatter(t["latency_ms"], t["val_acc"] * 100, c=t["size_mb"], cmap="viridis")
    plt.colorbar(sc, ax=ax[0], label="model size (MB)")
    ax[0].axvline(config.MAX_LATENCY_MS, color="red", ls="--", label="latency limit")
    ax[0].set_xlabel("CPU latency (ms)"); ax[0].set_ylabel("validation accuracy (%)")
    ax[0].set_title("Trained candidates"); ax[0].legend()
    ax[1].plot(range(1, len(t) + 1), t["score"].cummax(), marker="o", ms=3)
    ax[1].set_xlabel("trained candidate #"); ax[1].set_ylabel("best score so far"); ax[1].set_title("Search progress")
    ax[2].bar(["trained", "rejected\n(constraints)"], [len(t), len(rj)], color=["tab:green", "tab:red"])
    ax[2].set_title("Hardware-aware filtering")
    plt.tight_layout(); plt.savefig(p("fig_q2_nas_search.png"), dpi=150); plt.close()

# 4) Benchmark
if os.path.exists(p("benchmark_results.csv")):
    b = pd.read_csv(p("benchmark_results.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for a, col, ttl in zip(ax, ["latency_p50_ms", "size_mb", "test_acc"],
                           ["CPU latency p50 (ms, 1 thread)", "ONNX file size (MB)", "test accuracy"]):
        a.barh(b["model"], b[col]); a.set_title(ttl)
    plt.tight_layout(); plt.savefig(p("fig_benchmark.png"), dpi=150); plt.close()
print("figures saved in", R, [f for f in os.listdir(R) if f.endswith(".png")])
