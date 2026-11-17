"""hardware.py - REAL measurements of a candidate on the CPU + the deployment constraints.
  * latency  : median time of one forward pass (batch 1, 1 CPU thread), measured with a timer
  * size     : size in MB of the saved model weights (float32)
  * peak mem : estimated peak memory during inference = weights + the largest layer input+output (batch 1)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import config
import common as C
from search_space import build_model


def estimate_peak_mem_mb(net, dummy):
    records, hooks = [], []

    def hook(mod, inp, out):
        n_in = inp[0].numel() if len(inp) and isinstance(inp[0], torch.Tensor) else 0
        n_out = out.numel() if isinstance(out, torch.Tensor) else 0
        records.append(n_in + n_out)

    for m in net.modules():
        if len(list(m.children())) == 0:
            hooks.append(m.register_forward_hook(hook))
    with torch.inference_mode():
        net(dummy)
    for h in hooks:
        h.remove()
    act_bytes = max(records) * 4
    par_bytes = sum(p.numel() for p in net.parameters()) * 4
    return (act_bytes + par_bytes) / 1e6


def measure_hardware(cfg, runs=30):
    model = build_model(cfg, config.Q2_NUM_CLASSES).eval()
    pre = C.Preprocess(config.Q2_MEAN, config.Q2_STD, cfg["resolution"])
    net = C.Deployable(model, pre).eval()                 # stays on the CPU
    dummy = torch.rand(1, 3, 32, 32)
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)                              # an edge device has very little CPU power
    try:
        with torch.inference_mode():
            lat = C.timeit_ms(lambda: net(dummy), warmup=5, runs=runs)
        peak = estimate_peak_mem_mb(net, dummy)
    finally:
        torch.set_num_threads(prev_threads)
    return {"params_M": round(C.count_params_m(model), 4), "size_mb": round(C.state_dict_mb(model), 4),
            "latency_ms": round(lat["p50_ms"], 3), "peak_mem_mb": round(peak, 3)}


def check_constraints(hw):
    """returns a list of broken rules (empty list = architecture is allowed)"""
    bad = []
    if hw["latency_ms"] > config.MAX_LATENCY_MS:
        bad.append(f"latency {hw['latency_ms']:.2f}ms > {config.MAX_LATENCY_MS}ms")
    if hw["size_mb"] > config.MAX_MODEL_SIZE_MB:
        bad.append(f"size {hw['size_mb']:.2f}MB > {config.MAX_MODEL_SIZE_MB}MB")
    if hw["peak_mem_mb"] > config.MAX_PEAK_MEM_MB:
        bad.append(f"peak memory {hw['peak_mem_mb']:.2f}MB > {config.MAX_PEAK_MEM_MB}MB")
    return bad
