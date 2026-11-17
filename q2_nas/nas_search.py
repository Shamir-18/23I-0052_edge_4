"""nas_search.py - hardware-aware EVOLUTIONARY architecture search on CIFAR-10 ("regularized evolution").

How it works (in simple words)
  1. Create random architectures. Measure their real CPU speed + size. Throw away the ones that break the limits.
  2. Train every allowed architecture for a few epochs, give it a score (accuracy, minus a small penalty for slow/big).
  3. Repeat: pick the best of a few random population members (tournament), make a slightly changed copy (mutation),
     check limits, train, score. The oldest member leaves the population.
  4. Save the full search history (CSV/JSON) and the best architecture (JSON).
Run:  python q2_nas/nas_search.py
"""
import os, sys, time, random, json, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

import config
import common as C
from search_space import random_config, mutate, config_key, build_model
from hardware import measure_hardware, check_constraints


def score_of(acc, hw):
    return (acc - config.SCORE_LATENCY_WEIGHT * hw["latency_ms"] / config.MAX_LATENCY_MS
            - config.SCORE_SIZE_WEIGHT * hw["size_mb"] / config.MAX_MODEL_SIZE_MB)


def train_candidate(cfg, train_xy, val_xy):
    C.set_seed()                                           # same seed for every candidate -> fair comparison
    pre = C.Preprocess(config.Q2_MEAN, config.Q2_STD, cfg["resolution"])
    train = C.GPUData(train_xy[0], train_xy[1], pre, config.Q2_BATCH, train=True)
    val = C.GPUData(val_xy[0], val_xy[1], pre, config.Q2_BATCH, train=False)
    model = build_model(cfg, config.Q2_NUM_CLASSES).to(C.DEVICE).to(memory_format=torch.channels_last)

    def step(m, x, y):
        logits = m(x).float()
        return F.cross_entropy(logits, y), logits

    logs, secs = C.fit("candidate", model, step, train, val, config.NAS_CANDIDATE_EPOCHS,
                       config.NAS_LR, config.NAS_WD, optimizer="adamw", verbose=False)
    del model, train, val
    return logs[-1]["test_acc"], secs                      # ("test_acc" here = accuracy on the VALIDATION split)


def main():
    t_all = time.time()
    rng = random.Random(config.SEED)
    xtr, ytr, _, _, _ = C.load_cifar("cifar10")
    xt, yt, xv, yv = C.split_train_val(xtr, ytr, config.NAS_VAL_SIZE)
    train_xy = (xt[:config.NAS_SEARCH_TRAIN], yt[:config.NAS_SEARCH_TRAIN])
    val_xy = (xv, yv)
    print(f"Search data: {len(train_xy[0])} train / {len(xv)} validation images | device {C.DEVICE}")

    history, seen = [], set()
    population = collections.deque()
    budget = config.NAS_POPULATION + config.NAS_CYCLES
    state = {"trained": 0, "attempts": 0, "calls": 0}
    max_calls = budget * 60
    hist_csv = os.path.join(C.RESULTS_DIR, "nas_search_history.csv")

    def evaluate_candidate(cfg, origin):
        state["calls"] += 1
        key = config_key(cfg)
        if key in seen:
            return None
        seen.add(key)
        state["attempts"] += 1
        hw = measure_hardware(cfg)                          # REAL CPU latency + size + memory
        broken = check_constraints(hw)
        row = {"id": state["attempts"], "origin": origin, "status": "rejected" if broken else "trained",
               "reject_reason": "; ".join(broken), "resolution": cfg["resolution"],
               "num_stages": len(cfg["stages"]), "total_blocks": sum(s["blocks"] for s in cfg["stages"]),
               **hw, "val_acc": "", "score": "", "train_time_s": "", "config": json.dumps(cfg)}
        if broken:                                          # hardware-aware: never even trained
            history.append(row)
            C.save_csv(history, hist_csv)
            return None
        acc, secs = train_candidate(cfg, train_xy, val_xy)
        state["trained"] += 1
        sc = score_of(acc, hw)
        row.update(val_acc=round(acc, 4), score=round(sc, 4), train_time_s=round(secs, 1))
        history.append(row)
        C.save_csv(history, hist_csv)
        n_rej = sum(1 for r in history if r["status"] == "rejected")
        print(f"[{state['trained']:>3}/{budget}] {origin:<8} res={cfg['resolution']} stages={row['num_stages']} "
              f"acc={acc:.3f} score={sc:.3f} | lat={hw['latency_ms']:.2f}ms size={hw['size_mb']:.2f}MB "
              f"mem={hw['peak_mem_mb']:.1f}MB | rejected so far: {n_rej}", flush=True)
        return {"cfg": cfg, "score": sc, "acc": acc}

    # 1) first population: random architectures
    while len(population) < config.NAS_POPULATION and state["calls"] < max_calls:
        ind = evaluate_candidate(random_config(rng), "random")
        if ind:
            population.append(ind)
    if not population:
        raise RuntimeError("Every random architecture broke the constraints - relax the limits in config.py")

    # 2) evolution: tournament -> mutation -> constraint check -> train -> replace oldest
    while state["trained"] < budget and state["calls"] < max_calls:
        pool = rng.sample(list(population), min(config.NAS_TOURNAMENT, len(population)))
        parent = max(pool, key=lambda p: p["score"])
        ind = evaluate_candidate(mutate(parent["cfg"], rng), "mutation")
        if ind:
            population.append(ind)
            if len(population) > config.NAS_POPULATION:
                population.popleft()

    trained_rows = [r for r in history if r["status"] == "trained"]
    best = max(trained_rows, key=lambda r: r["score"])
    best_cfg = json.loads(best["config"])
    C.save_json({"config": best_cfg, "search_val_acc": best["val_acc"], "score": best["score"],
                 "hardware": {k: best[k] for k in ("params_M", "size_mb", "latency_ms", "peak_mem_mb")},
                 "constraints": {"max_latency_ms": config.MAX_LATENCY_MS, "max_size_mb": config.MAX_MODEL_SIZE_MB,
                                 "max_peak_mem_mb": config.MAX_PEAK_MEM_MB}},
                os.path.join(C.RESULTS_DIR, "nas_best_config.json"))
    C.save_json(history, os.path.join(C.RESULTS_DIR, "nas_search_history.json"))
    n_rej = sum(1 for r in history if r["status"] == "rejected")
    print(f"\nSEARCH DONE in {(time.time() - t_all) / 60:.1f} min | trained {len(trained_rows)} architectures, "
          f"rejected {n_rej} for breaking the hardware limits (never trained).")
    print("BEST:", json.dumps(best_cfg))
    print({k: best[k] for k in ("val_acc", "score", "params_M", "size_mb", "latency_ms", "peak_mem_mb")})


if __name__ == "__main__":
    main()
