"""compare_results.py - builds the table: student accuracy WITHOUT vs WITH distillation."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import common as C

rows = []
t = os.path.join(C.RESULTS_DIR, "q1_teacher_metrics.json")
if os.path.exists(t):
    m = C.load_json(t)
    rows.append({"model": "resnet50 (teacher)", "params_M": m["params_M"], "baseline_acc": "", "kd_acc": "",
                 "final_acc": m["final_test_acc"], "kd_gain": ""})
for n in config.STUDENTS:
    b = os.path.join(C.RESULTS_DIR, f"q1_baseline_{n}_metrics.json")
    k = os.path.join(C.RESULTS_DIR, f"q1_kd_{n}_metrics.json")
    if os.path.exists(b) and os.path.exists(k):
        mb, mk = C.load_json(b), C.load_json(k)
        rows.append({"model": n, "params_M": mb["params_M"], "baseline_acc": mb["final_test_acc"],
                     "kd_acc": mk["final_test_acc"], "final_acc": "",
                     "kd_gain": round(mk["final_test_acc"] - mb["final_test_acc"], 4),
                     "baseline_train_min": mb["train_time_min"], "kd_train_min": mk["train_time_min"]})
C.save_csv(rows, os.path.join(C.RESULTS_DIR, "q1_baseline_vs_kd.csv"))
for r in rows:
    print(r)
