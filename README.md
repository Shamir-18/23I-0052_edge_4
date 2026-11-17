# Assignment 4 - Edge Computing & Intelligent Systems

**Q1** Knowledge distillation: ResNet-50 teacher (CIFAR-100) -> MobileNetV2 / SqueezeNet / EfficientNet-Lite0 students, compared with students trained without distillation, then exported to ONNX and run with ONNX Runtime on the CPU.
**Q2** Hardware-aware Neural Architecture Search (evolutionary search, CIFAR-10) that measures real CPU latency and model size, rejects architectures that break the limits, trains the best one from scratch and exports it to ONNX.

Everything was run in a Kaggle notebook (GPU for training, CPU for latency/ONNX measurements). The notebook writes all the `.py` files below.

## Folder structure
```
config.py                 all settings (epochs, KD temperature/alpha/beta, NAS budget, hardware limits)
common.py                 shared code: GPU data pipeline, training loop, ONNX / ONNX-Runtime / timing helpers
q1_distillation/
  teacher.py              ResNet-50 teacher definition
  students.py             MobileNetV2, SqueezeNet, EfficientNet-Lite0
  distillation.py         KD loss: soft targets + temperature + hard labels + feature-based distillation
  train_teacher.py        trains the teacher
  train_baseline.py       trains the 3 students WITHOUT distillation
  train_kd.py             trains the 3 students WITH distillation
  compare_results.py      baseline-vs-KD table
  inference.py            unified inference for all students (PyTorch)
  export_onnx.py          exports students to ONNX and verifies them
q2_nas/
  search_space.py         search space + automatic architecture generator
  hardware.py             measured CPU latency, model size, peak memory + constraints
  nas_search.py           evolutionary hardware-aware search
  train_best.py           trains the best architecture from scratch
  inference.py            inference for the NAS model (PyTorch)
  export_onnx.py          exports the NAS model to ONNX and verifies it
onnx_inference/run_onnx_cpu.py   ONNX Runtime CPU inference for every ONNX model
benchmarking/benchmark.py        automatic benchmark -> results/benchmark_results.csv
report_figures.py         creates the figures used in the report
checkpoints/              trained weights (teacher stored in fp16 to stay under GitHub's 100 MB limit)
onnx_models/              exported ONNX models (input: raw image batch (N,3,32,32) in 0..1, output: class scores)
results/                  training logs, metrics, NAS history/best config, benchmark CSV, figures
```

## Install
```bash
pip install -r requirements.txt
```
A GPU is recommended for training (`train_*.py`, `nas_search.py`). ONNX inference and benchmarking only need the CPU.

## Run (in this order, from the repository root)
```bash
# ---------- Q1
python q1_distillation/train_teacher.py
python q1_distillation/train_baseline.py
python q1_distillation/train_kd.py
python q1_distillation/compare_results.py
python q1_distillation/inference.py --all
python q1_distillation/export_onnx.py
# ---------- Q2
python q2_nas/nas_search.py
python q2_nas/train_best.py
python q2_nas/inference.py
python q2_nas/export_onnx.py
# ---------- ONNX Runtime + benchmark + figures
python onnx_inference/run_onnx_cpu.py
python benchmarking/benchmark.py
python report_figures.py
```
Predict one picture:
```bash
python q1_distillation/inference.py --model mobilenetv2 --variant kd --image my_image.jpg
python onnx_inference/run_onnx_cpu.py --model onnx_models/q2_nas_best.onnx --image my_image.jpg
```
Every training script skips work whose checkpoint already exists (use `--force` to retrain). `FAST_DEV=1 python ...` runs a tiny smoke test and writes to `_dev_outputs/`.

## Design notes
* CIFAR images are enlarged to 96x96 for Q1 (ImageNet-pretrained backbones need bigger pictures). The teacher, the baseline students and the KD students all start from the same ImageNet weights, so the only difference between baseline and KD is the teacher's help.
* KD loss = alpha * T^2 * KL(soft targets) + (1-alpha) * cross-entropy + beta * feature loss (student feature map -> 1x1 conv connector -> normalised MSE against the teacher's last feature map).
* Accuracy is reported from the LAST epoch (no picking the best epoch on the test set). Q2 uses a separate validation split for the search; the test set is only used once at the end.
* Q2 latency = median of repeated CPU runs (batch 1, 1 thread); size = saved float32 weights; peak memory during the search is an estimate (weights + largest layer input/output), the benchmark script measures the real peak process memory per ONNX model.
* Preprocessing (resize + normalisation) is inside the ONNX models, so inference only needs the raw 32x32 image.
