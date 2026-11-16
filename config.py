"""
config.py - ONE place for every setting of the assignment.
Change numbers here (not inside the other scripts).
FAST_DEV=1 (environment variable) = tiny smoke test: 1 epoch, small data, outputs go to _dev_outputs/.
"""
import os

FAST_DEV = os.environ.get("FAST_DEV", "0") == "1"
SEED = 42

# ------------------------------------------------------------------ Q1 : knowledge distillation (CIFAR-100)
Q1_NUM_CLASSES = 100
Q1_IMG_SIZE = 96                 # CIFAR images (32x32) are enlarged to 96x96 before entering the networks
Q1_MEAN = (0.485, 0.456, 0.406)  # ImageNet statistics (we start from ImageNet-pretrained weights)
Q1_STD = (0.229, 0.224, 0.225)
Q1_BATCH = 256
TEACHER_EPOCHS = 15
STUDENT_EPOCHS = 20
STUDENTS = ["mobilenetv2", "squeezenet", "efficientnet_lite0"]
PRETRAINED = True                # same ImageNet start for teacher, baseline students and KD students
WEIGHT_DECAY = 5e-4
LR = {"resnet50": 0.05, "mobilenetv2": 0.05, "squeezenet": 0.01, "efficientnet_lite0": 0.03}

KD_TEMPERATURE = 4.0             # softens teacher/student probabilities
KD_ALPHA = 0.7                   # weight of soft-label loss  (1-alpha = weight of hard-label loss)
KD_FEATURE_BETA = 2.0            # weight of feature-based distillation loss

# ------------------------------------------------------------------ Q2 : hardware-aware NAS (CIFAR-10)
Q2_NUM_CLASSES = 10
Q2_MEAN = (0.4914, 0.4822, 0.4465)
Q2_STD = (0.2470, 0.2435, 0.2616)
Q2_BATCH = 256
NAS_VAL_SIZE = 5000              # part of the CIFAR-10 train set kept aside for validation
NAS_SEARCH_TRAIN = 20000         # images used to train every candidate during the search
NAS_CANDIDATE_EPOCHS = 5
NAS_POPULATION = 12
NAS_CYCLES = 48                  # total trained candidates = population + cycles
NAS_TOURNAMENT = 4
NAS_FINAL_EPOCHS = 40
NAS_LR = 3e-3                    # AdamW + one-cycle schedule for candidates and for the final model
NAS_WD = 0.01

# deployment constraints (measured on CPU, 1 thread, batch size 1)
MAX_LATENCY_MS = 8.0
MAX_MODEL_SIZE_MB = 2.0
MAX_PEAK_MEM_MB = 20.0
SCORE_LATENCY_WEIGHT = 0.05      # score = val_acc - w1*(latency/max) - w2*(size/max)
SCORE_SIZE_WEIGHT = 0.05

# ------------------------------------------------------------------ FAST_DEV overrides (smoke test only)
if FAST_DEV:
    TEACHER_EPOCHS = 1
    STUDENT_EPOCHS = 1
    NAS_VAL_SIZE = 500
    NAS_SEARCH_TRAIN = 1500
    NAS_CANDIDATE_EPOCHS = 1
    NAS_POPULATION = 3
    NAS_CYCLES = 3
    NAS_FINAL_EPOCHS = 1
    DEV_TRAIN_SUBSET = 2000
    DEV_TEST_SUBSET = 1000
