from pathlib import Path
import torch
# -----------------------
# Project Paths
# -----------------------

ROOT_DIR = Path(__file__).resolve().parent

DATASET_DIR = ROOT_DIR / "dataset"

CLEAN_DIR = DATASET_DIR / "clean"

DAMAGED_DIR = DATASET_DIR / "damaged"

OUTPUT_DIR = ROOT_DIR / "outputs"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

RESULTS_DIR = OUTPUT_DIR / "results"

# -----------------------
# Image Parameters
# -----------------------

IMAGE_SIZE = 256

CHANNELS = 3

# -----------------------
# Training Parameters
# -----------------------

BATCH_SIZE = 1

NUM_EPOCHS = 100

LEARNING_RATE = 2e-4

BETA1 = 0.5

BETA2 = 0.999

NUM_WORKERS = 0

# -----------------------
# Device
# -----------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"