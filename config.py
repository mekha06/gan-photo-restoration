from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent

DATASET_DIR = ROOT_DIR / "dataset"

CLEAN_DIR = Path("/kaggle/input/datasets/denislukovnikov/ffhq256-images-only/ffhq256")

DAMAGED_DIR = DATASET_DIR / "damaged"

OUTPUT_DIR = ROOT_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
SAMPLES_DIR = OUTPUT_DIR / "samples"
RESULTS_DIR = OUTPUT_DIR / "results"

IMAGE_SIZE = 256
CHANNELS = 3

BATCH_SIZE = 8
NUM_EPOCHS = 100

LEARNING_RATE = 2e-4
BETA1 = 0.5
BETA2 = 0.999

NUM_WORKERS = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"