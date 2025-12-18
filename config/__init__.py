# config/__init__.py

from .audio import SAMPLE_RATE, FRAME_LENGTH, FRAME_STEP, N_FRAMES, N_MELS
from .gnn import (
    BATCH_SIZE,
    N_DILATION_LAYERS,
    REDUCED_NODE_REP_BOOL,
    REDUCED_NODE_REP_K,
    WINDOW_SIZE_SIMPLE,
)
from .path import SCRIPT_DIR
