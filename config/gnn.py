# config/gnn.py
BATCH_SIZE = 64
# Set the number of dilation layers (i.e. k creates the undilated adjacency matrix and k-1 dilated adjacency matrices)
N_DILATION_LAYERS = 5
# Whether to reduce the node representation of the graph (i.e. pooling over the 98 frames in groups of size k)
REDUCED_NODE_REP_BOOL = False
REDUCED_NODE_REP_K = 0
# Parameters for the adjacency matrix creation
# Simple sliding window size (for 'window')
WINDOW_SIZE_SIMPLE = 5