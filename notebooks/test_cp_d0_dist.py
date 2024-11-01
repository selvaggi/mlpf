from src.dependencies.grid_graph.python.bin.grid_graph import edge_list_to_forward_star
from src.dependencies.parallel_cut_pursuit.python.wrappers import cp_d0_dist
import torch
from torch_geometric.data import Data
import dgl
import numpy as np

# Load data
dic = torch.load("0_0_2.pt", map_location="cpu")

# KNN
pos = dic["graph"].ndata["pos_hits_xyz"]
x = dic["graph"].ndata["h"][:, 3:]
knn_g = dgl.knn_graph(pos, 20)  # Each node has two predecessors
knn_g = dgl.remove_self_loop(knn_g)
i, j = knn_g.edges()
edge_index = torch.cat((i.view(1, -1), j.view(1, -1)), dim=0)
data = Data(pos=pos, x=x, edge_index=edge_index)

# Set parameters
regularization = 1
spatial_weight = 100
cutoff = 1
parallel = True
iterations = 10
k_adjacency = 5
verbose = False
d1 = data
n_dim = data.pos.shape[1]
n_feat = data.x.shape[1] if data.x is not None else 0
reg = regularization
cut = cutoff

# Convert edges to forward-star (or CSR) representation
source_csr, target, reindex = edge_list_to_forward_star(
    d1.num_nodes, d1.edge_index.T.contiguous().cpu().numpy()
)
source_csr = source_csr.astype("uint32")
target = target.astype("uint32")
edge_weights = (
    d1.edge_attr.cpu().numpy()[reindex] * reg if d1.edge_attr is not None else reg
)


# Recover attributes features from Data object
pos_offset = d1.pos.mean(dim=0)
if d1.x is not None:
    x = torch.cat((d1.pos - pos_offset, d1.x), dim=1)
else:
    x = d1.pos - pos_offset
x = torch.nn.functional.normalize(x, dim=1)
X1 = np.asfortranarray(x[:, 0:3].cpu().numpy().T)

coor_weights = np.ones(n_dim, dtype=np.float32)
coor_weights[:n_dim] *= spatial_weight

# run cp_d0_dist
Comp, rX, Obj, Time = cp_d0_dist.cp_d0_dist(
    0.1,
    X1,
    source_csr,
    target,
    edge_weights=edge_weights,
    cp_dif_tol=1e-6,
    K=3,
    cp_it_max=iterations,
    split_damp_ratio=0.1,
    verbose=True,
    max_num_threads=1,
    balance_parallel_split=False,
    compute_Time=True,
    compute_Obj=True,
)
