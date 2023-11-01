import dgl
import torch
import os


def create_and_store_graph_output(
    batch_g, model_output, y, local_rank, step, path_save
):
    batch_g.ndata["coords"] = model_output[:, 0:4]
    batch_g.ndata["beta"] = model_output[:, 4]
    graphs = dgl.unbatch(batch_g)
    batch_id = y[:, -1].view(-1)
    for i in range(0, len(graphs)):
        mask = batch_id == 0
        dic = {}
        dic["graph"] = graphs[i]
        dic["part_true"] = y[mask]
        torch.save(
            dic,
            path_save + "/" + str(local_rank) + "_" + str(step) + "_" + str(i) + ".pt",
        )
