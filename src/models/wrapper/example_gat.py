import torch
from src.models.gravnet_model import GravnetModel
from src.models.gatconv import GATConv

class GATWrapper(torch.nn.Module):
    def __init__(self, dev, **kwargs) -> None:
        super().__init__()
        self.emb_mlp = torch.nn.Linear(9, 128).to(dev)
        self.emb_out = torch.nn.Linear(128, 4).to(dev)
        self.mod = GATConv(in_channels=128, out_channels=128, heads=3, concat=False, **kwargs).to(dev)
        self.mod.input_dim = 9
        self.mod.output_dim = 4   # to be used by the loss model
        self.mod.clust_space_norm = "none"
        self.mod.post_pid_pool_module = torch.nn.Identity()
    def forward(self, g):
        x = g.ndata["h"]
        x = self.emb_mlp(x)
        edge_index = torch.stack(g.edges())
        x_new = self.mod(x=x, edge_index=edge_index) + x
        return self.emb_out(x_new)


def get_model(data_config, dev, **kwargs):
    print("Model options: ", kwargs)
    model = GATWrapper(dev, **kwargs)

    model_info = {
        "input_names": list(data_config.input_names),
        "input_shapes": {
            k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()
        },
        "output_names": ["softmax"],
        "dynamic_axes": {
            **{k: {0: "N", 2: "n_" + k.split("_")[0]} for k in data_config.input_names},
            **{"softmax": {0: "N"}},
        },
    }

    return model, model_info


def get_loss(data_config, **kwargs):

    return torch.nn.MSELoss()
