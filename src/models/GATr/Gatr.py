from os import path
import sys

# sys.path.append(
#     path.abspath("/afs/cern.ch/work/m/mgarciam/private/geometric-algebra-transformer/")
# )
sys.path.append(path.abspath("/mnt/proj3/dd-23-91/cern/geometric-algebra-transformer/"))

from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import embed_point, extract_scalar
import torch
import torch.nn as nn
from src.logger.plotting_tools import PlotCoordinates


class ExampleWrapper(torch.nn.Module):
    """Example wrapper around a GATr model.

    Expects input data that consists of a point cloud: one 3D point for each item in the data.
    Returns outputs that consists of one scalar number for the whole dataset.

    Parameters
    ----------
    blocks : int
        Number of transformer blocks
    hidden_mv_channels : int
        Number of hidden multivector channels
    hidden_s_channels : int
        Number of hidden scalar channels
    """

    def __init__(
        self, args, dev, blocks=20, hidden_mv_channels=16, hidden_s_channels=32
    ):
        super().__init__()
        self.gatr = GATr(
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=None,
            out_s_channels=None,
            hidden_s_channels=hidden_s_channels,
            num_blocks=blocks,
            attention=SelfAttentionConfig(),  # Use default parameters for attention
            mlp=MLPConfig(),  # Use default parameters for MLP
        )
        self.ScaledGooeyBatchNorm2_1 = nn.BatchNorm1d(self.input_dim, momentum=0.5)
        self.clustering = nn.Linear(16, self.output_dim - 1, bias=False)
        self.beta = nn.Linear(16, 1)

    def forward(self, g, step_count):
        """Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor with shape (*batch_dimensions, num_points, 3)
            Point cloud input data

        Returns
        -------
        outputs : torch.Tensor with shape (*batch_dimensions, 1)
            Model prediction: a single scalar for the whole point cloud.
        """
        inputs = g.ndata["pos_hits_xyz"]
        inputs = self.ScaledGooeyBatchNorm2_1(inputs)
        inputs = inputs.unsqueeze(0)
        # Embed point cloud in PGA
        embedded_inputs = embed_point(inputs).unsqueeze(-2)  # (..., num_points, 1, 16)

        # Pass data through GATr
        embedded_outputs, _ = self.gatr(
            embedded_inputs, scalars=None
        )  # (..., num_points, 1, 16)

        # Extract scalar and aggregate outputs from point cloud
        # nodewise_outputs = extract_scalar(embedded_outputs)  # (..., num_points, 1, 1)
        # # outputs = torch.mean(nodewise_outputs, dim=(-3, -2))  # (..., 1)
        embedded_outputs = embedded_outputs.view(-1, 16)
        x_cluster_coord = self.clustering(x)
        beta = self.beta(x)
        g.ndata["final_cluster"] = x_cluster_coord
        g.ndata["beta"] = beta.view(-1)
        if step_count % 5:
            PlotCoordinates(
                g,
                path="final_clustering",
                outdir=self.args.model_prefix,
                predict=self.args.predict,
            )
        x = torch.cat((x_cluster_coord, beta.view(-1, 1)), dim=1)
        return x
