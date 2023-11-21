import torch
import os.path as osp
import tqdm
import sys
import numpy as np

sys.path.append("/afs/cern.ch/work/m/mgarciam/private/mlpf/")
from src.dataset.dataset import SimpleIterDataset
from src.utils.utils import to_filelist
from torch.utils.data import DataLoader
import dgl
import uproot
import plotly
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
import networkx as nx
import xml.etree.ElementTree as ET
from dash import Dash, dcc, html


class Args:
    def __init__(self):
        self.data_train = [
            "/eos/user/m/mgarciam/datasets_mlpf/230923_20_25/pf_tree_2.root",
        ]
        self.data_val = [
            "/eos/user/m/mgarciam/datasets_mlpf/230923_20_25/pf_tree_2.root",
        ]
        # self.data_train = files_train
        self.data_config = "/afs/cern.ch/work/m/mgarciam/private/mlpf/config_files/config_2_newlinks.yaml"
        self.extra_selection = None
        self.train_val_split = 0.8
        self.data_fraction = 1
        self.file_fraction = 1
        self.fetch_by_files = False
        self.fetch_step = 0.01
        self.steps_per_epoch = None
        self.in_memory = False
        self.local_rank = None
        self.copy_inputs = False
        self.no_remake_weights = False
        self.batch_size = 10
        self.num_workers = 0
        self.demo = False
        self.laplace = False
        self.diffs = False
        self.class_edges = False


args = Args()
train_range = (0, args.train_val_split)
train_file_dict, train_files = to_filelist(args, "train")
train_data = SimpleIterDataset(
    train_file_dict,
    args.data_config,
    for_training=True,
    extra_selection=args.extra_selection,
    remake_weights=True,
    load_range_and_fraction=(train_range, args.data_fraction),
    file_fraction=args.file_fraction,
    fetch_by_files=args.fetch_by_files,
    fetch_step=args.fetch_step,
    infinity_mode=False,
    in_memory=args.in_memory,
    async_load=False,
    name="train",
)
iterator = iter(train_data)
# extract E_cal and H_cal dimentions for plotting
tree = ET.parse("/afs/cern.ch/work/m/mgarciam/private/mlpf/CLIC_o3_v14.xml")
root = tree.getroot()
for constant in root.findall(".//constant"):
    if constant.get("name") == "ECalBarrel_inner_radius":
        Ecal_Barrel_inner_radius = constant.get("value")
    if constant.get("name") == "ECalBarrel_outer_radius":
        Ecal_Barrel_outer_radius = constant.get("value")
    if constant.get("name") == "ECalBarrel_half_length":
        Ecal_Barrel_half_length = constant.get("value")
    if constant.get("name") == "HCalBarrel_inner_radius":
        HCal_Barrel_inner_radius = constant.get("value")
    if constant.get("name") == "HCalBarrel_outer_radius":
        HCal_Barrel_outer_radius = constant.get("value")
    if constant.get("name") == "HCalBarrel_half_length":
        HCal_Barrel_half_length = constant.get("value")
# remove '*mm' from values
Ecal_Barrel_inner_radius = int(Ecal_Barrel_inner_radius.replace("*mm", ""))
Ecal_Barrel_outer_radius = int(Ecal_Barrel_outer_radius.replace("*mm", ""))
Ecal_Barrel_half_length = int(Ecal_Barrel_half_length.replace("*mm", ""))
HCal_Barrel_inner_radius = int(HCal_Barrel_inner_radius.replace("*mm", ""))
HCal_Barrel_outer_radius = int(HCal_Barrel_outer_radius.replace("*mm", ""))
HCal_Barrel_half_length = int(HCal_Barrel_half_length.replace("*mm", ""))
# Define variables for plotting the E_cal and H_cal
resolution = 50
theta = np.linspace(0, 2 * np.pi, resolution)
E_height = Ecal_Barrel_half_length * 2
H_height = HCal_Barrel_half_length * 2
E_z = np.linspace(0, E_height, resolution)
H_z = np.linspace(0, H_height, resolution)
theta, E_z = np.meshgrid(theta, E_z)
theta, H_z = np.meshgrid(theta, H_z)
E_x_outer = Ecal_Barrel_outer_radius * np.cos(theta)
E_y_outer = Ecal_Barrel_outer_radius * np.sin(theta)
E_x_inner = Ecal_Barrel_inner_radius * np.cos(theta)
E_y_inner = Ecal_Barrel_inner_radius * np.sin(theta)
H_x_outer = HCal_Barrel_outer_radius * np.cos(theta)
H_y_outer = HCal_Barrel_outer_radius * np.sin(theta)
H_x_inner = HCal_Barrel_inner_radius * np.cos(theta)
H_y_inner = HCal_Barrel_inner_radius * np.sin(theta)
# define color for the H_cal and E_cal
h_colorscale = [[0, "rgb(255, 0, 0)"], [1, "rgb(255, 0, 0)"]]
e_colorscale = [[0, "rgb(255, 255, 0)"], [1, "rgb(255, 255, 0)"]]
# Show first 10 events
number_of_plots = 10
for i in range(number_of_plots):
    g, gt = next(iterator)
    list_graph = dgl.unbatch(g)
    number_of_iteration = list_graph[0]
    particle_id = number_of_iteration.ndata["particle_number"]
    hit_type = number_of_iteration.ndata["hit_type"]
    pos = number_of_iteration.ndata["pos_hits_xyz"]
    pos_particles = gt[:, 0:3]
    e_particle = gt[:, 3]
    # change symbol by hit_type
    marker_symbols = ["circle", "circle-open", "cross", "diamond"]
    max_hit_type = hit_type.argmax(1)
    ms = [marker_symbols[i.item()] for i in max_hit_type]
    # 10 plots particles and hits
    fig = make_subplots(rows=10, cols=1)
    # fig plot particles with size of particle energy color particle_id, shape hit_type
    fig.add_trace(
        go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            mode="markers",
            marker=dict(size=e_particle, color=particle_id, symbol=ms),
            name="particle",
            text=e_particle,
        )
    )
    # fig plot hits with size of particle energy color particle_id, shape hit_type
    fig.add_trace(
        go.Scatter3d(
            x=pos_particles[:, 0],
            y=pos_particles[:, 1],
            z=pos_particles[:, 2],
            mode="markers",
            marker=dict(size=e_particle, color=particle_id, symbol=ms),
            name="hits",
            text=e_particle,
        )
    )

    # plot H_cal and E_cal
    fig2 = go.Figure(
        data=[
            go.Surface(
                z=H_x_outer,
                y=H_y_outer,
                x=H_z,
                opacity=0.3,
                colorscale=h_colorscale,
                name="HCAL",
            ),
            go.Surface(
                z=H_x_inner,
                y=H_y_inner,
                x=H_z,
                opacity=0.3,
                colorscale=h_colorscale,
                name="HCAL",
            ),
            go.Surface(
                z=E_x_outer,
                y=E_y_outer,
                x=E_z,
                opacity=0.2,
                colorscale=e_colorscale,
                name="ECAL",
            ),
            go.Surface(
                z=E_x_inner,
                y=E_y_inner,
                x=E_z,
                opacity=0.2,
                colorscale=e_colorscale,
                name="ECAL",
            ),
        ]
    )
    fig2.update_layout(
        scene=dict(
            aspectmode="manual",
            # aspectratio=dict(x=1, y=1, z=1),
            xaxis_title="X mm ",
            yaxis_title="Y mm",
            zaxis_title="Z mm",
        )
    )
    # add first plot to second
    for trace in fig2.data:
        fig.add_trace(trace)

    # update plot size
    fig.update_layout(
        height=700,
        width=700,
        showlegend=True,
        autosize=True,
        # margin = dict( t= 0, b= 0, l=0, r= 0),
        template="plotly_white",
    )
    # 3D scene options
    fig.update_scenes(aspectratio=dict(x=1, y=1, z=1), aspectmode="manual")
    # dropdown menu for hits and particle_id
    button_layer_1_height = 1
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list(
                    [
                        dict(
                            label="Plots",
                            method="update",
                            args=[
                                {"visible": [True, True, True, True, True, True]},
                                {"title": "Plots"},
                            ],
                        ),
                        dict(
                            label="particle_id",
                            method="update",
                            args=[
                                {"visible": [True, False, False, False, False, False]},
                                {"title": "particles"},
                            ],
                        ),
                        dict(
                            label="hits",
                            method="update",
                            args=[
                                {"visible": [False, True, False, False, False, False]},
                                {"title": "hits"},
                            ],
                        ),
                        dict(
                            label="E_cal",
                            method="update",
                            args=[
                                {"visible": [True, True, True, True, False, False]},
                                {"title": "E_cal"},
                            ],
                        ),
                        dict(
                            label="H_cal",
                            method="update",
                            args=[
                                {"visible": [True, True, False, False, True, True]},
                                {"title": "H_cal"},
                            ],
                        ),
                    ]
                )
            )
        ]
    )
    fig.update_layout()
    pyo.plot(fig, filename="first10eventsplot.html")
