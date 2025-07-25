import argparse
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MLPF_development/mlpf/')))

#Temporary fix!######
from onnxscript.function_libs.torch_lib.ops import nn as _nn
from onnxscript import opset18 as op

def _gelu_tanh_fix(x):
    # float constant as scalar with value_float
    three = op.Constant(value_float=3.0)
    coeff = op.Constant(value_float=0.044715)
    return op.Mul(op.Pow(x, three), coeff) + x

def _gelu_none_fix(x):
    sqrt2 = op.Constant(value_float=1.4142135)
    return op.Mul(op.Erf(op.Div(x, sqrt2)), x)

_nn._aten_gelu_approximate_tanh = _gelu_tanh_fix
_nn._aten_gelu_approximate_none = _gelu_none_fix



##################################################################################

from src.models.GATr.Gatr_pf_e_noise_onnx import ExampleWrapper as GravnetModel

parser = argparse.ArgumentParser()
#parser arguments
parser.add_argument("-m","--outputpath",type=str, default="/home/lherrman/output/",help="path to output directory")
parser.add_argument("--correction",action="store_true",default=False,help="Train correction only",)
    
args = parser.parse_args()
filepath = args.outputpath + "clustering_1.onnx"
model_weights = "_epoch=4_step=57500.ckpt"
# args1 = (torch.randn((10, 3)), torch.randn((10, 1)), torch.randn((10, 3)))
torch._dynamo.config.verbose = True
model = GravnetModel.load_from_checkpoint(
                    model_weights,
                    args=args,
                    dev='cpu',  
                    map_location=torch.device("cpu")  
                )
model.eval()

pos_hits_xyz = torch.randn(10, 3).to("cpu")           # 3D coordinates
hit_type = torch.randint(0, 3, (10,)).to("cpu")        # random categorical types
h_scalar = torch.randn(10, 5).to("cpu")                # last 2 columns will be used (e.g., e, p)
dummy_attention_mask = torch.randn(10, 10).to("cpu") 



export_options = torch.onnx.ExportOptions(dynamic_shapes=True)

inputs = (
    pos_hits_xyz,
    hit_type,  
    h_scalar,
    dummy_attention_mask
)



#onnx_program = torch.onnx.dynamo_export(
#    model, pos_hits_xyz, hit_type, h_scalar, dummy_attention_mask, export_options=torch.onnx.ExportOptions(dynamic_shapes=True)
#)


# Export
torch.onnx.export(
    model,
    inputs,
    filepath,
    input_names=["pos_hits_xyz", "hit_type", "h_scalar", "attention_mask"],
    output_names=["output"],
    opset_version=17,  
    do_constant_folding=True,
    dynamic_axes={
        "pos_hits_xyz": {0: "batch"},
        "hit_type": {0: "batch"},
        "h_scalar": {0: "batch"},
        "attention_mask": {0: "batch", 1: "points"},
        "output": {0: "batch"}
    },
    verbose=True
)


print("filepath")
print(filepath)        
#onnx_program.save(filepath)

