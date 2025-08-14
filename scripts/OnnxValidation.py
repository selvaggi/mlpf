import argparse
import torch
import os
import sys
import onnxruntime as ort
import numpy 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MLPF_development/mlpf/')))
#lasse erst onnx laufen, dann python. das macht auch die evaluation


##################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-m","--model-prefix",type=str, default="/home/lherrman/onnxconversion/output",help="path to output directory")    
parser.add_argument("--onnx",action="store_true",help="set this flag to evaluate onnx") 
parser.add_argument("--pytorch",action="store_true",help="set this flag to evaluate pytorch") 
parser.add_argument("--evaluate",action="store_true",help="look at the outputs") 
parser.add_argument("--correction",action="store_true",default=False,help="Train correction only",)
parser.add_argument( "--gpus",type=str,default="0",help='device for the training/testing; to use CPU, set to empty string (""); to use multiple gpu, set it as a comma separated list, e.g., `1,2,3,4`',
)
args = parser.parse_args()

torch.manual_seed(42)
pos_hits_xyz = torch.randn(10, 3).to("cpu").to(torch.float32)          # 3D coordinates
hit_type = torch.randint(0, 1, (10,1)).to("cpu").to(torch.int64)        # random categorical types
h_scalar = torch.randn(10, 2).to("cpu").to(torch.float32)                # last 2 columns will be used (e.g., e, p)
# dummy_attention_mask = torch.randn(10, 10).to("cpu").to(torch.float32) 

input = torch.cat((
    pos_hits_xyz,
    hit_type,  
    h_scalar,
    # dummy_attention_mask
), dim=1)



if args.onnx:
    # output from ONNX model

    #Temporary fix!######
    from onnxscript.function_libs.torch_lib.ops import nn as _nn
    from onnxscript import opset18 as op

    # def _gelu_tanh_fix(x):
    #     # float constant as scalar with value_float
    #     three = op.Constant(value_float=3.0)
    #     coeff = op.Constant(value_float=0.044715)
    #     return op.Mul(op.Pow(x, three), coeff) + x

    # def _gelu_none_fix(x):
    #     sqrt2 = op.Constant(value_float=1.4142135)
    #     return op.Mul(op.Erf(op.Div(x, sqrt2)), x)

    # _nn._aten_gelu_approximate_tanh = _gelu_tanh_fix
    # _nn._aten_gelu_approximate_none = _gelu_none_fix
    
    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = True

    sess_options.inter_op_num_threads = 1
    sess_options.intra_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    onnx_model_path = "/afs/cern.ch/work/m/mgarciam/private/mlpf/clustering_1.onnx"
    session = ort.InferenceSession(onnx_model_path, sess_options)
    ort_inputs = {
    session.get_inputs()[0].name: input.cpu().numpy(),
    }  
    # input_names = [inp.name for inp in session.get_inputs()]

    # input_np = {
    #     input_names[0]: pos_hits_xyz.numpy().astype(numpy.float32),
    #     input_names[1]: hit_type.numpy().astype(numpy.int64),
    #     input_names[2]: h_scalar.numpy().astype(numpy.float32),
    #    # input_names[3]: dummy_attention_mask.numpy().astype(numpy.float32)
    # }

    outputs = session.run(None, ort_inputs)
    output_onnx = outputs[0]
    
    numpy.save("output_onnx.npy", output_onnx)  # original NumPy array


if args.pytorch:
    from src.models.GATr.Gatr_pf_e_noise_onnx import ExampleWrapper as GravnetModel
    #from src.models.GATr.Gatr_pf_e_noise_onnx import ExampleWrapper as GravnetModel
    from src.utils.train_utils import get_samples_steps_per_epoch, model_setup, set_gpus

    gpus, dev = set_gpus(args)

    # output from pure Python model
    # model_weights = "_epoch=0_step=500.ckpt"
    torch._dynamo.config.verbose = True
    model_weights = "/eos/user/m/mgarciam/datasets_mlpf/models_trained_CLD/gun_drlog_v9_dr01/_epoch=4_step=57500.ckpt"
    model = GravnetModel.load_from_checkpoint(
        model_weights,
        args=args,
        dev='cpu',
        map_location=torch.device("cpu"))
    model.eval()


    with torch.no_grad():
        output_torch = model(input)



    output_array = numpy.load("output_onnx.npy")  # Dateiname ggf. anpassen
    output_onnx = torch.tensor(output_array, dtype=torch.float32, device='cpu')
    print("Output Pure Python:", output_torch)
    print("Output ONNX Runtime:", output_onnx)

    print("PyTorch Output:")
    print("Type:", type(output_torch))                      
    print("Dtype:", output_torch.dtype)                     
    print("Shape:", output_torch.shape)                    

    print("\nONNX Output:")
    print("Type:", type(output_onnx))                        
    print("Dtype:", output_onnx.dtype)                      
    print("Shape:", output_onnx.shape)                     


    output_torch_np = output_torch.detach().cpu().numpy()
    print("\nMax Absolute Difference:", numpy.max(numpy.abs(output_torch_np - output_onnx.numpy())))
    print("Are outputs close?:", numpy.allclose(output_torch_np, output_onnx.numpy(), rtol=1e-5, atol=1e-6))

   # bei onnx modell klappt es aber wenn pytorch mit tracking model auswertest bekommst du unterschiede. Verstehe das