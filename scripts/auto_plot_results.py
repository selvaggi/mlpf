# Track the models as the checkpoints are saved, and produce results for each model

import argparse
import os
import time
import subprocess

timeout = 3600 * 8 # after 8 hr of no new checkpoints, stop the script
pause = 20 # check every 20 secs for new files

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="Path to the folder with the training in which checkpoints are saved")
parser.add_argument("--gpu", type=int, help="GPU to use for the evaluation")
parser.add_argument("--redo", action="store_true", help="Redo the evaluation for all checkpoints, even if they are already saved")
args = parser.parse_args()
gpu = int(args.gpu)
dir_ckpt = os.path.join(args.path, "eval_results")
if not os.path.exists(dir_ckpt):
    os.makedirs(dir_ckpt)
import os
# PYTHONOPATH=$PYTHONPATH:/afs/cern.ch/work/g/gkrzmanc/mlpf_2024
# quick fix for path problems
script_path = os.getcwd()
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + script_path
eval_cmd = """/home/gkrzmanc/gatr/bin/python -m src.train_lightning1 --data-test /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4002.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4003.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4004.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4005.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4006.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4007.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4008.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4009.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4010.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4011.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4012.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4013.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4014.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4015.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4016.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4017.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4018.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4019.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4020.root --data-config config_files/config_hits_track_v1.yaml -clust -clust_dim 3 --network-config src/models/wrapper/example_mode_gatr_e.py --model-prefix {model_prefix} --wandb-displayname evalHss_reprod_clust_only_newmodel_Clustering2810 --num-workers 0 --gpus {gpu} --batch-size 8 --start-lr 1e-3 --num-epochs 100 --optimizer ranger --fetch-step 0.1 --condensation --log-wandb --wandb-projectname mlpf_debug_eval --wandb-entity fcc_ml --frac_cluster_loss 0 --qmin 1 --use-average-cc-pos 0.99 --lr-scheduler reduceplateau --tracks --correction --ec-model gatr-neutrals --regress-pos --add-track-chis --load-model-weights  {ckpt_file}  --freeze-clustering --predict --regress-unit-p --PID-4-class"""
plotting_cmd = """ /home/gkrzmanc/gatr/bin/python src/evaluation/evaluate_mass_Hss.py  --path {path} """
while True:
    files = os.listdir(args.path)
    files = [f for f in files if f.endswith(".ckpt")]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(args.path, x)))
    if len(files) == 0:
        print("No files found, waiting...")
        time.sleep(pause)
        continue

    for f in files:
        ckpt_file = os.path.join(args.path, f)
        folder_name = f.split(".")[0].replace("=", "")
        current_folder_path = os.path.join(dir_ckpt, folder_name)
        if os.path.exists(current_folder_path) and not args.redo:
            print(f"Skipping {f}, already evaluated")
        else:
            print(f"Running evaluation for {ckpt_file}")
            cmd = eval_cmd.format(model_prefix=current_folder_path, gpu=gpu, ckpt_file=ckpt_file)
            cmdargs = cmd.split()
            proc = subprocess.Popen(cmdargs, stdout=subprocess.PIPE, shell=False)
            (out, err) = proc.communicate()
            # save stdout and stderr to files
            with open(os.path.join(current_folder_path, "stdout.txt"), "w") as f:
                if out is not None:
                    f.write(out.decode())
                else:
                    f.write("None")
            with open(os.path.join(current_folder_path, "stderr.txt"), "w") as f:
                if err is not None:
                    f.write(err.decode())
                else:
                    f.write("None")
            print(f"Finished evaluation for {ckpt_file}")
        plots_stdout = os.path.join(current_folder_path, "plots_stdout.txt")
        plots_stderr = os.path.join(current_folder_path, "plots_stderr.txt")
        # if plots stdout file exitss
        if os.path.exists(plots_stdout) and not args.redo:
            print(f"Plots already exist for {f}")
        else:
            print("Plotting for ", current_folder_path)
            cmd = plotting_cmd.format(path=current_folder_path)
            cmdargs = cmd.split()
            proc = subprocess.Popen(cmdargs, stdout=subprocess.PIPE, shell=False)
            (out, err) = proc.communicate()
            # save stdout and stderr to files
            with open(plots_stdout, "w") as f:
                if out is not None:
                    f.write(out.decode())
                else:
                    f.write("None")
            with open(plots_stderr, "w") as f:
                if err is not None:
                    f.write(err.decode())
                else:
                    f.write("None")
    time.sleep(pause)
    if time.time() - os.path.getmtime(os.path.join(args.path, files[-1])) > timeout:
        print("No new files found in the last 8 hrs, stopping")
        break
