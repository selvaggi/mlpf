# python scripts/auto_plot_results.py --datasets dr_05 --gpu 0 --latest-only --path /eos/user/g/gkrzmanc/results/2024/E_PID_02122024_dr05_GT_clusters

# Track the models as the checkpoints are saved, and produce results for each model
import argparse
import os
import time
import subprocess
from pathlib import Path

timeout = 3600 * 8  # After 8 hr of no new checkpoints, stop the script
pause = 20  # Check every 20 secs for new files

dataset_prefix = "/eos/experiment/fcc/ee/datasets/mlpf/CLD/"
datasets = {
    "dr_025":
        {#/eos/experiment/fcc/ee/datasets/mlpf/CLD/eval/181124_gun_dr_025_v2/
            "train": os.path.join(dataset_prefix, "train/181124_gun_dr_025_v1/"),
            "eval": os.path.join(dataset_prefix, "eval/181124_gun_dr_025_v2/")
        },
    "dr_log":
        {
            "train": os.path.join(dataset_prefix, "train/gun_log_dr_020_050_v2_201124/"),
            "eval": os.path.join(dataset_prefix, "eval/gun_log_dr_020_050_v2_201124/")
        },
    "dr_05":
        {
            "train": os.path.join(dataset_prefix, "train/gun_dr_050_v1_27112/"),
            "eval": os.path.join(dataset_prefix, "eval/181124_gun_dr_050_v2/")
        }
}

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="Path to the folder with the training in which checkpoints are saved")
parser.add_argument("--gpu", type=int, help="GPU to use for the evaluation")
parser.add_argument("--latest-only", action="store_true", help="Only evaluate the latest checkpoint")
parser.add_argument("--datasets", type=str, help="Datasets to evaluate on, comma-separated")
parser.add_argument("--n-files", type=int, default=50, help="Number of files to evaluate on")
#parser.add_argument("--redo", action="store_true", help="Redo the evaluation for all checkpoints, even if they are already saved")
#parser.add_argument("--redo-plots", action="store_true", help="Redo the plots, even if they are already saved")

args = parser.parse_args()
gpu = int(args.gpu)
dir_ckpt = os.path.join(args.path, "eval_results")
if not os.path.exists(dir_ckpt):
    os.makedirs(dir_ckpt)

import sys
class Logger(object):
    def __init__(self, filename='auto_plot_results_stdout.txt'):
        self.terminal = sys.stdout
        self.log = open(os.path.join(args.path, filename), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
sys.stdout = Logger()
sys.stderr = Logger("auto_plot_results_stderr.txt")

# PYTHONOPATH=$PYTHONPATH:/afs/cern.ch/work/g/gkrzmanc/mlpf_2024
# quick fix for path problems
script_path = os.getcwd()
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + script_path
#eval_cmd = """/home/gkrzmanc/gatr/bin/python -m src.train_lightning1 --data-test /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4002.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4003.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4004.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4005.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4006.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4007.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4008.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4009.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4010.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4011.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4012.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4013.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4014.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4015.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4016.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4017.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4018.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4019.root /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard_eval/pf_tree_4020.root --data-config config_files/config_hits_track_v1.yaml -clust -clust_dim 3 --network-config src/models/wrapper/example_mode_gatr_e.py --model-prefix {model_prefix} --wandb-displayname evalHss_reprod_clust_only_newmodel_Clustering2810 --num-workers 0 --gpus {gpu} --batch-size 8 --start-lr 1e-3 --num-epochs 100 --optimizer ranger --fetch-step 0.1 --condensation --log-wandb --wandb-projectname mlpf_debug_eval --wandb-entity fcc_ml --frac_cluster_loss 0 --qmin 1 --use-average-cc-pos 0.99 --lr-scheduler reduceplateau --tracks --correction --ec-model gatr-neutrals --regress-pos --add-track-chis --load-model-weights  {ckpt_file}  --freeze-clustering --predict --regress-unit-p --PID-4-class"""
eval_cmd = """/home/gkrzmanc/gatr/bin/python -m src.train_lightning1 --data-test {files}  --data-config config_files/config_hits_track_v4.yaml -clust -clust_dim 3   --use-gt-clusters --network-config src/models/wrapper/example_mode_gatr_e.py --model-prefix {model_prefix} --wandb-displayname Eval_with_auto_plot_results --num-workers 0 --gpus {gpu} --batch-size 8  --start-lr 1e-3 --num-epochs 100 --optimizer ranger --fetch-step 0.1 --condensation --log-wandb --wandb-projectname mlpf_debug_eval --wandb-entity fcc_ml --frac_cluster_loss 0 --qmin 1 --use-average-cc-pos 0.99 --lr-scheduler reduceplateau --tracks --correction --ec-model gatr-neutrals --regress-pos --add-track-chis --load-model-weights  {ckpt_file}  --freeze-clustering --predict  --regress-unit-p --PID-4-class --restrict_PID_charge """
#plotting_cmd = """ /home/gkrzmanc/gatr/bin/python src/evaluation/evaluate_mass_Hss.py --path {path} """
plotting_cmds = [""" /home/gkrzmanc/gatr/bin/python src/evaluation/refactor/plot_results.py --path {path} --preprocess class_correction,filt_LE_CH --output_dir filt_LE_CH_400bins_epsilon005  --mass-only """,
                 """ /home/gkrzmanc/gatr/bin/python src/evaluation/refactor/plot_results.py --path {path} --preprocess class_correction --output_dir mass_plots_400bins_epsilon005 --mass_only """,
                 """ /home/gkrzmanc/gatr/bin/python src/evaluation/refactor/plot_results.py --path {path} --preprocess class_correction --output_dir mass_plots_400bins_epsilon005 """]

while True:
    for dataset in args.datasets.split(","):
        files = os.listdir(args.path)
        files = [f for f in files if f.endswith(".ckpt")]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(args.path, x)))
        #if args.latest_only:
        #    #files = [files[-1]]
        #files = ["_epoch=1_step=8000.ckpt"]
        if len(files) == 0:
            print("No files found for dataset {}, waiting...".format(dataset))
            time.sleep(pause)
            continue
        for f in files:
            ckpt_file = os.path.join(args.path, f)
            folder_name = f.split(".")[0].replace("=", "")
            current_folder_path = os.path.join(dir_ckpt, dataset, folder_name)
            if current_folder_path.startswith(".") or ".sys" in current_folder_path:
                continue
            if os.path.exists(os.path.join(current_folder_path, "eval_done.txt")):# and not args.redo:
                print(f"Skipping {f}, already evaluated")
            else:
                Path(current_folder_path).mkdir(parents=True, exist_ok=True)
                print(f"Running evaluation for {ckpt_file}")
                file = [datasets[dataset]["eval"] + r"pf_tree_{}.root".format(x) for x in range(args.n_files)]
                cmd = eval_cmd.format(model_prefix=current_folder_path, gpu=gpu, ckpt_file=ckpt_file, files=" ".join(file))
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
                # write '1' to a file plotting_done.txt
                with open(os.path.join(current_folder_path, "eval_done.txt"), "w") as f:
                    f.write("1")
            plots_stdout = os.path.join(current_folder_path, "plots_stdout.txt")
            plots_stderr = os.path.join(current_folder_path, "plots_stderr.txt")
            # if plots stdout file exitss
            #if os.path.exists(plots_stdout) and not args.redo_plots:
            #    print(f"Plots already exist for {f}")
            if not os.path.exists(os.path.join(current_folder_path, "plotting_done.txt")):
                print("Plotting for ", current_folder_path)
                for plotting_cmd in plotting_cmds:
                    cmd = plotting_cmd.format(path=current_folder_path)
                print("Plot cmd:", cmd)
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
                print("Plotting done")
                with open(os.path.join(current_folder_path, "plotting_done.txt"), "w") as f:
                    f.write("1")
            if args.latest_only:
                break
        time.sleep(pause)
        if time.time() - os.path.getmtime(os.path.join(args.path, files[-1])) > timeout:
            print("No new files found in the last 8 hrs, stopping")
            break
