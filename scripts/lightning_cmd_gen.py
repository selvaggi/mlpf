# Generate a command for training
# format: python lightning_cmd_gen.py <train/test> <filter: */{0..10} etc.> <run/folder name> <gpus> <additional args>
# get the args
import sys
args = sys.argv[1:]
print("args", args)
prefix_test = "/eos/experiment/fcc/ee/datasets/mlpf/CLD/eval/020424_1015/pf_tree_"
prefix_train = "/eos/experiment/fcc/ee/datasets/mlpf/CLD/train/020424/pf_tree_"

cmd = "python -m src.train_lightning1 "
if args[0] == "train":
    cmd += "--data-train "
    prefix = prefix_train
elif args[0] == "test":
    cmd += "--data-test "
    prefix = prefix_test
else:
    raise Exception

prefix += args[1] + ".root"
cmd += prefix
cmd += " --wandb-displayname " + args[2]

if args[0] == "train":
    cmd += f" --data-config config_files/config_hits_track_predict_chis.yaml -clust -clust_dim 3 --network-config src/models/wrapper/example_mode_gatr_e.py --model-prefix /eos/user/g/gkrzmanc/2024/{args[2]} --num-workers 0 --gpus {args[3]} --batch-size 8 --start-lr 1e-3 --num-epochs 100 --optimizer ranger --fetch-step 0.01 --condensation --log-wandb --wandb-displayname {args[2]} --wandb-projectname mlpf_debug --wandb-entity fcc_ml --frac_cluster_loss 0 --qmin 3 --use-average-cc-pos 0.99 --lr-scheduler reduceplateau --tracks --correction --freeze-clustering --add-track-chis "
else:
    cmd += f" --data-config config_files/config_hits_track_predict_chis.yaml -clust -clust_dim 3 --network-config src/models/wrapper/example_mode_gatr_e.py --model-prefix /eos/user/g/gkrzmanc/2024/{args[2]} --num-workers 0 --gpus {args[3]} --batch-size 8 --start-lr 1e-3 --num-epochs 100 --optimizer ranger --fetch-step 0.5 --condensation --log-wandb --wandb-displayname {args[2]} --wandb-projectname mlpf_debug --wandb-entity fcc_ml --frac_cluster_loss 0 --qmin 3 --use-average-cc-pos 0.99 --lr-scheduler reduceplateau --tracks --correction --freeze-clustering --add-track-chis --predict "

cmd += " ".join(args[4:])
print("--------------------")
print(cmd)

