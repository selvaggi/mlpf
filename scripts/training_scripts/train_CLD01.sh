module load apptainer/1.1.5

export SINGULARITY_CACHEDIR=/mnt/proj3/dd-23-91/cern/cache 

singularity  shell -B /mnt/proj3/dd-23-91/ docker://dologarcia/colorsinglet:v4

cd /mnt/proj3/dd-23-91/cern/mlpf/ 

torchrun --standalone --nnodes=1 --nproc_per_node=4 -m src.train --data-train  "/mnt/proj3/dd-23-91/mlpf/CLD/train/221223_drmax01/pf_tree_{1..500}.root"  --data-config config_files/config_2_newlinks.yaml -clust -clust_dim 3 --network-config src/models/wrapper/example_gravnet_model_basic.py --model-prefix /mnt/proj3/dd-23-91/mlpf/models_trained_CLD/221223_drmax01/ --num-workers 0 --gpus 0,1,2,3 --batch-size 4 --start-lr 1e-3 --num-epochs 100 --optimizer ranger --fetch-step 0.16 --condensation --log-wandb --wandb-displayname full_training_clus_CLD_01 --wandb-projectname mlpf_debug --wandb-entity ml4hep --frac_cluster_loss 0 --qmin 3 --use-average-cc-pos 0.99 --lr-scheduler reduceplateau --backend nccl  --hgcalloss 