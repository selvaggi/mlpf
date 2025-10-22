#!/bin/bash
#SBATCH --job-name=testjob                
#SBATCH --output=logs-torchrun/%j.err   
#SBATCH --error=logs-torchrun/%j.out       
#SBATCH --nodes=1                          
#SBATCH --cpus-per-task=20           
#SBATCH --gres=gpu:4                      
#SBATCH --ntasks-per-node=4
#SBATCH --time=0:30:00
#SBATCH --qos=acc_ehpc              
#SBATCH --account=ehpc399         

# modules
module load MINIFORGE/24.3.0-0
source "/gpfs/apps/MN5/ACC/MINIFORGE/24.3.0-0/etc/profile.d/conda.sh"
conda activate /gpfs/apps/MN5/ACC/MINIFORGE/24.3.0-0/envs/mlpf


# change to working directory
cd "/gpfs/projects/ehpc399/MLPF/mlpf"


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SLURM_CPU_BIND=none
export CUDA_LAUNCH_BLOCKING=1
export GPUS_PER_NODE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NODE_RANK=$SLURM_PROCID
export NUM_PROCS=$((SLURM_NNODES * GPUS_PER_NODE))
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500


echo "NODE_RANK: {$NODE_RANK}"
echo "NNODES: {$SLURM_NNODES}"
echo "NUM_PROCS: {$NUM_PROCS}"
echo "MASTER_ADDR: {$MASTER_ADDR}"
echo "MASTER_PORT: {$MASTER_PORT}"


DATA_PATH="/gpfs/scratch/ehpc399/data/"
DATA_CONFIG_FILE="config_files/config_hits_track_v2_noise.yaml"
NETWORK_CONFIG_FILE="src/models/wrapper/example_mode_gatr_noise.py"
MODEL_PREFIX="output/testslurm/"

# Assign the absolute paths of all files to a variable
#filelist=$(find "$DATA_PATH" -type f -exec realpath {} \;)

# Print the variable
#echo "$filelist"



PYTHON_MODULE="src.train_lightning1"

train_command="torchrun \
    --nnodes $SLURM_NNODES \
    --nproc_per_node $GPUS_PER_NODE \
    --rdzv_id=$RANDOM \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m $PYTHON_MODULE \
    --data-train $DATA_PATH \
    --data-config $DATA_CONFIG_FILE \
    --network-config $NETWORK_CONFIG_FILE \
    --model-prefix $MODEL_PREFIX \
    --num-workers 1 \
    --batch-size 50 \
    --start-lr 1e-3 \
    --num-epochs 1 \
    --optimizer ranger \
    --condensation \
    --log-wandb \
    --clust \
    --clust_dim 3 \
    --gpus 0,1,2,3 \
    --fetch-step 0.01 \
    --wandb-displayname name \
    --wandb-projectname bscbenchmarking \
    --wandb-entity entity \
    --frac_cluster_loss 0 \
    --qmin 3 \
    --use-average-cc-pos 0.99 \
    --tracks \
    --train-val-split 0.7 \
    --prefetch-factor 16"

# This command runs both $gpu_monitor_command and $train_command in parallel
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1  --export=ALL bash -c " $train_command  "

if [ $? -ne 0 ]; then
    echo "Training failed. Exiting."
    exit 1
fi

echo "Training succeeded."


