Here are some instructions to start jobs and run the MLPF training on BSC:

Detailed information on the infrastructure is available here: 
https://www.bsc.es/supportkc/docs/MareNostrum5/intro/  

**Starting an interactive job:**

`salloc -A ehpc399 -t 00:30:00 -c 80 -n 1 -q acc_ehpc -J myjob --gres=gpu:1`

choose the parameters according to your needs

**Submitting a job to the cluster:**

`sbatch submit_job.sh`

**Wandb:**

Since BSC has no WAN, wandb needs to run in offline mode and synchronized after the run.

`sshfs -o workaround=rename user@transfer1.bsc.es:/gpfs/projects/ehpc399/mlpf/wandb destination/wandbsync`

on the machine with WAN:

`wandb sync --include-offline /path/to/sshfs/folder/wandb/offline-run-20250714_120000-XXXXXXXX/run-XXXXXXXX.wandb`
