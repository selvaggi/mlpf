#!/bin/bash

# python submit_jobs_train.py --config config_spread_300424.gun --outdir /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/300424/dr_01/40_50/ --condordir /eos/experiment/fcc/ee/datasets/mlpf/condor/train/300424/dr_01/40_50/ --njobs 2000 --nev 10 --queue workday


# python submit_jobs_train.py --config config_spread_300424_1.gun --outdir /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/300424/dr_03/40_50/ --condordir /eos/experiment/fcc/ee/datasets/mlpf/condor/train/300424/dr_03/40_50/ --njobs 2000 --nev 10 --queue workday

# python submit_jobs_train.py --config config_spread_1.gun --outdir /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/300424/dr_05/10_15/ --condordir /eos/experiment/fcc/ee/datasets/mlpf/condor/train/300424/dr_05/10_15/ --njobs 4000 --nev 100 --queue workday

# python submit_jobs_train.py --config config_spread_300424_0.gun --outdir /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/300424/dr_05/40_50_v0/ --condordir /eos/experiment/fcc/ee/datasets/mlpf/condor/train/300424/dr_05/40_50/ --njobs 4000 --nev 20 --queue workday

# python submit_jobs_train.py --config config_spread_ks.gun --outdir /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/081024_ks/  --condordir /eos/experiment/fcc/ee/datasets/mlpf/condor/train/081024_ks/  --njobs 2000 --nev 100 --queue workday

# python submit_jobs_train.py --config config_spread_181124.gun --outdir /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/181124_gun_dr_025_v1/  --condordir /eos/experiment/fcc/ee/datasets/mlpf/condor/181124_gun_dr_025_v1/  --njobs 4000 --nev 100 --queue workday

python submit_jobs_train.py --config config_spread_031224_fair.gun --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/gun_dr_log_logE_v0_290125/  --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/gun_dr_log_logE_v0_290125/  --njobs 8000 --nev 100 --queue workday

# python submit_jobs_train.py --config config_spread_181124.gun --outdir /eos/experiment/fcc/ee/datasets/mlpf/CLD/eval/181124_gun_dr_025_v0/  --condordir /eos/experiment/fcc/ee/datasets/mlpf/condor/181124_gun_dr_025_v0_eval/  --njobs 4000 --nev 100 --queue workday
