#!/bin/bash

# python submit_jobs_train.py --config config_spread_300424.gun --outdir /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/300424/dr_01/40_50/ --condordir /eos/experiment/fcc/ee/datasets/mlpf/condor/train/300424/dr_01/40_50/ --njobs 2000 --nev 10 --queue workday


# python submit_jobs_train.py --config config_spread_300424_1.gun --outdir /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/300424/dr_03/40_50/ --condordir /eos/experiment/fcc/ee/datasets/mlpf/condor/train/300424/dr_03/40_50/ --njobs 2000 --nev 10 --queue workday

# python submit_jobs_train.py --config config_spread_1.gun --outdir /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/300424/dr_05/10_15/ --condordir /eos/experiment/fcc/ee/datasets/mlpf/condor/train/300424/dr_05/10_15/ --njobs 4000 --nev 100 --queue workday

# python submit_jobs_train.py --config config_spread_300424_0.gun --outdir /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/300424/dr_05/40_50_v0/ --condordir /eos/experiment/fcc/ee/datasets/mlpf/condor/train/300424/dr_05/40_50/ --njobs 4000 --nev 20 --queue workday


# python submit_jobs_train.py --config config_spread_ks.gun --outdir /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/300424/Ks_v1/  --condordir /eos/experiment/fcc/ee/datasets/mlpf/condor/train/300424/Ks/  --njobs 100 --nev 100 --queue workday

python submit_jobs_train.py --config config_spread_ks_5.gun --outdir /eos/experiment/fcc/ee/datasets/mlpf/CLD/train/011024_Hcard/  --condordir /eos/experiment/fcc/ee/datasets/mlpf/condor/011024_Hcard/  --njobs 5000 --nev 100 --queue workday
