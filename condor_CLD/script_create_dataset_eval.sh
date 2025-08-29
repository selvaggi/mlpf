#!/bin/bash

python submit_jobs_train.py --config config_spread_031224_fair.gun  --outdir /eos/user/r/rchaafa/  --condordir /eos/user/r/rchaafa/   --njobs 10 --nev 10 --queue microcentury
# --queue workday

# python submit_jobs_train.py --config eval_2_particles_dr/config_01_cc_ee.gun --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/gun_dr_log_logE_v0_290125/  --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/gun_dr_log_logE_v0_290125/  --njobs 100 --nev 100 --queue workday
# python submit_jobs_eval.py --config eval_2_particles_dr/config_01_cc_ee.gun --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/eval/2_particles/Hss_250525/   --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/Hss_250525/  --njobs 400 --nev 10 --queue workday
# python submit_jobs_eval.py --config eval_2_particles_dr/config_001_cc_ee.gun --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/eval/2_particles/gun_001_cc_ee_180225_v0/   --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/gun_001_cc_ee_180225_v0/  --njobs 100 --nev 10 --queue workday
# python submit_jobs_eval.py --config eval_2_particles_dr/config_005_cc_ee.gun --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/eval/2_particles/gun_005_cc_ee_180225_v0/   --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/gun_005_cc_ee_180225_v0/  --njobs 100 --nev 10 --queue workday

# python submit_jobs_eval.py --config eval_2_particles_dr/config_01_cc.gun --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/eval/2_particles/gun_01_cc_180225_v0/   --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/gun_01_cc_180225_v0/  --njobs 100 --nev 10 --queue workday
# python submit_jobs_eval.py --config eval_2_particles_dr/config_001_cc.gun --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/eval/2_particles/gun_001_cc_180225_v0/   --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/gun_001_cc_180225_v0/  --njobs 100 --nev 10 --queue workday
# python submit_jobs_eval.py --config eval_2_particles_dr/config_005_cc.gun --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/eval/2_particles/gun_005_cc_180225_v0/   --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/gun_005_cc_180225_v0/  --njobs 100 --nev 10 --queue workday

# python submit_jobs_eval.py --config eval_2_particles_dr/config_01_cn.gun --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/eval/2_particles/gun_01_cn_180225_v0/   --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/gun_01_cn_180225_v0/  --njobs 100 --nev 10 --queue workday
# python submit_jobs_eval.py --config eval_2_particles_dr/config_001_cn.gun --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/eval/2_particles/gun_001_cn_180225_v0/   --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/gun_001_cn_180225_v0/  --njobs 100 --nev 10 --queue workday
# python submit_jobs_eval.py --config eval_2_particles_dr/config_005_cn.gun --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/eval/2_particles/gun_005_cn_180225_v0/   --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/gun_005_cn_180225_v0/  --njobs 100 --nev 10 --queue workday

