#!/bin/bash

# python submit_jobs_train.py --config p8_ee_Zuds_ecm91 --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/eval/Zuds_2025_09_29_key4hep_20250529_CLD_r20250526/  --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/Zuds_2025_09_29_key4hep_20250529_CLD_r20250526_v1_eval/  --njobs 50300 --nev 100 --queue tomorrow
# /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/Zss91_2025_09_25_key4hep_20250529_CLD_r20250526/
# /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/Zss91_2025_09_25_key4hep_20250529_CLD_r20250526/

python submit_jobs_train.py --config p8_ee_Zuds_ecm91 --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/allegro_v1/  --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/allegro_v0/  --njobs 800 --nev 100 --queue tomorrow





# bash run_sequence_CLD_train.sh /afs/cern.ch/work/m/mgarciam/private/mlpf/ p8_ee_Zss_ecm91 2 3 /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/Zss91_2025_09_25_key4hep_20250529_CLD_r20250526/ /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/Zss91_2025_09_25_key4hep_20250529_CLD_r20250526/

