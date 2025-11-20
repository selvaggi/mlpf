#!/bin/bash

python submit_jobs_train.py --sample gun --CLDGEO  CLD_o2_v07 --config config_spread_031224_fair.gun --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/gun_CLD_gentracking/  --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/gun_CLD_gentracking/  --njobs 10000 --nev 100 --queue tomorrow --cldconfig /afs/cern.ch/work/m/mgarciam/private/CLD_Config_versions/CLDConfig_latest/CLDConfig/ 

