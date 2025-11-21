#!/bin/bash
# execute it from the directory containing the script
# to have python: source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh

# gun
# python submit_jobs.py --mode train --config config_spread_031224_fair.gun --outdir /eos/experiment/fcc/users/g/gmarchio/ALLEGRO_o1_v03/mlpf/train/gun_dr_logE_v0_010925/  --condordir /eos/experiment/fcc/users/g/gmarchio/ALLEGRO_o1_v03/mlpf/condor/gun_dr_logE_v0_010925/  --njobs 1600 --nev 1000 --queue nextweek
# --queue tomorrow

python submit_jobs.py --mode train --config config_spread_211125.gun --outdir /eos/experiment/fcc/users/g/gmarchio/ALLEGRO_o1_v03/mlpf/train/gun_dr_logE_211125_test/  --condordir /eos/experiment/fcc/users/g/gmarchio/ALLEGRO_o1_v03/mlpf/condor/gun_dr_logE_211125_test/  --njobs 2 --nev 1000 --queue workday --output parquet

# test, 1 job
# python submit_jobs.py --mode train --config config_spread_031224_fair_test.gun --outdir /eos/experiment/fcc/users/g/gmarchio/ALLEGRO_o1_v03/mlpf/train/gun_dr_logE_v0_010925/  --condordir /eos/experiment/fcc/users/g/gmarchio/ALLEGRO_o1_v03/mlpf/condor/gun_dr_logE_v0_010925/  --njobs 1 --nev 10 --queue longlunch

# ee->Z->qq
# python submit_jobs.py --mode train --config p8_ee_Zuds_ecm91 --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/allegro_v1/  --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/allegro_v0/  --njobs 800 --nev 100 --queue tomorrow
