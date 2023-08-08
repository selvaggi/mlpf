#!/bin/bash

for sed in {1..1000}; do
./run_sequence.sh /eos/home-g/gkrzmanc/mlpf_data/condor/../ config_v1.gun 100  $sed /afs/cern.ch/work/g/gkrzmanc/public/pf_dataset_04082023_30_100K
done
