#!/bin/bash

for sed in {1..2}; do
./run_sequence.sh /eos/home-g/gkrzmanc/mlpf_data/condor/../ config_v1.gun 50  $sed /afs/cern.ch/work/g/gkrzmanc/public/pf_dataset_04082023_small_50events
done
