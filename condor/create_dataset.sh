#!/bin/bash

for sed in {1..10}; do
./run_sequence.sh /afs/cern.ch/work/m/mgarciam/private/mlpf/condor/../ config_v1.gun 1000  $sed /afs/cern.ch/work/m/mgarciam/public/mlpf_250723/
done 