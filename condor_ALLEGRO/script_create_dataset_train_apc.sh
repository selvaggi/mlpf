#!/bin/bash

# start jobs with >0 to avoid problem with seed=0
production=290825
nevents=10
startjob=1
njobs=1

stopjob=$((startjob + njobs -1))
mkdir -p logs/$production
for job in $(seq $startjob $stopjob); do
    source condor_ALLEGRO/run_sequence_ALLEGRO_train.sh /home/gmarchio/work/fcc/allegro/mlpf/mlpf config_spread_031224_fair.gun $nevents $job /home/gmarchio/work/fcc/allegro/mlpf/mlpf/output/$production tmp/$production &> logs/$production/job$job.log &
done
