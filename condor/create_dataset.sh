#!/bin/bash
# rm /afs/cern.ch/work/m/mgarciam/private/mlpf/condor/job/out_reco_edm4hep.root
# rm /afs/cern.ch/work/m/mgarciam/private/mlpf/condor/job/out_sim_edm4hep.root
# rm /afs/cern.ch/work/m/mgarciam/private/mlpf/condor/job/tree.root
# rm /afs/cern.ch/work/m/mgarciam/private/mlpf/condor/job/events.hepmc
# for sed in {1..100}; do
#     ./run_sequence.sh /afs/cern.ch/work/m/mgarciam/private/mlpf/condor/../ config_v1.gun 100  $sed /eos/user/m/mgarciam/datasets_mlpf/070823_20part/
#     rm /afs/cern.ch/work/m/mgarciam/private/mlpf/condor/job/out_reco_edm4hep.root
#     rm /afs/cern.ch/work/m/mgarciam/private/mlpf/condor/job/out_sim_edm4hep.root
#     rm /afs/cern.ch/work/m/mgarciam/private/mlpf/condor/job/tree.root
#     rm /afs/cern.ch/work/m/mgarciam/private/mlpf/condor/job/events.hepmc
# done 

./run_sequence_update_spread.sh /afs/cern.ch/work/m/mgarciam/private/mlpf config_spread.gun 100 1  /eos/user/m/mgarciam/datasets_mlpf/13112023_15_20/



