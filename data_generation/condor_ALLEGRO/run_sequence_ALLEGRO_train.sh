#!/bin/bash
# apcatlas01: source run_sequence_ALLEGRO_train.sh /home/gmarchio/work/fcc/allegro/mlpf/mlpf/ config_spread_031224_fair.gun 10 42 /home/gmarchio/work/fcc/allegro/mlpf/mlpf/output /home/gmarchio/work/fcc/allegro/mlpf/mlpf/tmp/
# lxplus: source run_sequence_ALLEGRO_train.sh /afs/cern.ch/user/g/gmarchio/work/fcc/allegro/mlpf/mlpf/ config_spread_211125.gun 10 42 /eos/experiment/fcc/users/g/gmarchio/ALLEGRO_o1_v03/mlpf/output/ /eos/experiment/fcc/users/g/gmarchio/ALLEGRO_o1_v03/mlpf/tmp/

HOMEDIR=${1} # path to where it's ran from (usally mlpf dir)
GUNCARD=${2} # name of gun card to use
NEV=${3} # number of events, if running with gun this is fixed in the gun card
SEED=${4} # seed for the random number generator
OUTPUTDIR=${5} # output directory
DIR=${6} # directory where the job is ran and intermediate files are created

gen=1
sim=1
rec=1
flatten=1

mkdir -p ${DIR}
mkdir -p ${DIR}/${SEED}
cd ${DIR}/${SEED}
pwd
SAMPLE="gun" 

cp -r $HOMEDIR/data_generation/preprocessing/ .

wrapperfunction() {
    # source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2025-01-28
    source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh
}
wrapperfunction

# Path to the ALLEGRO configuration files (needed for the reconstruction and ddsim)
# PATH_FCCCONFIG=$HOMEDIR/FCC-config/
PATH_ALLEGRO_DATA=$HOMEDIR/data_generation/condor_ALLEGRO/data/
if [ ! -d "$PATH_ALLEGRO_DATA" ]; then
    cd $HOMEDIR/data_generation/condor_ALLEGRO
    python downloadFilesForReco.py
    cd ${DIR}/${SEED}
fi

# Build gun  or Zcard
if [[ "${gen}" -ne 0 ]]
then
    if [[ "${SAMPLE}" == "gun" ]] 
    then 
        cp -r ${HOMEDIR}/data_generation/guns/gun_log_dr/gun.cpp .
        cp -r ${HOMEDIR}/data_generation/guns/gun_log_dr/CMakeLists.txt .
        PATH_GUN_CONFIG=${HOMEDIR}/data_generation/guns/gun_log_dr/config_files/${GUNCARD}
        mkdir -p build install
        cd build
        cmake .. -DCMAKE_INSTALL_PREFIX=../install
        make install -j 8
        cd ..
        ./build/gun ${PATH_GUN_CONFIG} 
    fi

    if [[ "${SAMPLE}" == "Zcard" ]]
    then
        cp ${HOMEDIR}/data_generation/pythia/${SAMPLE}.cmd card.cmd
        echo "Random:seed=${SEED}" >> card.cmd
        cat card.cmd
        k4run ${HOMEDIR}/data_generation/pythia/pythia.py -n $NEV --Dumper.Filename out.hepmc --Pythia8.PythiaInterface.pythiacard card.cmd
        cp out.hepmc events.hepmc
    fi
fi

if [[ "${sim}" -ne 0 && -f "events.hepmc" ]]
then
    # ddsim --compactFile $K4GEO/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml --outputFile out_sim_edm4hep.root --steeringFile ${PATH_FCCCONFIG}/FCCee/FullSim/ALLEGRO/ALLEGRO_o1_v03/allegro_steer.py --inputFiles events.hepmc --numberOfEvents ${NEV} --random.seed ${SEED}
    ddsim --compactFile $K4GEO/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml --outputFile out_sim_edm4hep.root --steeringFile ${HOMEDIR}/data_generation/condor_ALLEGRO/allegro_steer.py --inputFiles events.hepmc --numberOfEvents ${NEV} --random.seed ${SEED}
fi

if [[ "${rec}" -ne 0 && -f "out_sim_edm4hep.root" ]]
then
    ln -f -s $PATH_ALLEGRO_DATA .
    cp ${HOMEDIR}/data_generation/condor_ALLEGRO/run_ALLEGRO_reco.py .
    # added saveHits for debug
    k4run run_ALLEGRO_reco.py -n ${NEV} --IOSvc.Input out_sim_edm4hep.root --IOSvc.Output out_reco_edm4hep.root --includeHCal --includeMuon --saveCells --addTracks
    # --saveHits
    # save a lot of space by getting rid of sim file
    if [ -f "out_reco_edm4hep.root" ]; then
        rm -f out_sim_edm4hep.root
    fi
fi

if [[ "${flatten}" -ne 0 && -f "out_reco_edm4hep.root" ]]
then
    wrapperfunction2() {
        source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/setup.sh
    }
    wrapperfunction2
    python  -m preprocessing.dataset_creation_ALLEGRO --input out_reco_edm4hep.root  --outpath . --chunk_size 100
    chunks=$(( (NEV + 99) / 100 ))
    for ((i=0; i<chunks; i++)); do
        echo "Loop $i"
        cp out_reco_edm4hep_$i.parquet ${OUTPUTDIR}/out_reco_edm4hep_${SEED}_$i.parquet
    done
fi


# remove intermediate temporary directory
cd ..
# rm -rf ${SEED}
