#!/bin/bash

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
SAMPLE="Zcard" 

wrapperfunction() {
    source /cvmfs/sw.hsf.org/key4hep/setup.sh #-r 2025-01-28
}
wrapperfunction

if [ ! -f "out_reco_edm4hep_REC.edm4hep.root" ]; then
    # Path to the CLD configuration files (needed for the reconstruction and ddsim) this needs to be changed after git clone CLD config
    PATH_CLDCONFIG=/afs/cern.ch/work/m/mgarciam/private/CLD_Config_versions/CLDConfig_2025_05_26/CLDConfig/ 

    # Build gun  or Zcard
    if [[ "${SAMPLE}" == "gun" ]] 
    then 
        cp -r ${HOMEDIR}/guns/gun_log_dr/gun.cpp .
        cp -r ${HOMEDIR}/guns/gun_log_dr/CMakeLists.txt . 
        PATH_GUN_CONFIG=${HOMEDIR}/guns/gun_log_dr/config_files/${GUNCARD} 
        mkdir build install
        cd build
        cmake .. -DCMAKE_INSTALL_PREFIX=../install
        make install -j 8
        cd ..
        ./build/gun ${PATH_GUN_CONFIG} 
    fi

    if [[ "${SAMPLE}" == "Zcard" ]]
    then
        cp ${HOMEDIR}/Pythia_generation/${GUNCARD}.cmd card.cmd
        echo "Random:seed=${SEED}" >> card.cmd
        cat card.cmd
        k4run ${HOMEDIR}/Pythia_generation/pythia.py -n $NEV --Dumper.Filename out.hepmc --Pythia8.PythiaInterface.pythiacard card.cmd
        cp out.hepmc events.hepmc
    fi


    ddsim --compactFile $K4GEO/FCCee/CLD/compact/CLD_o2_v07/CLD_o2_v07.xml --outputFile out_sim_edm4hep.root --steeringFile ${PATH_CLDCONFIG}/cld_steer.py --inputFiles events.hepmc --numberOfEvents ${NEV} --random.seed ${SEED}

    cp -r ${PATH_CLDCONFIG}/* .
    k4run CLDReconstruction.py -n ${NEV}  --inputFiles out_sim_edm4hep.root --outputBasename out_reco_edm4hep
fi 

wrapperfunction() {
    source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/setup.sh
}
wrapperfunction

if [ ! -f "out_reco_edm4hep_REC.parquet" ]; then
    cp -r /afs/cern.ch/work/m/mgarciam/private/MLPF_datageneration/preprocessing/ .
    python  -m preprocessing.dataset_creation --input out_reco_edm4hep_REC_${SEED}.edm4hep.root  --outpath .   --dataset
fi

mkdir -p ${OUTPUTDIR}
python /afs/cern.ch/work/f/fccsw/public/FCCutils/eoscopy.py out_reco_edm4hep_REC_${SEED}.parquet ${OUTPUTDIR}/pf_tree_${SEED}.parquet
# python /afs/cern.ch/work/f/fccsw/public/FCCutils/eoscopy.py out_reco_edm4hep_REC.edm4hep.root ${OUTPUTDIR}/out_reco_edm4hep_REC_${SEED}.edm4hep.root


# rm -r ${DIR}/${SEED}

