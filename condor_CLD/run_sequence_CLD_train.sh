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
SAMPLE="gun" 

# Path to the CLD configuration files (needed for the reconstruction and ddsim) this needs to be changed after git clone CLD config
# PATH_CLDCONFIG=/afs/cern.ch/work/m/mgarciam/private/CLD_Config_versions/CLDConfig_030225/CLDConfig/
PATH_CLDCONFIG=/home/gmarchio/work/fcc/allegro/mlpf/CLDConfig/CLDConfig/
cp $HOMEDIR/condor/make_pftree_clic_bindings.py ./
cp $HOMEDIR/condor/tree_tools.py ./
wrapperfunction() {
    # source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2025-01-28
    source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh
}
wrapperfunction



# Build gun  or Zcard
if [[ "${gen}" -ne 0 ]]
then
    if [[ "${SAMPLE}" == "gun" ]]
    then
        cp -r ${HOMEDIR}/guns/gun_log_dr/gun.cpp .
        cp -r ${HOMEDIR}/guns/gun_log_dr/CMakeLists.txt .
        PATH_GUN_CONFIG=${HOMEDIR}/guns/gun_log_dr/config_files/${GUNCARD}
        mkdir -p build install
        cd build
        cmake .. -DCMAKE_INSTALL_PREFIX=../install
        make install -j 8
        cd ..
        ./build/gun ${PATH_GUN_CONFIG}
    fi

    if [[ "${SAMPLE}" == "Zcard" ]]
    then
        cp ${HOMEDIR}/Pythia_generation/${SAMPLE}.cmd card.cmd
        echo "Random:seed=${SEED}" >> card.cmd
        cat card.cmd
        k4run ${HOMEDIR}/Pythia_generation/pythia.py -n $NEV --Dumper.Filename out.hepmc --Pythia8.PythiaInterface.pythiacard card.cmd
        cp out.hepmc events.hepmc
    fi
fi

if [[ "${sim}" -ne 0 ]]
then
    ddsim --compactFile $K4GEO/FCCee/CLD/compact/CLD_o2_v07/CLD_o2_v07.xml --outputFile out_sim_edm4hep.root --steeringFile ${PATH_CLDCONFIG}/cld_steer.py --inputFiles events.hepmc --numberOfEvents ${NEV} --random.seed ${SEED}
fi

if [[ "${rec}" -ne 0 ]]
then
    cp -r ${PATH_CLDCONFIG}/* .
    k4run CLDReconstruction.py -n ${NEV}  --inputFiles out_sim_edm4hep.root --outputBasename out_reco_edm4hep
fi

if [[ "${flatten}" -ne 0 ]]
then
    python make_pftree_clic_bindings.py out_reco_edm4hep_REC.edm4hep.root tree5.root False False
    mkdir -p ${OUTPUTDIR}
    if [[ "$OUTPUTDIR" == /eos/* ]]; then
        python /afs/cern.ch/work/f/fccsw/public/FCCutils/eoscopy.py tree5.root ${OUTPUTDIR}/pf_tree_${SEED}.root
    else
        cp tree5.root ${OUTPUTDIR}/pf_tree_${SEED}.root
    fi
fi

# remove intermediate temporary directory
#cd ..
#rm -rf ${SEED}
