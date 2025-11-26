#!/bin/bash

####################################################################################################
HOMEDIR=${1} # path to where it's ran from e.g. /afs/cern.ch/user/f/fmokhtar/MLPF_datageneration
GUNCARD=${2}  # example p8_ee_Zuds_ecm91
NEV=${3} # 100
SEED=${4} # 1
OUTPUTDIR=${5} # dir where the final file is copied
DIR=${6} # dir where intermidiate files are created 
SAMPLE=${7} # can be "gun" or "Zcard"
GEO=${8} # default is ILD_l5_o1_v02
ILDCCONFIG=${9} # path to ILDConfig
if [ -z "$GEO" ]; then
    echo "Will use default geometry version ILD_l5_o1_v02"
    GEO=ILD_l5_o1_v02
else
    echo "Will use geometry version $GEO"
fi
# git clone the CLDConfig 

####################################################################################################

mkdir -p ${DIR}/${SEED}
cd ${DIR}/${SEED}

wrapperfunction() {
    # using nightlies change if using a specific CLD config  
    source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh 
}
wrapperfunction



# Build gun 
if [[ "${SAMPLE}" == "gun" ]] 
then 
    cp -r ${HOMEDIR}/data_generation/guns/gun_log_dr/gun.cpp .
    cp -r ${HOMEDIR}/data_generation/guns/gun_log_dr/CMakeLists.txt . 
    PATH_GUN_CONFIG=${HOMEDIR}/data_generation/guns/gun_log_dr/config_files/${GUNCARD} 
    mkdir build install
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=../install
    make install -j 8
    cd ..
    ./build/gun ${PATH_GUN_CONFIG} 
fi

if [[ "${SAMPLE}" == "Zcard" ]]
then
    xrdcp ${HOMEDIR}/pythia/${SAMPLE}.cmd card.cmd
    echo "Random:seed=${SEED}" >> card.cmd
    cat card.cmd
    k4run ${HOMEDIR}/pythia/pythia.py -n $NEV --Dumper.Filename out.hepmc --Pythia8.PythiaInterface.pythiacard card.cmd
    cp out.hepmc events.hepmc
fi



ddsim --compactFile $K4GEO/ILD/compact/$GEO/$GEO.xml --outputFile out_sim_edm4hep.root --steeringFile ${ILDCCONFIG}/ddsim_steer.py --inputFiles events.hepmc --numberOfEvents ${NEV} --random.seed ${SEED}

# copy large input files via xrootd (recommended)
xrdcp -r ${ILDCCONFIG}/* .

# running standard reconstruction
k4run ILDReconstruction.py -n ${NEV}  --inputFiles out_sim_edm4hep.root --outputBasename out_reco_edm4hep


wrapperfunction() {
    source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/setup.sh
}
wrapperfunction

if [ ! -f "out_reco_edm4hep_REC.parquet" ]; then
    cp -r ${HOMEDIR}/data_generation/preprocessing/ .
    python  -m preprocessing.dataset_creation --input  out_reco_edm4hep_REC.edm4hep.root  --outpath . 
fi

mkdir -p ${OUTPUTDIR}
mkdir -p ${OUTPUTDIR}/hepmc_files/

python /afs/cern.ch/work/f/fccsw/public/FCCutils/eoscopy.py out_reco_edm4hep_REC.parquet ${OUTPUTDIR}/pf_tree_${SEED}.parquet
python /afs/cern.ch/work/f/fccsw/public/FCCutils/eoscopy.py  events.hepmc ${OUTPUTDIR}/hepmc_files/events_${SEED}.hepmc
rm -r ${DIR}/${SEED}