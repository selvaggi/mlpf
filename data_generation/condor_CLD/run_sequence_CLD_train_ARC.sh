#!/bin/bash

####################################################################################################
HOMEDIR=${1} # path to where it's ran from e.g. /afs/cern.ch/user/f/fmokhtar/MLPF_datageneration
GUNCARD=${2}  # example p8_ee_Zuds_ecm91
NEV=${3} # 100
SEED=${4} # 1
OUTPUTDIR=${5} # dir where the final file is copied
DIR=${6} # dir where intermidiate files are created 
SAMPLE=${7} # can be "gun" or "Zcard"
CLDGEO=${8} # default is CLD_o2_v06 (CLD+ARC is CLD_o3_v01 https://github.com/key4hep/k4geo/blob/main/FCCee/CLD/compact/CLD_o3_v01/CLD_o3_v01.xml)
PATHCLDCONFIG=${9} # path to CLD config 
if [ -z "$CLDGEO" ]; then
    echo "Will use default CLD geometry version CLD_o2_v06"
    CLDGEO=CLD_o2_v06
else
    echo "Will use CLD geometry version $CLDGEO"
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


echo ${HOMEDIR}
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
    xrdcp ${HOMEDIR}/data_generation/pythia/${GUNCARD}.cmd card.cmd
    echo "Random:seed=${SEED}" >> card.cmd
    cat card.cmd
    k4run ${HOMEDIR}/data_generation/pythia/pythia.py -n $NEV --Dumper.Filename out.hepmc --Pythia8.PythiaInterface.pythiacard card.cmd
    cp out.hepmc events.hepmc
fi


echo ${PATHCLDCONFIG}
xrdcp -r ${PATHCLDCONFIG}/* .
ddsim --compactFile $K4GEO/FCCee/CLD/compact/$CLDGEO/$CLDGEO.xml  --outputFile out_sim_edm4hep.root --steeringFile ${PATHCLDCONFIG}/cld_steer.py --inputFiles events.hepmc --numberOfEvents ${NEV} --random.seed ${SEED}

ddsim --compactFile $K4GEO/FCCee/CLD/compact/CLD_o3_v01/CLD_o3_v01.xml  --outputFile out_sim_edm4hep_ARC.root --steeringFile ${PATHCLDCONFIG}/cld_arc_steer.py --inputFiles events.hepmc --numberOfEvents ${NEV} --random.seed ${SEED}

# copy large input files via xrootd (recommended)


# running both gen tracking and CT tracking
k4run CLDReconstruction.py -n ${NEV}  --inputFiles out_sim_edm4hep.root --outputBasename out_reco_edm4hep 

k4run CLDReconstruction.py -n ${NEV}  --inputFiles out_sim_edm4hep_ARC.root --outputBasename out_reco_edm4hep_ARC  

wrapperfunction() {
    source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/setup.sh
}
wrapperfunction

if [ ! -f "out_reco_edm4hep_REC.parquet" ]; then
    cp -r ${HOMEDIR}/data_generation/preprocessing/ .
    python  -m preprocessing.dataset_creation --input  out_reco_edm4hep_ARC_REC.edm4hep.root  --outpath .   --dataset

    python  -m preprocessing.dataset_creation --input  out_reco_edm4hep_REC.edm4hep.root  --outpath .   --dataset
fi

mkdir -p ${OUTPUTDIR}
mkdir -p ${OUTPUTDIR}/arc/
mkdir -p ${OUTPUTDIR}/05/

python /afs/cern.ch/work/f/fccsw/public/FCCutils/eoscopy.py out_reco_edm4hep_ARC_REC.parquet ${OUTPUTDIR}/arc/pf_tree_${SEED}_arc.parquet
python /afs/cern.ch/work/f/fccsw/public/FCCutils/eoscopy.py out_reco_edm4hep_REC.parquet ${OUTPUTDIR}/05/pf_tree_${SEED}.parquet
rm -r ${DIR}/${SEED}