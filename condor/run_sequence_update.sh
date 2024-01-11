#!/bin/bash

HOMEDIR=${1}
GUNCARD=${2}
NEV=${3}
SEED=${4}
OUTPUTDIR=${5}

mkdir /eos/user/m/mgarciam/datasets_mlpf/condor_dataset_test/${SEED}
cd /eos/user/m/mgarciam/datasets_mlpf/condor_dataset_test/${SEED}

cp -r ${HOMEDIR}/gun/gun.cpp .
cp -r ${HOMEDIR}/gun/compile_gun.x .

cp -r ${HOMEDIR}/gun/clic_steer.py .
cp -r ${HOMEDIR}/gun/clicRec_e4h_input.py .
cp -r ${HOMEDIR}/gun/PandoraSettings .
cp -r ${HOMEDIR}/scripts/make_pftree_clic.py .

cp -r ${HOMEDIR}/gun/${GUNCARD} .

echo ${INPUTFILE}

## first recompile gun locally 
echo "  "
echo " ================================================================================ "
echo "  "
echo "compiling gun"
echo "  "
echo " ===============================================================================  "
echo "  "

source compile_gun.x

# produce hepmc event file 
echo 'nevents '${NEV} >> ${GUNCARD}

echo "  "
echo " ================================================================================ "
echo "  "
echo "running gun"
echo "  "
echo " ===============================================================================  "
echo "  "

./gun ${GUNCARD}

echo "  "
echo " ================================================================================ "
echo "  "
echo "gun complete ..."
echo "  "
echo " ================================================================================ "
echo "  "
## 

# # When you want to revert the changes:

source /cvmfs/sw.hsf.org/spackages6/key4hep-stack/2023-01-15/x86_64-centos7-gcc11.2.0-opt/csapx/setup.sh

echo "  "
echo " ================================================================================ "
echo "  "
echo "run simulation ..."
echo "  "
echo " ================================================================================ "
echo "  "

ddsim --compactFile $LCGEO/CLIC/compact/CLIC_o3_v14/CLIC_o3_v14.xml --outputFile out_sim_edm4hep.root --steeringFile clic_steer.py --inputFiles events.hepmc --numberOfEvents ${NEV} --random.seed ${SEED}

echo "  "
echo " ================================================================================ "
echo "  "
echo "run reconstruction ..."
echo "  "
echo " ================================================================================ "
echo "  "

k4run clicRec_e4h_input.py -n ${NEV}  --EventDataSvc.input out_sim_edm4hep.root --PodioOutput.filename out_reco_edm4hep.root

echo "  "
echo " ================================================================================ "
echo "  "
echo "produce pf tree ..."
echo "  "
echo " ================================================================================ "
echo "  "

# python make_pftree_clic.py out_reco_edm4hep.root tree.root

# echo "  "
# echo " ================================================================================ "
# echo "  "
# echo "copying output file  ..."
# echo "  "
# echo " ================================================================================ "
# echo "  "

# mkdir -p ${OUTPUTDIR}
# python /afs/cern.ch/work/f/fccsw/public/FCCutils/eoscopy.py tree.root ${OUTPUTDIR}/pf_tree_${SEED}.root

# echo "  "
# echo " ================================================================================ "
# echo "  "
# echo "file copied ... "
# echo "  "
# echo " ================================================================================ "
# echo "  "