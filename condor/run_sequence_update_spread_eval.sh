#!/bin/bash

HOMEDIR=${1}
GUNCARD=${2}
NEV=${3}
SEED=${4}
OUTPUTDIR=${5}

DIR="/eos/user/m/mgarciam/datasets_mlpf/CLIC/condor/eval/211523/"
mkdir ${DIR}
mkdir ${DIR}${SEED}
cd ${DIR}${SEED}


cp -r ${HOMEDIR}/gun/gun_random_angle.cpp .
cp -r ${HOMEDIR}/gun/compile_gun_RA.x .

cp -r ${HOMEDIR}/gun/clic_steer.py .
cp -r ${HOMEDIR}/gun/clicRec_e4h_input.py .
cp -r ${HOMEDIR}/gun/PandoraSettings .
cp -r ${HOMEDIR}/condor/make_pftree_clic_bindings.py .
cp -r ${HOMEDIR}/condor/tree_tools.py .
cp -r ${HOMEDIR}/gun/${GUNCARD} .
echo ${HOMEDIR}/gun/${GUNCARD}
# echo ${INPUTFILE}

# first recompile gun locally 
echo "  "
echo " ================================================================================ "
echo "  "
echo "compiling gun"
echo "  "
echo " ===============================================================================  "
echo "  "

source compile_gun_RA.x

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

# When you want to revert the changes:

# # #source /cvmfs/sw.hsf.org/spackages6/key4hep-stack/2023-01-15/x86_64-centos7-gcc11.2.0-opt/csapx/setup.sh
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh

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

python make_pftree_clic_bindings.py out_reco_edm4hep.root tree.root True True

echo "  "
echo " ================================================================================ "
echo "  "
echo "copying output file  ..."
echo "  "
echo " ================================================================================ "
echo "  "

mkdir -p ${OUTPUTDIR}
#mkdir -p ${OUTPUTDIR}/gun_config/
#cp -r ${HOMEDIR}/gun/${GUNCARD} ${OUTPUTDIR}/gun_config/
python /afs/cern.ch/work/f/fccsw/public/FCCutils/eoscopy.py tree.root ${OUTPUTDIR}/pf_tree_${SEED}.root 

echo "  "
echo " ================================================================================ "
echo "  "
echo "file copied ... "
echo "  "
echo " ================================================================================ "
echo "  "