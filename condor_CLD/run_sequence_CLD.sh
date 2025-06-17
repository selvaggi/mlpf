#!/bin/bash

HOMEDIR=${1}
GUNCARD=${2}
NEV=${3}
SEED=${4}
OUTPUTDIR=${5}

mkdir -p /eos/user/m/mgarciam/datasets_mlpf/condor_dataset_CLD/
mkdir -p /eos/user/m/mgarciam/datasets_mlpf/condor_dataset_CLD/${SEED}
cd /eos/user/m/mgarciam/datasets_mlpf/condor_dataset_CLD/${SEED}


cp -r ${HOMEDIR}/gun/gun_random_angle.cpp .
cp -r ${HOMEDIR}/gun/compile_gun_RA.x .

cp -r /afs/cern.ch/work/m/mgarciam/private/CLDConfig/CLDConfig/cld_steer.py .
cp -r /afs/cern.ch/work/m/mgarciam/private/CLDConfig/CLDConfig/CLDReconstruction.py .
cp -r /afs/cern.ch/work/m/mgarciam/private/CLDConfig/CLDConfig/PandoraSettings .
#cp -r ${HOMEDIR}/scripts/make_pftree_clic.py .

cp -r ${HOMEDIR}/gun/${GUNCARD} .
echo ${HOMEDIR}/gun/${GUNCARD}
echo ${INPUTFILE}

## first recompile gun locally 
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

source /cvmfs/sw.hsf.org/spackages6/key4hep-stack/2023-01-15/x86_64-centos7-gcc11.2.0-opt/csapx/setup.sh

echo "  "
echo " ================================================================================ "
echo "  "
echo "run simulation ..."
echo "  "
echo " ================================================================================ "
echo "  "
ddsim --compactFile $K4GEO/FCCee/CLD/compact/CLD_o2_v05/CLD_o2_v05.xml --outputFile out_sim_edm4hep.root --steeringFile cld_steer.py --inputFiles events.hepmc --numberOfEvents ${NEV} --random.seed ${SEED}

echo "  "
echo " ================================================================================ "
echo "  "
echo "run reconstruction ..."
echo "  "
echo " ================================================================================ "
echo "  "

k4run CLDReconstruction.py -n ${NEV}  --EventDataSvc.input out_sim_edm4hep.root --PodioOutput.filename out_reco_edm4hep.root

# echo "  "
# echo " ================================================================================ "
# echo "  "
# echo "produce pf tree ..."
# echo "  "
# echo " ================================================================================ "
# echo "  "

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
