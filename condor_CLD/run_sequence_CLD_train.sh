#!/bin/bash

HOMEDIR=${1}
GUNCARD=${2}
NEV=${3}
SEED=${4}
OUTPUTDIR=${5}
DIR=${6}
mkdir ${DIR}
mkdir ${DIR}/${SEED}
cd ${DIR}/${SEED}

SAMPLE="Zcard" #main card


# cp -r ${HOMEDIR}/gun/gun_random_angle.cpp .
# cp -r ${HOMEDIR}/gun/compile_gun_RA.x .
if [[ "${SAMPLE}" == "gun" ]] 
    cp -r ${HOMEDIR}/gun/gun.cpp .
    cp -r ${HOMEDIR}/gun/CMakeLists.txt . 
fi 


if [[ "${SAMPLE}" == "Zcard" ]]
then 
      cp ${HOMEDIR}/Pythia_generation/${SAMPLE}.cmd card.cmd
      echo "Random:seed=${SEED}" >> card.cmd
      cat card.cmd
      cp ${HOMEDIR}/Pythia_generation/pythia.py ./
fi

cp -r /afs/cern.ch/work/m/mgarciam/private/CLDConfig/CLDConfig/cld_steer.py .
cp -r /afs/cern.ch/work/m/mgarciam/private/CLDConfig/CLDConfig/CLDReconstruction.py .
mkdir ${DIR}/${SEED}/PandoraSettingsCLD
cp /afs/cern.ch/work/m/mgarciam/private/CLDConfig/CLDConfig/PandoraSettingsCLD/* ./PandoraSettingsCLD/
cp -r ${HOMEDIR}/condor/make_pftree_clic_bindings.py .
cp -r ${HOMEDIR}/condor/tree_tools.py .
cp -r ${HOMEDIR}/gun/${GUNCARD} .


echo "  "
echo " ================================================================================ "
echo "  "
echo "compiling gun"
echo "  "
echo " ===============================================================================  "
echo "  "

wrapperfunction() {
    source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh -r 2024-03-07
}
wrapperfunction

# source compile_gun.x

# Build gun 
if [[ "${SAMPLE}" == "gun" ]] 
then 
    mkdir build install
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=../install
    make install -j 8
    cd ..
    ./build/gun ${GUNCARD} 
fi

if [[ "${SAMPLE}" == "Zcard" ]]
then
    k4run pythia.py -n $NEV --Dumper.Filename out.hepmc --Pythia8.PythiaInterface.pythiacard card.cmd
    cp out.hepmc events.hepmc
fi




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

k4run CLDReconstruction.py -n ${NEV}  --inputFiles out_sim_edm4hep.root --outputBasename out_reco_edm4hep

echo "  "
echo " ================================================================================ "
echo "  "
echo "produce pf tree ..."
echo "  "
echo " ================================================================================ "
echo "  "

python make_pftree_clic_bindings.py out_reco_edm4hep_edm4hep.root tree.root False False

echo "  "
echo " ================================================================================ "
echo "  "
echo "copying output file  ..."
echo "  "
echo " ================================================================================ "
echo "  "

mkdir -p ${OUTPUTDIR}
python /afs/cern.ch/work/f/fccsw/public/FCCutils/eoscopy.py tree.root ${OUTPUTDIR}/pf_tree_${SEED}.root

echo "  "
echo " ================================================================================ "
echo "  "
echo "file copied ... "
echo "  "
echo " ================================================================================ "
echo "  "