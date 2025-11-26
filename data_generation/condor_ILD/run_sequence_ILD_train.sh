#!/usr/bin/env bash
set -eo pipefail

# Read arguments from condor submit file
file=$1
energy=$2
process=$3
shift 3 # avoid passing cli arguments to the key4hep load

# Define paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P )"
PROJECT_ROOT="${SCRIPT_DIR}/../.."
TEMP_DATA_DIR="${PROJECT_ROOT}/data/ILD/intermediate/e${energy}/job_${process}"

# Create temporary directory
rm -rf ${TEMP_DATA_DIR}
mkdir -p ${TEMP_DATA_DIR}

# Patch missing CaloHit-MCTruth links and convert to edm4hep
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh -r 2025-11-26
k4run ${PROJECT_ROOT}/data_generation/condor_ILD/old_ild_prod_to_edm4hep.py --inputFiles=${file} --outputFileBase=${TEMP_DATA_DIR}/${process}

# Run dataset creation script
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/setup.sh
cd ${PROJECT_ROOT}/data_generation/
python -m preprocessing.dataset_creation_ILD --input ${TEMP_DATA_DIR}/${process}_REC.edm4hep.root --outpath ${TEMP_DATA_DIR}

# Cleanup and move final parquet file
mv ${TEMP_DATA_DIR}/*.parquet ${TEMP_DATA_DIR}/../${process}.parquet
rm -rf ${TEMP_DATA_DIR}