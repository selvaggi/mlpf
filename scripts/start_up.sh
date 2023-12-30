#!/bin/bash

gpuopt=""
files=$(ls -l /dev/nvidia* 2> /dev/null | egrep -c '\n')
if [[ "$files" != "0" ]]
then
gpuopt="--nv"
fi

#this is a singularity problem only fixed recently
unset LD_LIBRARY_PATH
unset PYTHONPATH
sing=`which singularity`
unset PATH
cd
export SINGULARITY_CACHEDIR=/afs/cern.ch/work/m/mgarciam/private/Containers/cache/
singularity shell --nv -B /eos -B /afs  docker://dologarcia/colorsinglet:v4
