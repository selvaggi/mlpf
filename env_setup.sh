# using this to avoid weird threading errors
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
conda activate /home/gkrzmanc/env
export LD_LIBRARY_PATH=/eos/home-g/gkrzmanc/miniforge3/lib:/home/gkrzmanc/env/lib:
