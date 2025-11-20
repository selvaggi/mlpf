import glob
import os


# def chunks(lst, n):
#     """Yield successive n-sized chunks from lst."""
#     for i in range(0, len(lst), n):
#         yield lst[i : i + n]


# def write_script(infiles, outpath):
#     s = []
#     s += ["#!/bin/bash"]
#     s += ["#SBATCH --partition short"]
#     s += ["#SBATCH --cpus-per-task 1"]
#     s += ["#SBATCH --mem-per-cpu 4G"]
#     s += ["#SBATCH -o logs/slurm-%x-%j-%N.out"]
#     s += ["set -e"]

#     for inf in infiles:
#         s += [
#             "singularity exec -B /local /home/software/singularity/pytorch.simg:2024-08-02 python3 "
#             + f"scripts/fccee_cld/postprocessing.py --input {inf} --outpath {outpath}"
#         ]
#     ret = "\n".join(s)

#     ret += "\n"
#     return ret


samples = "/eos/experiment/fcc/users/m/mgarciam/mlpf/condor/tt365_2025_09_23_key4hep_20250529_CLD_r20250526/"
infiles = list(glob.glob(f"{samples}/*/out_reco_edm4hep_REC.edm4hep.root"))
print(infiles)
# ichunk = 1
# for sample, outpath in samples:
#     infiles = list(glob.glob(f"{sample}/*.root"))
#     os.makedirs(outpath, exist_ok=True)
#     for infiles_chunk in chunks(infiles, 20):
#         scr = write_script(infiles_chunk, outpath)
#         ofname = f"jobscripts/postproc_{ichunk}.sh"
#         with open(ofname, "w") as outfi:
#             outfi.write(scr)
#         ichunk += 1