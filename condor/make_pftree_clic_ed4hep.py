import sys
import math
import ROOT
from array import array
from ROOT import TFile, TTree
import numpy as np
from podio import root_io
import edm4hep


debug = True

"""
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh
"""

## global params
CALO_RADIUS_IN_MM = 1500

# "/eos/user/m/mgarciam/datasets_mlpf/condor_dataset_CLD/1/out_reco_edm4hep_edm4hep.root"

reader = root_io.Reader(
    "/eos/user/m/mgarciam/datasets_mlpf/condor_dataset_CLD/1/out_reco_edm4hep_edm4hep.root"
)

i = 0
for event in reader.get("events"):
    dc_hits = event.get("ECALBarrel")
    mc_particles = event.get("MCParticles")
    j = 0
    for particle in mc_particles:
        print(particle.getObjectID().index, particle.getPDG())
    for dc_hit in dc_hits:
        print("   New hit: ")
        position = dc_hit.getPosition()
        x = position.x
        y = position.y
        z = position.z
        print("position", x, y, z)
        print(dir(dc_hit))
        mcParticle = dc_hit.getMCParticle()
        print(mcParticle.getPDG())
        object_id = mcParticle.getObjectID()
        # print(object_id.index)
        genlink0 = object_id.index
        print(mcParticle.getPDG(), genlink0)
        if j > 3:
            break
        j += 1
