import sys
from array import array
import math
from ROOT import TFile, TTree

debug = False

"""
source /cvmfs/fcc.cern.ch/sw/latest/setup.sh
python make_pftree.py /afs/cern.ch/user/b/brfranco/work/public/Fellow/FCCSW/221123/LAr_scripts/FCCSW_ecal/output_fullCalo_SimAndDigi_withCluster_MagneticField_False_pMin_5000_MeV_ThetaMinMax_50_130_pdgId_211_pythiaFalse_NoiseFalse.root tree.root
"""

# source /cvmfs/fcc.cern.ch/sw/latest/setup.sh

## global params
MIN_CALO_ENERGY = 0.002

if len(sys.argv) < 2:
    print(" Usage: make_pftree.py input_file output_file")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

# Opening the input file containing the tree (output of stage1.py)
infile = TFile.Open(input_file, "READ")

ev = infile.Get("events")
numberOfEntries = ev.GetEntries()

out_root = TFile(output_file, "RECREATE")
t = TTree("events", "pf tree lar")

event_number = array("i", [0])
n_hit = array("i", [0])
n_part = array("i", [0])

max_hit = 10000
max_part = 1000

hit_x = array("f", max_hit * [0.0])
hit_y = array("f", max_hit * [0.0])
hit_z = array("f", max_hit * [0.0])
hit_t = array("f", max_hit * [0.0])
hit_p = array("f", max_hit * [0.0])
hit_e = array("f", max_hit * [0.0])
hit_theta = array("f", max_hit * [0.0])
hit_phi = array("f", max_hit * [0.0])

### store here whether track: 0 /ecal: 1/hcal: 2
hit_type = array("f", max_hit * [0.0])

### store here the position of the corresponding gen particle associated to the hit
hit_link1 = array("f", max_hit * [0.0])
hit_link2 = array("f", max_hit * [0.0])
hit_link3 = array("f", max_hit * [0.0])
hit_link4 = array("f", max_hit * [0.0])
hit_link5 = array("f", max_hit * [0.0])


## store here true information
part_p = array("f", max_part * [0.0])
part_e = array("f", max_part * [0.0])
part_theta = array("f", max_part * [0.0])
part_phi = array("f", max_part * [0.0])
part_m = array("f", max_part * [0.0])
part_pid = array("f", max_part * [0.0])


t.Branch("event_number", event_number, "event_number/I")
t.Branch("n_hit", n_hit, "n_hit/I")
t.Branch("n_part", n_part, "n_part/I")

t.Branch("hit_x", hit_x, "hit_x[n_hit]/F")
t.Branch("hit_y", hit_y, "hit_y[n_hit]/F")
t.Branch("hit_z", hit_z, "hit_z[n_hit]/F")
t.Branch("hit_t", hit_t, "hit_t[n_hit]/F")
t.Branch("hit_p", hit_p, "hit_p[n_hit]/F")
t.Branch("hit_e", hit_e, "hit_e[n_hit]/F")
t.Branch("hit_theta", hit_theta, "hit_theta[n_hit]/F")
t.Branch("hit_phi", hit_phi, "hit_phi[n_hit]/F")
t.Branch("hit_type", hit_type, "hit_type[n_hit]/F")
t.Branch("hit_link1", hit_link1, "hit_link1[n_hit]/F")
t.Branch("hit_link2", hit_link2, "hit_link2[n_hit]/F")
t.Branch("hit_link3", hit_link3, "hit_link3[n_hit]/F")
t.Branch("hit_link4", hit_link4, "hit_link4[n_hit]/F")
t.Branch("hit_link5", hit_link5, "hit_link5[n_hit]/F")

t.Branch("part_p", part_p, "part_p[n_part]/F")
t.Branch("part_theta", part_theta, "part_theta[n_part]/F")
t.Branch("part_phi", part_phi, "part_phi[n_part]/F")
t.Branch("part_m", part_m, "part_m[n_part]/F")
t.Branch("part_pid", part_pid, "part_pid[n_part]/F")

for i, e in enumerate(ev):

    event_number[0] = i

    if (i + 1) % 1000 == 0:
        print(" ... processed {} events ...".format(i + 1))

    n_hit[0] = 0

    ## TODO: UPDATE WHEN MORE GRANULAR INFORMATION AVAILABLE
    n_part[0] = len(ev.genParticles)

    """
    for j in range(max_hit):
    """

    for j, part in enumerate(ev.genParticles):
        part_p[j] = math.sqrt(part.momentum.x ** 2 + part.momentum.y ** 2 + part.momentum.z ** 2)
        part_theta[j] = math.acos(part.momentum.z / part_p[j])
        part_phi[j] = math.atan2(part.momentum.y, part.momentum.x)
        part_m[j] = part.mass
        part_pid[j] = part.PDG

    ## TODO: WHEN TRACKS WILL BE AVAILABLES, FOR NOW SIMPLY TAKE FIRST GENPARTICLE AS SUCH
    """
    for track in ev.Tracks ...
    """

    hit_x[0] = 0.0
    hit_y[0] = 0.0
    hit_z[0] = 0.0
    hit_t[0] = 0.0
    hit_p[0] = part_p[0]
    hit_e[0] = math.sqrt(part_p[0] ** 2 + part_m[0] ** 2)
    hit_theta[0] = part_theta[0]
    hit_phi[0] = part_phi[0]
    hit_type[0] = 0  # 0 for tracks
    hit_link1[0] = 0  # linked to first particle by default now
    hit_link2[0] = -1
    hit_link3[0] = -1
    hit_link4[0] = -1
    hit_link5[0] = -1

    n_hit[0] += 1

    ## now loop ovoer ECAL hits
    j = 0
    for hit in ev.ECalBarrelPositionedHits:
        if hit.energy < MIN_CALO_ENERGY:
            continue

        index = n_hit[0] + j
        r = math.sqrt(hit.position.x ** 2 + hit.position.y ** 2 + hit.position.z ** 2)
        theta = math.acos(hit.position.z / r)
        phi = math.atan2(hit.position.y, hit.position.x)

        hit_x[index] = hit.position.x
        hit_y[index] = hit.position.y
        hit_z[index] = hit.position.z
        hit_t[index] = 0.0
        hit_p[index] = hit.energy
        hit_e[index] = hit.energy
        hit_theta[index] = theta
        hit_phi[index] = phi
        hit_type[index] = 1  # 1 for ECAL
        hit_link1[index] = 0  # linked to first particle by default now
        hit_link2[index] = -1
        hit_link3[index] = -1
        hit_link4[index] = -1
        hit_link5[index] = -1

        n_hit[0] += 1
        j += 1

    ## now loop over HCAL hits
    j = 0
    for hit in ev.HCalBarrelPositionedHits:
        if hit.energy < MIN_CALO_ENERGY:
            continue

        index = n_hit[0] + j

        r = math.sqrt(hit.position.x ** 2 + hit.position.y ** 2 + hit.position.z ** 2)
        theta = math.acos(hit.position.z / r)
        phi = math.atan2(hit.position.y, hit.position.x)

        hit_x[index] = hit.position.x
        hit_y[index] = hit.position.y
        hit_z[index] = hit.position.z
        hit_t[index] = 0.0
        hit_p[index] = hit.energy
        hit_e[index] = hit.energy
        hit_theta[index] = theta
        hit_phi[index] = phi
        hit_type[index] = 2  # 2 for HCAL
        hit_link1[index] = 0  # linked to first particle by default now
        hit_link2[index] = -1
        hit_link3[index] = -1
        hit_link4[index] = -1
        hit_link5[index] = -1

        n_hit[0] += 1
        j += 1

    print(i, n_hit[0], n_part[0])
    t.Fill()


t.SetDirectory(out_root)
t.Write()
