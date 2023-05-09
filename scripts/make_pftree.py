import sys
import math
import ROOT
from array import array
from ROOT import TFile, TTree

debug = False

"""
source /cvmfs/fcc.cern.ch/sw/latest/setup.sh
python scripts/make_pftree.py /afs/cern.ch/user/b/brfranco/work/public/Fellow/FCCSW/221123/LAr_scripts/FCCSW_ecal/output_fullCalo_SimAndDigi_withCluster_MagneticField_False_pMin_5000_MeV_ThetaMinMax_50_130_pdgId_211_pythiaFalse_NoiseFalse.root tree.root
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

vector_float = ROOT.std.vector("float")()
vector_vector_float = ROOT.std.vector(ROOT.std.vector("float"))()

event_number = array("i", [0])
n_hit = array("i", [0])
n_part = array("i", [0])

hit_x = ROOT.std.vector("float")()
hit_y = ROOT.std.vector("float")()
hit_z = ROOT.std.vector("float")()
hit_t = ROOT.std.vector("float")()
hit_p = ROOT.std.vector("float")()
hit_e = ROOT.std.vector("float")()
hit_theta = ROOT.std.vector("float")()
hit_phi = ROOT.std.vector("float")()

### store here whether track: 0 /ecal: 1/hcal: 2
hit_type = ROOT.std.vector("int")()

### store here the position of the corresponding gen particles associated to the hit
hit_genlink = ROOT.std.vector(ROOT.std.vector("int"))()

## store here true information
part_p = ROOT.std.vector("float")()
part_e = ROOT.std.vector("float")()
part_theta = ROOT.std.vector("float")()
part_phi = ROOT.std.vector("float")()
part_m = ROOT.std.vector("float")()
part_pid = ROOT.std.vector("float")()


t.Branch("event_number", event_number, "event_number/I")
t.Branch("n_hit", n_hit, "n_hit/I")
t.Branch("n_part", n_part, "n_part/I")

t.Branch("hit_x", hit_x)
t.Branch("hit_y", hit_y)
t.Branch("hit_z", hit_z)
t.Branch("hit_t", hit_t)
t.Branch("hit_p", hit_p)
t.Branch("hit_e", hit_e)
t.Branch("hit_theta", hit_theta)
t.Branch("hit_phi", hit_phi)
t.Branch("hit_type", hit_type)

# Create a branch for the hit_genlink_flat
t.Branch("hit_genlink", hit_genlink)

t.Branch("part_p", part_p)
t.Branch("part_theta", part_theta)
t.Branch("part_phi", part_phi)
t.Branch("part_m", part_m)
t.Branch("part_pid", part_pid)

for i, e in enumerate(ev):

    event_number[0] = i

    # clear all the vectors
    hit_x.clear()
    hit_y.clear()
    hit_z.clear()
    hit_t.clear()
    hit_p.clear()
    hit_e.clear()
    hit_theta.clear()
    hit_phi.clear()
    hit_type.clear()
    hit_genlink.clear()
    part_p.clear()
    part_e.clear()
    part_theta.clear()
    part_phi.clear()
    part_m.clear()
    part_pid.clear()

    if (i + 1) % 1000 == 0:
        print(" ... processed {} events ...".format(i + 1))

    n_hit[0] = 0

    ## TODO: UPDATE WHEN MORE GRANULAR INFORMATION AVAILABLE
    n_part[0] = len(ev.genParticles)

    """
    for j in range(max_hit):
    """

    for j, part in enumerate(ev.genParticles):
        part_p.push_back(math.sqrt(part.momentum.x ** 2 + part.momentum.y ** 2 + part.momentum.z ** 2))
        part_theta.push_back(math.acos(part.momentum.z / part_p[j]))
        part_phi.push_back(math.atan2(part.momentum.y, part.momentum.x))
        part_m.push_back(part.mass)
        part_pid.push_back(part.PDG)

    ## TODO: WHEN TRACKS WILL BE AVAILABLES, FOR NOW SIMPLY TAKE FIRST GENPARTICLE AS SUCH
    """
    for track in ev.Tracks ...
    """

    hit_x.push_back(0.0)
    hit_y.push_back(0.0)
    hit_z.push_back(0.0)
    hit_t.push_back(0.0)
    hit_p.push_back(part_p[0])
    hit_e.push_back(math.sqrt(part_p[0] ** 2 + part_m[0] ** 2))
    hit_theta.push_back(part_theta[0])
    hit_phi.push_back(part_phi[0])
    hit_type.push_back(0)  # 0 for tracks

    link_vector = ROOT.std.vector("int")()
    link_vector.push_back(0)
    hit_genlink.push_back(link_vector) # linked to first particle by default now

    n_hit[0] += 1

    ## now loop ovoer ECAL hits
    for hit in ev.ECalBarrelPositionedHits:
        if hit.energy < MIN_CALO_ENERGY:
            continue

        r = math.sqrt(hit.position.x ** 2 + hit.position.y ** 2 + hit.position.z ** 2)
        theta = math.acos(hit.position.z / r)
        phi = math.atan2(hit.position.y, hit.position.x)

        hit_x.push_back(hit.position.x)
        hit_y.push_back(hit.position.y)
        hit_z.push_back(hit.position.z)
        hit_t.push_back(0.0)
        hit_p.push_back(hit.energy)
        hit_e.push_back(hit.energy)
        hit_theta.push_back(theta)
        hit_phi.push_back(phi)
        hit_type.push_back(1)  # 1 for ECAL

        link_vector = ROOT.std.vector("int")()
        link_vector.push_back(0)
        hit_genlink.push_back(link_vector) # linked to first particle by default now

        n_hit[0] += 1

    ## now loop over HCAL hits
    for hit in ev.HCalBarrelPositionedHits:
        if hit.energy < MIN_CALO_ENERGY:
            continue

        r = math.sqrt(hit.position.x ** 2 + hit.position.y ** 2 + hit.position.z ** 2)
        theta = math.acos(hit.position.z / r)
        phi = math.atan2(hit.position.y, hit.position.x)

        hit_x.push_back(hit.position.x)
        hit_y.push_back(hit.position.y)
        hit_z.push_back(hit.position.z)
        hit_t.push_back(0.0)
        hit_p.push_back(hit.energy)
        hit_e.push_back(hit.energy)
        hit_theta.push_back(theta)
        hit_phi.push_back(phi)
        hit_type.push_back(2) # 2 for HCAL

        link_vector = ROOT.std.vector("int")()
        link_vector.push_back(0)
        hit_genlink.push_back(link_vector) # linked to first particle by default now

        n_hit[0] += 1

    print(i, n_hit[0], n_part[0])
    t.Fill()


t.SetDirectory(out_root)
t.Write()
