import sys
import math
import ROOT
from array import array
from ROOT import TFile, TTree

# TODO
# is last track state position at calo?
# Bz should be stored in the in the tree
# should allow for multiple gen links to hit (probablyhas to be done in the previous edm4hep formation stage)

debug = True

"""
source /cvmfs/fcc.cern.ch/sw/latest/setup.sh

python scripts/make_pftree_clic.py /afs/cern.ch/work/s/selvaggi/private/particleflow/fcc/out_reco_edm4hep.root tree.root 

"""
c_light =  2.99792458e8
Bz = 4.
mchp = 0.139570

def omega_to_pt(omega):
    a = c_light * 1e3 * 1e-15
    return a * Bz / abs(omega)

def track_momentum(trackstate):
    pt = omega_to_pt(trackstate.omega)
    phi = trackstate.phi
    pz = trackstate.tanLambda * pt
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    p = math.sqrt(px*px + py*py + pz*pz)
    energy = math.sqrt(p*p + mchp*mchp)
    theta = math.acos(pz/p)
    #print(p, theta, phi, energy)
    return p, theta, phi, energy

def find_gen_link(j, id, gen_link_indexreco, gen_link_indexmc, gen_link_weight, genpart_indexes):
    
    reco_positions = []
    ## extract position of gen particles in the gen-trk trk link collection
    for i, l in enumerate(gen_link_indexreco):
        if l.index == j and l.collectionID == id:
            reco_positions.append(i)

    # now extract corresponid mc part position in the gen-trk gen collection
    gen_positions = []
    for idx in reco_positions:
        gen_positions.append(gen_link_indexmc[idx].index)

    # now make sure that the corresponding gen part exists and will be stored in the tree    
    indices = []
    for pos in gen_positions:
        if pos in genpart_indexes:
            indices.append(genpart_indexes[pos])
    
    #print(id, indices)
    return indices

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

genparts = "MCParticles"

## track stuff
tracks = ("SiTracks_Refitted",45)
trackstates = "SiTracks_1"
gen_track_links0 = "SiTracksMCTruthLink#0"
gen_track_links1 = "SiTracksMCTruthLink#1"
gen_track_weights = "SiTracksMCTruthLink"

## calo stuff
ecal_barrel = ("ECALBarrel",46)
ecal_endcap = ("ECALEndcap",47)
ecal_other = ("ECALOther",48)
hcal_barrel = ("HCALBarrel",49)
hcal_endcap = ("HCALEndcap",50)
hcal_other = ("HCALOther",51)
gen_calo_links0 = "CalohitMCTruthLink#0"
gen_calo_links1 = "CalohitMCTruthLink#1"
gen_calo_weights = "CalohitMCTruthLink"

numberOfEntries = ev.GetEntries()

out_root = TFile(output_file, "RECREATE")
t = TTree("events", "pf tree lar")

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

    ## STORE ALL STATUS 1 GENPARTICLES
    n_part[0] = 0

    if debug:
        print("")
        print(" ----- new event ----------")
        print("")
    genpart_indexes = dict()  ## remember in which position these gen particles were located in the original tree
    for j, part in enumerate(getattr(ev,genparts)):
        if part.generatorStatus == 1 and abs(part.PDG) not in [12,14,16]:
            p = math.sqrt(part.momentum.x ** 2 + part.momentum.y ** 2 + part.momentum.z ** 2)
            theta = math.acos(part.momentum.z / p)
            phi = math.atan2(part.momentum.y, part.momentum.x)
            part_p.push_back(p)
            part_theta.push_back(theta)
            part_phi.push_back(phi)
            part_m.push_back(part.mass)
            part_pid.push_back(part.PDG)
            genpart_indexes[j] = n_part[0]

            if debug:
                print("genpart: N: {}, PID: {}, P: {:.2e}, Theta: {:.2e}, Phi: {:.2e}, M: {:.2e}".format(n_part[0], part.PDG, p, theta, phi, part.mass))
            n_part[0] += 1



    n_hit[0] = 0
    ## STORE ALL RECONSTRUCTED TRACKS
    
    gen_track_link_indextr = getattr(ev, gen_track_links0)
    gen_track_link_indexmc = getattr(ev, gen_track_links1)
    gen_track_link_weight = getattr(ev, gen_track_weights)
    
    track_coll = tracks[0]
    track_collid = tracks[1]
    
    for j, track in enumerate(getattr(ev,track_coll)):
    
        # there are 4 track states , accessible via 4*j, 4*j+1, 4*j+2, 4*j+3 
        # TODO check that this is the last track state, presumably, the one that gives coordinates at calo
        trackstate = getattr(ev,trackstates)[4*j+3]

        #print(trackstate.referencePoint.x, trackstate.referencePoint.y, trackstate.referencePoint.z)
        
        x = trackstate.referencePoint.x
        y = trackstate.referencePoint.y
        z = trackstate.referencePoint.z
        R = math.sqrt(x**2 + y**2)
        r = math.sqrt(x**2 + y**2 + z**2)

        hit_x.push_back(x)
        hit_y.push_back(y)
        hit_z.push_back(z)
        hit_t.push_back(trackstate.time)
       

        track_mom = track_momentum(trackstate)

        hit_p.push_back(track_mom[0])
        hit_theta.push_back(track_mom[1])
        hit_phi.push_back(track_mom[2])
        hit_e.push_back(-1)

        hit_type.push_back(0)  # 0 for tracks

        gen_indices = find_gen_link(j, track_collid, gen_track_link_indextr, gen_track_link_indexmc, gen_track_link_weight, genpart_indexes)

        link_vector = ROOT.std.vector("int")()
        for idx in gen_indices:
            link_vector.push_back(idx)        
        #print(gen_indices)
        hit_genlink.push_back(link_vector) # linked to first particle by default now
    
        if debug:
            print("track: N: {}, P: {:.2e}, Theta: {:.2e}, Phi: {:.2e}, X: {:.2e}, Y: {:.2e}, R: {:.2e}, Z: {:.2e}, r: {:.2e}".format(n_hit[0], track_mom[0], track_mom[1], track_mom[2], x, y, R, z, r))

        n_hit[0] += 1


    gen_calohit_link_indexhit = getattr(ev, gen_calo_links0)
    gen_calohit_link_indexmc = getattr(ev, gen_calo_links1)
    gen_calohit_link_weight = getattr(ev, gen_calo_weights)

    calohit_collections = [ecal_barrel[0], hcal_barrel[0], ecal_endcap[0], hcal_endcap[0], ecal_other[0], hcal_other[0]]
    calohit_collection_ids = [ecal_barrel[1], hcal_barrel[1], ecal_endcap[1], hcal_endcap[1], ecal_other[1], hcal_other[1]]
    
    for k, calohit_coll in enumerate(calohit_collections):
        for j, calohit in enumerate(getattr(ev,calohit_coll)):

            x = calohit.position.x
            y = calohit.position.y
            z = calohit.position.z
            R = math.sqrt(x**2 + y**2)
            r = math.sqrt(x**2 + y**2 + z**2)

            hit_x.push_back(x)
            hit_y.push_back(y)
            hit_z.push_back(z)
            hit_t.push_back(calohit.time)
            hit_p.push_back(-1)
            hit_e.push_back(calohit.energy)
             
            theta = math.acos(z/r)
            phi = math.atan2(y, x)
            
            hit_theta.push_back(theta)
            hit_phi.push_back(phi)
            hit_type.push_back(1)  # 0 for tracks

            gen_indices = find_gen_link(j, calohit_collection_ids[k], gen_calohit_link_indexhit, gen_calohit_link_indexmc, gen_calohit_link_weight, genpart_indexes)

            link_vector = ROOT.std.vector("int")()
            for idx in gen_indices:
                link_vector.push_back(idx)
            
            hit_genlink.push_back(link_vector) # linked to first particle by default now

            if debug:
                print("calo : ID: {}, N: {}, E: {:.2e}, X: {:.2e}, Y: {:.2e}, R: {:.2e}, Z: {:.2e}, r: {:.2e}".format(calohit_collection_ids[k], n_hit[0], calohit.energy, x, y, R, z, r))

            n_hit[0] += 1

    if debug:
        print("total number of hits: {}".format(n_hit[0]))
    t.Fill()

t.SetDirectory(out_root)
t.Write()
