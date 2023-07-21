import sys
import math
import ROOT
from array import array
from ROOT import TFile, TTree
import numpy as np

# TODO
# is last track state position at calo?
# Bz should be stored in the in the tree
# should allow for multiple gen links to hit (probablyhas to be done in the previous edm4hep formation stage)

debug = True

"""
source /cvmfs/fcc.cern.ch/sw/latest/setup.sh

python scripts/make_pftree_clic.py /afs/cern.ch/work/s/selvaggi/private/particleflow/fcc/out_reco_edm4hep.root tree.root 

"""
c_light = 2.99792458e8
Bz = 4.0
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
    p = math.sqrt(px * px + py * py + pz * pz)
    energy = math.sqrt(p * p + mchp * mchp)
    theta = math.acos(pz / p)
    # print(p, theta, phi, energy)
    return p, theta, phi, energy


def get_genparticle_daughters(i, mcparts, daughters):

    p = mcparts[i]

    daughter_positions = []
    for j in range(p.daughters_begin, p.daughters_end):
        # print(j, daughters[j].index)
        daughter_positions.append(daughters[j].index)
        # break

    return daughter_positions


def get_genparticle_parents(i, mcparts, parents):

    p = mcparts[i]

    parent_positions = []
    for j in range(p.parents_begin, p.parents_end):
        # print(j, daughters[j].index)
        parent_positions.append(parents[j].index)
        # break

    return parent_positions


def find_mother_particle(j, gen_part_coll, gen_parent_link_indexmc):
    parent_p = j
    counter = 0
    while len(np.reshape(np.array(parent_p), -1)) < 1.5:
        if type(parent_p) == list:
            parent_p = parent_p[0]
        parent_p_r = get_genparticle_parents(
            parent_p,
            gen_part_coll,
            gen_parent_link_indexmc,
        )
        pp_old = parent_p
        counter = counter + 1
        if len(np.reshape(np.array(parent_p_r), -1)) < 1.5:
            print(parent_p, parent_p_r)
        parent_p = parent_p_r

    return pp_old


def find_gen_link(
    j,
    id,
    gen_link_indexreco,
    gen_link_indexmc,
    gen_link_weight,
    genpart_indexes,
    calo=False,
    gen_part_coll=None,
    gen_parent_link_indexmc=None,
):

    reco_positions = []
    ## extract position of gen particles in the gen-trk trk link collection
    for i, l in enumerate(gen_link_indexreco):
        if l.index == j and l.collectionID == id:
            reco_positions.append(i)

    # now extract corresponid mc part position in the gen-trk gen collection
    gen_positions = []
    gen_weights = []
    for idx in reco_positions:
        gen_positions.append(gen_link_indexmc[idx].index)
        gen_weights.append(gen_link_weight[idx].weight)

    # now make sure that the corresponding gen part exists and will be stored in the tree
    # print(gen_positions, gen_weights)
    indices = []

    for i, pos in enumerate(gen_positions):
        if pos in genpart_indexes:
            if calo:
                mother = find_mother_particle(
                    genpart_indexes[pos], gen_part_coll, gen_parent_link_indexmc
                )
                indices.append(mother)
            else:
                indices.append(genpart_indexes[pos])

    # print(id, indices, gen_positions, gen_weights)

    indices += [-1] * (5 - len(indices))
    gen_weights += [-1] * (5 - len(gen_weights))

    return indices, gen_weights


## global params
CALO_RADIUS_IN_MM = 1500

if len(sys.argv) < 2:
    print(" Usage: make_pftree.py input_file output_file")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

# Opening the input file containing the tree (output of stage1.py)
infile = TFile.Open(input_file, "READ")

ev = infile.Get("events")

genparts = "MCParticles"
genparts_parents = "MCParticles#0"
genparts_daughters = "MCParticles#1"


## track stuff
tracks = ("SiTracks_Refitted", 45)
trackstates = "SiTracks_1"
gen_track_links0 = "SiTracksMCTruthLink#0"
gen_track_links1 = "SiTracksMCTruthLink#1"
gen_track_weights = "SiTracksMCTruthLink"

## calo stuff
ecal_barrel = ("ECALBarrel", 46)
ecal_endcap = ("ECALEndcap", 47)
ecal_other = ("ECALOther", 48)
hcal_barrel = ("HCALBarrel", 49)
hcal_endcap = ("HCALEndcap", 50)
hcal_other = ("HCALOther", 51)
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


### store here the position of the corresponding gen particles associated to the hit in flat format (same info as above but easier to read)
hit_genlink0 = ROOT.std.vector("int")()
hit_genlink1 = ROOT.std.vector("int")()
hit_genlink2 = ROOT.std.vector("int")()
hit_genlink3 = ROOT.std.vector("int")()
hit_genlink4 = ROOT.std.vector("int")()

## this is the fraction of the energy depoisited by that gen particle in this hit
hit_genweight0 = ROOT.std.vector("float")()
hit_genweight1 = ROOT.std.vector("float")()
hit_genweight2 = ROOT.std.vector("float")()
hit_genweight3 = ROOT.std.vector("float")()
hit_genweight4 = ROOT.std.vector("float")()


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

t.Branch("hit_genlink0", hit_genlink0)
t.Branch("hit_genlink1", hit_genlink1)
t.Branch("hit_genlink2", hit_genlink2)
t.Branch("hit_genlink3", hit_genlink3)
t.Branch("hit_genlink4", hit_genlink4)
t.Branch("hit_genweight0", hit_genweight0)
t.Branch("hit_genweight1", hit_genweight1)
t.Branch("hit_genweight2", hit_genweight2)
t.Branch("hit_genweight3", hit_genweight3)
t.Branch("hit_genweight4", hit_genweight4)

t.Branch("part_p", part_p)
t.Branch("part_theta", part_theta)
t.Branch("part_phi", part_phi)
t.Branch("part_m", part_m)
t.Branch("part_pid", part_pid)

event_number[0] = 0
for i, e in enumerate(ev):
    if debug:
        if i == 100:
            break

    number_of_hist_with_no_genlinks = 0

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

    hit_genlink0.clear()
    hit_genlink1.clear()
    hit_genlink2.clear()
    hit_genlink3.clear()
    hit_genlink4.clear()

    hit_genweight0.clear()
    hit_genweight1.clear()
    hit_genweight2.clear()
    hit_genweight3.clear()
    hit_genweight4.clear()

    if (i + 1) % 1000 == 0:
        print(" ... processed {} events ...".format(i + 1))

    ## STORE ALL STATUS 1 GENPARTICLES
    n_part[0] = 0

    if debug:
        print("")
        print(" ----- new event: {} ----------".format(event_number[0]))
        print("")

    genpart_indexes_pre = (
        dict()
    )  ## key: index in gen particle collection, value: position in stored gen particle array
    indexes_genpart_pre = (
        dict()
    )  ## key: position in stored gen particle array, value: index in gen particle collection

    genpart_indexes = (
        dict()
    )  ## key: index in gen particle collection, value: position in stored gen particle array
    indexes_genpart = (
        dict()
    )  ## key: position in stored gen particle array, value: index in gen particle collection

    gen_parent_link_indexmc = getattr(e, genparts_parents)
    gen_daughter_link_indexmc = getattr(e, genparts_daughters)
    gen_part_coll = getattr(e, genparts)

    n_part_pre = 0
    for j, part in enumerate(gen_part_coll):

        p = math.sqrt(
            part.momentum.x**2 + part.momentum.y**2 + part.momentum.z**2
        )
        theta = math.acos(part.momentum.z / p)
        phi = math.atan2(part.momentum.y, part.momentum.x)

        if debug:
            print(
                "all genparts: N: {}, PID: {}, Q: {}, P: {:.2e}, Theta: {:.2e}, Phi: {:.2e}, M: {:.2e}, X(m): {:.3f}, Y(m): {:.3f}, R(m): {:.3f}, Z(m): {:.3f}, status: {}, parents: {}, daughters: {}".format(
                    j,
                    part.PDG,
                    part.charge,
                    p,
                    theta,
                    phi,
                    part.mass,
                    part.vertex.x * 1e-03,
                    part.vertex.y * 1e-03,
                    math.sqrt(part.vertex.x**2 + part.vertex.y**2) * 1e-03,
                    part.vertex.z * 1e-03,
                    part.generatorStatus,
                    get_genparticle_parents(
                        j,
                        gen_part_coll,
                        gen_parent_link_indexmc,
                    ),
                    get_genparticle_daughters(
                        j,
                        gen_part_coll,
                        gen_daughter_link_indexmc,
                    ),
                )
            )
            # part.daughters_begin,  part.daughters_end, part.parents_begin,  part.parents_end, D1: {}, D2: {}, M1: {}, M2: {}

        ## store all gen parts for now
        genpart_indexes_pre[j] = n_part_pre
        indexes_genpart_pre[n_part_pre] = j
        n_part_pre += 1

        """
        # exclude neutrinos (and pi0 for now)
        if part.generatorStatus == 1 and abs(part.PDG) not in [12, 14, 16, 111]:

            genpart_indexes_pre[j] = n_part_pre
            indexes_genpart_pre[n_part_pre] = j
            n_part_pre += 1

        # extract the photons from the pi0
        elif part.generatorStatus == 1 and part.PDG == 111:

            daughters = get_genparticle_daughters(
                j, gen_part_coll, gen_daughter_link_indexmc
            )

            if len(daughters) != 2:
                print("STRANGE PI0 DECAY")

            for d in daughters:
                a = gen_part_coll[d]
                genpart_indexes_pre[d] = n_part_pre
                indexes_genpart_pre[n_part_pre] = d
                n_part_pre += 1
        """

    # TODO: for now exclude gen particle that have decayed/interacted before the calo
    for j in range(n_part_pre):

        part = gen_part_coll[indexes_genpart_pre[j]]

        daughters = get_genparticle_daughters(
            indexes_genpart_pre[j], gen_part_coll, gen_daughter_link_indexmc
        )

        # check if particles has interacted, if it did remove it from the list of gen particles
        # if len(daughters) > 0:
        #    continue

        p = math.sqrt(
            part.momentum.x**2 + part.momentum.y**2 + part.momentum.z**2
        )
        theta = math.acos(part.momentum.z / p)
        phi = math.atan2(part.momentum.y, part.momentum.x)

        part_p.push_back(p)
        part_theta.push_back(theta)
        part_phi.push_back(phi)
        part_m.push_back(part.mass)
        part_pid.push_back(part.PDG)

        genpart_indexes[indexes_genpart_pre[j]] = n_part[0]
        indexes_genpart[n_part[0]] = indexes_genpart_pre[j]
        n_part[0] += 1

    if debug:
        print("")
        print(genpart_indexes)
        for j in range(n_part[0]):
            part = gen_part_coll[indexes_genpart[j]]
            p = math.sqrt(
                part.momentum.x**2 + part.momentum.y**2 + part.momentum.z**2
            )
            theta = math.acos(part.momentum.z / p)
            phi = math.atan2(part.momentum.y, part.momentum.x)
            print(
                "stored genparts: N: {}, PID: {}, P: {:.2e}, Theta: {:.2e}, Phi: {:.2e}, M: {:.2e}".format(
                    j, part.PDG, p, theta, phi, part.mass
                )
            )

    n_hit[0] = 0
    ## STORE ALL RECONSTRUCTED TRACKS

    gen_track_link_indextr = getattr(e, gen_track_links0)
    gen_track_link_indexmc = getattr(e, gen_track_links1)
    gen_track_link_weight = getattr(e, gen_track_weights)

    track_coll = tracks[0]
    track_collid = tracks[1]

    if debug:
        print("")
    for j, track in enumerate(getattr(e, track_coll)):

        # there are 4 track states , accessible via 4*j, 4*j+1, 4*j+2, 4*j+3
        # TODO check that this is the last track state, presumably, the one that gives coordinates at calo

        # first store track state at vertex
        trackstate = getattr(e, trackstates)[4 * j]

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

        hit_type.push_back(0)  # 0 for tracks at vertex

        gen_indices, gen_weights = find_gen_link(
            j,
            track_collid,
            gen_track_link_indextr,
            gen_track_link_indexmc,
            gen_track_link_weight,
            genpart_indexes,
        )

        link_vector = ROOT.std.vector("int")()
        for idx in gen_indices:
            link_vector.push_back(idx)

        ngen = len(link_vector)

        if ngen == 0:
            number_of_hist_with_no_genlinks += 1
            if debug:
                print("  -> WARNING: this track with no gen-link")

        hit_genlink.push_back(link_vector)  # linked to first particle by default now

        genlink = -1
        if ngen > 0:
            genlink = link_vector[0]

        if len(gen_indices) > 0:
            hit_genlink0.push_back(gen_indices[0])
        if len(gen_indices) > 1:
            hit_genlink1.push_back(gen_indices[1])
        if len(gen_indices) > 2:
            hit_genlink2.push_back(gen_indices[2])
        if len(gen_indices) > 3:
            hit_genlink3.push_back(gen_indices[3])
        if len(gen_indices) > 4:
            hit_genlink4.push_back(gen_indices[4])

        if len(gen_indices) > 0:
            hit_genweight0.push_back(gen_weights[0])
        if len(gen_indices) > 1:
            hit_genweight1.push_back(gen_weights[1])
        if len(gen_indices) > 2:
            hit_genweight2.push_back(gen_weights[2])
        if len(gen_indices) > 3:
            hit_genweight3.push_back(gen_weights[3])
        if len(gen_indices) > 4:
            hit_genweight4.push_back(gen_weights[4])

        if debug:
            print(
                "track at vertex: N: {}, P: {:.2e}, Theta: {:.2e}, Phi: {:.2e}, X(m): {:.3f}, Y(m): {:.3f}, R(m): {:.3f}, Z(m): {:.3f}, r(m): {:.3f}, gen links: {}".format(
                    n_hit[0],
                    track_mom[0],
                    track_mom[1],
                    track_mom[2],
                    x * 1e-03,
                    y * 1e-03,
                    R * 1e-03,
                    z * 1e-03,
                    r * 1e-03,
                    list(link_vector),
                )
            )

        n_hit[0] += 1

        ## now access trackstate at calo
        trackstate = getattr(e, trackstates)[4 * j + 3]

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

        hit_type.push_back(1)  # 0 for tracks at calo

        gen_indices, gen_weights = find_gen_link(
            j,
            track_collid,
            gen_track_link_indextr,
            gen_track_link_indexmc,
            gen_track_link_weight,
            genpart_indexes,
        )

        link_vector = ROOT.std.vector("int")()
        for idx in gen_indices:
            link_vector.push_back(idx)

        ngen = len(link_vector)

        if ngen == 0:
            number_of_hist_with_no_genlinks += 1
            if debug:
                print("  -> WARNING: this track with no gen-link")

        hit_genlink.push_back(link_vector)  # linked to first particle by default now

        genlink = -1
        if ngen > 0:
            genlink = link_vector[0]

        if len(gen_indices) > 0:
            hit_genlink0.push_back(gen_indices[0])
        if len(gen_indices) > 1:
            hit_genlink1.push_back(gen_indices[1])
        if len(gen_indices) > 2:
            hit_genlink2.push_back(gen_indices[2])
        if len(gen_indices) > 3:
            hit_genlink3.push_back(gen_indices[3])
        if len(gen_indices) > 4:
            hit_genlink4.push_back(gen_indices[4])

        if len(gen_indices) > 0:
            hit_genweight0.push_back(gen_weights[0])
        if len(gen_indices) > 1:
            hit_genweight1.push_back(gen_weights[1])
        if len(gen_indices) > 2:
            hit_genweight2.push_back(gen_weights[2])
        if len(gen_indices) > 3:
            hit_genweight3.push_back(gen_weights[3])
        if len(gen_indices) > 4:
            hit_genweight4.push_back(gen_weights[4])

        if debug:
            print(
                "track at calo: N: {}, P: {:.2e}, Theta: {:.2e}, Phi: {:.2e}, X(m): {:.3f}, Y(m): {:.3f}, R(m): {:.3f}, Z(m): {:.3f}, r(m): {:.3f}, gen links: {}".format(
                    n_hit[0],
                    track_mom[0],
                    track_mom[1],
                    track_mom[2],
                    x * 1e-03,
                    y * 1e-03,
                    R * 1e-03,
                    z * 1e-03,
                    r * 1e-03,
                    list(link_vector),
                )
            )

        n_hit[0] += 1

    gen_calohit_link_indexhit = getattr(e, gen_calo_links0)
    gen_calohit_link_indexmc = getattr(e, gen_calo_links1)
    gen_calohit_link_weight = getattr(e, gen_calo_weights)

    calohit_collections = [
        ecal_barrel[0],
        hcal_barrel[0],
        ecal_endcap[0],
        hcal_endcap[0],
        ecal_other[0],
        hcal_other[0],
    ]
    calohit_collection_ids = [
        ecal_barrel[1],
        hcal_barrel[1],
        ecal_endcap[1],
        hcal_endcap[1],
        ecal_other[1],
        hcal_other[1],
    ]

    for k, calohit_coll in enumerate(calohit_collections):
        if debug:
            print("")
        for j, calohit in enumerate(getattr(e, calohit_coll)):

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

            theta = math.acos(z / r)
            phi = math.atan2(y, x)

            hit_theta.push_back(theta)
            hit_phi.push_back(phi)

            htype = 2  # 2 if ECAL, 3 if HCAL
            if "HCAL" in calohit_coll:
                htype = 3

            hit_type.push_back(htype)  # 0 for calo hits

            gen_indices, gen_weights = find_gen_link(
                j,
                calohit_collection_ids[k],
                gen_calohit_link_indexhit,
                gen_calohit_link_indexmc,
                gen_calohit_link_weight,
                genpart_indexes,
                calo=True,
                gen_part_coll=gen_part_coll,
                gen_parent_link_indexmc=gen_parent_link_indexmc,
            )

            link_vector = ROOT.std.vector("int")()
            for idx in gen_indices:
                link_vector.push_back(idx)

            ngen = len(link_vector)

            if ngen == 0:
                number_of_hist_with_no_genlinks += 1
                # if debug:
                #    print("  -> WARNING: this calo hit has no gen-link")

            hit_genlink.push_back(
                link_vector
            )  # linked to first particle by default now

            genlink = -1
            if ngen > 0:
                genlink = link_vector[0]

            if len(gen_indices) > 0:
                hit_genlink0.push_back(gen_indices[0])
            if len(gen_indices) > 1:
                hit_genlink1.push_back(gen_indices[1])
            if len(gen_indices) > 2:
                hit_genlink2.push_back(gen_indices[2])
            if len(gen_indices) > 3:
                hit_genlink3.push_back(gen_indices[3])
            if len(gen_indices) > 4:
                hit_genlink4.push_back(gen_indices[4])

            if len(gen_indices) > 0:
                hit_genweight0.push_back(gen_weights[0])
            if len(gen_indices) > 1:
                hit_genweight1.push_back(gen_weights[1])
            if len(gen_indices) > 2:
                hit_genweight2.push_back(gen_weights[2])
            if len(gen_indices) > 3:
                hit_genweight3.push_back(gen_weights[3])
            if len(gen_indices) > 4:
                hit_genweight4.push_back(gen_weights[4])

            # if debug:
            #     print(
            #         "calo hit type: {}, N: {}, E: {:.2e}, X(m): {:.3f}, Y(m): {:.3f}, R(m): {:.3f}, Z(m): {:.3f}, r(m): {:.3f}, gen links: {}".format(
            #             htype,
            #             n_hit[0],
            #             calohit.energy,
            #             x * 1e-03,
            #             y * 1e-03,
            #             R * 1e-03,
            #             z * 1e-03,
            #             r * 1e-03,
            #             list(link_vector),
            #         )
            #     )

            n_hit[0] += 1

    if debug:
        print("total number of hits: {}".format(n_hit[0]))
        print(
            "total number of hits with no gen links: {}".format(
                number_of_hist_with_no_genlinks
            )
        )
        print(": {}".format(number_of_hist_with_no_genlinks))

    if n_hit[0] <= number_of_hist_with_no_genlinks:
        print(
            "  --> WARNING: all hists in this event have no gen link associated or simply no hits, skipping event"
        )

    else:
        event_number[0] += 1
        t.Fill()

t.SetDirectory(out_root)
t.Write()
