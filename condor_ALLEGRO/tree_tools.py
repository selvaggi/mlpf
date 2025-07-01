import sys
import math
import ROOT
from array import array
from ROOT import TFile, TTree
import numpy as np
from podio import root_io
import edm4hep

# constants
c_light = 2.99792458e8      # speed of light
Bz_clic = 4.0               # B field, CLIC
Bz_cld = 2.0                # B field, CLD
mchp = 0.139570             # muon mass

# convert track curvature to pT
def omega_to_pt(omega, isclic):
    if isclic:
        Bz = Bz_clic
    else:
        Bz = Bz_cld
    a = c_light * 1e3 * 1e-15
    return a * Bz / abs(omega)

# get track momentum from track state
def track_momentum(trackstate, isclic=True):
    pt = omega_to_pt(trackstate.omega, isclic)
    phi = trackstate.phi
    pz = trackstate.tanLambda * pt
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    p = math.sqrt(px * px + py * py + pz * pz)
    energy = math.sqrt(p * p + mchp * mchp)
    theta = math.acos(pz / p)
    # print(p, theta, phi, energy)
    return p, theta, phi, energy, px, py, pz

# get indices of daughters of genparticle with position i in collection mcparts
def get_genparticle_daughters(i, mcparts):

    p = mcparts[i]
    daughters = p.getDaughters()
    daughter_positions = []
    # for j in range(p.daughters_begin, p.daughters_end):
    #     # print(j, daughters[j].index)
    #     daughter_positions.append(daughters[j].index)
    #     # break
    for daughter in daughters:
        daughter_positions.append(daughter.getObjectID().index)

    return daughter_positions


def find_pandora_cluster_of_hit(hit_index, hit_collection, cluster_collection):
    cluster_energy_found = 0
    for index_c, cluster in enumerate(cluster_collection):
        cluster_hits = cluster.getHits()
        cluster_energy = cluster.getEnergy()
        for index_h, hit in enumerate(cluster_hits):
            object_id_hit = hit.getObjectID()
            if (
                hit_index == object_id_hit.index
                and object_id_hit.collectionID == hit_collection
            ):
                pandora_cluster_index = index_c
                cluster_energy_found = cluster_energy
                break
            else:
                pandora_cluster_index = -1
                cluster_energy_found = 0
        if pandora_cluster_index >= 0:
            break
    return pandora_cluster_index, cluster_energy_found


# def check_pandora_pfos(event):
#     pandora_pfo = "PandoraPFOs"
#     pandora_pfos_event = event.get(pandora_pfo)
#     for index, pfo in enumerate(pandora_pfos_event):
#         clusters_pfo = pfo.getClusters()
#         for index_c, cluster in enumerate(clusters_pfo):
#             cluster_hits = cluster.getHits()
#             for index_h, hit in enumerate(cluster_hits):
#         # clusters = pfo.getClusters()
#         # for cluster in clusters:
#         #     print("clusters", dir(cluster))
#         #     print("id", cluster.getObjectID().index)
#         #     cluster_energy = cluster.getEnergy()
#         #     print("cluster energy", cluster_energy)
#         break


def find_pandora_pfo_and_cluster_of_hit(
    hit_index, hit_collection, cluster_collection, pfo_collection
):
    pandora_cluster_index = -1
    pfo_index = -1
    cluster_energy_found = 0
    pfo_energy_found = 0
    reference_point_found = None
    momentum_found = None
    pid_reconstructed_pfo = -1
    for index_pfo, pfo in enumerate(pfo_collection):
        # print("index pfo ", index_pfo)
        clusters_pfo = pfo.getClusters()
        pfo_energy = pfo.getEnergy()
        reference_point = pfo.getReferencePoint()
        momentum = pfo.getMomentum()
        pid_reconstructed_pfo = pfo.getPDG()
        for index_c, cluster in enumerate(clusters_pfo):
            # print("index cluster ", index_c)
            cluster_hits = cluster.getHits()
            cluster_energy = cluster.getEnergy()
            cluster_id = cluster.getObjectID().index
            for index_h, hit in enumerate(cluster_hits):
                object_id_hit = hit.getObjectID()
                if (
                    hit_index == object_id_hit.index
                    and object_id_hit.collectionID == hit_collection
                ):
                    pandora_cluster_index = cluster_id
                    cluster_energy_found = cluster_energy
                    pfo_energy_found = pfo_energy
                    reference_point_found = reference_point
                    momentum_found = momentum
                    pfo_index = index_pfo
                    break
                else:
                    pandora_cluster_index = -1
                    cluster_energy_found = 0
                    pfo_energy_found = 0
                    pfo_index = -1
            if pandora_cluster_index >= 0:
                break
        if pandora_cluster_index >= 0:
            break
    # print("PFO", pfo_index, pfo_energy_found)
    return (
        pandora_cluster_index,
        cluster_energy_found,
        pfo_energy_found,
        pfo_index,
        reference_point_found,
        momentum_found,
        pid_reconstructed_pfo
    )


def find_pandora_pfo_track(hit_index, hit_collection, pfo_collection):
    pandora_cluster_index = -1
    pandora_pfo_index = -1
    pfo_energy_found = 0
    pfo_momentum_found = None
    pfo_reference_point_found = None
    pandora_pid = None
    for index_pfo, pfo in enumerate(pfo_collection):
        tracks_pfo = pfo.getTracks()
        pfo_energy = pfo.getEnergy()
        pfo_momentum = pfo.getMomentum()
        pfo_reference_point = pfo.getReferencePoint()
        pandora_pid = pfo.getPDG()
        for index_t, track in enumerate(tracks_pfo):
            # print("index cluster ", index_c)
            track_index = track.getObjectID().index
            track_collection_id = track.getObjectID().collectionID

            if hit_index == track_index and track_collection_id == hit_collection:
                pandora_pfo_index = index_pfo
                pfo_energy_found = pfo_energy
                pfo_momentum_found = pfo_momentum
                pfo_reference_point_found = pfo_reference_point
                break
            else:
                pandora_pfo_index = -1
                pfo_energy_found = 0
                pfo_momentum_found = None
                pfo_reference_point_found = None
        if pandora_pfo_index >= 0:
            break
    # print(pandora_cluster_index, pfo_energy_found, pandora_pfo_index)
    return (
        pandora_cluster_index,
        pfo_energy_found,
        pandora_pfo_index,
        pfo_reference_point_found,
        pfo_momentum_found,
        pandora_pid
    )


def get_genparticle_parents(i, mcparts):

    p = mcparts[i]
    parents = p.getParents()
    # print(p.parents_begin(), p.parents_end())
    parent_positions = []
    # for j in range(p.parents_begin(), p.parents_end()):
    #     # print(j, daughters[j].index)
    #     parent_positions.append(parents[j].index)
    #     # break
    for parent in parents:
        parent_positions.append(parent.getObjectID().index)

    return parent_positions


def find_mother_particle(j, gen_part_coll):
    parent_p = j
    counter = 0
    while len(np.reshape(np.array(parent_p), -1)) < 1.5:
        if type(parent_p) == list:
            if len(parent_p) > 0:
                parent_p = parent_p[0]
            else:
                break
        parent_p_r = get_genparticle_parents(
            parent_p,
            gen_part_coll,
        )
        pp_old = parent_p
        counter = counter + 1
        # if len(np.reshape(np.array(parent_p_r), -1)) < 1.5:
        #     print(parent_p, parent_p_r)
        parent_p = parent_p_r
    # if j != pp_old:
    #     print("old parent and new parent", j, pp_old)
    return pp_old


def find_gen_link(
    j,
    id,
    SiTracksMCTruthLink,
    # gen_link_indexmc,
    # gen_link_weight,
    genpart_indexes,
    calo=False,
    gen_part_coll=None,
):
    # print(id)
    gen_positions = []
    gen_weights = []
    for i, l in enumerate(SiTracksMCTruthLink):
        # rec_ = l.getRec()
        rec_ = l.getFrom()
        object_id = rec_.getObjectID()
        index = object_id.index
        collectionID = object_id.collectionID
        # print(index, j, collectionID, id)
        if index == j and collectionID == id:
            # print(j, "found match")
            # gen_positions.append(l.getSim().getObjectID().index)
            gen_positions.append(l.getTo().getObjectID().index)
            weight = l.getWeight()
            gen_weights.append(weight)

    indices = []
    for i, pos in enumerate(gen_positions):
        if pos in genpart_indexes:
            if calo:
                mother = find_mother_particle(genpart_indexes[pos], gen_part_coll)
                indices.append(mother)
                indices.append(genpart_indexes[pos])
            else:
                indices.append(genpart_indexes[pos])

    indices += [-1] * (5 - len(indices))
    gen_weights += [-1] * (5 - len(gen_weights))

    return indices, gen_weights


def initialize(t):
    event_number = array("i", [0])
    n_hit = array("i", [0])
    n_part = array("i", [0])

    hit_chis = ROOT.std.vector("float")()
    hit_x = ROOT.std.vector("float")()
    hit_y = ROOT.std.vector("float")()
    hit_z = ROOT.std.vector("float")()
    hit_px = ROOT.std.vector("float")()
    hit_py = ROOT.std.vector("float")()
    hit_pz = ROOT.std.vector("float")()
    hit__pandora_x = ROOT.std.vector("float")()
    hit__pandora_y = ROOT.std.vector("float")()
    hit__pandora_z = ROOT.std.vector("float")()
    hit__pandora_px = ROOT.std.vector("float")()
    hit__pandora_py = ROOT.std.vector("float")()
    hit__pandora_pz = ROOT.std.vector("float")()
    pandora_pid = ROOT.std.vector("float")()
    hit_t = ROOT.std.vector("float")()
    hit_p = ROOT.std.vector("float")()
    hit_e = ROOT.std.vector("float")()
    hit_theta = ROOT.std.vector("float")()
    hit_phi = ROOT.std.vector("float")()
    hit_pandora_cluster_energy = ROOT.std.vector("float")()
    hit_pandora_pfo_energy = ROOT.std.vector("float")()
    ### store here whether track: 0 /ecal: 1/hcal: 2
    hit_type = ROOT.std.vector("int")()
    calohit_col = ROOT.std.vector("int")()

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
    part_px = ROOT.std.vector("float")()
    part_py = ROOT.std.vector("float")()
    part_pz = ROOT.std.vector("float")()
    part_ks_dataset = ROOT.std.vector("float")()
    part_vertex_x = ROOT.std.vector("float")()
    part_vertex_y = ROOT.std.vector("float")()
    part_vertex_z = ROOT.std.vector("float")()
    part_e = ROOT.std.vector("float")()
    part_theta = ROOT.std.vector("float")()
    part_phi = ROOT.std.vector("float")()
    part_m = ROOT.std.vector("float")()
    part_pid = ROOT.std.vector("float")()
    part_isDecayedInCalorimeter = ROOT.std.vector("float")()
    part_isDecayedInTracker = ROOT.std.vector("float")()

    t.Branch("event_number", event_number, "event_number/I")
    t.Branch("n_hit", n_hit, "n_hit/I")
    t.Branch("n_part", n_part, "n_part/I")

    t.Branch("hit_chis", hit_chis)
    t.Branch("hit_x", hit_x)
    t.Branch("hit_y", hit_y)
    t.Branch("hit_z", hit_z)
    t.Branch("hit_px", hit_px)
    t.Branch("hit_py", hit_py)
    t.Branch("hit_pz", hit_pz)
    t.Branch("hit__pandora_x", hit__pandora_x)
    t.Branch("hit__pandora_y", hit__pandora_y)
    t.Branch("pandora_pid", pandora_pid)
    t.Branch("hit__pandora_z", hit__pandora_z)
    t.Branch("hit__pandora_px", hit__pandora_px)
    t.Branch("hit__pandora_py", hit__pandora_py)
    t.Branch("hit__pandora_pz", hit__pandora_pz)
    t.Branch("hit_t", hit_t)
    t.Branch("hit_p", hit_p)
    t.Branch("hit_e", hit_e)
    t.Branch("hit_theta", hit_theta)
    t.Branch("hit_phi", hit_phi)
    t.Branch("hit_pandora_cluster_energy", hit_pandora_cluster_energy)
    t.Branch("hit_pandora_pfo_energy", hit_pandora_pfo_energy)
    t.Branch("hit_type", hit_type)
    t.Branch("calohit_col", calohit_col)

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
    t.Branch("part_ks_dataset", part_ks_dataset)
    t.Branch("part_px", part_px)
    t.Branch("part_py", part_py)
    t.Branch("part_pz", part_pz)
    t.Branch("part_vertex_x", part_vertex_x)
    t.Branch("part_vertex_y", part_vertex_y)
    t.Branch("part_vertex_z", part_vertex_z)
    t.Branch("part_e", part_e)
    t.Branch("part_theta", part_theta)
    t.Branch("part_phi", part_phi)
    t.Branch("part_m", part_m)
    t.Branch("part_pid", part_pid)
    t.Branch("part_isDecayedInCalorimeter", part_isDecayedInCalorimeter)
    t.Branch("part_isDecayedInTracker", part_isDecayedInTracker)

    dic = {
        "hit_chis": hit_chis,
        "hit_x": hit_x,
        "hit_y": hit_y,
        "hit_z": hit_z,
        "hit_px": hit_px,
        "hit_py": hit_py,
        "hit_pz": hit_pz,
        "pandora_pid": pandora_pid, 
        "hit__pandora_x": hit__pandora_x,
        "hit__pandora_y": hit__pandora_y,
        "hit__pandora_z": hit__pandora_z,
        "hit__pandora_px": hit__pandora_px,
        "hit__pandora_py": hit__pandora_py,
        "hit__pandora_pz": hit__pandora_pz,
        "hit_t": hit_t,
        "hit_p": hit_p,
        "hit_e": hit_e,
        "hit_theta": hit_theta,
        "hit_phi": hit_phi,
        "hit_type": hit_type,
        "calohit_col": calohit_col,
        "hit_genlink0": hit_genlink0,
        "hit_genlink1": hit_genlink1,
        "hit_genlink2": hit_genlink2,
        "hit_genlink3": hit_genlink3,
        "hit_genlink4": hit_genlink4,
        "hit_genlink": hit_genlink,
        "hit_genweight0": hit_genweight0,
        "hit_genweight1": hit_genweight1,
        "hit_genweight2": hit_genweight2,
        "hit_genweight3": hit_genweight3,
        "hit_genweight4": hit_genweight4,
        "hit_pandora_cluster_energy": hit_pandora_cluster_energy,
        "hit_pandora_pfo_energy": hit_pandora_pfo_energy,
        "part_p": part_p,
        "part_ks_dataset":part_ks_dataset,
        "part_px": part_px,
        "part_py": part_py,
        "part_pz": part_pz,
        "part_vertex_x": part_vertex_x,
        "part_vertex_y": part_vertex_y,
        "part_vertex_z": part_vertex_z,
        "part_theta": part_theta,
        "part_phi": part_phi,
        "part_m": part_m,
        "part_e": part_e,
        "part_pid": part_pid,
        "part_isDecayedInCalorimeter": part_isDecayedInCalorimeter,
        "part_isDecayedInTracker": part_isDecayedInTracker,
    }
    return (event_number, n_hit, n_part, dic, t)


def clear_dic(dic):
    for key in dic:
        dic[key].clear()
    return dic


def gen_particles_find(event, debug):
    genparts = "MCParticles"
    genparts_parents = "_MCParticles_parents"
    genparts_daughters = "_MCParticles_daughters"
    # gen_parent_link_indexmc = event.get(genparts_parents)
    # gen_daughter_link_indexmc = event.get(genparts_daughters)
    gen_part_coll = event.get(genparts)
    genpart_indexes_pre = (
        dict()
    )  ## key: index in gen particle collection, value: position in stored gen particle array
    indexes_genpart_pre = (
        dict()
    )  ## key: position in stored gen particle array, value: index in gen particle collection
    total_e = 0
    n_part_pre = 0
    e_pp = np.zeros(11)
    for j, part in enumerate(gen_part_coll):
        momentum = part.getMomentum()
        p = math.sqrt(momentum.x**2 + momentum.y**2 + momentum.z**2)
        if debug:
            if j < 11 and j > 1:
                e_pp[j] = p
                total_e = total_e + p
        theta = math.acos(momentum.z / p)
        phi = math.atan2(momentum.y, momentum.x)
        if debug:
            print(
                "all genparts: N: {}, PID: {}, Q: {}, P: {:.2e}, Theta: {:.2e}, Phi: {:.2e}, M: {:.2e}, X(m): {:.3f}, Y(m): {:.3f}, R(m): {:.3f}, Z(m): {:.3f}, status: {}, parents: {}, daughters: {}, decayed_traacker: {}".format(
                    j,
                    part.getPDG(),
                    part.getCharge(),
                    p,
                    theta,
                    phi,
                    part.getMass(),
                    part.getVertex().x * 1e-03,
                    part.getVertex().y * 1e-03,
                    math.sqrt(part.getVertex().x ** 2 + part.getVertex().y ** 2)
                    * 1e-03,
                    part.getVertex().z * 1e-03,
                    part.getGeneratorStatus(),
                    get_genparticle_parents(
                        j,
                        gen_part_coll,
                    ),
                    get_genparticle_daughters(
                        j,
                        gen_part_coll,
                    ),
                    part.isDecayedInTracker() * 1,
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
    return (
        genpart_indexes_pre,
        indexes_genpart_pre,
        n_part_pre,
        total_e,
        e_pp,
        gen_part_coll,
    )


def store_gen_particles(
    n_part_pre,
    gen_part_coll,
    indexes_genpart_pre,
    dic,
    n_part,
    debug,
):
    ##
    ## ['BITBackscatter', 'BITCreatedInSimulation', 'BITDecayedInCalorimeter', 'BITDecayedInTracker', 'BITEndpoint', 'BITLeftDetector',
    # 'BITOverlay', 'BITStopped', 'BITVertexIsNotEndpointOfParent',
    # '__add__', '__assign__', '__bool__', '__class__', '__delattr__', '__destruct__', '__dict__', '__dir__', '__dispatch__',
    # '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__invert__',
    #  '__le__', '__lt__', '__module__', '__mul__', '__ne__', '__neg__', '__new__', '__pos__', '__python_owns__', '__radd__', '__reduce__',
    # '__reduce_ex__', '__repr__', '__rmul__', '__rsub__', '__rtruediv__', '__setattr__', '__sizeof__', '__smartptr__', '__str__', '__sub__',
    # '__subclasshook__', '__truediv__', '__weakref__', 'addToDaughters', 'addToParents', 'clone', 'colorFlow', 'daughters_begin', 'daughters_end',
    #  'daughters_size', 'endpoint', 'getCharge', 'getColorFlow', 'getDaughters', 'getEndpoint', 'getEnergy', 'getGeneratorStatus', 'getMass',
    # 'getMomentum', 'getMomentumAtEndpoint', 'getObjectID', 'getPDG', 'getParents', 'getSimulatorStatus', 'getSpin', 'getTime', 'getVertex',
    # 'hasLeftDetector', 'id', 'isAvailable', 'isBackscatter', 'isCreatedInSimulation', 'isDecayedInCalorimeter', 'isDecayedInTracker',
    # 'isOverlay', 'isStopped', 'momentum', 'momentumAtEndpoint', 'operator MCParticle', 'parents_begin', 'parents_end', 'parents_size',
    #  'setBackscatter', 'setCharge', 'setColorFlow', 'setCreatedInSimulation', 'setDecayedInCalorimeter', 'setDecayedInTracker',
    # 'setEndpoint', 'setGeneratorStatus', 'setHasLeftDetector', 'setMass', 'setMomentum', 'setMomentumAtEndpoint', 'setOverlay',
    # 'setPDG', 'setSimulatorStatus', 'setSpin', 'setStopped', 'setTime', 'setVertex', 'setVertexIsNotEndpointOfParent', 'set_bit',
    #  'spin', 'unlink', 'vertex', 'vertexIsNotEndpointOfParent']
    # TODO: for now exclude gen particle that have decayed/interacted before the calo
    genpart_indexes = (
        dict()
    )  ## key: index in gen particle collection, value: position in stored gen particle array
    indexes_genpart = (
        dict()
    )  ## key: position in stored gen particle array, value: index in gen particle collection

    for j in range(n_part_pre):

        part = gen_part_coll[indexes_genpart_pre[j]]
        daughters = get_genparticle_daughters(indexes_genpart_pre[j], gen_part_coll)

        # check if particles has interacted, if it did remove it from the list of gen particles
        # if len(daughters) > 0:
        #    continue
        momentum = part.getMomentum()
        p = math.sqrt(momentum.x**2 + momentum.y**2 + momentum.z**2)
        theta = math.acos(momentum.z / p)
        phi = math.atan2(momentum.y, momentum.x)
        m = part.getMass()
        e = math.sqrt(m**2 + p**2)
        dic["part_p"].push_back(p)
        dic["part_px"].push_back(momentum.x)
        dic["part_py"].push_back(momentum.y)
        dic["part_pz"].push_back(momentum.z)
        dic["part_vertex_x"].push_back(part.getVertex().x)
        dic["part_vertex_y"].push_back(part.getVertex().y)
        dic["part_vertex_z"].push_back(part.getVertex().z)
        dic["part_theta"].push_back(theta)
        dic["part_phi"].push_back(phi)
        dic["part_m"].push_back(m)
        dic["part_e"].push_back(e)
        dic["part_pid"].push_back(part.getPDG())
        dic["part_ks_dataset"].push_back(0)
        dic["part_isDecayedInCalorimeter"].push_back(
            part.isDecayedInCalorimeter() * 1.0
        )
        dic["part_isDecayedInTracker"].push_back(part.isDecayedInTracker() * 1)

        genpart_indexes[indexes_genpart_pre[j]] = n_part[0]
        indexes_genpart[n_part[0]] = indexes_genpart_pre[j]
        n_part[0] += 1

    # if debug:
    #     print("")
    #     # print(genpart_indexes)
    #     for j in range(n_part[0]):
    #         part = gen_part_coll[indexes_genpart[j]]
    #         momentum = part.getMomentum()
    #         p = math.sqrt(momentum.x**2 + momentum.y**2 + momentum.z**2)
    #         theta = math.acos(momentum.z / p)
    #         phi = math.atan2(momentum.y, momentum.x)
    #         print(
    #             "stored genparts: N: {}, PID: {}, P: {:.2e}, Theta: {:.2e}, Phi: {:.2e}, M: {:.2e}".format(
    #                 j, part.getPDG(), p, theta, phi, part.getMass()
    #             )
    #         )
    return dic, genpart_indexes


def store_tracks(
    event,
    debug,
    dic,
    genpart_indexes,
    n_hit,
    number_of_hist_with_no_genlinks,
    store_pandora_hits=False,
    CLIC="True",
):
    if CLIC == "True":
        isclic = True
    else:
        isclic = False
    ## track stuff
    tracks = ("TracksFromGenParticles", 45)
    # trackstates = "SiTracks_1"
    SiTracksMCTruthLink = "TracksFromGenParticlesAssociation"
    # gen_track_links1 = "SiTracksMCTruthLink#1"
    # gen_track_weights = "SiTracksMCTruthLink"
    pandora_pfo = "PandoraPFOs"
    # gen_calo_links1 = "CalohitMCTruthLink#1"
    # gen_calo_weights = "CalohitMCTruthLink"
    # pandora_pfos_event = event.get(pandora_pfo)
    gen_track_link_indextr = event.get(SiTracksMCTruthLink)
    # gen_track_link_indexmc = event.get(gen_track_links1)
    # gen_track_link_weight = event.get(gen_track_weights)

    track_coll = tracks[0]
    track_collid = tracks[1]

    if debug:
        print("")
    for j, track in enumerate(event.get(track_coll)):
        # there are 4 track states , accessible via 4*j, 4*j+1, 4*j+2, 4*j+3
        # TODO check that this is the last track state, presumably, the one that gives coordinates at calo
        # first store track state at vertex

        trackstate = track.getTrackStates()[0]
        referencePoint = trackstate.referencePoint
        x = referencePoint.x
        y = referencePoint.y
        z = referencePoint.z
        R = math.sqrt(x**2 + y**2)
        r = math.sqrt(x**2 + y**2 + z**2)

        chi_s = track.getChi2()

        dic["hit_chis"].push_back(chi_s)
        dic["hit_x"].push_back(x)
        dic["hit_y"].push_back(y)
        dic["hit_z"].push_back(z)
        dic["hit_t"].push_back(trackstate.time)

        track_mom = track_momentum(trackstate, isclic=isclic)

        dic["hit_p"].push_back(track_mom[0])
        dic["hit_theta"].push_back(track_mom[1])
        dic["hit_phi"].push_back(track_mom[2])
        dic["hit_px"].push_back(track_mom[4])
        dic["hit_py"].push_back(track_mom[5])
        dic["hit_pz"].push_back(track_mom[6])
        dic["hit_e"].push_back(-1)

        dic["hit_type"].push_back(0)  # 0 for tracks at vertex
        dic["calohit_col"].push_back(0)
        # print(gen_track_link_indextr[0].MCRecoTrackParticleAssociation())
        gen_indices, gen_weights = find_gen_link(
            j,
            track.getObjectID().collectionID,
            gen_track_link_indextr,
            # gen_track_link_indexmc,
            # gen_track_link_weight,
            genpart_indexes,
        )
        # print("store_pandora_hits", store_pandora_hits)
        if store_pandora_hits == "True":
            (
                pandora_index,
                pandora_pfo_energy,
                pandora_index_pfo,
                reference_point_pandora,
                momentum_pandora,
                pid_pandora
            ) = find_pandora_pfo_track(
                track.getObjectID().index,
                track.getObjectID().collectionID,
                pandora_pfos_event,
            )
            dic["hit_pandora_cluster_energy"].push_back(0)
            dic["hit_pandora_pfo_energy"].push_back(pandora_pfo_energy)
            if reference_point_pandora is not None:
                dic["hit__pandora_px"].push_back(momentum_pandora.x)
                dic["hit__pandora_py"].push_back(momentum_pandora.y)
                dic["hit__pandora_pz"].push_back(momentum_pandora.z)
                dic["hit__pandora_x"].push_back(reference_point_pandora.x)
                dic["hit__pandora_y"].push_back(reference_point_pandora.y)
                dic["hit__pandora_z"].push_back(reference_point_pandora.z)
                dic["pandora_pid"].push_back(pid_pandora)
            else:
                dic["hit__pandora_px"].push_back(-1)
                dic["hit__pandora_py"].push_back(-1)
                dic["hit__pandora_pz"].push_back(-1)
                dic["hit__pandora_x"].push_back(-1)
                dic["hit__pandora_y"].push_back(-1)
                dic["hit__pandora_z"].push_back(-1)
                dic["pandora_pid"].push_back(-1)

        link_vector = ROOT.std.vector("int")()
        for idx in gen_indices:
            link_vector.push_back(idx)

        ngen = len(link_vector)

        if ngen == 0:
            number_of_hist_with_no_genlinks += 1
            if debug:
                print("  -> WARNING: this track with no gen-link")

        dic["hit_genlink"].push_back(
            link_vector
        )  # linked to first particle by default now

        genlink = -1
        if ngen > 0:
            genlink = link_vector[0]
       
        if len(gen_indices) > 0:
            dic["hit_genlink0"].push_back(gen_indices[0])

        if store_pandora_hits == "True":
            dic["hit_genlink1"].push_back(pandora_index_pfo)
        else:
            if len(gen_indices) > 1:
                dic["hit_genlink1"].push_back(gen_indices[1])
        if store_pandora_hits == "True":
            # print("storing calo hit")
            dic["hit_genlink2"].push_back(pandora_index_pfo)

        else:
            if len(gen_indices) > 2:
                dic["hit_genlink2"].push_back(gen_indices[2])
        if len(gen_indices) > 3:
            dic["hit_genlink3"].push_back(gen_indices[3])
        if len(gen_indices) > 4:
            dic["hit_genlink4"].push_back(gen_indices[4])

        if len(gen_indices) > 0:
            dic["hit_genweight0"].push_back(gen_weights[0])
        if len(gen_indices) > 1:
            dic["hit_genweight1"].push_back(gen_weights[1])
        if len(gen_indices) > 2:
            dic["hit_genweight2"].push_back(gen_weights[2])
        if len(gen_indices) > 3:
            dic["hit_genweight3"].push_back(gen_weights[3])
        if len(gen_indices) > 4:
            dic["hit_genweight4"].push_back(gen_weights[4])

        n_hit[0] += 1

        ## now access trackstate at calo
        trackstate = track.getTrackStates()[3]

        x = trackstate.referencePoint.x
        y = trackstate.referencePoint.y
        z = trackstate.referencePoint.z
        R = math.sqrt(x**2 + y**2)
        r = math.sqrt(x**2 + y**2 + z**2)

        dic["hit_chis"].push_back(chi_s)
        dic["hit_x"].push_back(x)
        dic["hit_y"].push_back(y)
        dic["hit_z"].push_back(z)
        dic["hit_t"].push_back(trackstate.time)

        track_mom = track_momentum(trackstate, isclic=isclic)

        dic["hit_p"].push_back(track_mom[0])
        dic["hit_theta"].push_back(track_mom[1])
        dic["hit_phi"].push_back(track_mom[2])
        dic["hit_e"].push_back(-1)
        dic["hit_px"].push_back(track_mom[4])
        dic["hit_py"].push_back(track_mom[5])
        dic["hit_pz"].push_back(track_mom[6])

        dic["hit_type"].push_back(1)  # 0 for tracks at calo
        dic["calohit_col"].push_back(0)
        gen_indices, gen_weights = find_gen_link(
            j,
            track.getObjectID().collectionID,
            gen_track_link_indextr,
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

        dic["hit_genlink"].push_back(
            link_vector
        )  # linked to first particle by default now
        if store_pandora_hits == "True":
            dic["hit_pandora_cluster_energy"].push_back(0)
            dic["hit_pandora_pfo_energy"].push_back(pandora_pfo_energy)
            if reference_point_pandora is not None:
                dic["hit__pandora_px"].push_back(momentum_pandora.x)
                dic["hit__pandora_py"].push_back(momentum_pandora.y)
                dic["hit__pandora_pz"].push_back(momentum_pandora.z)
                dic["hit__pandora_x"].push_back(reference_point_pandora.x)
                dic["hit__pandora_y"].push_back(reference_point_pandora.y)
                dic["hit__pandora_z"].push_back(reference_point_pandora.z)
                dic["pandora_pid"].push_back(pid_pandora)
            else:
                dic["hit__pandora_px"].push_back(-1)
                dic["hit__pandora_py"].push_back(-1)
                dic["hit__pandora_pz"].push_back(-1)
                dic["hit__pandora_x"].push_back(-1)
                dic["hit__pandora_y"].push_back(-1)
                dic["hit__pandora_z"].push_back(-1)
                dic["pandora_pid"].push_back(-1)

        genlink = -1
        if ngen > 0:
            genlink = link_vector[0]

        if len(gen_indices) > 0:
            dic["hit_genlink0"].push_back(gen_indices[0])
        if store_pandora_hits == "True":
            dic["hit_genlink1"].push_back(pandora_index_pfo)
        else:
            if len(gen_indices) > 1:
                dic["hit_genlink1"].push_back(gen_indices[1])
        if store_pandora_hits == "True":
            # print("storing calo hit")
            dic["hit_genlink2"].push_back(pandora_index_pfo)
        else:
            if len(gen_indices) > 2:
                dic["hit_genlink2"].push_back(gen_indices[2])
        if len(gen_indices) > 3:
            dic["hit_genlink3"].push_back(gen_indices[3])
        if len(gen_indices) > 4:
            dic["hit_genlink4"].push_back(gen_indices[4])

        if len(gen_indices) > 0:
            dic["hit_genweight0"].push_back(gen_weights[0])
        if len(gen_indices) > 1:
            dic["hit_genweight1"].push_back(gen_weights[1])
        if len(gen_indices) > 2:
            dic["hit_genweight2"].push_back(gen_weights[2])
        if len(gen_indices) > 3:
            dic["hit_genweight3"].push_back(gen_weights[3])
        if len(gen_indices) > 4:
            dic["hit_genweight4"].push_back(gen_weights[4])

        # if debug:
        #     print(
        #         "track at calo: N: {}, P: {:.2e}, Theta: {:.2e}, Phi: {:.2e}, X(m): {:.3f}, Y(m): {:.3f}, R(m): {:.3f}, Z(m): {:.3f}, r(m): {:.3f}, gen links: {}".format(
        #             n_hit[0],
        #             track_mom[0],
        #             track_mom[1],
        #             track_mom[2],
        #             x * 1e-03,
        #             y * 1e-03,
        #             R * 1e-03,
        #             z * 1e-03,
        #             r * 1e-03,
        #             list(link_vector),
        #         )
        #     )

        n_hit[0] += 1

    return n_hit, dic, number_of_hist_with_no_genlinks


def store_calo_hits(
    event,
    debug,
    dic,
    n_hit,
    genpart_indexes,
    gen_part_coll,
    number_of_hist_with_no_genlinks,
    store_pandora_hits,
    CLIC,
):
    ## calo stuff
    ecal_barrel = ("ECalBarrelModuleThetaMergedPositioned", 46)
    ecal_endcap = ("ECalEndcapTurbinePositioned", 47)
    print("using CLIC dataset", CLIC)
    if CLIC == "True":
        ecal_other = ("ECALOther", 48)
    hcal_barrel = ("HCalBarrelReadoutPositioned", 49)
    hcal_endcap = ("HCalEndcapReadoutPositioned", 50)
    # only for CLD
    # hcal_other = ("HCALOther", 51)
    gen_calo_links0 = "CaloHitMCParticleLinks"
    # pandora_clusters = "PandoraClusters"
    # pandora_pfo = "PandoraPFOs"
    gen_calohit_link_indexhit = event.get(gen_calo_links0)
    # pandora_clusters_event = event.get(pandora_clusters)
    # pandora_pfos_event = event.get(pandora_pfo)
    print("checking clic again", CLIC)
    if CLIC == "True":
        print("here")
        calohit_collections = [
            ecal_barrel[0],
            hcal_barrel[0],
            ecal_endcap[0],
            hcal_endcap[0],
            ecal_other[0],
            hcal_other[0],
        ]
    else:
        print("here2")
        calohit_collections = [
            ecal_barrel[0],
            hcal_barrel[0],
            ecal_endcap[0],
            hcal_endcap[0],
            # hcal_other[0],
            "MuonTaggerBarrelPhiThetaPositioned",
            "MuonTaggerEndcapPhiThetaPositioned",
            # "MUON" #add muon collections => thisi for CLD
        ]

    total_calohit_ = np.zeros(11)
    total_calohit_pandora = np.zeros(15)
    for calohit_col_index, calohit_coll in enumerate(calohit_collections):
        if debug:
            print("")
        # print(calohit_coll, len(event.get(calohit_coll)))
        for j, calohit in enumerate(event.get(calohit_coll)):
            # print(j, calohit.getObjectID().index)
            # print(dir(calohit))
            # print(calohit.getObjectID().collectionID, calohit_collection_ids[k])
            position = calohit.getPosition()
            x = position.x
            y = position.y
            z = position.z
            R = math.sqrt(x**2 + y**2)
            r = math.sqrt(x**2 + y**2 + z**2)
            hit_collection = calohit.getObjectID().collectionID

            dic["hit_chis"].push_back(0)
            dic["hit_x"].push_back(x)
            dic["hit_y"].push_back(y)
            dic["hit_z"].push_back(z)
            dic["hit_t"].push_back(calohit.getTime())
            dic["hit_p"].push_back(-1)
            dic["hit_e"].push_back(calohit.getEnergy())
            dic["hit_px"].push_back(0.0)
            dic["hit_py"].push_back(0.0)
            dic["hit_pz"].push_back(0.0)
            theta = math.acos(z / r)
            phi = math.atan2(y, x)

            dic["hit_theta"].push_back(theta)
            dic["hit_phi"].push_back(phi)

            htype = 2  # 2 if ECAL, 3 if HCAL
            if "HCAL" in calohit_coll:
                htype = 3
            elif  "MUON" in calohit_coll:
                htype = 4

            dic["hit_type"].push_back(htype)  # 0 for calo hits
            dic["calohit_col"].push_back(calohit_col_index + 1)
            gen_indices, gen_weights = find_gen_link(
                j,
                hit_collection,
                gen_calohit_link_indexhit,
                genpart_indexes,
                calo=True,
                gen_part_coll=gen_part_coll,
            )
            # print(gen_indices)
            link_vector = ROOT.std.vector("int")()
            for idx in gen_indices:
                link_vector.push_back(idx)
            # if j > 3:
            #     break

            ngen = len(link_vector)

            if ngen == 0:
                number_of_hist_with_no_genlinks += 1
                # if debug:
                #    print("  -> WARNING: this calo hit has no gen-link")

            dic["hit_genlink"].push_back(
                link_vector
            )  # linked to first particle by default now

            genlink = -1
            if ngen > 0:
                genlink = link_vector[0]
            if gen_indices[0] ==4: 
                print(gen_indices[0],gen_indices[1], calohit_col_index + 1)
            if store_pandora_hits == "True":
                # print("looking for calo hit")
                (
                    pandora_cluster,
                    pandora_cluster_energy,
                    pfo_energy,
                    pandora_pfo_index,
                    reference_point_pandora,
                    momentum_pandora,
                    pandora_pid
                ) = find_pandora_pfo_and_cluster_of_hit(
                    j, hit_collection, pandora_clusters_event, pandora_pfos_event
                )
                dic["hit_pandora_cluster_energy"].push_back(pandora_cluster_energy)
                dic["hit_pandora_pfo_energy"].push_back(pfo_energy)
                if reference_point_pandora is not None:
                    dic["hit__pandora_px"].push_back(momentum_pandora.x)
                    dic["hit__pandora_py"].push_back(momentum_pandora.y)
                    dic["hit__pandora_pz"].push_back(momentum_pandora.z)
                    dic["hit__pandora_x"].push_back(reference_point_pandora.x)
                    dic["hit__pandora_y"].push_back(reference_point_pandora.y)
                    dic["hit__pandora_z"].push_back(reference_point_pandora.z)
                    dic["pandora_pid"].push_back(pandora_pid)
                else:
                    dic["hit__pandora_px"].push_back(-1)
                    dic["hit__pandora_py"].push_back(-1)
                    dic["hit__pandora_pz"].push_back(-1)
                    dic["hit__pandora_x"].push_back(-1)
                    dic["hit__pandora_y"].push_back(-1)
                    dic["hit__pandora_z"].push_back(-1)
                    dic["pandora_pid"].push_back(-1)

            if len(gen_indices) > 0:
                dic["hit_genlink0"].push_back(gen_indices[0])
            if store_pandora_hits == "True":
                # print("storing calo hit")
                dic["hit_genlink1"].push_back(pandora_cluster)
            else:
                if len(gen_indices) > 1:
                    dic["hit_genlink1"].push_back(0)
            if store_pandora_hits == "True":
                # print("storing calo hit")
                dic["hit_genlink2"].push_back(pandora_pfo_index)
            else:
                if len(gen_indices) > 2:
                    dic["hit_genlink2"].push_back(0)
            if len(gen_indices) > 3:
                dic["hit_genlink3"].push_back(gen_indices[1])
            if len(gen_indices) > 4:
                dic["hit_genlink4"].push_back(gen_indices[4])

            if len(gen_indices) > 0:
                dic["hit_genweight0"].push_back(gen_weights[0])
            if len(gen_indices) > 1:
                dic["hit_genweight1"].push_back(gen_weights[1])
            if len(gen_indices) > 2:
                dic["hit_genweight2"].push_back(gen_weights[2])
            if len(gen_indices) > 3:
                dic["hit_genweight3"].push_back(gen_weights[3])
            if len(gen_indices) > 4:
                dic["hit_genweight4"].push_back(gen_weights[4])

            # find_pandora_cluster_of_hit(j, hit_collection, pandora_clusters_event)

            # if debug:
            #     total_calohit_e = total_calohit_e + calohit.getEnergy()
            #     if list(link_vector)[0] < 11 and list(link_vector)[0] > 1:
            #         total_calohit_[list(link_vector)[0]] = (
            #             total_calohit_[list(link_vector)[0]] + calohit.getEnergy()
            #         )
            #     total_calohit_pandora[pandora_cluster] = (
            #         total_calohit_pandora[pandora_cluster] + calohit.getEnergy()
            #     )

            # print(
            #     "calo hit type: {}, N: {}, E: {:.2e}, X(m): {:.3f}, Y(m): {:.3f}, R(m): {:.3f}, Z(m): {:.3f}, r(m): {:.3f}, gen links: {}".format(
            #         htype,
            #         n_hit[0],
            #         calohit.energy,
            #         x * 1e-03,
            #         y * 1e-03,
            #         R * 1e-03,
            #         z * 1e-03,
            #         r * 1e-03,
            #         list(link_vector),
            #     )
            # )

            n_hit[0] += 1

    return (
        n_hit,
        dic,
        total_calohit_,
        number_of_hist_with_no_genlinks,
        total_calohit_pandora,
    )
