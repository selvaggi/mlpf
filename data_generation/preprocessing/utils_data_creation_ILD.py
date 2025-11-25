
import os
import numpy as np
import awkward
import uproot
import vector
import tqdm
from scipy.sparse import coo_matrix
track_coll = "SiTracks_Refitted"
mc_coll = "MCParticles"

particle_feature_order = [
"PDG", 
"generatorStatus",
"charge",
"pt",
"eta",
"phi",
"sin_phi",
"cos_phi",
"energy",
"simulatorStatus",
"mass", 
"p", 
"momentum.x", 
"momentum.y",
"momentum.z",
"vertex.x",
"vertex.y",
"vertex.z",
"endpoint.x",
"endpoint.y",
"endpoint.z",
]
PandoraPFO_feature_order = [
    "PDG", 
    "momentum.x", 
    "momentum.y",
    "momentum.z",
    "referencePoint.x", 
    "referencePoint.y",
    "referencePoint.z",
    "energy",
    "p"
]
track_feature_order = [
    "elemtype", #0
    "pt", #1
    "eta",#2
    "sin_phi", #3
    "cos_phi", #4
    "p", # at vertex
    "px", # at vertex
    "py",#7
    "pz",#8
    "referencePoint.x", # store the reference at vertex
    "referencePoint.y", #10
    "referencePoint.z",#11
    "referencePoint_calo.x", 
    "referencePoint_calo.y", 
    "referencePoint_calo.z", #store the reference at calo
    "chi2", 
    "ndf", 
    "tanLambda",
    "D0", 
    "omega", 
    "Z0", 
    "time", 
    "px_calo", # at vertex
    "py_calo",#7
    "pz_calo",#8
]
hit_feature_order = [
    "elemtype",
    "et",
    "eta",
    "sin_phi",
    "cos_phi",
    "energy",
    "position.x",
    "position.y",
    "position.z",
    "time",
    "subdetector",
    "type",
]

def get_feature_matrix(feature_dict, features):
    feats = []
    for feat in features:
        feat_arr = awkward.to_numpy(feature_dict[feat])
        feats.append(feat_arr)
    feats = np.array(feats)
    return feats.T



def sanitize(arr):
    arr[np.isnan(arr)] = 0.0
    arr[np.isinf(arr)] = 0.0


def get_reco_properties(prop_data, iev):
    reco_arr = prop_data["PandoraPFOs"][iev]
    reco_arr = {k.replace("PandoraPFOs.", ""): reco_arr[k] for k in reco_arr.fields}

    reco_p4 = vector.awk(
        awkward.zip({"mass": reco_arr["mass"], "x": reco_arr["momentum.x"], "y": reco_arr["momentum.y"], "z": reco_arr["momentum.z"]})
    )
    reco_arr["pt"] = reco_p4.pt
    reco_arr["eta"] = reco_p4.eta
    reco_arr["phi"] = reco_p4.phi
    reco_arr["energy"] = reco_p4.energy

    msk = reco_arr["PDG"] != 0
    reco_arr = awkward.Record({k: reco_arr[k][msk] for k in reco_arr.keys()})
    return reco_arr

def build_dummy_array(num, dtype=np.int64):
    return awkward.Array(
        awkward.contents.ListOffsetArray(
            awkward.index.Index64(np.zeros(num + 1, dtype=np.int64)),
            awkward.from_numpy(np.array([], dtype=dtype), highlevel=False),
        )
    )

def get_genparticles_and_adjacencies( prop_data, hit_data, calohit_links, sitrack_links, iev, collectionIDs, eval_dataset=False, dic=None, truth_tracking=False):
    gen_features = gen_to_features(prop_data, iev)
    
    hit_features, genparticle_to_hit, hit_idx_local_to_global = get_calohit_matrix_and_genadj(hit_data, calohit_links, iev, collectionIDs)
    track_features = track_to_features(prop_data, iev)
    genparticle_to_trk = genparticle_track_adj( sitrack_links, iev, truth_tracking)
    print(genparticle_to_trk)
    n_gp = awkward.count(gen_features["PDG"])
    n_track = awkward.count(track_features["type"])
    n_hit = awkward.count(hit_features["type"])
    if eval_dataset:
        pandora_features = pandora_to_features(prop_data, iev)
        hit_to_pfo = hit_pfo_adj(prop_data, hit_idx_local_to_global, iev)
        n_pfo = awkward.count(pandora_features["PDG"])
        pfo_to_calohit_matrix = coo_matrix((hit_to_pfo[2], (hit_to_pfo[1], hit_to_pfo[0])), shape=(n_pfo, n_hit))
        pfo_to_calohit = pfo_to_calohit_matrix.toarray().argmax(axis=0)
        pfo_to_calohit_nolink_mask  = (pfo_to_calohit_matrix.sum(axis=0).reshape(-1))==0
        pfo_to_calohit_nolink_mask = np.array(pfo_to_calohit_nolink_mask).reshape(-1)
        pfo_to_calohit[pfo_to_calohit_nolink_mask] = -1 #if no link set to -1

        pfo_to_track = track_pfo_adj(prop_data, hit_idx_local_to_global, iev)
        pfo_to_track_matrix = coo_matrix((pfo_to_track[2], (pfo_to_track[1], pfo_to_track[0])), shape=(n_pfo, n_track))
        pfo_to_track= pfo_to_track_matrix.toarray().argmax(axis=0).reshape(-1)
        pfo_to_track_nolink_mask  = (pfo_to_track_matrix.sum(axis=0))==0
        pfo_to_track_nolink_mask = np.array(pfo_to_track_nolink_mask).reshape(-1)
        pfo_to_track[pfo_to_track_nolink_mask] = -1 #if no link set to -1
    else:
        pandora_features = None
        pfo_to_calohit = None
        pfo_to_track = None
        pfo_to_track = None
    # hit_to_cluster = hit_cluster_adj(dataset, prop_data, hit_idx_local_to_global, iev)
    # cluster_features = cluster_to_features(prop_data, hit_features, hit_to_cluster, iev)
    

    # # collect hits of st=1 daughters to the st=1 particles
    # mask_status1 = gen_features["generatorStatus"] == 1

    # if gen_features["index"] is not None:  # if there are even daughters
    #     genparticle_to_hit, genparticle_to_trk = add_daughters_to_status1(gen_features, genparticle_to_hit, genparticle_to_trk)
    # n_cluster = awkward.count(cluster_features["type"])

    if len(genparticle_to_trk[0]) > 0:
        gp_to_track_matrix = coo_matrix((genparticle_to_trk[2], (genparticle_to_trk[0], genparticle_to_trk[1])), shape=(n_gp, n_track))
        gp_to_track = gp_to_track_matrix.max(axis=1).todense()
        gp_to_track_index = gp_to_track_matrix.toarray().argmax(axis=0).reshape(-1)

        print("gp_to_track_index",gp_to_track_index)
    else:
        gp_to_track = np.zeros((n_gp, 1))
    # one hit has contribution from different MCs
    gp_to_calohit = coo_matrix((genparticle_to_hit[2], (genparticle_to_hit[0], genparticle_to_hit[1])), shape=(n_gp, n_hit))
    # count hits per MC can't count enegy because there are more links than hits (one hit has contribution from different MCs)
    gp_to_calohit_hitcount = coo_matrix((np.ones_like(genparticle_to_hit[2]), (genparticle_to_hit[0], genparticle_to_hit[1])), shape=(n_gp, n_hit))
    gp_hitcount = gp_to_calohit_hitcount.toarray().sum(axis=1) #hit count of particles
    gp_to_calohit = gp_to_calohit.toarray().argmax(axis=0).reshape(-1) #hit to MC link 
    gp_to_recoE = coo_matrix((hit_features["energy"], (gp_to_calohit, np.arange(n_hit))), shape=(n_gp, n_hit)).toarray().sum(axis=1)
    gp_to_calohit_beforecalomother = gp_to_calohit

    gp_to_calohit = np.array(gen_features["index_calomother"])[gp_to_calohit] #assign to the MC parent that was produced before calo (index of calomother)
    gp_to_calohit_beforecalomother = gp_to_calohit_beforecalomother!=gp_to_calohit
    gp_to_recoE = coo_matrix((hit_features["energy"], (gp_to_calohit, np.arange(n_hit))), shape=(n_gp, n_hit)).toarray().sum(axis=1)
    
    #! deprecated (bases the definition of reconstructable in cluster E)
    # calohit_to_cluster = coo_matrix((hit_to_cluster[2], (hit_to_cluster[0], hit_to_cluster[1])), shape=(n_hit, n_cluster))
    # gp_to_cluster = (gp_to_calohit * calohit_to_cluster).sum(axis=1)
    # 60% of the hits of a track must come from the genparticle
    # gp_in_tracker = np.array(gp_to_track >= 0.6)[:, 0]
    # at least 10% of the energy of the genparticle should be matched to a calorimeter cluster
    # gp_in_calo = (np.array(gp_to_cluster)[:, 0] / gen_features["energy"]) > 0.1
    # did the particle leave hits or track? (=interacted with detector) 
    # gp_interacted_with_detector = gp_in_tracker | gp_in_calo
    # mask_visible = awkward.to_numpy(mask_status1 & gp_interacted_with_detector)

    # particle has more than 10 MeV enegy in the calo
    gp_in_calo = np.array(gp_to_recoE>0.01) 
    gp_in_tracker = np.array(gp_to_track >= 0.1)[:, 0]
    gp_in_tracker_not = np.array(gp_to_track == 0.0)[:, 0]
    gp_interacted_with_detector = gp_in_tracker*gp_in_calo+gp_in_calo
    gp_electrons_without_track_but_E = gp_in_calo*gp_in_tracker_not*(np.abs(gen_features["PDG"])==11)
    gp_electrons_with_track = gp_in_calo*gp_in_tracker*(np.abs(gen_features["PDG"])==11)
    mask_visible = awkward.to_numpy( gp_interacted_with_detector)
    mask_visible_notrack = awkward.to_numpy( gp_electrons_without_track_but_E)
    mask_visible_track = awkward.to_numpy( gp_electrons_with_track)
    idx_all_masked = np.where(mask_visible)[0]
    # idx_all_masked_notrack = np.where(mask_visible_notrack)[0]
    # idx_all_masked_track = np.where(mask_visible_track)[0]
    # gen_features_np =  awkward.to_numpy(gen_features["energy"])
    # gen_features_phi =  awkward.to_numpy(gen_features["phi"])
    # if len (gen_features_np[idx_all_masked_notrack])>0:
    #     dic["index_no_track"].append(np.where(idx_all_masked_notrack))
    #     dic["energy_no_track"].append(gen_features_np[idx_all_masked_notrack])
    #     dic["phi_no_track"].append(gen_features_phi[idx_all_masked_notrack])
    # if len(gen_features_np[idx_all_masked_track])>0:
    #     dic["energy_track"].append( gen_features_np[idx_all_masked_track])
    #     dic["phi_track"].append(gen_features_phi[idx_all_masked_track])
    genpart_idx_all_to_filtered = {idx_all: idx_filtered for idx_filtered, idx_all in enumerate(idx_all_masked)}
    if np.array(mask_visible).sum() == 0:
        print("event does not have even one 'visible' particle. will skip event")
        return None
    

    if len(np.array(mask_visible)) == 1:
        # event has only one particle (then index will be empty because no daughters)
        gen_features = awkward.Record({feat: (gen_features[feat][mask_visible] if feat != "index" else None) for feat in gen_features.keys()})
    else:
        gen_features = awkward.Record({feat: gen_features[feat][mask_visible] for feat in gen_features.keys()})

    # get the track/cluster -> genparticle map
    # assign 0,..N indices to adjacency, -1 if genparticle not in filtered list
    hit_to_gp = index_to_range(gp_to_calohit, genpart_idx_all_to_filtered)
    track_to_gp = index_to_range(gp_to_track_index, genpart_idx_all_to_filtered)

    return EventData(
        gen_features,
        hit_features,
        track_features,
        hit_to_gp,
        track_to_gp,
        pandora_features, 
        pfo_to_calohit, 
        pfo_to_track, 
        gp_to_calohit_beforecalomother
    ), dic 

def isProducedInCalo(vertices, BarrelRadius=2150, EndCapZ=2307):

    x, y, z = vertices[:,0], vertices[:,1], vertices[:,2]
    radius = np.sqrt(x**2 + y**2 + z**2)
    return (radius > BarrelRadius) | (np.abs(z) > EndCapZ)

def define_produced_before_calo_map(MCParticles_p4, gen_arr, parents):
    index = []
    isproducedincalo = True
    for particle_idx in range(0,len(MCParticles_p4.pt)):
        particle_idx_search = particle_idx
        while isproducedincalo:
            vertex = np.array([gen_arr["vertex.x"][particle_idx_search],gen_arr["vertex.y"][particle_idx_search],gen_arr["vertex.z"][particle_idx_search]]).reshape(1,3)
            isproducedincalo = isProducedInCalo(vertex)
            if isproducedincalo:
                parents_begin = gen_arr["parents_begin"][particle_idx_search]
                parents_end = gen_arr["parents_end"][particle_idx_search]
                particle_idx_search = parents[parents_begin:parents_end][0]
        index.append(particle_idx_search)
        isproducedincalo = True
    return index


def pandora_to_features(prop_data, iev):
    pandora_arr = prop_data["PandoraPFOs"][iev]
    pandora_arr = {k.replace("PandoraPFOs" + ".", ""): pandora_arr[k] for k in pandora_arr.fields}
    pandora_arr["p"] = np.sqrt(pandora_arr["momentum.x"]**2 + pandora_arr["momentum.x"]**2 + pandora_arr["momentum.z"]**2)
    ret = {
        "energy": pandora_arr["energy"],
        "PDG": pandora_arr["PDG"],
        "referencePoint.x": pandora_arr["referencePoint.x"],
        "referencePoint.y": pandora_arr["referencePoint.y"],
        "referencePoint.z": pandora_arr["referencePoint.z"],
        "momentum.x": pandora_arr["momentum.x"],
        "momentum.y": pandora_arr["momentum.y"],
        "momentum.z": pandora_arr["momentum.z"],
        "p": pandora_arr["p"]
    }
    return ret 

def gen_to_features(prop_data, iev):

    
    gen_arr = prop_data[mc_coll][iev]

    gen_arr = {k.replace(mc_coll + ".", ""): gen_arr[k] for k in gen_arr.fields}

    MCParticles_p4 = vector.awk(
        awkward.zip({"mass": gen_arr["mass"], "x": gen_arr["momentum.x"], "y": gen_arr["momentum.y"], "z": gen_arr["momentum.z"]})
    )

    parents = prop_data["_MCParticles_parents/_MCParticles_parents.index"][iev]
    gen_arr["pt"] = MCParticles_p4.pt
    gen_arr["p"] = np.sqrt(gen_arr["momentum.x"]**2 + gen_arr["momentum.y"]**2 + gen_arr["momentum.z"]**2)
    gen_arr["eta"] = MCParticles_p4.eta
    gen_arr["phi"] = MCParticles_p4.phi
    gen_arr["energy"] = MCParticles_p4.energy
    gen_arr["sin_phi"] = np.sin(gen_arr["phi"])
    gen_arr["cos_phi"] = np.cos(gen_arr["phi"])

    index = define_produced_before_calo_map(MCParticles_p4, gen_arr, parents)
       
       

    # placeholder flag
    gen_arr["ispu"] = np.zeros_like(gen_arr["phi"])

    ret = {
        "PDG": gen_arr["PDG"],
        "generatorStatus": gen_arr["generatorStatus"],
        "charge": gen_arr["charge"],
        "pt": gen_arr["pt"],
        "p": gen_arr["p"],
        "eta": gen_arr["eta"],
        "phi": gen_arr["phi"],
        "sin_phi": gen_arr["sin_phi"],
        "cos_phi": gen_arr["cos_phi"],
        "energy": gen_arr["energy"],
        "ispu": gen_arr["ispu"],
        "mass": gen_arr["mass"], 
        "simulatorStatus": gen_arr["simulatorStatus"],
        "gp_to_track": np.zeros(len(gen_arr["PDG"]), dtype=np.float64),
        "gp_to_cluster": np.zeros(len(gen_arr["PDG"]), dtype=np.float64),
        "jet_idx": np.zeros(len(gen_arr["PDG"]), dtype=np.int64),
        "daughters_begin": gen_arr["daughters_begin"],
        "daughters_end": gen_arr["daughters_end"],
        "index_calomother": np.array(index), 
        "momentum.x"    : gen_arr["momentum.x"],
        "momentum.y"    : gen_arr["momentum.y"],
        "momentum.z"    : gen_arr["momentum.z"],
        "vertex.x"      : gen_arr["vertex.x"],
        "vertex.y"      : gen_arr["vertex.y"],
        "vertex.z"      : gen_arr["vertex.z"],
        "endpoint.x"    : gen_arr["endpoint.x"],
        "endpoint.y"    : gen_arr["endpoint.y"],
        "endpoint.z"    : gen_arr["endpoint.z"],
    }


    ret["index"] = prop_data["_MCParticles_daughters/_MCParticles_daughters.index"][iev]
    
    return ret



def get_calohit_matrix_and_genadj(hit_data, calohit_links, iev, collectionIDs):
    feats = ["type", "cellID", "energy", "energyError", "time", "position.x", "position.y", "position.z"]

    hit_idx_global = 0
    hit_idx_global_to_local = {}
    hit_feature_matrix = []
    for col in sorted(hit_data.keys()):
        icol = collectionIDs[col]
        hit_features = hits_to_features(hit_data[col], iev, col, feats)
        hit_feature_matrix.append(hit_features)
        for ihit in range(len(hit_data[col][col + ".energy"][iev])):
            hit_idx_global_to_local[hit_idx_global] = (icol, ihit)
            hit_idx_global += 1

    hit_idx_local_to_global = {v: k for k, v in hit_idx_global_to_local.items()}
    hit_feature_matrix = awkward.Record(
        {k: awkward.concatenate([hit_feature_matrix[i][k] for i in range(len(hit_feature_matrix))]) for k in hit_feature_matrix[0].fields}
    )

    # add all edges from genparticle to calohit
    calohit_to_gen_weight = calohit_links["CalohitMCTruthLink.weight"][iev]
    
    calohit_to_gen_calo_colid = calohit_links["_CalohitMCTruthLink_from/_CalohitMCTruthLink_from.collectionID"][iev]
    calohit_to_gen_gen_colid = calohit_links["_CalohitMCTruthLink_to/_CalohitMCTruthLink_to.collectionID"][iev]
    calohit_to_gen_calo_idx = calohit_links["_CalohitMCTruthLink_from/_CalohitMCTruthLink_from.index"][iev]
    calohit_to_gen_gen_idx = calohit_links["_CalohitMCTruthLink_to/_CalohitMCTruthLink_to.index"][iev]

    genparticle_to_hit_matrix_coo0 = []
    genparticle_to_hit_matrix_coo1 = []
    genparticle_to_hit_matrix_w = []
    for calo_colid, calo_idx, gen_colid, gen_idx, w in zip(
        calohit_to_gen_calo_colid,
        calohit_to_gen_calo_idx,
        calohit_to_gen_gen_colid,
        calohit_to_gen_gen_idx,
        calohit_to_gen_weight,
    ):
        genparticle_to_hit_matrix_coo0.append(gen_idx)
        genparticle_to_hit_matrix_coo1.append(hit_idx_local_to_global[(calo_colid, calo_idx)])
        genparticle_to_hit_matrix_w.append(w)
   
    return (
        hit_feature_matrix,
        (
            np.array(genparticle_to_hit_matrix_coo0),
            np.array(genparticle_to_hit_matrix_coo1),
            np.array(genparticle_to_hit_matrix_w),
        ),
        hit_idx_local_to_global,
    )


def hits_to_features(hit_data, iev, coll, feats):
    feat_arr = {f: hit_data[coll + "." + f][iev] for f in feats}

    # set the subdetector type
    sdcoll = "subdetector"
    feat_arr[sdcoll] = np.zeros(len(feat_arr["type"]), dtype=np.int32)
    if coll.startswith("ECAL"):
        feat_arr[sdcoll][:] = 1
    elif coll.startswith("HCAL"):
        feat_arr[sdcoll][:] = 2
    elif coll.startswith("MUON"):
        feat_arr[sdcoll][:] = 3
    else:
        feat_arr[sdcoll][:] = 4

    # hit elemtype is always 2
    feat_arr["elemtype"] = 2 * np.ones(len(feat_arr["type"]), dtype=np.int32)

    # precompute some approximate et, eta, phi
    pos_mag = np.sqrt(feat_arr["position.x"] ** 2 + feat_arr["position.y"] ** 2 + feat_arr["position.z"] ** 2)
    px = (feat_arr["position.x"] / pos_mag) * feat_arr["energy"]
    py = (feat_arr["position.y"] / pos_mag) * feat_arr["energy"]
    pz = (feat_arr["position.z"] / pos_mag) * feat_arr["energy"]
    feat_arr["et"] = np.sqrt(px**2 + py**2)
    feat_arr["eta"] = 0.5 * np.log((feat_arr["energy"] + pz) / (feat_arr["energy"] - pz))
    feat_arr["sin_phi"] = py / feat_arr["energy"]
    feat_arr["cos_phi"] = px / feat_arr["energy"]

    return awkward.Record(feat_arr)




def track_to_features(prop_data, iev):
    track_arr = prop_data[track_coll][iev]
    feats_from_track = ["type", "chi2", "ndf"]
    ret = {feat: track_arr[track_coll + "." + feat] for feat in feats_from_track}
    n_tr = len(ret["type"])

    # get the index of the first track state
    trackstate_idx = prop_data[track_coll][track_coll + ".trackStates_begin"][iev]
    # get the properties of the track at the first track state (at the origin)
    for k in ["tanLambda", "D0", "phi", "omega", "Z0", "time", "referencePoint.x", "referencePoint.y", "referencePoint.z"]:
        ret[k] = awkward.to_numpy(prop_data["_SiTracks_Refitted_trackStates"]["_SiTracks_Refitted_trackStates." + k][iev][trackstate_idx])
    
    ret["referencePoint_calo.x"] = awkward.to_numpy(prop_data["_SiTracks_Refitted_trackStates"]["_SiTracks_Refitted_trackStates.referencePoint.x"][iev][trackstate_idx+3])
    ret["referencePoint_calo.y"] = awkward.to_numpy(prop_data["_SiTracks_Refitted_trackStates"]["_SiTracks_Refitted_trackStates.referencePoint.y"][iev][trackstate_idx+3])
    ret["referencePoint_calo.z"] = awkward.to_numpy(prop_data["_SiTracks_Refitted_trackStates"]["_SiTracks_Refitted_trackStates.referencePoint.z"][iev][trackstate_idx+3])
    ret["phi_calo"] = awkward.to_numpy(prop_data["_SiTracks_Refitted_trackStates"]["_SiTracks_Refitted_trackStates.phi"][iev][trackstate_idx+3])
    ret["tanLambda_calo"] = awkward.to_numpy(prop_data["_SiTracks_Refitted_trackStates"]["_SiTracks_Refitted_trackStates.tanLambda"][iev][trackstate_idx+3])
    ret["omega_calo"] = awkward.to_numpy(prop_data["_SiTracks_Refitted_trackStates"]["_SiTracks_Refitted_trackStates.omega"][iev][trackstate_idx+3])

    ret["pt"] = awkward.to_numpy(track_pt(ret["omega"]))
    # from the track state at IP (location 1)
    ret["px"] = awkward.to_numpy(np.cos(ret["phi"])) * ret["pt"] 
    ret["py"] = awkward.to_numpy(np.sin(ret["phi"])) * ret["pt"]
    ret["pz"] = awkward.to_numpy(ret["tanLambda"]) * ret["pt"]

    ret["pt_calo"] = awkward.to_numpy(track_pt(ret["omega_calo"]))
    ret["px_calo"] = awkward.to_numpy(np.cos(ret["phi_calo"])) * ret["pt_calo"] 
    ret["py_calo"] = awkward.to_numpy(np.sin(ret["phi_calo"])) * ret["pt_calo"]
    ret["pz_calo"] = awkward.to_numpy(ret["tanLambda_calo"]) * ret["pt_calo"]

    ret["p"] = np.sqrt(ret["px"] ** 2 + ret["py"] ** 2 + ret["pz"] ** 2)
    cos_theta = np.divide(ret["pz"], ret["p"], where=ret["p"] > 0)
    theta = np.arccos(cos_theta)
    tt = np.tan(theta / 2.0)
    eta = awkward.to_numpy(-np.log(tt, where=tt > 0))
    eta[tt <= 0] = 0.0
    ret["eta"] = eta

    ret["sin_phi"] = np.sin(ret["phi"])
    ret["cos_phi"] = np.cos(ret["phi"])

    # track is always type 1
    ret["elemtype"] = 1 * np.ones(n_tr, dtype=np.float32)

    return awkward.Record(ret)

def track_pt(omega):
    a = 2.99792e-4
    b = 2  # B-field in tesla, for CLD
    return a * np.abs(b / omega)

def genparticle_track_adj(sitrack_links, iev, truth_tracking):
    print("here", truth_tracking)
    if truth_tracking:
        trk_to_gen_trkidx = sitrack_links["_SiTracks_Refitted_Relation_from/_SiTracks_Refitted_Relation_from.index"][iev]
        trk_to_gen_genidx = sitrack_links["_SiTracks_Refitted_Relation_to/_SiTracks_Refitted_Relation_to.index"][iev]
        trk_to_gen_w = sitrack_links["SiTracks_Refitted_Relation.weight"][iev]
    else:
        trk_to_gen_trkidx = sitrack_links["_SiTracksMCTruthLink_from/_SiTracksMCTruthLink_from.index"][iev]
        trk_to_gen_genidx = sitrack_links["_SiTracksMCTruthLink_to/_SiTracksMCTruthLink_to.index"][iev]
        trk_to_gen_w = sitrack_links["SiTracksMCTruthLink.weight"][iev]

    genparticle_to_track_matrix_coo0 = awkward.to_numpy(trk_to_gen_genidx)
    genparticle_to_track_matrix_coo1 = awkward.to_numpy(trk_to_gen_trkidx)
    genparticle_to_track_matrix_w = awkward.to_numpy(trk_to_gen_w)
    print(genparticle_to_track_matrix_coo0)
    return genparticle_to_track_matrix_coo0, genparticle_to_track_matrix_coo1, genparticle_to_track_matrix_w



def filter_adj(adj, all_to_filtered):
    i0s_new = []
    i1s_new = []
    ws_new = []
    for i0, i1, w in zip(*adj):
        if i0 in all_to_filtered:
            i0_new = all_to_filtered[i0]
            i0s_new.append(i0_new)
            i1s_new.append(i1)
            ws_new.append(w)
    return np.array(i0s_new), np.array(i1s_new), np.array(ws_new)

def index_to_range(arr, mapping):
    map_func = np.vectorize(lambda x: mapping.get(x, -1))
    mapped_arr = map_func(arr)
    return mapped_arr



class EventData:
    def __init__(
        self,
        gen_features,
        hit_features,
        track_features,
        hit_to_gp,
        track_to_gp,
        pandora_features=None,
        pfo_to_calohit = None, 
        pfo_to_track = None, 
        gp_to_calohit_beforecalomother = None
    ):
        self.gen_features = gen_features  # feature matrix of the genparticles
        self.hit_features = hit_features  # feature matrix of the calo hits
        self.track_features = track_features  # feature matrix of the tracks
        self.hit_to_gp = hit_to_gp  # array linking hit to gen MC
        self.track_to_gp = track_to_gp  # array linking track to gen MC
        self.pandora_features = pandora_features  # feature matrix of the PandoraPFOs
        self.pfo_to_calohit = pfo_to_calohit # array linking pfo to calohit
        self. pfo_to_track = pfo_to_track # array linking pfo to track
        self.gp_to_calohit_beforecalomother = gp_to_calohit_beforecalomother


def hit_pfo_adj(prop_data, hit_idx_local_to_global, iev):


    clusters_begin = prop_data["PandoraPFOs"]["PandoraPFOs.clusters_begin"][iev]
    clusters_end = prop_data["PandoraPFOs"]["PandoraPFOs.clusters_end"][iev]
    idx_arr_cluster = prop_data["_PandoraPFOs_clusters/_PandoraPFOs_clusters.index"][iev]
    coll_arr = prop_data["_PandoraClusters_hits/_PandoraClusters_hits.collectionID"][iev]
    idx_arr = prop_data["_PandoraClusters_hits/_PandoraClusters_hits.index"][iev]
    hits_begin = prop_data["PandoraClusters"]["PandoraClusters.hits_begin"][iev]
    hits_end = prop_data["PandoraClusters"]["PandoraClusters.hits_end"][iev]
    # index in the array of all hits
    hit_to_cluster_matrix_coo0 = []
    # index in the cluster array
    hit_to_cluster_matrix_coo1 = []

    # weight
    hit_to_cluster_matrix_w = []

    # loop over all pfos 
    for ipfo in range(len(clusters_begin)):
        cluster_begin = clusters_begin[ipfo]
        cluster_end = clusters_end[ipfo]
        idx_range = idx_arr_cluster[cluster_begin:cluster_end]
        for index_cluster, icluster in enumerate(idx_range):
            # get the slice in the hit array corresponding to this cluster
            hbeg = hits_begin[icluster]
            hend = hits_end[icluster]
            idx_range = idx_arr[hbeg:hend]
            coll_range = coll_arr[hbeg:hend]

            # add edges from hit to cluster
            for icol, idx in zip(coll_range, idx_range):
                hit_to_cluster_matrix_coo0.append(hit_idx_local_to_global[(icol, idx)])
                hit_to_cluster_matrix_coo1.append(ipfo)
                hit_to_cluster_matrix_w.append(1.0)
    return hit_to_cluster_matrix_coo0, hit_to_cluster_matrix_coo1, hit_to_cluster_matrix_w


def track_pfo_adj(prop_data, hit_idx_local_to_global, iev):
    tracks_begin = prop_data["PandoraPFOs"]["PandoraPFOs.tracks_begin"][iev]
    tracks_end = prop_data["PandoraPFOs"]["PandoraPFOs.tracks_end"][iev]
    idx_arr_track = prop_data["_PandoraPFOs_tracks/_PandoraPFOs_tracks.index"][iev]
   
    # index in the array of all hits
    track_to_pfo_matrix_coo0 = []
    # index in the track array
    track_to_pfo_matrix_coo1 = []
    # weight
    track_to_pfo_matrix_w = []

    # loop over all pfos 
    for ipfo in range(len(tracks_begin)):
        track_begin = tracks_begin[ipfo]
        track_end = tracks_end[ipfo]
        idx_range = idx_arr_track[track_begin:track_end]
        for index_track, itrack in enumerate(idx_range):
            track_to_pfo_matrix_coo0.append(itrack)
            track_to_pfo_matrix_coo1.append(ipfo)
            track_to_pfo_matrix_w.append(1.0)
    return track_to_pfo_matrix_coo0, track_to_pfo_matrix_coo1, track_to_pfo_matrix_w