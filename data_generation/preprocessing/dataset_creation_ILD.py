import os
import numpy as np
import awkward
import uproot
import tqdm
from preprocessing.utils_data_creation_ILD import get_feature_matrix, sanitize, get_reco_properties, build_dummy_array
from preprocessing.utils_data_creation_ILD import track_feature_order, hit_feature_order, particle_feature_order, PandoraPFO_feature_order
from preprocessing.utils_data_creation_ILD import  get_genparticles_and_adjacencies

from preprocessing.utils_data_creation_ILD import MC_PARTICLE_COL, PANDORA_PFO_COL, CLUSTERS_COL
from preprocessing.utils_data_creation_ILD import TRACKS_COL, TRACK_TO_MC_LINK_COL
from preprocessing.utils_data_creation_ILD import CALO_HIT_COLS, CALOHIT_TO_MC_LINK_COL
import time 


def process_one_file(fn, ofn, eval_dataset):

    # output exists, do not recreate
    # if os.path.isfile(ofn):
    #     return
    # print(fn)

    fi = uproot.open(fn)

    arrs = fi["events"]
    collectionIDs = {
        k: v
        for k, v in zip(
            fi.get("podio_metadata").arrays("events___CollectionTypeInfo.name")["events___CollectionTypeInfo.name"][0],
            fi.get("podio_metadata").arrays("events___CollectionTypeInfo.collectionID")["events___CollectionTypeInfo.collectionID"][0]
        )
    }

    prop_data = arrs.arrays(
        [
            MC_PARTICLE_COL,
            f"{MC_PARTICLE_COL}.PDG",
            f"{MC_PARTICLE_COL}.momentum.x",
            f"{MC_PARTICLE_COL}.momentum.y",
            f"{MC_PARTICLE_COL}.momentum.z",
            f"{MC_PARTICLE_COL}.mass",
            f"{MC_PARTICLE_COL}.charge",
            f"{MC_PARTICLE_COL}.generatorStatus",
            f"{MC_PARTICLE_COL}.simulatorStatus",
            f"{MC_PARTICLE_COL}.daughters_begin",
            f"{MC_PARTICLE_COL}.daughters_end",
            f"_{MC_PARTICLE_COL}_daughters/_{MC_PARTICLE_COL}_daughters.index",  # similar to f"{MC_PARTICLE_COL}#1.index" in clic
            f"_{MC_PARTICLE_COL}_parents/_{MC_PARTICLE_COL}_parents.index",  # similar to f"{MC_PARTICLE_COL}#1.index" in clic
            TRACKS_COL,
            f"_{TRACKS_COL}_trackStates",
            f"_{PANDORA_PFO_COL}_tracks/_{PANDORA_PFO_COL}_tracks.index",
            f"{CLUSTERS_COL}",
            f"_{CLUSTERS_COL}_hits/_{CLUSTERS_COL}_hits.index",
            f"_{CLUSTERS_COL}_hits/_{CLUSTERS_COL}_hits.collectionID",
            f"{PANDORA_PFO_COL}",
            f"_{PANDORA_PFO_COL}_clusters/_{PANDORA_PFO_COL}_clusters.index",
            # f"{TRACKS_COL}_dQdx",
        ]
    )
    calohit_links = arrs.arrays(
        [
            f"{CALOHIT_TO_MC_LINK_COL}.weight",
            f"_{CALOHIT_TO_MC_LINK_COL}_to/_{CALOHIT_TO_MC_LINK_COL}_to.collectionID",
            f"_{CALOHIT_TO_MC_LINK_COL}_to/_{CALOHIT_TO_MC_LINK_COL}_to.index",
            f"_{CALOHIT_TO_MC_LINK_COL}_from/_{CALOHIT_TO_MC_LINK_COL}_from.collectionID",
            f"_{CALOHIT_TO_MC_LINK_COL}_from/_{CALOHIT_TO_MC_LINK_COL}_from.index",
        ]
    )
    sitrack_links = arrs.arrays(
        [
            f"{TRACK_TO_MC_LINK_COL}.weight",
            f"_{TRACK_TO_MC_LINK_COL}_to/_{TRACK_TO_MC_LINK_COL}_to.collectionID",
            f"_{TRACK_TO_MC_LINK_COL}_to/_{TRACK_TO_MC_LINK_COL}_to.index",
            f"_{TRACK_TO_MC_LINK_COL}_from/_{TRACK_TO_MC_LINK_COL}_from.collectionID",
            f"_{TRACK_TO_MC_LINK_COL}_from/_{TRACK_TO_MC_LINK_COL}_from.index",
        ]
    )
    # maps the recoparticle track/cluster index (in tracks_begin,end and clusters_begin,end)
    # to the index in the track/cluster collection
    idx_rp_to_cluster = arrs[f"_{PANDORA_PFO_COL}_clusters/_{PANDORA_PFO_COL}_clusters.index"].array()
    idx_rp_to_track = arrs[f"_{PANDORA_PFO_COL}_tracks/_{PANDORA_PFO_COL}_tracks.index"].array()

    hit_data = {}
    for CALO_HIT_COL in CALO_HIT_COLS:
        hit_data[CALO_HIT_COL] = arrs[CALO_HIT_COL].array() 
    ret = []
    i =0 
    dic = {}
    dic["energy_track"] = []
    dic["index_no_track"] = []
    dic["energy_no_track"] = []
    dic["phi_no_track"] = []
    dic["phi_track"] = []
    for iev in tqdm.tqdm(range(arrs.num_entries), total=arrs.num_entries):
        # print("Processing event ", iev)
        # if i ==5:
            # get the reco particles
        reco_arr = get_reco_properties( prop_data, iev)
        reco_type = np.abs(reco_arr["PDG"])
    
    
        # get the genparticles and the links between genparticles and tracks/clusters
        gpdata, dic  = get_genparticles_and_adjacencies( prop_data, hit_data, calohit_links, sitrack_links, iev, collectionIDs, eval_dataset, dic)


        n_tracks = len(gpdata.track_features["type"])
        n_hits = len(gpdata.hit_features["type"])
        n_gps = len(gpdata.gen_features["PDG"])
        print("hits={} tracks={} gps={}".format(n_hits, n_tracks, n_gps))

        track_to_gp = gpdata.track_to_gp
        hit_to_gp = gpdata.hit_to_gp


        X_track = get_feature_matrix(gpdata.track_features, track_feature_order)
        X_hit = get_feature_matrix(gpdata.hit_features, hit_feature_order)
        X_gen = get_feature_matrix(gpdata.gen_features, particle_feature_order)
        if eval_dataset:
            X_pandora = get_feature_matrix(gpdata.pandora_features, PandoraPFO_feature_order)
        ygen_track = track_to_gp
        ygen_hit = hit_to_gp
    #     ycand_track = rps_track
    #     ycand_hit = rps_hit

        sanitize(X_track)
        sanitize(X_hit)
        sanitize(X_gen)
        sanitize(ygen_track)
        sanitize(ygen_hit)
        if eval_dataset:
            sanitize(X_pandora) 
            sanitize(gpdata.pfo_to_calohit)
            sanitize(gpdata.pfo_to_track)

        this_ev = {
            "X_track": X_track,
            "X_hit": X_hit,
            "X_gen": X_gen, 
            "ygen_track": ygen_track,
            "ygen_hit": ygen_hit,
            "ygen_hit_calomother": gpdata.gp_to_calohit_beforecalomother
        }
        if eval_dataset:
            this_ev["X_pandora"] = X_pandora
            this_ev["pfo_calohit"] = gpdata.pfo_to_calohit
            this_ev["pfo_track"] = gpdata.pfo_to_track
            

        this_ev = awkward.Record(this_ev)
        ret.append(this_ev)
        # i = i +1
 
       
    ret = {k: awkward.from_iter([r[k] for r in ret]) for k in ret[0].fields}
    for k in ret.keys():
        if len(awkward.flatten(ret[k])) == 0:
            ret[k] = build_dummy_array(len(ret[k]), np.float32)
    ret = awkward.Record(ret)
    awkward.to_parquet(ret, ofn, compression="snappy")
    # np.save('data.npy', dic, allow_pickle=True)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input file ROOT file", required=True)
    parser.add_argument("--outpath", type=str, default="raw", help="output path")
    parser.add_argument("--dataset", action="store_true", default=False, help="is dataset for eval")
    args = parser.parse_args()
    return args


def process(args):
    infile = args.input
    outfile = os.path.join(args.outpath, os.path.basename(infile).split(".")[0] + ".parquet")
    eval_dataset = args.dataset 
    tic = time.time()
    process_one_file(infile, outfile, eval_dataset)
    toc = time.time()
    print("Processing time: ", toc - tic)


if __name__ == "__main__":
    args = parse_args()
    process(args)

