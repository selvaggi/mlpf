import os
import numpy as np
import awkward
import uproot
import tqdm
from preprocessing.utils_data_creation_ALLEGRO import (
    get_feature_matrix, sanitize, get_reco_properties, build_dummy_array,
    track_feature_order, hit_feature_order, particle_feature_order,
    PandoraPFO_feature_order, get_genparticles_and_adjacencies, mc_coll
)
import time

def process_one_file(fn, ofn_base, eval_dataset, chunk_size=100):
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
            mc_coll,
            "MCParticles.PDG",
            "MCParticles.momentum.x", "MCParticles.momentum.y", "MCParticles.momentum.z",
            "MCParticles.mass", "MCParticles.charge",
            "MCParticles.generatorStatus", "MCParticles.simulatorStatus",
            "MCParticles.daughters_begin", "MCParticles.daughters_end",
            "_MCParticles_daughters/_MCParticles_daughters.index",
            "_MCParticles_parents/_MCParticles_parents.index",
            "TracksFromGenParticles", "_TracksFromGenParticles_trackStates",
        ]
    )

    calohit_links = arrs.arrays([
        "CaloHitMCParticleLinks.weight",
        "_CaloHitMCParticleLinks_to/_CaloHitMCParticleLinks_to.collectionID",
        "_CaloHitMCParticleLinks_to/_CaloHitMCParticleLinks_to.index",
        "_CaloHitMCParticleLinks_from/_CaloHitMCParticleLinks_from.collectionID",
        "_CaloHitMCParticleLinks_from/_CaloHitMCParticleLinks_from.index",
    ])

    sitrack_links = arrs.arrays([
        "TracksFromGenParticlesAssociation.weight",
        "_TracksFromGenParticlesAssociation_to/_TracksFromGenParticlesAssociation_to.collectionID",
        "_TracksFromGenParticlesAssociation_to/_TracksFromGenParticlesAssociation_to.index",
        "_TracksFromGenParticlesAssociation_from/_TracksFromGenParticlesAssociation_from.collectionID",
        "_TracksFromGenParticlesAssociation_from/_TracksFromGenParticlesAssociation_from.index",
    ])

    hit_data = {
        "ECalBarrelModuleThetaMergedPositioned": arrs["ECalBarrelModuleThetaMergedPositioned"].array(),
        "ECalEndcapTurbinePositioned": arrs["ECalEndcapTurbinePositioned"].array(),
        "HCalBarrelReadoutPositioned": arrs["HCalBarrelReadoutPositioned"].array(),
        "HCalEndcapReadoutPositioned": arrs["HCalEndcapReadoutPositioned"].array(),
        "MuonTaggerBarrelPhiThetaPositioned": arrs["MuonTaggerBarrelPhiThetaPositioned"].array(),
        "MuonTaggerEndcapPhiThetaPositioned": arrs["MuonTaggerEndcapPhiThetaPositioned"].array(),
    }

    dic = {
        "energy_track": [],
        "index_no_track": [],
        "energy_no_track": [],
        "phi_no_track": [],
        "phi_track": [],
    }

    total_events = arrs.num_entries
    chunk_index = 0
    ret = []

    for iev in tqdm.tqdm(range(total_events), total=total_events):
        gpdata, dic = get_genparticles_and_adjacencies(
            prop_data, hit_data, calohit_links, sitrack_links, iev, collectionIDs, eval_dataset, dic
        )

        X_track = get_feature_matrix(gpdata.track_features, track_feature_order)
        X_hit = get_feature_matrix(gpdata.hit_features, hit_feature_order)
        X_gen = get_feature_matrix(gpdata.gen_features, particle_feature_order)
        ygen_track = gpdata.track_to_gp
        ygen_hit = gpdata.hit_to_gp

        sanitize(X_track)
        sanitize(X_hit)
        sanitize(X_gen)
        sanitize(ygen_track)
        sanitize(ygen_hit)

        this_ev = {
            "X_track": X_track,
            "X_hit": X_hit,
            "X_gen": X_gen,
            "ygen_track": ygen_track,
            "ygen_hit": ygen_hit,
            "ygen_hit_calomother": gpdata.gp_to_calohit_beforecalomother
        }

        if eval_dataset:
            X_pandora = get_feature_matrix(gpdata.pandora_features, PandoraPFO_feature_order)
            sanitize(X_pandora)
            sanitize(gpdata.pfo_to_calohit)
            sanitize(gpdata.pfo_to_track)
            this_ev["X_pandora"] = X_pandora
            this_ev["pfo_calohit"] = gpdata.pfo_to_calohit
            this_ev["pfo_track"] = gpdata.pfo_to_track

        this_ev = awkward.Record(this_ev)
        ret.append(this_ev)

        # ✅ Save every `chunk_size` events
        if (iev + 1) % chunk_size == 0 or iev == total_events - 1:
            if len(ret) > 0:
                ret_chunk = {k: awkward.from_iter([r[k] for r in ret]) for k in ret[0].fields}
                for k in ret_chunk.keys():
                    if len(awkward.flatten(ret_chunk[k])) == 0:
                        ret_chunk[k] = build_dummy_array(len(ret_chunk[k]), np.float32)
                ret_chunk = awkward.Record(ret_chunk)

                ofn = f"{ofn_base}_{chunk_index}.parquet"
                awkward.to_parquet(ret_chunk, ofn, compression="snappy")
                print(f"✅ Saved {ofn} ({len(ret)} events)")
                ret = []  # clear buffer
                chunk_index += 1


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input ROOT file", required=True)
    parser.add_argument("--outpath", type=str, default="raw", help="Output path")
    parser.add_argument("--dataset", action="store_true", default=False, help="Is eval dataset?")
    parser.add_argument("--chunk_size", type=int, default=100, help="Events per output file")
    return parser.parse_args()


def process(args):
    infile = args.input
    outbase = os.path.join(args.outpath, os.path.basename(infile).split(".")[0])
    eval_dataset = args.dataset
    tic = time.time()
    process_one_file(infile, outbase, eval_dataset, args.chunk_size)
    toc = time.time()
    print("Processing time: ", toc - tic)


if __name__ == "__main__":
    args = parse_args()
    process(args)
