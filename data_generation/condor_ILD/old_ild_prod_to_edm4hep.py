from Gaudi.Configuration import INFO, WARNING, DEBUG
from Configurables import EventDataSvc, MarlinProcessorWrapper, LcioEvent
from k4FWCore import ApplicationMgr, IOSvc
from k4MarlinWrapper.io_helpers import IOHandlerHelper
from k4FWCore.parseArgs import parser

# Parser for command line arguments
parser.add_argument(
    "--inputFiles",
    action="extend",
    nargs="+",
    metavar=["file1", "file2"],
    help="One or multiple input files",
)

parser.add_argument(
    "--outputFileBase",
    help="Base name of all the produced output files",
    default="StandardReco",
)
reco_args = parser.parse_known_args()[0]

alg_list = []
evt_svc = EventDataSvc("EventDataSvc")
svc_list = [evt_svc]
io_svc = IOSvc()

io_handler = IOHandlerHelper(alg_list, io_svc)

io_handler.add_reader(reco_args.inputFiles)

def parse_collection_patch_file(patch_file):
  # Copied from ILDConfig
  # https://github.com/iLCSoft/ILDConfig/blob/2c492445fccbd8fae1afd16592d23de917cb0920/StandardConfig/production/py_utils.py#L144
  with open(patch_file, "r") as pfile:
      patch_colls = [l.split() for l in pfile.readlines()]

  # Flatten the list of lists into one large list
  return [s for strings in patch_colls for s in strings]



# https://github.com/iLCSoft/ILDConfig/pull/170
# Run RecoMCTruthLinker to create the CaloHit - MC truth links


CaloHitToMcTruthLinker = MarlinProcessorWrapper("CaloHitToMcTruthLinker")
CaloHitToMcTruthLinker.ProcessorType = "RecoMCTruthLinker"
CaloHitToMcTruthLinker.Parameters = {
  "MCParticleCollection": ["MCParticle"],
  "TrackCollection": ["MarlinTrkTracks"],
  "ClusterCollection": ["PandoraClusters"],
  "RecoParticleCollection": ["PandoraPFOs"],

  ### input collections
  "SimTrackerHitCollections": [ "VXDCollection", "SITCollection", "FTD_PIXELCollection", "FTD_STRIPCollection", "TPCCollection", "SETCollection"],

  "TrackerHitsRelInputCollections": [ "VXDTrackerHitRelations", "SITTrackerHitRelations", "FTDPixelTrackerHitRelations", "FTDSpacePointRelations", "TPCTrackerHitRelations", "SETSpacePointRelations"],

  # excluding LumiCal, BeamCal, and LHCal collections for now, because their time needs manual fixing.
  # "BeamCalCollection", "LHCalCollection", "LumiCalCollection", 
  "SimCaloHitCollections": ["ECalBarrelSiHitsEven",
                            "ECalBarrelSiHitsOdd",
                            "ECalEndcapSiHitsEven",
                            "ECalEndcapSiHitsOdd",
                            "EcalEndcapRingCollection",
                            "HcalBarrelRegCollection",
                            "HcalEndcapsCollection",
                            "HcalEndcapRingCollection",
                            "YokeBarrelCollection",
                            "YokeEndcapsCollection"
                            ],


  # "RelationLHcalHit", "RelationLcalHit", "RelationBCalHit"
  "SimCalorimeterHitRelationNames": [ "EcalBarrelRelationsSimRec",
                                      "EcalEndcapRingRelationsSimRec",
                                      "EcalEndcapsRelationsSimRec",
                                      "HcalBarrelRelationsSimRec",
                                      "HcalEndcapRingRelationsSimRec",
                                      "HcalEndcapsRelationsSimRec",
                                      "RelationMuonHit"
                                    ],


  ### output collections
  "MCTruthTrackLinkName": [""],
  "TrackMCTruthLinkName": [""],
  "ClusterMCTruthLinkName": [""],
  "MCTruthClusterLinkName": [""],
  "RecoMCTruthLinkName": [""],
  "MCTruthRecoLinkName": [""],
  "MCParticlesSkimmedName": [""],
  "CalohitMCTruthLinkName": ["CalohitMCTruthLink"],

  ### steering parameters
  "FullRecoRelation": ["true"],
  "KeepDaughtersPDG": ["22", "111", "310", "13", "211", "321"],
  "UseTrackerHitRelations": ["true"],
  "UsingParticleGun": ["false"],
}


# Make sure that all collections are always available by patching in missing ones on-the-fly
collPatcherRec = MarlinProcessorWrapper("CollPatcherREC", ProcessorType="PatchCollections")
collPatcherRec.Parameters = {
    "PatchCollections": parse_collection_patch_file("data_generation/condor_ILD/collections_rec_level.txt")
}

alg_list.append(CaloHitToMcTruthLinker)
alg_list.append(collPatcherRec)

io_handler.add_edm4hep_writer(f"{reco_args.outputFileBase}_REC.edm4hep.root", ["keep *"])

io_handler.finalize_converters()

ApplicationMgr(TopAlg=alg_list,
               EvtSel="NONE",
               EvtMax=-1,
               ExtSvc=svc_list,
               OutputLevel=WARNING,
               )
