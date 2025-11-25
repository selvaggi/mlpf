// How to:
// Update path to your edm4hep files in files variable below.
// Run as a ROOT script: $ root count_sim_decays.cpp

#include "podio/ROOTReader.h"
#include "podio/Frame.h"

#include "edm4hep/MCParticleCollection.h"

//if the list is too long put it in a separate header file and include.
// #include "data_files.h"
auto files = std::vector<std::string>{
    "/path/to/my/file1.edm4hep.root",
    "/path/to/my/file2.edm4hep.root",
    "/path/to/my/file3.edm4hep.root",
}

// TODO: if particle dies in the tracker (daughters and parent do not reach calorimeter) then ignore, as we don't care for particle flow.
// TODO: Maybe introduce additional minimal criteria when we care?
// E.g. at least one daughter or parent leaves more than X hits in the calorimeter or has more than Y GeV energy?

bool isK0sDecay(const edm4hep::MCParticle& mc){
    if ( not (std::abs(mc.getPDG()) == 310) ) return false;
    if ( not (mc.isDecayedInTracker()) ) return false;

    auto daughters = mc.getDaughters();
    if (daughters.size() != 2) return false;
    auto pdg1 = std::abs(daughters[0].getPDG());
    auto pdg2 = std::abs(daughters[1].getPDG());
    if (not (pdg1 == 211 && pdg2 == 211) ) return false;
    return true;
}

bool isLambdaDecay(const edm4hep::MCParticle& mc){
    if ( not (std::abs(mc.getPDG()) == 3122) ) return false;
    if ( not mc.isDecayedInTracker() ) return false;

    auto daughters = mc.getDaughters();
    if (daughters.size() != 2) return false;
    auto pdg1 = std::abs(daughters[0].getPDG());
    auto pdg2 = std::abs(daughters[1].getPDG());
    if ( not( (pdg1 == 2212 && pdg2 == 211) || (pdg1 == 211 && pdg2 == 2212) ) ) return false;
    return true;
}

bool isGammaConversion(const edm4hep::MCParticle& mc){
    if ( not (std::abs(mc.getPDG()) == 22) ) return false;
    if ( not mc.isDecayedInTracker() ) return false;

    auto daughters = mc.getDaughters();
    if (daughters.size() != 2) return false;
    auto pdg1 = std::abs(daughters[0].getPDG());
    auto pdg2 = std::abs(daughters[1].getPDG());
    if ( not( (pdg1 == 11 && pdg2 == 11) ) ) return false;
    return true;
}

bool isPionDecay(const edm4hep::MCParticle& mc){
    if ( not (std::abs(mc.getPDG()) == 211) ) return false;
    if ( not mc.isDecayedInTracker() ) return false;

    auto daughters = mc.getDaughters();
    if (daughters.size() != 2) return false;
    auto pdg1 = std::abs(daughters[0].getPDG());
    auto pdg2 = std::abs(daughters[1].getPDG());
    if ( not( (pdg1 == 13 && pdg2 == 14) || (pdg1 == 14 && pdg2 == 13) ) ) return false;
    return true;
}

bool isBremstrahlung(const edm4hep::MCParticle& mc){
    if ( (std::abs(mc.getCharge()) == 0.f) ) return false;
    if ( not mc.isDecayedInTracker() ) return false;
    if ( not (mc.getGeneratorStatus() == 1 || mc.getGeneratorStatus() == 0) ) return false;

    auto daughters = mc.getDaughters();
    if (daughters.size() != 1) return false;
    auto pdg = std::abs(daughters[0].getPDG());
    if (pdg != 22) return false;
    if ( not daughters[0].vertexIsNotEndpointOfParent() ) return false;
    return true;
}

bool isMaterialInteraction(const edm4hep::MCParticle& mc){
    if ( not mc.isDecayedInTracker() ) return false;
    auto parentEnergy = mc.getEnergy();
    auto daughtersEnergy = 0.;
    for ( const auto& d : mc.getDaughters() ) daughtersEnergy += d.getEnergy();
    if ( std::abs(parentEnergy - daughtersEnergy) < 0.5 ) return false;
    return true;
}

bool isBackscatter(const edm4hep::MCParticle& mc){
    return mc.isBackscatter();
}

bool isGhost(const edm4hep::MCParticle& mc){
    // https://github.com/AIDASoft/DD4hep/issues/1431
    auto r = std::sqrt(mc.getEndpoint()[0]*mc.getEndpoint()[0] + mc.getEndpoint()[1]*mc.getEndpoint()[1] + mc.getEndpoint()[2]*mc.getEndpoint()[2]);
    auto isFakeParticle = ( r < 0.001 && mc.getGeneratorStatus() == 1 && ! mc.isDecayedInCalorimeter() && ! mc.isDecayedInTracker() && ! mc.hasLeftDetector() );
    return isFakeParticle;
}


void count_sim_decays() {
    auto reader = podio::ROOTReader();
    reader.openFiles(files);
    std::cout<<"Number of events: " << reader.getEntries("events") << std::endl;

    auto decayNames = std::vector<std::string>{"K_short", "Lambda", "GammaConversion", "Pion Decay", "Brem", "Material Int", "Backscatter", "Ghost"};
    auto decayFuncs = std::vector<std::function<bool(const edm4hep::MCParticle&)> >{isK0sDecay, isLambdaDecay, isGammaConversion, isPionDecay, isBremstrahlung, isMaterialInteraction, isBackscatter, isGhost};
    auto decayHappenedFlags = std::vector<bool>(decayNames.size());

    auto nEventsAnalyzed = 0;
    //Store how many times given scenario occurs, e.g. K0s decays and in how many events given scenario occurs
    std::map<std::string, int > decayCounts;
    std::map<std::string, int > evtDecayCounts;

    for (size_t i = 0; i < reader.getEntries("events"); ++i){
        if ( i % 1000 == 0 ) std::cout<<"Event: " << i << std::endl;
        // if ( i == 100) break; # earyly stop for debugging
        auto event = podio::Frame(reader.readNextEntry("events"));
        auto& mcs = event.get<edm4hep::MCParticleCollection>("MCParticle");

        decayHappenedFlags = std::vector<bool>(decayNames.size()); // reset every event
        for(const auto& mc : mcs){
            for(int j=0; j < decayNames.size(); ++j){
                auto isDecay = decayFuncs[j];
                auto decayName = decayNames[j];
                auto alreadyHappend = decayHappenedFlags[j];
                if ( not isDecay(mc) ) continue;
                decayCounts[decayName]++;

                if (alreadyHappend) continue;
                evtDecayCounts[decayName]++;
                decayHappenedFlags[j] = true;
            }
        }
        nEventsAnalyzed++;
    }

    std::cout<<std::right<<std::setw(20)<<"Decay Name:"<<std::setw(30)<<"N Occurances / N events"<<std::setw(30)<<"Avg decays/event"<<std::setw(50)<<"Fraction of events with at least one decay"<<std::endl;
    for(int j=0; j < decayNames.size(); ++j){
        auto decayName = decayNames[j];
        auto nDecays = decayCounts[decayName];
        auto nEventsWithDecay = evtDecayCounts[decayName];
        auto ratio = std::format("{}/{}", nDecays, nEventsAnalyzed);
        auto avgDecays = std::format("{:.2f}", 1.*nDecays/nEventsAnalyzed);
        auto evtFraction = std::format("{:.1f} %", 100.*nEventsWithDecay/nEventsAnalyzed);
        std::cout<<std::right<<std::setw(20)<<decayName<<std::setw(30)<<ratio<<std::setw(30)<<avgDecays<<std::setw(50)<<evtFraction<<std::endl;
    }
}
