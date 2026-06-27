import ROOT
import numpy as np
from config import ReconstructionConfig

def readData(config):
    event_dtype = np.dtype([
        ("name", "U32"),
        ("x0", "f8"),
        ("y0", "f8"),
        ("px0", "f8"),
        ("py0", "f8"),
        ("angle", "f8"),
        ("energy_in", "f8"),
        ("energy_out", "f8"),
        ("mean_x", "f8"),
        ("mean_y", "f8"),
        ("sigma_x", "f8"),
        ("sigma_y", "f8"),
    ])

    events = np.zeros(config.N, dtype=event_dtype)

    events = getInitialConditions(events, config)
    events = readRootFile(events, config)
    return events

def readRootFile(events, config):
    nameStrings = events["name"]
    rootFile = ROOT.TFile(config.file_name)
    i = 0
    for name in nameStrings:
        print(name)
        root_tuple = rootFile.Get(name)
        root_tuple.GetEntry(1)
        events["sigma_y"][i] = root_tuple.posZ / 10 #Im not sure if Im supposed to divide these by 10 to convert to cm, I need to check geant
        events["sigma_x"][i] = root_tuple.posX / 10 #Im not sure if Im supposed to divide these by 10 to convert to cm, I need to check geant

        root_tuple.GetEntry(0)
        events["mean_y"][i] = root_tuple.posZ / 10 #Im not sure if Im supposed to divide these by 10 to convert to cm, I need to check geant
        events["mean_x"][i] = root_tuple.posX / 10 #Im not sure if Im supposed to divide these by 10 to convert to cm, I need to check geant

        events["energy_in"][i] = config.beam_energy_mev
        events["energy_out"][i] = root_tuple.Energy
        i += 1

    return events
    




def getInitialConditions(events, config):
    with open(config.log_name, "r") as file:
        i = -1
        for line in file:
            line = line.strip()
            #print(line)
            if line[0:5] == "orbit":
                i += 1
                run = line
                print(run)
                events["name"][i] = run
            if line[0:16] == "Phantom Rotation":
                anglevals = line[16:].split(" ")
                events["angle"][i] = float(anglevals[2])
            if line[0:14] == "Beam Position:" or line[0:14] == "Beam_Position:":
                initialVals = line[15:].split(" ")
                x0 = float(initialVals[0]) 
                y0 = float(initialVals[2]) 
                events["x0"][i] = x0
                events["y0"][i] = y0
            if line[0:15] == "Beam Direction:":
                initialVals = line[16:].split(" ")
                x0 = float(initialVals[0])
                y0 = float(initialVals[2])
                events["px0"][i] = x0
                events["py0"][i] = y0
            
           
    return events