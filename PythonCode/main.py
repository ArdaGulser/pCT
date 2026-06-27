from config import ReconstructionConfig
from Data.data_reader import readData
from MLP.PMLP import evaluateMLP
from ProcessPaths.ProcessPathsController import processPaths
from ART.ARTAlgorithm import ART

config = ReconstructionConfig()

events = readData(config)

paths = evaluateMLP(events, config)

p, H = processPaths(events, paths, config)
#print(H[2000], p[2000])
ART(H, p, config)



