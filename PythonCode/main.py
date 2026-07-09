from config import ReconstructionConfig
from Data.data_reader import readData
from MLP.PMLP import evaluateMLP
from ProcessPaths.ProcessPathsController import processPaths
from ART.ARTAlgorithm import ART

if __name__ == "__main__":
    config = ReconstructionConfig()

    events = readData(config)

    paths = evaluateMLP(events, config)

    p, H = processPaths(events, paths, config)

    ART(H, p, config)



