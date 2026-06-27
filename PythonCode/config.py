from dataclasses import dataclass
import string

@dataclass
class ReconstructionConfig:

    file_name: string = "RootFiles/ManyInserts4k.root" #Name of root file for data

    log_name: string = "RootFiles/PointCTLog4k.txt" #Name of text file for initial conditions

    N: int = 4000 #Number of proton events

    n: int = 25 #the image is an n x n matrix

    image_width_cm: float = 12.0 #Gives dimensions of recosntructiln in physical space

    beam_energy_mev: float = 200 #Starting energy of beam

    initial_spread: float = 0.2 #Initial Spread of the beam in cm

    lambda_ART: float = 0.001 #Lambda for ART Algo

    art_tolerance: float = 1e-3 #Stopping point for ART

    art_max_iter: int = 5000 #Max number of iterations for ART
