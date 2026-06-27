import numpy as np
from scipy import integrate as spi
import MLP.IntegrateBethe as ib
from MLP.IntegrateT.IntegrateTA0 import integrate_T_A0
from MLP.IntegrateT.IntegrateTA1 import integrate_T_A1
from MLP.IntegrateT.IntegrateTA2 import integrate_T_A2
from MLP.RotatePoints import rotatePoints


waterScatteringLength = 0.0115 #1/cm
T_hat_bone = 1.805 #I think its a relative thing so no units?
Mp = 938.272 # MeV/c^2
Me = 0.510999  # MeV/c^2

def estimate_exit_angle(spread):
    #This function is the most rudimentary way of getting the exit angle, we need a better way
    return np.arctan(spread/12)



def evaluateMLP(events, config):
    nameStrings = events["name"]
    paths = []
    i = 0
    for name in nameStrings:
        print(name)
        currentMLP = []

        

        #Get Equivalent Point Beam Spread and use it for approx exit position
        radius = np.sqrt(np.abs(events["sigma_y"][i]**2 - config.initial_spread)) / 4

        x0 = events["x0"][i]
        y0 = events["y0"][i]
        x2 = events["mean_x"][i]
        y2 = events["mean_y"][i] + radius #We just shift the y exit position by the beam spread
        t0 = 0
        t2 = estimate_exit_angle(radius)

        #Calculate the MLP
        #MLPath, UVecs = YLPFormula(x0, 20, x2, y0, t0, y2, t2, (x2-x0)/200, config)

        #----DEBUG_SECTION, DELETE LATER--------#
        #Be sure to uncomment YLP formula call
        MLPath = np.linspace(y0, y2, num=20)
        UVecs = np.linspace(x0, x2, num=20)
        #---------------------------------------#

        MLPX_pix, MLPY_pix = rotatePoints(events["angle"][i], MLPath, UVecs, config)


        # Build path list
        for j in range(len(MLPX_pix)):
            data = (MLPX_pix[j], MLPY_pix[j])
            currentMLP.append(data)

        paths.append(currentMLP)

   
        i += 1

    return paths



def find_closest(lst, value):
    arr = np.array(lst)
    idx = (np.abs(arr - value)).argmin()
    return idx


def YLPFormula(U0, pathPoints, U2, LD0, T0, LD2, T2, du, config):
    """
    Calculates the Most Likely Path of Proton through a phantom of water and bone
   
    U0, U1, and U2: Initial, desired, and final depths respectively

    LD0: initial and final displacement respectively

    T0, T2: Initial and Final angles respectively

    du: step size for integration

    """
    integrated_bethe = ib.integrate_bethe(U0, U2, du, 200)
    
    epsilon = 1e-8

    returns = integrate_T_A0(U0, U2, du, config.beam_energy_mev, integrated_bethe)
    integratedDepthList = returns[0]
    t0 = returns[2]
    

    #Initialize arrays
    MLP = [LD0]
    MLPerror = []
    U2Index = len(t0) - 1
    depthValues = [U0]
    u1 = np.linspace(U0, U2, num=pathPoints)
    
    for depth in u1:
        if depth == U0 or depth == U2:
            continue
        
        U1Index = find_closest(integratedDepthList, depth)
        A0_0 = t0[U1Index] + epsilon

        A1_0 = integrate_T_A1(U0, depth, du, config.beam_energy_mev, integrated_bethe)[3]
        A2_0 = integrate_T_A2(U0, depth, du, config.beam_energy_mev, integrated_bethe)[3]

        A1_0 += epsilon
        A2_0 += epsilon

        A0_1 = t0[U2Index - U1Index] + epsilon

        A1_1 = integrate_T_A1(depth, U2, du, config.beam_energy_mev, integrated_bethe)[3]
        A2_1 = integrate_T_A2(depth, U2, du, config.beam_energy_mev, integrated_bethe)[3]
        A1_1 += epsilon
        A2_1 += epsilon



        U1 = depth
        depthValues.append(U1)


 

   
    
        sigmaMatrix1Inverse = np.linalg.inv(np.array([[A2_0, A1_0],
                                                    [A1_0, A0_0]]))
        
    
        
        sigmaMatrix2Inverse = np.linalg.inv(np.array([[A2_1, A1_1],
                                                    [A1_1, A0_1]]))

        
        R0 = np.array([[1, U1-U0],
                    [0, 1]])
        R1 = np.array([[1, U2-U1],
                    [0, 1]])
        y0 = np.array([[LD0], [T0]])  
        y2 = np.array([[LD2], [T2]])
        
        result = np.linalg.inv(sigmaMatrix1Inverse+R1.T@sigmaMatrix2Inverse@R1) @ (sigmaMatrix1Inverse@R0@y0 + R1.T@sigmaMatrix2Inverse@y2)
        error = 2*np.linalg.inv(sigmaMatrix1Inverse + R1.T@sigmaMatrix2Inverse@R1)
        MLP.append(result[0][0])
        MLPerror.append(error)
    

    MLP.append(LD2)
    depthValues.append(U2)
    return MLP, depthValues










