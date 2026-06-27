import numpy as np


def evaluate_T(u_initial, u_upperBound, du, E_initial, energy):
    waterScatteringLength = 0.0115 #1/cm
    T_hat_bone = 0.267646 #I think its a relative thing so no units?
    
    Mp = 938.272 # MeV/c^2
    Me = 0.510999  # MeV/c^2
    alpha = 7.2974*(10**(-3) )
    
    currentProtonEnergy = E_initial - energy
    tau = currentProtonEnergy / Mp
    return ((2*np.pi*(Me**2))/alpha)*(((tau+1)/(tau+2))**2)*(1/(currentProtonEnergy**2))*(waterScatteringLength)*(T_hat_bone)


def weighted_average(E_forward, E_backward):
    E_new = np.zeros_like(E_forward)
    E_new[0] = E_forward[0]
    E_new[-1] = E_backward[-1]
    N = len(E_new)
    for j in range(1,N-1):
        E_new[j] = (N - j) / N * E_forward[j] + j / N * E_backward[j]
    
    return E_new