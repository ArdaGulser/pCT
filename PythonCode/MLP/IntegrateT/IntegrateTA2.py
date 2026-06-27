import numpy as np
from MLP.IntegrateT.IntegrateTHelper import evaluate_T, weighted_average

def integrate_T_forward_euler_A2(u_initial, u_final, du, E_initial, energy_list):
    u_values = np.arange(u_initial, u_final, du)
    
    E_values = np.zeros_like(u_values)
    E_values[0] = E_initial
    
    E_loss = np.zeros_like(u_values)
    
    for i in range(1, len(u_values)):
        E_values[i] = E_values[i-1] + ((u_final - u_values[i])**2)*evaluate_T(u_initial, u_values[i], du, E_initial, energy_list[i]) * du  # Forward Euler step
        E_loss[i] = -E_values[i] + E_initial
    
    return u_values, E_values, E_loss

def integrate_T_backward_euler_A2(u_initial, u_final, du, E_initial, energy_list):
    u_values = np.arange(u_initial, u_final, du)
    
    E_values = np.zeros_like(u_values)
    E_values[0] = E_initial
    
    E_loss = np.zeros_like(u_values)
    
    for i in range(1, len(u_values)):
        E_values[i] = E_values[i-1] + ((u_final - u_values[i])**2)*evaluate_T(u_initial, u_values[i], du, E_initial, energy_list[i]) * du  # Backward Euler step (Implicit)
        E_loss[i] = -E_values[i] + E_initial
    
    return u_values, E_values, E_loss

def integrate_T_A2(u_initial, u_final, du, E_initial, energy_list):
    u_values, E_forward, E_loss_forward = integrate_T_forward_euler_A2(u_initial, u_final, du, E_initial, energy_list)
    u_values, E_backward, E_loss_backward = integrate_T_backward_euler_A2(u_initial, u_final, du, E_initial, energy_list)
    E_values = weighted_average(E_forward, E_backward)
    E_loss = weighted_average(E_loss_forward,E_loss_backward)

    return u_values, E_values, E_loss, E_loss[len(E_loss)-1]