from MLP.IntegrateBethe import evaluate_bethe, beta


#Test Area to see if works
def euler_integrate_inverse_stopping_power(E_in, E_out, dE=0.1):
    """
    Euler method to integrate 1/S(E) from E_out to E_in.

    Parameters:
        E_in (float): Initial energy in MeV (must be > E_out)
        E_out (float): Final energy in MeV
        dE (float): Step size in MeV (default 0.1)

    Returns:
        float: Approximate path length in cm
    """
    if E_in <= 0 or E_out <= 0:
        raise ValueError("Energies must be positive.")
    if E_in < E_out:
        raise ValueError("E_in must be greater than E_out.")
    if dE <= 0:
        raise ValueError("Step size dE must be positive.")

    E = E_out
    length = 0.0
    
    while E < E_in:
        S = evaluate_bethe(beta(E))
        
        dL = dE / S 
        
        length += dL
        E += dE

    return length
#------------------------