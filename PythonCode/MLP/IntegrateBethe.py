import numpy as np


m_p = 938.2720813       # proton rest mass [MeV]
m_e = 0.5109989461      # electron rest mass [MeV]
K = 0.307075            # MeV cm^2 / g  (ICRU/PDG constant)
Z_over_A = 10.0 / 18.01528  # effective Z/A for water (H2O)
rho = 1.0               # g/cm^3 (water)
I_eV = 75.0             # mean excitation energy in eV (typical for water)
I = I_eV * 1e-6         # convert eV -> MeV
z = 1  # projectile charge number (proton)

def beta(E):
    """Relativistic beta from kinetic energy E (MeV)."""
    Mp = 938.272 # Mass of proton (MeV/c^2)
    return np.sqrt(1-(Mp/(E+Mp))**2)

def W_max(beta, gamma):
    """Maximum transferable kinetic energy to an electron (MeV)."""
    if beta == 0:
        return 0.0
    # formula: Wmax = 2 m_e beta^2 gamma^2 / (1 + 2 gamma m_e / m_p + (m_e/m_p)^2)
    denom = 1.0 + 2.0 * gamma * (m_e / m_p) + (m_e / m_p)**2
    return (2.0 * m_e * beta**2 * gamma**2) / denom

def evaluate_bethe(beta_val):
    """Return dE/dx [MeV/cm] using Bethe formula given beta."""
    gamma = 1.0 / np.sqrt(1.0 - beta_val**2)
    W = W_max(beta_val, gamma)
    arg = (2*m_p*beta_val**2*gamma**2*W)/(I**2)
    return -K * z**2 * (Z_over_A) * (1/beta_val**2) * (0.5*np.log(arg) - beta_val**2) * rho

def dEdx(E):
    """Derivative dE/dx using Bethe formula."""
    return evaluate_bethe(beta(E))

def integrate_bethe_forward_euler(u_initial, u_final, du, E_initial):
    u_values = np.arange(u_initial, u_final, du)
    E_values = np.zeros_like(u_values)
    E_loss = np.zeros_like(u_values)

    E_values[0] = E_initial

    for i in range(1, len(u_values)):
        
            
        
        fE = dEdx(E_values[i-1])
        E_values[i] = E_values[i-1] + fE * du
        E_loss[i] = E_initial - E_values[i]

    return E_loss

def integrate_bethe_backward_euler(u_initial, u_final, du, E_initial):

    u_values = np.arange(u_initial, u_final, du)
    E_values = np.zeros_like(u_values)
    E_loss = np.zeros_like(u_values)

    E_values[0] = E_initial

    for i in range(1, len(u_values)):
       
    
        # predictor (forward Euler)
        E_pred = E_values[i-1] + dEdx(E_values[i-1]) * du
        # corrector (evaluate slope at predicted point)
        f_pred = dEdx(E_pred)
        E_values[i] = E_values[i-1] + f_pred * du
        E_loss[i] = E_initial - E_values[i]

    return E_loss



def weighted_average(E_forward, E_backward):
    E_new = np.zeros_like(E_forward)
    E_new[0] = E_forward[0]
    E_new[-1] = E_backward[-1]
    N = len(E_new)
    for j in range(1,N-1):
        E_new[j] = (N - j) / N * E_forward[j] + j / N * E_backward[j]
    
    return E_new

def integrate_bethe(u_initial, u_final, du, E_initial):
    E_loss_forward = integrate_bethe_forward_euler(u_initial, u_final, du, E_initial)
    E_loss_backward = integrate_bethe_backward_euler(u_initial, u_final, du, E_initial)
    E_loss = weighted_average(E_loss_forward,E_loss_backward)

    return E_loss