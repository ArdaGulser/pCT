from ProcessPaths.IntegrateStoppingPower import euler_integrate_inverse_stopping_power
from ProcessPaths.PathLengthPerPixel import compute_path_length_per_pixel
import numpy as np

def processPaths(events, paths, config):

    N = config.N
    n = config.n
    H = np.zeros((N, n*n))
    p = np.zeros(N)
    nameStrings = events["name"]
    i = 0
    for name in nameStrings:
        print(name)
        p[i] = -1 * euler_integrate_inverse_stopping_power(events["energy_in"][i], events["energy_out"][i])
        H[i] = compute_path_length_per_pixel(paths[i], config)
        i += 1

    return p, H
