from concurrent.futures import ProcessPoolExecutor
import numpy as np

from ProcessPaths.IntegrateStoppingPower import (
    euler_integrate_inverse_stopping_power,
)
from ProcessPaths.PathLengthPerPixel import (
    compute_path_length_per_pixel,
)


def _process_single(args):
    i, event, path, config = args

    print(event["name"])

    p = -euler_integrate_inverse_stopping_power(
        event["energy_in"],
        event["energy_out"],
    )

    H = compute_path_length_per_pixel(path, config)

    return i, p, H


def processPaths(events, paths, config):

    N = config.N
    n = config.n

    p = np.zeros(N)
    H = np.zeros((N, n * n))

    tasks = [
        (i, events[i], paths[i], config)
        for i in range(N)
    ]

    with ProcessPoolExecutor() as executor:
        for i, p_i, H_i in executor.map(_process_single, tasks):
            p[i] = p_i
            H[i] = H_i

    return p, H