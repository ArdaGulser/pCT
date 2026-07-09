from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import numpy as np

from ProcessPaths.IntegrateStoppingPower import (
    euler_integrate_inverse_stopping_power,
)
from ProcessPaths.PathLengthPerPixel import (
    compute_path_length_per_pixel,
)


def _process_single(i, event, path, config):
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

    with ProcessPoolExecutor() as executor:
        for i, p_i, H_i in executor.map(
            _process_single,
            range(N),
            events,
            paths,
            repeat(config),
            chunksize=50,
        ):
            p[i] = p_i
            H[i] = H_i

    return p, H