import numpy as np

def rotatePoints(rA, MLPath, UVecs, config):

    
    # Change shape for matrix operations
    x = np.asarray(UVecs).reshape(-1)
    y = np.asarray(MLPath).reshape(-1)
    MLPoints = np.vstack((x, y)) 

    # Rotate points by -rA around cylinder center (0,0)
    R = np.array([[np.cos(-rA), -np.sin(-rA)],
                  [np.sin(-rA),  np.cos(-rA)]])
    P_rot = R @ MLPoints
    MLPX, MLPY = P_rot[0], P_rot[1]

    # Normalize to pixel indices
    x_min, x_max = -(config.image_width_cm / 2), config.image_width_cm/2
    y_min, y_max = -(config.image_width_cm / 2), config.image_width_cm/2
    MLPX_pix = np.clip((MLPX - x_min) / (x_max - x_min) * (config.n-1), 0, config.n-1)
    MLPY_pix = np.clip((MLPY - y_min) / (y_max - y_min) * (config.n-1), 0, config.n-1)

    return MLPX_pix, MLPY_pix