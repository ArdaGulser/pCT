import numpy as np

def get_pixel_index(x, y, n):
    row = min(max(int(y), 0), n - 1)
    col = min(max(int(x), 0), n - 1)
    return row, col

def compute_path_length_per_pixel(path_points, config, step_size=0.01):
    """
    path_points: list of (x,y) coordinates along the path
    step_size: how finely to sample between points
    
    Returns:
        length_map: n x n array with length of path inside each pixel
    """

    n = config.n
    length_map = np.zeros((n, n))
    
    for i in range(len(path_points)-1):
        x0 = path_points[i][0]
        y0 = path_points[i][1]
        x1 = path_points[i+1][0]
        y1 = path_points[i+1][1]
        dist = np.hypot(x1 - x0, y1 - y0)
        num_steps = max(int(dist / step_size), 1)
        
        
        xs = np.linspace(x0, x1, num_steps)
        ys = np.linspace(y0, y1, num_steps)
        
        for j in range(num_steps-1):
            r0, c0 = get_pixel_index(xs[j], ys[j], n)
            r1, c1 = get_pixel_index(xs[j+1], ys[j+1], n)
            segment_length = np.hypot(xs[j+1] - xs[j], ys[j+1] - ys[j])
            
            if (r0 == r1) and (c0 == c1):
                
                length_map[r0, c0] += segment_length
            else:
                
                length_map[r0, c0] += segment_length / 2
                length_map[r1, c1] += segment_length / 2

    length_vector = length_map.reshape(n*n)
    return length_vector