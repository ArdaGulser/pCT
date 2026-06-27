import numpy as np
import matplotlib.pyplot as plt




# ART setup
def ART(H,p, config):
    num_projections = config.N
    n = config.n
    eta = np.zeros(n*n)
    num_iterations = config.art_max_iter
    lambda_relax = config.lambda_ART
    tol = config.art_tolerance

    # ART iterations
    iter = 0
    for k in range(num_iterations):
        eta_old = eta
        for i in range(num_projections):
            h_i = H[i, :]
            h_i_norm_sq = np.dot(h_i, h_i)
            if h_i_norm_sq == 0:
                continue
            r_i = p[i] - np.dot(h_i, eta)
            add = lambda_relax * (r_i / h_i_norm_sq) * h_i
            eta = eta + add
        eta_new = eta
        conv = np.linalg.norm(eta_new-eta_old)
        print("Current Conv " + str(conv) + " Iter no " + str(k))
        iter = k
        if conv < tol:
            print("Finished in " + str(k) + " iterations")
            break

    if iter == num_iterations - 1:
        print("Did not converge in " + str(iter) + " iterations.")
    
        
    # Reshape and plot
    reconstructed_image = eta.reshape((n, n))

    
  
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title('Reconstructed Image')
    plt.colorbar(label="Relative Electron Density")
    plt.tight_layout()
    plt.show()
