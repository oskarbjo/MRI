import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

plt.close('all')


# Grid parameters
nx, ny = 50, 50  # Number of grid points in x and y directions
Lx, Ly = 1.0, 1.0  # Physical size of the domain
dx = Lx / (nx - 1)  # Grid spacing in x
dy = Ly / (ny - 1)  # Grid spacing in y
x = np.linspace(-Lx/2,Lx/2,nx)
y = np.linspace(-Ly/2,Ly/2,ny)


# Magnetic permeability (example: uniform or spatially varying)
mu0 = 1.2566e-6
mu = np.ones((nx, ny))  # Uniform permeability (can modify to vary spatially)


# Adding a region of higher permeability: 
#Square:
mu[nx//3 : 2*nx//3, ny//3 : 2*ny//3] = 5.0 
# Circle:
# radius=0.1
# for i in range(0, len(x)):
#     for j in range(0, len(x)):
#         if(np.sqrt(x[i]**2 + y[j]**2) <= radius):
#             mu[i,j] = 4.0
#Point:
# mu[round(nx/2),round(ny/2)] = 5
#Full volume:
# mu[:,:]=5

# Choose background field strength
background_field = 3 #Teslas

# Finite difference loop parameters
max_iter = 500  # Maximum number of iterations
tolerance = 1  # Convergence tolerance

# Set initial scalar potential based on the background field: Assume field is parallel to y direction
Phi = -background_field * np.outer(y, np.ones(nx)) / (mu0 * mu)

# Iterative solver 
for iteration in range(max_iter):
    Phi_new = np.copy(Phi)
    
    # Loop over interior points (exclude boundaries)
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            
            
            # Compute finite differences
            dPhi_dx2 = (Phi[i+1, j] - 2*Phi[i, j] + Phi[i-1, j]) / dx**2
            dPhi_dy2 = (Phi[i, j+1] - 2*Phi[i, j] + Phi[i, j-1]) / dy**2
            
            dMu_dx = (mu[i+1, j] - mu[i-1, j]) / (2 * dx)
            dMu_dy = (mu[i, j+1] - mu[i, j-1]) / (2 * dy)
            
            dPhi_dx = (Phi[i+1, j] - Phi[i-1, j]) / (2 * dx)
            dPhi_dy = (Phi[i, j+1] - Phi[i, j-1]) / (2 * dy)
            
            # Discretized equation
            # Phi_new[i, j] = ( 1/mu[i,j] * (dMu_dx * dPhi_dx + dMu_dy*dPhi_dy)
            #                  + 2*dPhi_dx/dx + 2*dPhi_dy/dy) * 1 / (2 * (1/dx**2 + 1/dy**2))
            Phi_new[i, j] = (
                mu[i, j] * (
                    (Phi[i+1, j] + Phi[i-1, j]) / dx**2
                    + (Phi[i, j+1] + Phi[i, j-1]) / dy**2
                )
                + ((mu[i+1, j] - mu[i-1, j]) * (Phi[i+1, j] - Phi[i-1, j])) / (4 * dx**2)
                + ((mu[i, j+1] - mu[i, j-1]) * (Phi[i, j+1] - Phi[i, j-1])) / (4 * dy**2)
            ) / (
                mu[i, j] * (2 / dx**2 + 2 / dy**2)
            )
            
            #### MATERIAL BOUNDARY CONDITIONS:
            
            # Handle interfaces in the x-direction
            if mu[i, j] != mu[i-1, j]: #Interface to the left
                Phi_new[i, j] = (
                    mu[i-1, j] * Phi_new[i-1, j] + mu[i, j] * Phi_new[i+1, j]
                ) / (mu[i-1, j] + mu[i, j])
            if mu[i, j] != mu[i+1, j]: #Interface to the right
                Phi_new[i, j] = (
                    mu[i+1  , j] * Phi_new[i+1, j] + mu[i, j] * Phi_new[i-1, j]
                ) / (mu[i+1, j] + mu[i, j])
     
            # Handle interfaces in the y-direction
            if mu[i, j] != mu[i, j-1]: #Interface below
                Phi_new[i, j] = (
                    mu[i, j-1] * Phi_new[i, j-1] + mu[i, j] * Phi_new[i, j+1]
                ) / (mu[i, j-1] + mu[i, j])
            if mu[i, j] != mu[i, j+1]: #Interface above
                Phi_new[i, j] = (
                    mu[i, j+1] * Phi_new[i, j+1] + mu[i, j] * Phi_new[i, j-1]
                ) / (mu[i, j+1] + mu[i, j])

    
            # #### END BOUNDARY CONDITIONS
            
            
    #Calculate effective B-field:
    Hy, Hx = np.gradient(-Phi, dx, dy)
    Bx = mu * mu0 * Hx
    By = mu * mu0 * Hy
    
    # Check for convergence
    error = np.max(np.abs(Phi_new - Phi))
    print(error)
    print(iteration)
    if error < tolerance:
        print(f"Converged after {iteration} iterations.")
        break
    
    Phi = Phi_new

#Calculate effective B-field:
Hy, Hx = np.gradient(-Phi, dx, dy)
Bx = mu * mu0 * Hx
By = mu * mu0 * Hy

# Optional: Visualization

plt.figure()
plt.imshow(Phi, extent=[0, Lx, 0, Ly], origin="lower", cmap="viridis")
plt.colorbar(label="Scalar Potential (Phi)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Magnetic Scalar Potential")


plt.figure()
plt.imshow(np.sqrt(Bx**2 + By**2),norm=LogNorm(vmin=1.0, vmax=2.7))

plt.figure()
plt.imshow(Hy,norm=LogNorm())
# plt.imshow(Bx,norm=LogNorm(vmin=1.0, vmax=2.7))


plt.show()


