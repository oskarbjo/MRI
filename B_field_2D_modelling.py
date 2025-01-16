import numpy as np
import matplotlib.pyplot as plt

def symlog(x):
    """ Returns the symmetric log10 value """
    return np.sign(x) * np.log10(np.abs(x))

Npts = 401
x = np.linspace(-5,5,Npts)
y = np.linspace(-5,5,Npts)
X, Y = np.meshgrid(x,y)
XY = np.vstack([X.ravel(), Y.ravel()])

mag = np.asarray([0,1]) #Point dipole direction and length

#If object is circular, choose its radius:
radius = 2

#If object is quadratic, choose its side length:
L = 2

dipole_scale = 1000

net_field = np.zeros(np.shape(XY))

#Loop through all points on map, add local field contribution to total
for i,m in enumerate(x):
    for j,n in enumerate(y):
        r_src = np.sqrt(n**2 + m**2)
        # if r_src <= radius: #Geometric condition: Circle
        # if np.abs(m) < L/2 and np.abs(n) < L/2: #Geometric condition: Square
        # if np.abs(m)==2 and n == 0: #Geometric condition: 2 Points
        # if (np.abs(m)==2 and np.abs(n) == 2) or (np.abs(m)==2 and np.abs(n) == 0) or (np.abs(m)==0 and np.abs(n) == 2) or (np.abs(m)==0 and np.abs(n) == 0): #Geometric condition: 4 Points
        if (np.abs(m)==0 and np.abs(n) == 0): #Geometric condition: 4 Points
            local_dipole_vector = np.asarray([m,n])
            r = XY - local_dipole_vector[:,None]
            r_abs = np.sqrt(r[0,:]**2 + r[1,:]**2)
            r_abs[np.where(r_abs < 0.001)] = np.nan #Avoid dividing with very small numbers
            local_field_contribution_map = dipole_scale*(3*(XY-local_dipole_vector[:,None])*(mag @ (XY-local_dipole_vector[:,None]))/(r_abs**5) - mag[:,None]/(r_abs**3))
            net_field = np.nansum((net_field,local_field_contribution_map),0)

net_field = np.reshape(net_field,[2,Npts,Npts])
net_field_abs = np.sqrt(net_field[0,:,:]**2 + net_field[1,:,:]**2)


plt.figure()
plt.imshow(net_field_abs,norm='log')

plt.figure()
plt.quiver(X, Y, symlog(net_field[0,:,:]), symlog(net_field[1,:,:]),scale=800)