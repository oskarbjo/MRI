
import numpy as np
from matplotlib import pyplot as plt


# rawdata=np.asarray(np.loadtxt(r"C:\Users\Oskar\Downloads\mrimage1d.txt"))
rawdata2 = np.loadtxt(r"C:\Users\Oskar\Downloads\mrimage2d.txt", dtype=complex)
# d=np.asarray([rawdata[:,0]+rawdata[:,1]*1j])
# Kspace = d.reshape(256,256)
XYspace = np.roll(np.roll(np.fft.ifft2(rawdata2),127,axis=0),127,axis=1)

    

plt.figure()
plt.imshow(np.abs(rawdata2))


plt.figure()
plt.imshow(np.abs(XYspace))

plt.show()


print(' ')