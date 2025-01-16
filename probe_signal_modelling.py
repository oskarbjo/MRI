import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter, freqz
# %% Init

#Function definitions:
def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

#Parameters:
lambda0 = 42.58e6 #Gyromagnetic ratio, Hz/Tesla
t2 = 0.1e-3 #seconds
droplet_diam = 1e-3 #meters
Npts = 20
field_strength = 3 #Tesla
gradient_strength = 0.1 #T/m
dt = 1e-9
sig_dur = 0.2e-3
t = np.arange(0,sig_dur,dt)
IF = 127.8e6

#Create a 2D grid and mask for the signal source droplet
x=np.linspace(-droplet_diam, droplet_diam,Npts)
y=np.linspace(-droplet_diam, droplet_diam,Npts)
meshgrid = np.meshgrid(x,y)
droplet_mask = np.sqrt(meshgrid[0]**2 + meshgrid[1]**2) < droplet_diam/2 #Circular mask
# droplet_mask = np.logical_and(np.abs(meshgrid[0]) < droplet_diam/2, np.abs(meshgrid[1]) < droplet_diam/2) #Square mask




# %% Signal from homogeneous field

#Map a field strength to all points:
field_map = np.ones(np.shape(meshgrid[0]))*field_strength
#Create empty array which should contain all individual signals for each point source
signal_map = np.zeros([int(sig_dur/dt),len(x),len(y)]) #Each pixel corresponds to one time signal or free induction decay (FID)

#Generic signal model: sin(omega*time) * exp(-time/t2), where omega = 2*pi*freq = 2*pi*lambda0*field(x,y)
#Calculate each individual signal contribution:
for i,m in enumerate(x):
    for j,n in enumerate(y):
        omega_local = 2*np.pi*lambda0*field_map[i,j]
        signal_map[:,i,j] = np.cos(omega_local*t) * np.exp(-t/t2)
        
#Our received signal is the sum of all individual spins:
RF_signal = np.sum(np.sum(signal_map,1),1)

plt.figure()
plt.plot(RF_signal)

#We demodulate in quadrature to get our complex-valued signal (I = real part, Q = imag part)
downconverted_signal = [RF_signal * np.sin(2*np.pi*IF*t), RF_signal * np.cos(2*np.pi*IF*t)]

# This signal should be low pass filtered:
I = butter_lowpass_filter(downconverted_signal[0], 1e6, 1/dt)
Q = butter_lowpass_filter(downconverted_signal[1], 1e6, 1/dt)

#Plot time domain FIDs
plt.figure()
plt.plot(t,I)
plt.plot(t,Q)

#Plot I and Q in complex plane
plt.figure()
plt.plot(I,Q)



# %% Signal from gradient field

#Map a field strength to all points:
field_map = np.ones(np.shape(meshgrid[0]))*field_strength + meshgrid[0]*gradient_strength
#Create empty array which should contain all individual signals for each point source
signal_map = np.zeros([int(sig_dur/dt),len(x),len(y)]) #Each pixel corresponds to one time signal or free induction decay (FID)

#Generic signal model: sin(omega*time) * exp(-time/t2), where omega = 2*pi*freq = 2*pi*lambda0*field(x,y)
#Calculate each individual signal contribution:
for i,m in enumerate(x):
    for j,n in enumerate(y):
        omega_local = 2*np.pi*lambda0*field_map[i,j]
        signal_map[:,i,j] = np.cos(omega_local*t) * np.exp(-t/t2)
        
#Our received signal is the sum of all individual spins:
RF_signal = np.sum(np.sum(signal_map,1),1)

plt.figure()
plt.plot(RF_signal)

#We demodulate in quadrature to get our complex-valued signal (I = real part, Q = imag part)
downconverted_signal = [RF_signal * np.sin(2*np.pi*IF*t), RF_signal * np.cos(2*np.pi*IF*t)]

# This signal should be low pass filtered:
I = butter_lowpass_filter(downconverted_signal[0], 1e6, 1/dt)
Q = butter_lowpass_filter(downconverted_signal[1], 1e6, 1/dt)


plt.figure()
plt.plot(t,I)
plt.plot(t,Q)


plt.figure()
plt.plot(I,Q)


