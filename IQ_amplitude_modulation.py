
import numpy as np
from matplotlib import pyplot as plt

def generateRFSignal(f0,BW,Nsignals,t):
    RFsignal = np.zeros(len(t))
    for i in range(0,Nsignals):
        f = f0 - BW/2 + BW/2 * i / Nsignals
        RFsignal = RFsignal + np.sin(2*np.pi*f*t)
    RFsignal = RFsignal * np.exp(-t/(0.3*np.max(t)))
    return RFsignal


f0 = 10e6   #LO freq
IF = 2e6    #Intermediate freq
f1 = f0+IF  #Larmor freq
dt = 5e-9
t = np.arange(0,10000e-9,dt)
RFsignal = generateRFSignal(f1, 1000e3, 100,t)
LO_I = np.sin(2*np.pi*f0*t)
LO_Q = np.sin(2*np.pi*f0*t + np.pi/2)
I = np.multiply(RFsignal,LO_I)
Q = np.multiply(RFsignal,LO_Q)

f = np.linspace(-0.5/dt,0.5/dt,len(RFsignal))
RFsignal_fft = np.roll(np.fft.fft(RFsignal),int(len(np.fft.fft(RFsignal))/2))
LO_I_fft = np.roll(np.fft.fft(LO_I),int(len(np.fft.fft(LO_I))/2))
LO_Q_fft = np.roll(np.fft.fft(LO_Q),int(len(np.fft.fft(LO_Q))/2))
I_fft = np.roll(np.fft.fft(I),int(len(np.fft.fft(I))/2))
Q_fft = np.roll(np.fft.fft(Q),int(len(np.fft.fft(Q))/2))

#Recreating RF:
r = np.multiply(I,np.sin(2*np.pi*f0*t)) + np.multiply(Q,np.sin(2*np.pi*f0*t + np.pi/2))
LPF = (np.abs(f) >= 0.5/dt - IF).astype(int)
# LPF = np.ones(np.size(Q_fft))
Q_fft_LPF = np.roll(np.fft.fft(Q) * LPF, 0)
Q_LPF = np.fft.ifft(Q_fft_LPF)

fig, axs = plt.subplots(4)
fig.suptitle('Vertically stacked subplots')
axs[0].plot(t, RFsignal)
axs[0].plot(t, r)
axs[1].plot(t, LO_Q)
axs[2].plot(t, Q)
axs[3].plot(t,Q_LPF)

fig, axs = plt.subplots(4)
fig.suptitle('Vertically stacked subplots')
axs[0].plot(f,np.abs(RFsignal_fft))
axs[1].plot(f,np.abs(LO_Q_fft))
axs[2].plot(f,np.abs(Q_fft))
axs[3].plot(f,np.abs(Q_fft_LPF))

plt.show()


