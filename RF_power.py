
from matplotlib import pyplot as plt
import numpy as np
from scipy import integrate


K = 50e-6
Df = 1e3
f = 127.74e6


t0 = 1e-12
# t0 = -2.5e-3
t1 = 0.5e-3 #duration/2
TR = 4e-3   #true TR
dutyCycle = (t1-t0)/(TR/2)
t = np.arange(t0,t1,1e-10)

B1 = K * np.sin(Df * np.pi * t) / (Df * np.pi * t) * np.sin(2 * np.pi * f * t)
B1prime = (K/(Df * np.pi * t)) * (Df * np.pi * np.cos(Df* np.pi * t)*np.sin(2 * np.pi * f * t) + 2 * np.pi * f * np.sin(Df * np.pi * t) * np.cos(2 * np.pi * f * t)) - K/(Df*np.pi*np.power(t,2)) * np.sin(2 * np.pi * f * t) * np.sin(Df * np.pi * t)

area = 0.025*0.025 #sqm
N = 1
R = 2*N
P = lambda x: np.power(N * area * ((K/(Df * np.pi * x)) * (Df * np.pi * np.cos(Df* np.pi * x)*np.sin(2 * np.pi * f * x) + 2 * np.pi * f * np.sin(Df * np.pi * x) * np.cos(2 * np.pi * f * x)) - K/(Df*np.pi*np.power(x,2)) * np.sin(2 * np.pi * f * x) * np.sin(Df * np.pi * x)),2)/R
integral = integrate.quad(P,t0,t1)
plt.figure()
plt.plot(t,B1)

plt.figure()
plt.plot(t,P(t))
plt.xlabel('Time [s]')
plt.ylabel('Power [W]')
print('Average input power: ' + str(2*dutyCycle*integral[0]))


plt.show()
