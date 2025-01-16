
from matplotlib import pyplot as plt
import numpy as np
from scipy import integrate


K = 10e-6 #Peak field strength
Df = 1e3
f = 127.74e6


t0 = 1e-13
# t0 = -2.5e-3
t1 = 2.5e-3 #RF pulse duration/2
TR = 100e-3   #TR
dutyCycle = (t1-t0)/(TR/2)
t = np.arange(t0,t1,1e-10)

eta = 3.767300000000000e+02
mu = 1.256600000000000e-06
B1 = K * np.sin(Df * np.pi * t) / (Df * np.pi * t) * np.sin(2 * np.pi * f * t)
Poynting_plot =  eta * np.square(K * np.sin(Df * np.pi * t) / (Df * np.pi * t) * np.sin(2 * np.pi * f * t) / mu)
Poynting = lambda tt: eta * np.square(K * np.sin(Df * np.pi * tt) / (Df * np.pi * tt) * np.sin(2 * np.pi * f * tt) / mu)
Poynting_integral = integrate.quad(Poynting,t0,t1)
B1prime = (K/(Df * np.pi * t)) * (Df * np.pi * np.cos(Df* np.pi * t)*np.sin(2 * np.pi * f * t) + 2 * np.pi * f * np.sin(Df * np.pi * t) * np.cos(2 * np.pi * f * t)) - K/(Df*np.pi*np.power(t,2)) * np.sin(2 * np.pi * f * t) * np.sin(Df * np.pi * t)

area = 0.025*0.025 #sqm
N = 1
R = 2*N #Restistance ~1 Ohm
P = lambda x: np.power(N * area * ((K/(Df * np.pi * x)) * (Df * np.pi * np.cos(Df* np.pi * x)*np.sin(2 * np.pi * f * x) + 2 * np.pi * f * np.sin(Df * np.pi * x) * np.cos(2 * np.pi * f * x)) - K/(Df*np.pi*np.power(x,2)) * np.sin(2 * np.pi * f * x) * np.sin(Df * np.pi * x)),2)/R
integral = integrate.quad(P,t0,t1)
plt.figure()
plt.plot(t,B1)

plt.figure()
plt.plot(t,Poynting_plot)

plt.figure()
plt.plot(t,P(t))
plt.xlabel('Time [s]')
plt.ylabel('Power [W]')
print('Dutycycle: ' + str(dutyCycle))
print('Peak power density: ' + str(np.max(np.abs(Poynting_plot))))
print('Average field power density: ' + str(2*dutyCycle*Poynting_integral[0]/(t1-t0))) #divide by time elapsed to go from Nm to W
print('Average input power: ' + str(2*dutyCycle*integral[0]))


plt.show()
