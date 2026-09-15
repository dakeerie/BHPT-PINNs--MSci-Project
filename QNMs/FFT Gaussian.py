import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift, ifft, ifftshift

def packet(E0,t,w0,T):
    return E0*np.cos(w0*t)*np.exp(-t**2/T**2)

def analytical_FT(E0,w,w0,T):
    return (E0*T/(2.0*np.sqrt(2.0)))*(np.exp(-(w - w0)**2*T**2/4.0) + np.exp(-(w + w0)**2*T**2/4.0))

E0 = 1.0
w0 = 1.0
T = 10.0

 ###############################################################################
 # time axis
n = 800
dt = 300/n
t = np.arange(-n/2, n/2) * dt

 # angular frequency axis
dw = 2.0*np.pi/(n*dt)
print(dw)
w = np.arange(-n/2, n/2)*dw
print(w)

signal = packet(E0,t,w0,T) 
transform = fftshift(fft(signal))*dt/np.sqrt(2.0*np.pi)

plt.figure()

# time series
plt.subplot(121)
plt.plot(t, packet(E0,t,w0,T))
plt.xlabel(r"$t$") 
plt.ylabel(r"$E(t)$")
plt.xlim(-30,30)

# spectrum
plt.subplot(122)
plt.plot(w, analytical_FT(E0,w,w0,T)**2,'-',label='Analytical')
plt.plot(w, np.abs(transform)**2, 'ro', label = 'Numerical')
plt.xlim(0,2)
plt.xlabel(r"$\omega$") 
plt.ylabel(r"$|E(\omega)|^2$")

plt.tight_layout()
plt.show()