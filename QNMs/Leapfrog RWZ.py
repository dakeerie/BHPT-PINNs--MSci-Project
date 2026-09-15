import numpy as np
from scipy.special import lambertw
import matplotlib.pyplot as plt

j = complex(0, 1)

def r_to_rstar(r, M):
    r = np.array(r)
    if np.any(r <= 2*M):
        raise ValueError("All r must be > 2M")
    return r + 2*M*np.log(r/(2*M) - 1)

def rstar_to_r(rs, M):
    rs = np.array(rs)
    return (2*M*(1 + lambertw(np.exp(rs/(2*M) - 1)))).real

def Vpm(r, M, l, parity):
    if parity == 'axial':
        V = (1-2*M/r)*(l*(l+1)/r**2 - 6*M/r**3)
    elif parity == 'polar':
        n = 1/2*(l-1)*(l+2)
        num = 2*(1-2*M/r)*(
            9*M**3 +
            9*n*M**2*r + 
            3*n**2*M*r**2 + 
            n**2*(1 + n)*r**3
        )
        den = r**3*(3*M + n*r)**2
        V = num/den
    else:
        raise ValueError("parity needs to be either: axial or polar")
    return V

def FD_xderivative(Psi, dx):
    dPsi = np.zeros_like(Psi)
    dPsi[1:-1] = (Psi[2:] - Psi[:-2])/(2*dx)
    dPsi[0] = (Psi[1] - Psi[0])/dx
    dPsi[-1] = (Psi[-1] - Psi[-2])/dx
    return dPsi

#Leapfrog solver
#Parameters
M = 1
c = 1
l = 2

#Define grid
dx = 0.1
dt = 0.5*dx
rstar = np.arange(-400*M, (700+dx)*M, dx)
r = rstar_to_r(rstar, M)
Nr = len(rstar) #spatial points
Nt = 8000 #timesteps
print(Nr, Nt)

#Initial data
width = 10*M
rstar0 = 110*M
omega = 0.5
Psi0 = 1/(np.sqrt(2*np.pi*width**2))*np.exp(-(rstar - rstar0)**2/(2*width**2))*np.exp(-j*omega*rstar)
dPsi0 = (-j*omega - (rstar - rstar0)/width**2)*Psi0


# plt.figure()
# # plt.plot(rstar, dPsi0, label = 'Analytical')
# # plt.plot(rstar, FD_xderivative(Psi0, dx), label = 'Finite Difference')
# plt.plot(rstar, np.abs(dPsi0 - FD_xderivative(Psi0, dx)), label = 'Difference' )
# plt.legend()
# plt.grid()
# plt.show()


#Solver function
def leapfrog(rstar: np.ndarray, tsteps: int, mass: float, mode: int, parity: str, 
            y0: np.ndarray, dydt0: np.ndarray, dx: float, dt: float):
    
    if dt > dx:
        raise ValueError("CFL condition, dt <= dx must hold for stable solver")
    r = rstar_to_r(rstar, mass)
    V = Vpm(r, mass, mode, parity)

    psi_arr = np.zeros((len(rstar), tsteps), dtype = y0.dtype)
    psi_arr[:, 0] = y0
    psi_arr[1:-1, 1] = psi_arr[1:-1, 0] + dt*dydt0[1:-1]
    0.5*(dt/dx)**2*(psi_arr[2:, 0] + 
        psi_arr[:-2, 0] - 2*psi_arr[1:-1, 0]) - 0.5*V[1:-1]*psi_arr[1:-1, 0]*dt**2
    
    for q in range(1, tsteps - 1):
        psi_arr[1:-1, q + 1] = (2*psi_arr[1:-1, q] - psi_arr[1:-1, q - 1]
            + (dt/dx)**2*(psi_arr[2:, q] + psi_arr[:-2, q] - 2*psi_arr[1:-1, q])
            - V[1:-1]*psi_arr[1:-1, q]*dt**2
        )
        psi_arr[0, q + 1] = (1-dt/dx)*psi_arr[0, q] + dt/dx*psi_arr[1, q]
        psi_arr[-1, q + 1] = (1 - dt/dx)*psi_arr[-1, q] + dt/dx*psi_arr[-2, q]
    
    return psi_arr

#Solve
#Take dPsi/dr = dPsi/dt at t = 0 (ingoing advection equation)
Psi = leapfrog(rstar, Nt, M, l, 'axial', Psi0, dPsi0, dx, dt)
Psi = Psi.transpose()
r = rstar_to_r(rstar, M)
V_axial = Vpm(r, M, 2, 'axial')

plt.figure()
for t in range(Nt):
    if t % 1000 == 0:
        plt.plot(rstar, Psi[t, :], label = rf'$\Psi$ at t = {t}')
plt.plot(rstar, V_axial, label = 'Axial potential for l = 2')
plt.xlabel(r'$r_*$', fontsize = 20)
plt.ylabel(r'$\Psi$ & $V$', fontsize = 20)
plt.legend()
plt.grid(True)
plt.show()

i_obs = np.argmin(np.abs(rstar - 200*M))
print(i_obs)


extraction = Psi[:, i_obs]
times = np.linspace(0, Nt, Nt)
print(times.shape, extraction.shape)

plt.figure()
plt.plot(times, extraction, '.-', label = r'Time series at $r = 200 M$')
# plt.plot(rstar, Psi[2000,:], label = r'Spatial pulse at $t = 2000$')
plt.xlabel('Time', fontsize = 20)
plt.ylabel(r'$\Psi$', fontsize = 20)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()