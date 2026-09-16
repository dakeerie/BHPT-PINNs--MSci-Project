import numpy as np 
from scipy.special import lambertw
from scipy.optimize import minimize_scalar

j = complex(0, 1)

def r_to_rstar(r, M):
    r = np.asarray(r)
    if np.any(r <= 2*M):
        raise ValueError("All r must be > 2M")
    return r + 2*M*np.log(r/(2*M) - 1)

def rstar_to_r(rs, M):
    rs = np.array(rs)
    return (2*M*(1 + lambertw(np.exp(rs/(2*M) - 1)))).real

def r_to_x(r, M):
    return 1 - 2*M/r

def x_to_r(x, M):
    return 2*M/(1-x)

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

def Vpm_peak(M, l: int, parity, coord):

    if parity not in ('axial', 'polar'):
        raise ValueError("parity needs to be either: axial or polar")
    if coord not in ('r', 'x'):
        raise ValueError("coord needs to be either: r or x")
    
    if parity == 'axial':
        coeffs = np.array([l*(l + 1), -3*M*(3 + l*(l + 1)), 24*M**2])
        peak = np.max(np.roots(coeffs))
    else:
        res = minimize_scalar(
            lambda r: -Vpm(r, M, l, "polar"), 
            bounds=(2.1*M, 5*M), 
            method='bounded'
        )
        peak = res.x

    if coord == 'x':
        peak = r_to_x(peak, M)
    
    return peak

def gaussian_wave(x, t, x0, sigma, omega):
    spatial = np.exp(-(x-x0)**2/(2*sigma**2))
    temporal = np.exp(-j*omega*(x-t))
    normal = 1/np.sqrt(2*np.pi*sigma**2)
    y = normal*spatial*temporal
    dydt = j*omega*y
    return y, dydt