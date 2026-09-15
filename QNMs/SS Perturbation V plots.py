import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": "" # Clear the helvet and sansmath packages
})

def axial(r, M, l):
    V_axial = (1-2*M/r)*(l*(l+1)/r**2 - 6*M/r**3)
    return V_axial

def polar(r, M, l):
    n = 1/2*(l-1)*(l+2)
    mode = n**2*(n+1)*r**3 + 3*M*n**2*r**2 + 9*M**2*n*r + 9*M**3
    coeff = 2*(1-2*M/r)/(r**3*(n*r + 3*M)**2)
    V_polar = coeff*mode
    return V_polar

l_array = [2, 3, 4, 5, 6]
M = 0.5
eps = 1e-6
r = np.linspace(2*M + eps, 20*M, 3000)

def r_to_rstar(r, M):
    r = np.asarray(r)
    if np.any(r <= 2*M):
        raise ValueError("All r must be > 2M")
    return r + 2*M*np.log(r/(2*M) - 1)

r_star = r_to_rstar(r, M)

# plt.plot(r/M, r_star/M)
# plt.plot(r/M, r/M)
# plt.show()

plt.figure(figsize = [8, 6])
for i in range(len(l_array)):
    lbl_p = 'Polar' if i == 0 else None
    lbl_a = 'Axial' if i == 0 else None
    polar_V = polar(r, M, l_array[i])
    axial_V = axial(r, M, l_array[i])
    plt.plot(r_star/(2*M), 4*M**2*polar_V, color = 'black', linewidth = 1.5, linestyle = '--', zorder = 3, label = lbl_p)
    plt.plot(r_star/(2*M), 4*M**2*axial_V, color = 'red', linewidth = 1.5, label = lbl_a)
plt.xlabel(r'$r_*/2M$', fontsize = 28)
plt.ylabel(r'$4M^2V^{\pm}$', fontsize = 28)
plt.title(r'Effective Potentials', fontsize = 28)
plt.xlim(-5, 10)
plt.grid()
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.legend(fontsize = 25, loc = 'center right')
plt.tight_layout()
plt.savefig('potentialplot.png', format = 'png')
plt.close()

