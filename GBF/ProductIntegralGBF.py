import numpy as np
from scipy.special import lambertw
import matplotlib.pyplot as plt

def u_from_ustar(ustar):
    """Calculates dimensionless u = r/2M from dimensionless tortoise coordinate u_* = r_*/2M"""
    arg = np.exp(ustar - 1.0)
    w = np.real(lambertw(arg))
    return 1.0 + w

def RW_potential(u, mode):
    #Dimensionless potential- see appendix B of Gray and Visser paper 
    return (1.0 - 1.0/u)*(mode*(mode + 1)/u**2 -3/u**3)

def calculate_GBF(x, mode, ustar_min = -50.0, ustar_max = 350.0, N = 10000):
    """Calculates the GBF for dimensionless frequency x = M*omega using the path-ordered exponential/product integral approach.
        Parameters: 
            x: float, dimensionless frequency x = M*omega
            mode: int, multipole mode l (>=2 here)
            ustar_min: float, lower boundary approximating -infinity (event horizon)
            ustar_max: float, upper boundary approximating +infinity (spatial infinity)
            N: int, number of integration grid points

        Returns:
            T: float, transmission probability i.e. GBF
            alpha: complex, Bogoliubov coefficient alpha(x)
            beta: complex, Bogoliubov coefficient beta(x)
    """
    ustar_grid = np.linspace(ustar_min, ustar_max, N)
    h = (ustar_max - ustar_min)/(N - 1)

    u_grid = u_from_ustar(ustar_grid)
    V_grid = RW_potential(u_grid, mode)

    #Pre-calculate matrix factors
    #A(u_*) = -i/(4x)*V(u_*)*[[1, exp(-4ixu_*)], [-exp(4ixu_*), -1]]
    coefficient = -1j*V_grid/(4.0*x)
    phase = np.exp(4j*x*ustar_grid)

    #Matrix elements for M_k = I + A(u*_k)*h
    M00 = 1.0 + coefficient*h
    M01 = coefficient*h*np.conj(phase)
    M10 = -coefficient*h*phase
    M11 = 1.0 - coefficient*h

    #Evaluate product integral: E = M_{N-1} @ M_{N - 2} @ ... @ M_0
    E = np.identity(2, dtype = complex)
    for k in range(N):
        Mk = np.array([[M00[k], M01[k]], [M10[k], M11[k]]])
        E = Mk @ E

    #Extract Bogoliubov coefficients
    alpha = E[0, 0]
    beta = E[1, 0]

    T = 1.0/(np.abs(alpha)**2)
    return T, alpha, beta


if __name__ =="__main__":
    l = 2

    print(f"Calculating Greybody Factors for spin-2, mode = {l}...")
    print("-"*60)
    print(f"{r'x = M omega':^12} | {r'T_{2, 2}(M omega)':^15} | {'|alpha|^2 - |beta|^2':^20}")
    print("-"*60)

    x_values = np.linspace(0.05, 0.8, 200)
    T_list = []

    for x in x_values:
        T, alpha, beta = calculate_GBF(x, mode = l, ustar_max = 1000, N = 10000)
        T_list.append(T)
        unitarity = np.abs(alpha)**2 - np.abs(beta)**2
        print(f"{x:^12.3f} | {T:^15.6f} | {unitarity:^20.8f}")

    plt.figure(figsize = (8, 5))
    plt.plot(x_values/0.5, T_list, label = 'l = 2, s = 2')
    plt.xlabel(r'$\omega$', fontsize = 16)
    plt.ylabel(r'$\Gamma_{2, 2}(\omega)$', fontsize = 16)
    plt.title('Grey-body Factor'
              '\n'
              r'$M = 0.5$, $s = 2$, $l = 2$', fontsize = 16)
    plt.grid()
    plt.legend()
    plt.show()