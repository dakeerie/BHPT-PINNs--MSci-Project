
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp
from Functions import *
import csv

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"],
#     "text.latex.preamble": "" # Clear the helvet and sansmath packages
# })

# modes = np.arange(2, 5)
# omega = np.linspace(0.01, 2.0, 30)

modes = [2]
omega = [0.3]
mass = 0.5

def taylor_coeffs(mass, omega, mode):
    Lambda = mode*(mode + 1)
    Omega = 4*mass*omega
    c1 = (Lambda - 3)/(1 - 1j*Omega)
    c2 = ((Lambda + 1)*c1 + 3)/(4 - 1j*2*Omega)
    return c1.real, c1.imag, c2.real, c2.imag

def system(x, Y, M, l, om):

    u_re, u_im, du_re, du_im = Y

    u = u_re + 1j*u_im
    du = du_re + 1j*du_im

    A = x*(1 - x)**2
    B = (1 - x)*(1 - 3*x) - 4*M*1j*om
    C = -(l*(l + 1) - 3*(1 - x))

    d2u = -(B*du + C*u)/A

    return np.array([du.real, du.imag, d2u.real, d2u.imag])

# def robin_BC_h(u0, M, l, om):

#     B0 = 1 - 4*M*1j*om
#     C0 = -(l*(l + 1) - 3)

#     du0 = -C0*u0/B0
#     return u0, du0

eps = 1e-6
# u0 = complex(1, 0)

print("Initialising solver...")

solutions = {}
GBFs = {}
probs = {}
rows = []

for mode in modes:

    solutions[mode] = {}
    GBFs[mode], probs[mode] = [], []

    for om in omega:
        c1_re, c1_im, c2_re, c2_im = taylor_coeffs(mass, om, mode)
        u_BC = 1 + complex(c1_re, c1_im)*eps + complex(c2_re, c2_im)*eps**2
        du_BC = complex(c1_re, c1_im) + 2*complex(c2_re, c2_im)*eps
        initial_state = [u_BC.real, u_BC.imag, du_BC.real, du_BC.imag]

        print(f"Solving for  l = {int(mode)} with omega = {om:.3f} ...") 
        sol = solve_ivp(system, 
                t_span = (eps, 1.0 - 1e-4),
                y0 = initial_state,
                method = 'DOP853',
                args = (mass, mode, om),
                rtol = 1e-9,
                atol = 1e-11,
                dense_output = True
                )

        if (not sol.success) or np.any(~np.isfinite(sol.y)):
            print(f"*** FAILED: l = {mode}, omega = {om:.4f} - status = {sol.status}: {sol.message}")
            GBF = prob = np.nan
            alpha = beta = complex(np.nan, np.nan)

        else:
            x_vals = sol.t
            u_sol = sol.y[0] + 1j*sol.y[1]
            du_sol = sol.y[2] + 1j*sol.y[3]

            solutions[mode][om] = (x_vals, u_sol, du_sol)

            x_end = x_vals[-1]
            y = 1.0 - x_end
            u_end = u_sol[-1]
            du_end = du_sol[-1]

            L = mode*(mode + 1)
            Omega = 4*mass*om

            a1 = -1j*L/Omega
            a2 = (-L*(L - 2) +1j*3*Omega)/(2*Omega**2)

            u1 = 1.0 + a1*y + a2*y**2
            du1_dx = -(a1 + 2.0*a2*y)

            rstar_end = 2*mass/(1 - x_end) + 2*mass*np.log(x_end/(1 - x_end))

            D =1j*Omega/(y**2*x_end) + np.conj(du1_dx/u1)
            alpha = (u_end - du_end/D)/(u1 - du1_dx/D)
            beta = (u_end - alpha*u1)/(np.exp(2j*om*rstar_end)*np.conj(u1))
            prob = np.abs(alpha)**2 - np.abs(beta)**2
            GBF = 1.0/(np.abs(alpha)**2)
            GBFs[mode].append(GBF)
            probs[mode].append(prob)
            rows.append({'l': int(mode), 'omega': om, 'GBF': GBF, 'log10GBF': np.log10(GBF), 'prob': prob,
                    'alpha_re': alpha.real, 'alpha_im': alpha.imag, 'beta_re': beta.real, 'beta_im': beta.imag,
                    'x_end': x_end, 'r_end': 2*mass/(1 - x_end), 'rtol': 1e-7, 'success': bool(sol.success)})

    print(f"Finished solving for l = {int(mode)}.")
    GBFs[mode] = np.array(GBFs[mode])
    probs[mode] = np.array(probs[mode])

fields = list(rows[0].keys())
with open('numerical_gbf.csv', 'w', newline = '') as f:
    w = csv.DictWriter(f, fieldnames = fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: (f'{v:.12e}' if isinstance(v, float) else v) for k, v in r.items()})
print(f'Wrote {len(rows)} rows to numerical_gbf.csv')

print("Solver successful. Plotting results...")

omega_last = omega[-1]
mode_last = modes[-1]
x_last, u_last, du_last = solutions[mode_last][omega_last]

plt.figure(figsize = [6,4])
plt.plot(x_last, u_last.real, label = 'Re(u)')
plt.plot(x_last, u_last.imag, label = 'Im(u)')
plt.plot(x_last, x_last*(1 - x_last)**2*(mode_last*(mode_last + 1) - 3*(1 - x_last))/(4*mass**2), label = 'V(x)', linestyle = '--')
plt.xlabel('x', fontsize = 18)
plt.ylabel('u(x)', fontsize = 18)
plt.legend(loc = 'best')
plt.title(f'Wavefunction u(x) for l = {mode_last} with omega = {omega_last}')
plt.grid()
plt.tight_layout()
plt.savefig('NumericalGBFWavefunction.png', format = 'png')

plt.figure(figsize = [6,4])
for l in modes:
    plt.plot(omega, GBFs[l], 'o-', label = f"$l = {int(l)}$", markersize = 4)
plt.xlabel(r'$\omega$', fontsize = 18)
plt.ylabel(r'$\Gamma(\omega)$', fontsize = 18)
plt.title('Greybody Factor vs Frequency', fontsize = 20)
plt.legend(loc = 'lower right')
# plt.ylim(-0.05, 1.05)
plt.tight_layout()
plt.grid()
plt.savefig('NumericalGBF.png', format = 'png')
plt.close()

plt.figure(figsize = [6,4])
for l in modes:
    plt.plot(omega, GBFs[l], 'o-', label = f"$l = {int(l)}$", markersize = 4)
plt.xlabel(r'$\omega$', fontsize = 18)
plt.ylabel(r'$\Gamma(\omega)$', fontsize = 18)
plt.title('Greybody Factor vs Frequency', fontsize = 20)
plt.legend(loc = 'lower right')
plt.yscale('log')
plt.tight_layout()
plt.grid()
plt.savefig('NumericalGBFlog.png', format = 'png')
plt.close()