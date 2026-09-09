
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp
from Functions import *
import csv
import pickle

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"],
#     "text.latex.preamble": "" # Clear the helvet and sansmath packages
# })

# modes = np.arange(2, 5)
# omega = np.linspace(0.01, 2.0, 30)

modes = [2]
omega = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
mass = 0.5

RTOL = 1e-9
ATOL = 1e-11

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

def extraction(sol, x_extract, mass, mode, omega):

    if x_extract <= sol.t[0] or x_extract > sol.t[-1]:
        raise ValueError(f"x_extract = {x_extract} is outside the available solution range: [{sol.t[0]}, {sol.t[-1]}]")

    Y = sol.sol(x_extract)
    u_end = Y[0] + 1j*Y[1]
    du_end = Y[2] + 1j*Y[3]

    y = 1.0 - x_extract
    L = mode*(mode + 1)
    Omega = 4*mass*omega

    a1 = -1j*L/Omega
    a2 = (-L*(L - 2) + 1j*3*Omega)/(2*Omega**2)

    u1 = 1.0 + a1*y + a2*y**2
    du1_dx = -(a1 + 2*a2*y)

    rstar = 2*mass/(1 - x_extract) + 2*mass*np.log(x_extract/(1 - x_extract))

    D = 1j*Omega/(y**2*x_extract) + np.conj(du1_dx/u1)

    alpha = (u_end - du_end/D)/(u1 - du1_dx/D)
    beta = (u_end - alpha*u1)/(np.exp(1j*2*omega*rstar)*np.conj(u1))

    flux_check = np.abs(alpha)**2 - np.abs(beta)**2
    GBF = 1.0/np.abs(alpha)**2

    return GBF, alpha, beta, flux_check

print("Initialising solver...")

solutions = {}
results = {}
rows = []

for mode in modes:

    solutions[mode] = {}
    results[mode] = {}

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
                rtol = RTOL,
                atol = ATOL,
                dense_output = True
                )

        if (not sol.success) or np.any(~np.isfinite(sol.y)):
            print(f"*** FAILED: l = {mode}, omega = {om:.4f} - status = {sol.status}: {sol.message}")
            GBF = np.nan
            alpha = beta = complex(np.nan, np.nan)
            solutions[mode][om] = sol
            results[mode][om] = {"GBF": np.nan, "flux_check": np.nan, "alpha": complex(np.nan, np.nan), "beta": complex(np.nan, np.nan),  "success": False,
            }
            continue

        solutions[mode][om] = sol

        x_vals = sol.t
        u_sol = sol.y[0] + 1j*sol.y[1]
        du_sol = sol.y[2] + 1j*sol.y[3]

        x_end = sol.t[-1]
        GBF, alpha, beta, flux_check = extraction(sol, x_end, mass, mode, om)

        results[mode][om] = {'GBF': GBF, 'flux_check': flux_check,'alpha': alpha, 'beta': beta,
                    'x_end': x_end, 'r_end': 2*mass/(1 - x_end), 'method': "DOP853", "eps": eps, 'rtol': RTOL, 'atol': ATOL, 'success': bool(sol.success)}
        rows.append({'l': int(mode), 'omega': om, 'GBF': GBF, 'log10GBF': np.log10(GBF), 'flux_check': flux_check,
                    'alpha_re': alpha.real, 'alpha_im': alpha.imag, 'beta_re': beta.real, 'beta_im': beta.imag,
                    'x_end': x_end, 'r_end': 2*mass/(1 - x_end), 'method': "DOP853", "eps": eps, 'rtol': RTOL, 'atol': ATOL, 'success': bool(sol.success)})

    print(f"Finished solving for l = {int(mode)}.")

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

sol_last = solutions[mode_last][omega_last]
x_last = sol_last.t
u_last = sol_last.y[0] + 1j*sol_last.y[1]
du_last = sol_last.y[2] + 1j*sol_last.y[3]

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
plt.close()

plt.figure(figsize = [6,4])
for l in modes:
    GBF_values = np.array([results[l][om]["GBF"] for om in omega])
    plt.plot(omega, GBF_values, 'o-', label = f"$l = {int(l)}$", markersize = 4)
plt.xlabel(r'$\omega$', fontsize = 18)
plt.ylabel(r'$\Gamma(\omega)$', fontsize = 18)
plt.title('Greybody Factor vs Frequency', fontsize = 20)
plt.legend(loc = 'lower right')
# plt.ylim(-0.05, 1.05)
plt.grid()
plt.tight_layout()
plt.savefig('NumericalGBF.png', format = 'png')
plt.close()

plt.figure(figsize = [6,4])
for l in modes:
    GBF_values = np.array([results[l][om]["GBF"] for om in omega])
    plt.plot(omega, GBF_values, 'o-', label = f"$l = {int(l)}$", markersize = 4)
plt.xlabel(r'$\omega$', fontsize = 18)
plt.ylabel(r'$\Gamma(\omega)$', fontsize = 18)
plt.title('Greybody Factor vs Frequency', fontsize = 20)
plt.legend(loc = 'lower right')
plt.yscale('log')
plt.tight_layout()
plt.grid()
plt.savefig('NumericalGBFlog.png', format = 'png')
plt.close()

with open("solutions.pkl", "wb") as f:
    pickle.dump(solutions, f)

with open("results.pkl", "wb") as f:
    pickle.dump(results, f)

#For convergence test
convergence_l = 2
convergence_omega = 0.50

extraction_points = [0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999]

if convergence_l not in solutions:
    raise ValueError(f"l = {convergence_l} was not included in the original calculation. Available values: {list(solutions.keys())}")

if convergence_omega not in solutions[convergence_l]:
    raise ValueError(f"omega = {convergence_omega} was not included in the original calculation. Available values: {list(solutions[convergence_l].keys())}")

sol_test = solutions[convergence_l][convergence_omega]

print()
print("="*60)
print("Extraction point convergence test")
print("="*60)

print(f"l     = {convergence_l}")
print(f"omega = {convergence_omega}")
print()

print(
    f"{'x':>12}"
    f"{'r':>12}"
    f"{'GBF':>20}"
    f"{'GBF rel. diff.':>20}"
    f"{'Flux check':>20}"
    f"{'Flux abs. error':>20}"  
)
print("-"*104)

convergence_results = []
for x_extract in extraction_points:

    try:
        GBF, alpha, beta, flux_check = extraction(
            sol_test,
            x_extract,
            mass,
            convergence_l,
            convergence_omega
        )
        r_extract = 2*mass/(1 - x_extract)

        convergence_results.append({
            "x_extract": x_extract,
            "r_extract": r_extract,
            "GBF": GBF,
            "alpha": alpha,
            "beta": beta,
            "flux_check": flux_check
        })

    except Exception as e:
        print(
            f"{x_extract:12.7f}"
            f"{'FAILED':>20}"
            f"{'':>20}"
            f"{str(e)}"
        )

if len(convergence_results) == 0:
    raise RuntimeError("No successful extraction points were found.")

GBF_final = convergence_results[-1]["GBF"]

for result in convergence_results:

    GBF = result["GBF"]
    relative_difference = abs(GBF - GBF_final)/abs(GBF_final)
    result["relative_difference"] = relative_difference

    print(
        f"{result['x_extract']:12.7f}"
        f"{result['r_extract']:12.3f}"
        f"{GBF:20.12e}"
        f"{relative_difference:20.6e}"
        f"{result['flux_check']:20.12e}"
        f"{abs(result['flux_check'] - 1.0):20.6e}"
    )

print("-"*104)
print(
    f"Final GBF "
    f"(x = {convergence_results[-1]['x_extract']:.7f}): "
    f"{GBF_final:.12e}"
)

x_extract_values = np.array([result["x_extract"]for result in convergence_results])
GBF_values = np.array([result["GBF"] for result in convergence_results])
relative_difference_values = np.array([result["relative_difference"] for result in convergence_results])

#Convergence
plt.figure(figsize=[7, 5])
plt.plot(x_extract_values, GBF_values,"o-")
plt.xlabel(r"$x_{extract}$", fontsize = 18)
plt.ylabel(r"$\Gamma$", fontsize = 18)
plt.title(f"GBF Extraction Convergence \n"
    rf"$l = {convergence_l}$, "
    rf"$\omega = {convergence_omega}$", fontsize = 16)
plt.grid()
plt.tight_layout()
plt.savefig("GBF_extraction_convergence.png", dpi=300, format="png")
plt.close()

#Relative Difference
plt.figure(figsize=[7, 5])
plt.plot(x_extract_values, relative_difference_values, "o-")
plt.xlabel(r"$x_{extract}$", fontsize = 18)
plt.ylabel("Relative difference", fontsize = 18)
plt.title(
    f"GBF Relative Difference\n"
    rf"$l = {convergence_l}$, "
    rf"$\omega = {convergence_omega}$", fontsize = 16)
plt.yscale("log")
plt.grid()
plt.tight_layout()
plt.savefig("GBF_extraction_relative_difference.png", dpi=300, format="png")
plt.close()

print()
print("Saved:")
print("GBF_extraction_convergence.png")
print("GBF_extraction_relative_difference.png")

GBF_stored = results[convergence_l][convergence_omega]["GBF"]
GBF_extracted = convergence_results[-1]["GBF"]

print()
print("="*60)
print("Consistency check")
print("="*60)
print(f"Stored GBF:     {GBF_stored:.12e}")
print(f"Extracted GBF:  {GBF_extracted:.12e}")
print(
    f"Absolute difference:     "
    f"{abs(GBF_stored - GBF_extracted):.6e}")

with open("convergence.pkl", "wb") as f:
    pickle.dump(convergence_results, f)

convergence_metadata = {
    "l": convergence_l,
    "omega": convergence_omega,
    "extraction_points": extraction_points,
    "mass": mass,
    "eps": eps,
    "RTOL": RTOL,
    "ATOL": ATOL,
}

with open("convergence_metadata.pkl", "wb") as f:
    pickle.dump(convergence_metadata, f)

print()
print("Saved:")
print("  solutions.pkl")
print("  results.pkl")
print("  convergence.pkl")
print("  convergence_metadata.pkl")