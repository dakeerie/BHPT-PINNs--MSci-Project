import torch as t
import numpy as np 
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
import torch.distributions as dist
import os
import argparse
from matplotlib import rc
import csv
import json
from datetime import datetime
import math
from scipy.special import lambertw
from scipy.integrate import solve_ivp

mass = 0.5
mode = 2
x_max = 0.99
ENVELOPE_M_OMEGA_B = 0.1

def static_solution_coeffs(mode):
    #Horizon-regular static solution coefficients for the dimensionless spin-2 ODE with y = 1 - x = 2M/r and static wavefunction
    # u_s(y) = sum_{k = 3}^{l + 1} a_k y^{-k}. Coefficients normalised so that u_s(1) = 1 and coefficients are independent of mass.
    #A recurrence relation can be derived for the coefficients a_k.

    L = mode*(mode + 1)
    coeffs = {mode + 1: 1.0} 
    for j in range(mode + 1, 3, -1): #Maximum power is always l + 1 so minimum power is 3 as l>=2
        denominator = (j - 1)*(j - 2) - L
        if denominator == 0:
            raise ZeroDivisionError(f"Singular static-coefficient recurrence at l = {mode}, j = {j}.")
        coeffs[j - 1] = (j - 3)*(j + 1)/denominator*coeffs[j]

    norm = sum(coeffs.values())
    if not np.isfinite(norm) or abs(norm) < 1e-14:
        raise ValueError(f"Cannot normalise static solution coefficients for l = {mode}.")
    
    return {k: float(v/norm) for k, v in coeffs.items()}


def characteristic_radius_scale(mode):
    #c_l = ((2l + 1)!!)**(1/(l+1)), used in the characteristic radius: epsilon = 2Momega/c_l
    l = int(mode)
    return float(np.prod(np.arange(1, 2*mode + 2, 2)))**(1/(mode + 1))

def numpy_envelope_components(x_values, omega_value, mass, mode):
    #Numpy components of amplitude_envelope for plotting diagnostics
    x_values = np.asarray(x_values, dtype = float)
    y = np.maximum(1.0 - x_values, 1e-12)
    u_static = np.zeros_like(y)
    for k, coeff in static_solution_coeffs(int(mode)).items():
        u_static += coeff*y**(-k)

    c_l = characteristic_radius_scale(mode)
    epsilon = 2*mass*float(omega_value)/c_l
    far_factor = ((1.0 + epsilon**2)/(1.0 + (epsilon/y)**2))**(0.5*(mode + 1))
    gate_value = 1.0/(1.0 + (mass*float(omega_value)/ENVELOPE_M_OMEGA_B)**(2*(mode + 1)))
    raw_envelope = u_static*far_factor
    envelope = np.exp(gate_value*np.log(np.maximum(raw_envelope, np.finfo(float).tiny)))

    if not np.all(np.isfinite(u_static)) or np.any(u_static <= 0):
            raise FloatingPointError(f"Static solution is non-finite or non-positive for mode={mode}")
    
    for name, values in {"u_static": u_static, "far_factor": far_factor, "raw_envelope": raw_envelope, "envelope": envelope}.items():
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(f"{name} is non-finite for mode={mode}, omega={omega_value}")
        
        if np.any(values <= 0):
            raise FloatingPointError(f"{name} is non-positive for mode={mode}, omega={omega_value}")

    return {"u_static": u_static, "far_factor": far_factor, "gate": np.full_like(x_values, gate_value), "raw_envelope": raw_envelope, "envelope": envelope, "correction_prefactor": x_values**5*envelope}

def reference_u_minus_T4(mass, mode, omega, x_grid, x0 = 1e-3):
    L = mode*(mode + 1)
    Om = 4.0*mass*omega

    c = [1.0 + 0j, (L-3)/(1 - 1j*Om)]
    c.append(((L + 1)*c[1] + 3)/(4 - 2j*Om))
    c.append((L + 9)*c[2]/(3*(3 - 1j*Om)))
    c.append(((L + 21)*c[3] - 5*c[2])/(4*(4 - 1j*Om)))
    T4 = lambda x: sum(c[n]*x**n for n in range(5))
    u0 = T4(x0)
    du0 = sum(n*c[n]*x0**(n - 1) for n in range(1, 5))

    def rhs(x, s):
        u, du = s
        return [du, -(((1 - x)*(1 - 3*x) - 1j*Om)*du + (-L + 3*(1 - x))*u)/(x*( 1- x)**2)]

    sol = solve_ivp(rhs, (x0, x_grid[-1]), [u0, du0], t_eval = x_grid, method = 'DOP853', rtol = 1e-11, atol = 1e-30)
    return sol.y[0] - T4(x_grid)

correction_array = []

def scan_gate_centre(mass, mode, x_max, Mw_grid, Mw_b_grid, x_lo=0.3):
    global ENVELOPE_M_OMEGA_B
    x_grid = np.linspace(x_lo, x_max, 200)
    nums = {Mw: np.abs(reference_u_minus_T4(mass, mode, Mw/mass, x_grid)) for Mw in Mw_grid}
    saved, scores = ENVELOPE_M_OMEGA_B, []
    try:
        for Mw_b in Mw_b_grid:
            ENVELOPE_M_OMEGA_B = Mw_b
            logs = np.array([np.log10(nums[Mw]/(x_grid**5*numpy_envelope_components(x_grid, Mw/mass, mass, mode)["envelope"])) for Mw in Mw_grid])
            spread_x = np.mean(np.std(logs, axis=1))     # variation along x at fixed omega
            spread_w = np.std(logs[:, -1])               # variation across omega at x_max
            scores.append(np.hypot(spread_x, spread_w))
    finally:
        ENVELOPE_M_OMEGA_B = saved
    return np.array(scores)

Mw_grid = np.geomspace(0.005, 0.7, 40)
Mw_b_grid = np.linspace(0.05, 0.8, 40)
scores = scan_gate_centre(mass, mode, x_max, Mw_grid, Mw_b_grid)
print("best Mω_b:", Mw_b_grid[np.argmin(scores)])