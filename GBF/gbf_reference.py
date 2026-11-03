"""Trusted reference greybody factors for spin-2 (Regge-Wheeler) perturbations of Schwarzschild.

Step 1 of the PINN improvement plan: build reference values that can be trusted at low frequency,
independently of the PINN, before using them to judge the PINN.

Two independent methods are provided:
    1. Direct ODE integration of the Regge-Wheeler equation from the horizon to a large matching radius,
       matched there to an optimally truncated far-zone series (the "gold standard").
    2. The Helton-Stuckwisch product integral (copied unchanged from the PINN script), with a study of how
       its result depends on the outer cut-off u*_max.

Conventions (identical to the PINN script):
    rho   = r/(2M),            x = 1 - 1/rho,          y = 1 - x = 1/rho
    rho*  = r*/(2M) = rho + ln(rho - 1)
    k     = 2*M*omega,         Omega = 4*M*omega = 2k,  L = l(l + 1)
    psi   = exp(-i k rho*) u(x),  with u(x = 0) = 1   (unit transmitted amplitude at the horizon)
    psi   = alpha exp(-i k rho*) u1(y) + beta exp(+i k rho*) conj(u1(y))   at large rho
    GBF   = 1/|alpha|^2

Run as a script, e.g.
    python gbf_reference.py --mass 1 --mode 2 --omega_min 0.01 --omega_max 1.2 --query_steps 200
"""

import argparse
import csv
import math
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import lambertw


# ============================================================================
# Coordinates
# ============================================================================

def rho_minus_one(rhostar):
    """Return W = rho - 1 for a given rho* = rho + ln(rho - 1).

    W solves W + ln(W) = rho* - 1, i.e. W = LambertW(exp(rho* - 1)). W is returned (rather than rho)
    so that x = W/(1 + W) keeps full relative precision near the horizon, where rho - 1 is tiny.
    """
    z = float(rhostar) - 1.0
    if z < 30.0:
        return float(np.real(lambertw(math.exp(z))))
    # exp(z) would overflow for large z: Newton iteration on F(W) = W + ln W - z, F'(W) = 1 + 1/W
    w = z - math.log(z)
    for _ in range(50):
        step = (w + math.log(w) - z)*w/(w + 1.0)
        w -= step
        if abs(step) <= 1e-16*w:
            break
    return w


def coordinates_from_rhostar(rhostar):
    """Return (x, y, rho) at a given rho*, each computed without cancellation."""
    w = rho_minus_one(rhostar)
    return w/(1.0 + w), 1.0/(1.0 + w), 1.0 + w


def rw_potential_xy(x, y, mode):
    """Dimensionless Regge-Wheeler potential (2M)^2 V = (1 - 1/rho)(L/rho^2 - 3/rho^3) = x y^2 (L - 3y)."""
    L = mode*(mode + 1)
    return x*y*y*(L - 3.0*y)


# ============================================================================
# Far-zone (infinity-ingoing) series u1(y) = sum_n a_n y^n
# ============================================================================

def far_zone_series(y, Momega, mode, n_max = 2000):
    """Optimally truncated asymptotic series for the infinity-ingoing branch u1(y) and du1/dy.

    Recurrence (derived from the PINN ODE written in y = 1 - x):
        i Omega (n + 1) a_{n+1} = [L - n(n + 1)] a_n + (n^2 - 4) a_{n-1},   a_0 = 1, a_{-1} = 0.
    The scaled terms t_n = a_n y^n are iterated directly to avoid overflow:
        t_{n+1} = ([L - n(n + 1)] y t_n + (n^2 - 4) y^2 t_{n-1}) / (i Omega (n + 1)).
    The series is asymptotic, so summation stops at the smallest term once the flat-space part (n <= l + 1)
    has been included. Returns (u1, du1_dy, n_terms_used, |last term included|).
    """
    L = mode*(mode + 1)
    Omega = 4.0*Momega

    t_prev = 0.0 + 0.0j      # t_{-1}
    t_curr = 1.0 + 0.0j      # t_0
    u1 = t_curr
    du1_dy = 0.0 + 0.0j
    last_nonzero_size = 1.0  # |t_0|
    n_used = 0

    for n in range(n_max):
        t_next = ((L - n*(n + 1))*y*t_curr + (n*n - 4)*y*y*t_prev)/(1j*Omega*(n + 1))
        size = abs(t_next)

        # Individual terms can vanish exactly (e.g. a_3 = 0 for l = 2), so only non-zero terms are used to
        # decide convergence. After the exact flat-space part (n + 1 > l + 1), stop before the terms grow.
        if size > 0.0 and n + 1 > mode + 1:
            if size > last_nonzero_size:
                break
            if size < 1e-18*abs(u1):
                u1 += t_next
                du1_dy += (n + 1)*t_next/y
                n_used = n + 1
                last_nonzero_size = size
                break

        u1 += t_next
        du1_dy += (n + 1)*t_next/y
        n_used = n + 1
        t_prev, t_curr = t_curr, t_next
        if size > 0.0:
            last_nonzero_size = size

    return u1, du1_dy, n_used, last_nonzero_size


# ============================================================================
# Method 1: direct ODE integration with Wronskian matching
# ============================================================================

def _rw_rhs(rhostar, state, k, mode):
    """First-order form of psi'' + (k^2 - V)psi = 0 in rho*: state = [psi, dpsi/drho*]."""
    x, y, _ = coordinates_from_rhostar(rhostar)
    V = rw_potential_xy(x, y, mode)
    return np.array([state[1], (V - k*k)*state[0]], dtype = np.complex128)


def reference_gbf_ode(Momega, mode, rhostar_start = -40.0, omega_r_match = None, rtol = 1e-12, atol = 1e-30):
    """Greybody factor from direct integration of the Regge-Wheeler equation.

    1. Start at rho*_0 (deep in the near-horizon region) with psi = exp(-i k rho*) (1 + c1 x).
    2. Integrate outward to rho_m, chosen so that omega*r_m = k*rho_m is large enough for the far-zone
       series to be accurate to near machine precision.
    3. Match to psi_in = exp(-i k rho*) u1(y) and psi_out = conj(psi_in) using Wronskians:
           alpha = W[psi, psi_out]/(2ik),   beta = W[psi_in, psi]/(2ik).
    """
    Momega = float(Momega)
    if not (np.isfinite(Momega) and Momega > 0.0):
        raise ValueError("Momega must be finite and positive")
    if mode < 2:
        raise ValueError("mode must be >= 2 for spin-2 perturbations")

    L = mode*(mode + 1)
    k = 2.0*Momega

    # Matching radius: omega*r_m = k*rho_m (see derivation). Default leaves the optimal truncation error
    # of the far-zone series ~exp(-2 omega r_m) far below double precision.
    if omega_r_match is None:
        omega_r_match = 25.0 + 2.0*L
    rho_m = max(omega_r_match/k, 20.0)
    rhostar_m = rho_m + math.log(rho_m - 1.0)

    # Horizon initial data from u = 1 + c1 x + O(x^2), c1 = (L - 3)/(1 - i Omega)
    x0, _, _ = coordinates_from_rhostar(rhostar_start)
    c1 = (L - 3)/(1.0 - 1j*2.0*k)
    phase0 = np.exp(-1j*k*rhostar_start)
    u_h = 1.0 + c1*x0
    du_h_drhostar = c1*x0*(1.0 - x0)**2          # du/dx * dx/drho*, with dx/drho* = x(1 - x)^2
    psi0 = phase0*u_h
    dpsi0 = phase0*(-1j*k*u_h + du_h_drhostar)

    sol = solve_ivp(_rw_rhs, (rhostar_start, rhostar_m), np.array([psi0, dpsi0], dtype = np.complex128),
                    method = "DOP853", rtol = rtol, atol = atol, args = (k, mode))
    if not sol.success:
        raise RuntimeError(f"ODE integration failed at Momega = {Momega}: {sol.message}")
    psi_m, dpsi_m = sol.y[0, -1], sol.y[1, -1]

    # Far-zone basis at the matching point
    y_m = 1.0/rho_m
    u1, du1_dy, n_terms, tail = far_zone_series(y_m, Momega, mode)
    phase_m = np.exp(-1j*k*rhostar_m)
    psi_in = phase_m*u1
    dpsi_in = phase_m*(-1j*k*u1 - y_m*y_m*(1.0 - y_m)*du1_dy)     # dy/drho* = -y^2 (1 - y)
    psi_out, dpsi_out = np.conj(psi_in), np.conj(dpsi_in)

    wronskian_exact = 2j*k
    alpha = (psi_m*dpsi_out - dpsi_m*psi_out)/wronskian_exact
    beta = (psi_in*dpsi_m - dpsi_in*psi_m)/wronskian_exact

    wronskian_basis = psi_in*dpsi_out - dpsi_in*psi_out
    abs_a2, abs_b2 = abs(alpha)**2, abs(beta)**2

    return {"Momega": Momega,
            "GBF": 1.0/abs_a2,
            "alpha": complex(alpha),
            "beta": complex(beta),
            "flux_error_scaled": (abs_a2 - abs_b2 - 1.0)/(abs_a2 + abs_b2),
            "basis_wronskian_error": abs(wronskian_basis - wronskian_exact)/abs(wronskian_exact),
            "series_terms": n_terms,
            "series_tail": tail,
            "rho_match": rho_m,
            "rhs_evaluations": int(sol.nfev)}


def reference_gbf_with_error(Momega, mode):
    """Baseline reference plus an error estimate from three independent perturbations of the numerics:
    a looser ODE tolerance, a doubled matching radius and a later horizon starting point."""
    L = mode*(mode + 1)
    base = reference_gbf_ode(Momega, mode)
    variants = [reference_gbf_ode(Momega, mode, rtol = 1e-10),
                reference_gbf_ode(Momega, mode, omega_r_match = 2.0*(25.0 + 2.0*L)),
                reference_gbf_ode(Momega, mode, rhostar_start = -30.0)]
    base["GBF_rel_error_estimate"] = max(abs(v["GBF"] - base["GBF"])/base["GBF"] for v in variants)
    return base


# ============================================================================
# Extraction-point quantities used by the PINN (Step 2)
# ============================================================================

def far_zone_basis_at_x(x_ext, Momega, mode, series_tol = 1e-14, rtol = 1e-12, atol = 1e-30):
    """Infinity-ingoing basis function u1 and du1/dx at x_ext, in PINN conventions (u1 -> 1 as y -> 0).

    If the optimally truncated series is accurate at y_ext (smallest term <= series_tol*|u1|) it is used
    directly, with du1/dx = -du1/dy. Otherwise psi_in = exp(-i k rho*) u1 is evaluated from the series at a
    large radius rho_m and integrated inward to rho_ext, and u1 and du1/dx are recovered from psi_in.
    Returns (u1, du1_dx, info).
    """
    x_ext = float(x_ext)
    Momega = float(Momega)
    if not (0.0 < x_ext < 1.0):
        raise ValueError("x_ext must lie in (0, 1)")

    L = mode*(mode + 1)
    k = 2.0*Momega
    y_ext = 1.0 - x_ext
    rho_ext = 1.0/y_ext

    u1, du1_dy, n_terms, tail = far_zone_series(y_ext, Momega, mode)
    if np.isfinite(tail) and tail <= series_tol*abs(u1):
        return complex(u1), complex(-du1_dy), {"method": "series", "series_terms": n_terms, "series_tail": tail}

    # Series not accurate enough at x_ext: start from a radius where it is, then integrate inward
    rho_m = max((25.0 + 2.0*L)/k, 20.0, 2.0*rho_ext)
    rhostar_m = rho_m + math.log(rho_m - 1.0)
    y_m = 1.0/rho_m
    u1_m, du1_dy_m, n_terms, tail = far_zone_series(y_m, Momega, mode)
    phase_m = np.exp(-1j*k*rhostar_m)
    psi_m = phase_m*u1_m
    dpsi_m = phase_m*(-1j*k*u1_m - y_m*y_m*(1.0 - y_m)*du1_dy_m)

    rhostar_ext = 1.0/y_ext + math.log(x_ext/y_ext)          # rho + ln(rho - 1), with rho - 1 = x/y
    sol = solve_ivp(_rw_rhs, (rhostar_m, rhostar_ext), np.array([psi_m, dpsi_m], dtype = np.complex128),
                    method = "DOP853", rtol = rtol, atol = atol, args = (k, mode))
    if not sol.success:
        raise RuntimeError(f"Inward basis integration failed at Momega = {Momega}: {sol.message}")
    psi_e, dpsi_e = sol.y[0, -1], sol.y[1, -1]

    phase_e = np.exp(1j*k*rhostar_ext)
    u1_e = phase_e*psi_e                                     # u1 = exp(+i k rho*) psi_in
    du1_drhostar = phase_e*(dpsi_e + 1j*k*psi_e)
    du1_dx = du1_drhostar/(x_ext*y_ext*y_ext)                # dx/drho* = x y^2
    return complex(u1_e), complex(du1_dx), {"method": "inward_integration", "series_terms": n_terms,
                                            "series_tail": tail, "rho_match": rho_m}


def two_term_basis_at_x(x_ext, Momega, mode):
    """The original PINN basis u1 = 1 + a1 y + a2 y^2 and du1/dx, kept for comparison only."""
    L = mode*(mode + 1)
    Omega = 4.0*Momega
    a1 = -1j*L/Omega
    a2 = -(3.0 + (2.0 - L)*a1)/(1j*2.0*Omega)
    y = 1.0 - float(x_ext)
    return complex(1.0 + a1*y + a2*y**2), complex(-(a1 + 2.0*a2*y))


def extract_alpha_beta(u, du_dx, u1, du1_dx, x_ext, Momega):
    """Wronskian extraction of alpha and beta from u = alpha u1 + beta exp(2 i k rho*) conj(u1).

    With G = x y^2/(2 i k) (= g(x)/(2 i omega) in the PINN's units) and primes denoting d/dx:
        alpha = u conj(u1) + G [u conj(u1') - u' conj(u1)]
        beta  = exp(-2 i k rho*) G [u1 u' - u1' u]
    These divide by the exact constant Wronskian 2ik, so (unlike (u - u'/D)/(u1 - u1'/D)) they do not
    lose precision when omega*r_ext << 1.
    """
    x = float(x_ext)
    y = 1.0 - x
    k = 2.0*float(Momega)
    G = x*y*y/(2j*k)
    alpha = u*np.conj(u1) + G*(u*np.conj(du1_dx) - du_dx*np.conj(u1))
    rhostar = 1.0/y + math.log(x/y)
    beta = np.exp(-2j*k*rhostar)*G*(u1*du_dx - du1_dx*u)
    return complex(alpha), complex(beta)


def extract_alpha_beta_original(u, du_dx, u1, du1_dx, x_ext, Momega):
    """The PINN's original formula alpha = (u - u'/D)/(u1 - u1'/D), kept for comparison only."""
    x = float(x_ext)
    y = 1.0 - x
    k = 2.0*float(Momega)
    D = 1j*2.0*k/(y*y*x) + np.conj(du1_dx/u1)
    alpha = (u - du_dx/D)/(u1 - du1_dx/D)
    rhostar = 1.0/y + math.log(x/y)
    beta = (u - alpha*u1)/(np.exp(2j*k*rhostar)*np.conj(u1))
    return complex(alpha), complex(beta)


def basis_wronskian_error(u1, du1_dx, x_ext, Momega):
    """|W[psi_in, psi_out] - 2ik|/(2k) at x_ext, a check on the accuracy of the basis.
    W = 2ik |u1|^2 + x y^2 (u1 conj(u1') - u1' conj(u1)) in terms of u1 (see derivation)."""
    x = float(x_ext)
    y = 1.0 - x
    k = 2.0*float(Momega)
    W = 2j*k*abs(u1)**2 + x*y*y*(u1*np.conj(du1_dx) - du1_dx*np.conj(u1))
    return abs(W - 2j*k)/(2.0*k)


def scaled_flux_error(alpha, beta):
    """(|alpha|^2 - |beta|^2 - 1)/(|alpha|^2 + |beta|^2): flux-identity error relative to the size of the terms."""
    a2, b2 = abs(alpha)**2, abs(beta)**2
    return (a2 - b2 - 1.0)/(a2 + b2)


def exact_u_at_x(x_ext, Momega, mode, rhostar_start = -40.0, rtol = 1e-12, atol = 1e-30):
    """Exact horizon-normalised u(x_ext) and du/dx(x_ext): what a perfectly trained PINN would output."""
    x_ext = float(x_ext)
    L = mode*(mode + 1)
    k = 2.0*float(Momega)
    y_ext = 1.0 - x_ext

    x0, _, _ = coordinates_from_rhostar(rhostar_start)
    c1 = (L - 3)/(1.0 - 1j*2.0*k)
    phase0 = np.exp(-1j*k*rhostar_start)
    u_h = 1.0 + c1*x0
    psi0 = phase0*u_h
    dpsi0 = phase0*(-1j*k*u_h + c1*x0*(1.0 - x0)**2)

    rhostar_ext = 1.0/y_ext + math.log(x_ext/y_ext)
    sol = solve_ivp(_rw_rhs, (rhostar_start, rhostar_ext), np.array([psi0, dpsi0], dtype = np.complex128),
                    method = "DOP853", rtol = rtol, atol = atol, args = (k, mode))
    if not sol.success:
        raise RuntimeError(f"Horizon integration failed at Momega = {Momega}: {sol.message}")
    psi_e, dpsi_e = sol.y[0, -1], sol.y[1, -1]

    phase_e = np.exp(1j*k*rhostar_ext)
    u = phase_e*psi_e                                        # u = exp(+i k rho*) psi
    du_dx = phase_e*(dpsi_e + 1j*k*psi_e)/(x_ext*y_ext*y_ext)
    return complex(u), complex(du_dx)


def extraction_self_check(x_ext, Momega, mode):
    """Apply the new and the old (two-term) extraction to the exact solution at x_ext and compare with
    the reference GBF. This isolates extraction error from network error."""
    reference = reference_gbf_ode(Momega, mode)["GBF"]
    u, du = exact_u_at_x(x_ext, Momega, mode)

    u1, du1, info = far_zone_basis_at_x(x_ext, Momega, mode)
    alpha_new, beta_new = extract_alpha_beta(u, du, u1, du1, x_ext, Momega)
    gbf_new = 1.0/abs(alpha_new)**2

    u1_old, du1_old = two_term_basis_at_x(x_ext, Momega, mode)
    alpha_old, _ = extract_alpha_beta_original(u, du, u1_old, du1_old, x_ext, Momega)
    gbf_old = 1.0/abs(alpha_old)**2

    return {"Momega": float(Momega), "x_ext": float(x_ext), "GBF_reference": reference,
            "GBF_new_extraction": gbf_new, "GBF_two_term_extraction": gbf_old,
            "rel_error_new": abs(gbf_new - reference)/reference, "rel_error_two_term": abs(gbf_old - reference)/reference,
            "scaled_flux_error_new": scaled_flux_error(alpha_new, beta_new), "basis_method": info["method"]}


# ============================================================================
# Low-frequency asymptotic formula (copied from the PINN script)
# ============================================================================

def low_frequency_GBF_coeff(mode):
    numerator = 2**(2*mode + 2)*math.factorial(mode - 2)*math.factorial(mode)*math.factorial(mode + 2)
    denominator = math.factorial(2*mode)*math.factorial(2*mode + 1)
    return (numerator/denominator)**2


def low_frequency_limit_GBF_Momega(Momega, mode):
    return low_frequency_GBF_coeff(mode)*np.asarray(Momega)**(2*mode + 2)


# ============================================================================
# Method 2: product integral (copied unchanged from the PINN script) + u*_max study
# ============================================================================

def u_from_ustar(ustar):
    """Calculates dimensionless u = r/2M from dimensionless tortoise coordinate u_* = r_*/2M"""
    ustar_arr = np.asarray(ustar, dtype = float)
    original_shape = ustar_arr.shape
    z = (ustar_arr - 1.0).reshape(-1)
    w = np.empty_like(z)

    direct = z < 500.0
    if np.any(direct):
        w[direct] = np.real(lambertw(np.exp(z[direct])))

    large = ~direct
    if np.any(large):
        z_large = z[large]
        w_large = z_large - np.log(z_large) + np.log(z_large)/z_large
        for _ in range(8):
            residual = w_large + np.log(w_large) - z_large
            w_large -= residual*w_large/(w_large + 1.0)
        w[large] = w_large

    return (1.0 + w).reshape(original_shape)


def RW_potential(u, mode):
    return (1.0 - 1.0/u)*(mode*(mode + 1)/u**2 - 3/u**3)


def negative_integrand_values(ustar_nodes, Momega, mode):
    u = u_from_ustar(ustar_nodes)
    potential = RW_potential(u, mode)
    coeff = 1j*potential/(4.0*Momega)
    phase = np.exp(4j*Momega*ustar_nodes)

    A = np.empty((ustar_nodes.size, 2, 2), dtype = np.complex128)
    A[:, 0, 0] = coeff
    A[:, 0, 1] = coeff*np.conjugate(phase)
    A[:, 1, 0] = -coeff*phase
    A[:, 1, 1] = -coeff
    return A


def product_integral_GBF(x, mode, ustar_min = -50.0, ustar_max = 350.0, N = 10000):
    x = float(x)
    h = (ustar_max - ustar_min)/N
    ustar_nodes = ustar_min + (h/4.0)*np.arange(4*N + 1)
    A_nodes = negative_integrand_values(ustar_nodes, x, mode)

    I = np.eye(2, dtype = np.complex128)
    H = I.copy()

    for k in range(N):
        A0, A1, A2, A3, A4 = A_nodes[4*k:4*k + 5]
        K1 = h*A4
        K2 = h*(A3 @ (4.0*I + K1))
        K3 = h*(A2 @ (8.0*(I - K1) + 3.0*K2))
        K4 = h*(A1 @ (32.0*I + 18.0*K1 + 3.0*h*(A2 @ (6.0*I - K1 + K2)) - 3.0*K2))
        K5 = h*(A0 @ (28.0*(I - K1) - 3*h*(A2 @ (16.0*I + 4.0*K1 + K2)) + K4 + 18.0*K2))
        step = I + (28.0*K1 + 32.0*K2 + 6.0*K3 + 4.0*K4 + K5)/360.0
        H = H @ step

    alpha = np.conjugate(H[0, 0])
    beta = -H[1, 0]
    return float(1.0/(np.abs(alpha)**2)), complex(alpha), complex(beta)


def product_integral_ustar_max_study(Momega, mode, reference_gbf, ustar_max_values = (350.0, 700.0, 1400.0, 2800.0, 5600.0),
                                     h = 0.04, ustar_min = -50.0):
    """Product-integral GBF as a function of the outer cut-off u*_max at fixed step h (default h = 400/10000,
    as in the PINN script), compared against the ODE reference."""
    rows = []
    for ustar_max in ustar_max_values:
        N = int(round((ustar_max - ustar_min)/h))
        T, alpha, beta = product_integral_GBF(Momega, mode, ustar_min = ustar_min, ustar_max = ustar_max, N = N)
        rows.append({"Momega": float(Momega), "ustar_max": float(ustar_max), "N": N, "GBF_PI": T,
                     "rel_diff_vs_reference": abs(T - reference_gbf)/reference_gbf,
                     "tail_parameter": mode*(mode + 1)/(4.0*Momega*ustar_max)})
    return rows


# ============================================================================
# Optional comparison with the existing numerical CSV (same filters as the PINN script)
# ============================================================================

def load_numerical_gbf(csv_path, mass, mode, x_extract):
    numerical_results = {}
    with open(csv_path, "r", newline = "") as f:
        for row in csv.DictReader(f):
            if not np.isclose(float(row["mass"]), mass, rtol = 0.0, atol = 1e-12):
                continue
            if int(row["mode"]) != mode:
                continue
            if not np.isclose(float(row["x_extract"]), x_extract, rtol = 0.0, atol = 1e-12):
                continue
            if row["success"].strip().lower() not in {"true", "1"}:
                continue
            gbf = float(row["GBF"])
            if np.isfinite(gbf):
                numerical_results[round(float(row["omega"]), 8)] = gbf
    return numerical_results


# ============================================================================
# Script entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description = "Trusted reference GBFs for the spin-2 Regge-Wheeler equation")
    parser.add_argument("--mass", type = float, required = True)
    parser.add_argument("--mode", type = int, required = True)
    parser.add_argument("--omega_min", type = float, default = 0.01)
    parser.add_argument("--omega_max", type = float, default = 1.2)
    parser.add_argument("--query_steps", type = int, default = 200, help = "Same log-spaced grid as the PINN query")
    parser.add_argument("--error_every", type = int, default = 10, help = "Run the 3-variant error estimate on every n-th frequency")
    parser.add_argument("--pi_study_Momega", type = float, nargs = "*", default = None,
                        help = "M*omega values for the product-integral u*_max study (default: lowest, geometric-mid and 0.2)")
    parser.add_argument("--numerical_csv", type = str, default = None)
    parser.add_argument("--x_extract", type = float, default = None, help = "Needed to filter the numerical CSV")
    parser.add_argument("--out_dir", type = str, default = None)
    args = parser.parse_args()

    if args.mass <= 0 or args.mode < 2 or not (0 < args.omega_min < args.omega_max):
        raise ValueError("Require mass > 0, mode >= 2 and 0 < omega_min < omega_max")

    mass, mode = args.mass, args.mode
    out_dir = args.out_dir or os.path.join("./Reference", f"l{mode}", f"M{mass:g}_w{args.omega_min:g}-{args.omega_max:g}")
    os.makedirs(out_dir, exist_ok = True)

    # ---- 1. ODE reference on the PINN query grid ------------------------------------------------
    omega_grid = np.logspace(np.log10(args.omega_min), np.log10(args.omega_max), args.query_steps)
    rows = []
    for i, omega in enumerate(omega_grid):
        Momega = mass*omega
        with_error = (i % args.error_every == 0) or (i == len(omega_grid) - 1)
        res = reference_gbf_with_error(Momega, mode) if with_error else reference_gbf_ode(Momega, mode)
        res["omega"] = float(omega)
        res["GBF_asymptotic"] = float(low_frequency_limit_GBF_Momega(Momega, mode))
        res.setdefault("GBF_rel_error_estimate", np.nan)
        rows.append(res)

    ref_csv = os.path.join(out_dir, "reference_GBF.csv")
    with open(ref_csv, "w", newline = "") as f:
        writer = csv.writer(f)
        writer.writerow(["omega", "Momega", "GBF", "GBF_rel_error_estimate", "alpha_re", "alpha_im", "beta_re", "beta_im",
                         "flux_error_scaled", "basis_wronskian_error", "series_terms", "series_tail", "rho_match",
                         "GBF_asymptotic", "GBF_over_asymptotic"])
        for r in rows:
            writer.writerow([f"{r['omega']:.10f}", f"{r['Momega']:.10e}", f"{r['GBF']:.15e}",
                             f"{r['GBF_rel_error_estimate']:.3e}",
                             f"{r['alpha'].real:.15e}", f"{r['alpha'].imag:.15e}", f"{r['beta'].real:.15e}", f"{r['beta'].imag:.15e}",
                             f"{r['flux_error_scaled']:.3e}", f"{r['basis_wronskian_error']:.3e}", r["series_terms"],
                             f"{r['series_tail']:.3e}", f"{r['rho_match']:.6e}",
                             f"{r['GBF_asymptotic']:.15e}", f"{r['GBF']/r['GBF_asymptotic']:.10f}"])
    print(f"Saved ODE reference to {ref_csv}")

    err = np.array([r["GBF_rel_error_estimate"] for r in rows])
    flux = np.array([abs(r["flux_error_scaled"]) for r in rows])
    basis = np.array([r["basis_wronskian_error"] for r in rows])
    print(f"Max GBF relative error estimate (sampled): {np.nanmax(err):.2e}")
    print(f"Max |scaled flux error|: {flux.max():.2e} | max basis Wronskian error: {basis.max():.2e}")

    # ---- 2. Product-integral u*_max study --------------------------------------------------------
    Momega_grid = mass*omega_grid
    study_Momega = args.pi_study_Momega
    if study_Momega is None:
        study_Momega = [Momega_grid[0], math.sqrt(Momega_grid[0]*Momega_grid[-1]), 0.2]
    pi_rows = []
    for Mw in study_Momega:
        ref = reference_gbf_ode(Mw, mode)["GBF"]
        pi_rows.extend(product_integral_ustar_max_study(Mw, mode, ref))

    pi_csv = os.path.join(out_dir, "product_integral_ustar_max_study.csv")
    with open(pi_csv, "w", newline = "") as f:
        writer = csv.writer(f)
        writer.writerow(["Momega", "ustar_max", "N", "GBF_PI", "rel_diff_vs_reference", "tail_parameter"])
        for r in pi_rows:
            writer.writerow([f"{r['Momega']:.10e}", r["ustar_max"], r["N"], f"{r['GBF_PI']:.15e}",
                             f"{r['rel_diff_vs_reference']:.3e}", f"{r['tail_parameter']:.3e}"])
    print(f"Saved product-integral u*_max study to {pi_csv}")
    for r in pi_rows:
        print(f"  M*omega = {r['Momega']:.4g} | u*_max = {r['ustar_max']:7.0f} | rel. diff vs ODE = {r['rel_diff_vs_reference']:.2e}")

    # ---- 3. Optional comparison with the existing numerical CSV ----------------------------------
    numerical = {}
    if args.numerical_csv is not None:
        if args.x_extract is None:
            raise ValueError("--x_extract is required with --numerical_csv")
        numerical = load_numerical_gbf(args.numerical_csv, mass, mode, args.x_extract)
        print(f"Loaded {len(numerical)} numerical CSV values")

    # ---- 4. Plots ---------------------------------------------------------------------------------
    gbf = np.array([r["GBF"] for r in rows])
    asym = np.array([r["GBF_asymptotic"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize = (13, 5))
    axes[0].loglog(Momega_grid, gbf, label = "ODE reference")
    axes[0].loglog(Momega_grid, asym, "--", label = r"$C_\ell (M\omega)^{2\ell+2}$")
    axes[1].loglog(Momega_grid, gbf/asym, label = "ODE reference / asymptotic")

    if numerical:
        num_M, num_ratio = [], []
        for omega, g in zip(omega_grid, gbf):
            key = round(float(omega), 8)
            if key in numerical:
                num_M.append(mass*omega)
                num_ratio.append(numerical[key]/g)
        if num_M:
            axes[1].loglog(num_M, num_ratio, "x", label = "numerical CSV / ODE reference")

    axes[0].set(xlabel = r"$M\omega$", ylabel = r"$\Gamma$", title = rf"Reference GBF, $\ell = {mode}$")
    axes[1].axhline(1.0, color = "k", lw = 0.8)
    axes[1].set(xlabel = r"$M\omega$", ylabel = "ratio", title = "Ratios")
    for ax in axes:
        ax.grid(True, which = "both", alpha = 0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reference_GBF.png"), dpi = 200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize = (7, 5))
    for Mw in study_Momega:
        sub = [r for r in pi_rows if r["Momega"] == float(Mw)]
        ax.loglog([r["ustar_max"] for r in sub], [max(r["rel_diff_vs_reference"], 1e-17) for r in sub], "o-",
                  label = rf"$M\omega = {Mw:.3g}$")
    ax.set(xlabel = r"$u_*^{\max}$", ylabel = "|PI - ODE| / ODE", title = rf"Product integral cut-off study, $\ell = {mode}$")
    ax.grid(True, which = "both", alpha = 0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "product_integral_ustar_max_study.png"), dpi = 200)
    plt.close(fig)
    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
