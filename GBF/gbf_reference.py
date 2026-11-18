"""Exact far-zone basis, Wronskian extraction and ODE reference for the spin-2 Regge-Wheeler PINN (PINNGBFWX.py).

Provides what the PINN imports:
    far_zone_basis_at_x       infinity-ingoing basis u1, du1/dx at the extraction point (optimally truncated
                              asymptotic series, or inward integration from a radius where the series is accurate)
    extract_alpha_beta        Wronskian extraction of alpha and beta
    reference_gbf_ode         ODE reference GBF (horizon-to-large-radius integration matched to the far-zone series)
    exact_u_at_x              exact horizon-normalised u and du/dx at a point (what a perfect PINN would output)
    extraction_self_check     extraction applied to the exact solution, compared with the reference
    two_term_basis_at_x, extract_alpha_beta_original   the original two-term basis and formula, kept for comparison
    scaled_flux_error         scale-free flux-identity error
    basis tracking            counts of series / inward-integration basis computations (basis_tracking_summary, ...)

Conventions (identical to the PINN script):
    rho   = r/(2M),            x = 1 - 1/rho,          y = 1 - x = 1/rho
    rho*  = r*/(2M) = rho + ln(rho - 1)
    k     = 2*M*omega,         Omega = 4*M*omega = 2k,  L = l(l + 1)
    psi   = exp(-i k rho*) u(x),  with u(x = 0) = 1 (unit transmitted amplitude at the horizon)
    psi   = alpha exp(-i k rho*) u1(y) + beta exp(+i k rho*) conj(u1(y)) at large rho, where conj(u1(y)) = u2
    GBF   = 1/|alpha|^2
"""

import csv
import math
import os
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import lambertw

#Coordinates
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

# Far-zone (infinity-ingoing) series u1(y) = sum_n a_n y^n
def far_zone_series(y, Momega, mode, n_max = 2000):
    """Truncated asymptotic series for the infinity-ingoing branch u1(y) and du1/dy.

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

# Method 1: direct ODE integration with Wronskian matching
def _rw_rhs(rhostar, state, k, mode):
    """First-order form of psi'' + (k^2 - V)psi = 0 in rho*: state = [psi, dpsi/drho*]."""
    x, y, _ = coordinates_from_rhostar(rhostar)
    V = rw_potential_xy(x, y, mode)
    return np.array([state[1], (V - k*k)*state[0]], dtype = np.complex128)

def reference_gbf_ode(Momega, mode, rhostar_start = -40.0, omega_r_match = None, rtol = 1e-12, atol = 1e-30):
    """Greybody factor from direct integration of the Regge-Wheeler equation.

    1. Start at rho*_0 with psi = exp(-i k rho*) (1 + c1 x).
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
    _BASIS_TRACKING["counts"]["reference_gbf_ode"] += 1

    L = mode*(mode + 1)
    k = 2.0*Momega

    # Matching radius: omega*r_m = k*rho_m. Default leaves the optimal truncation error
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


# Optional tracking of far-zone basis evaluations
# far_zone_basis_at_x either sums the asymptotic series at x_ext ("series") or, where that is not accurate, starts
# the series at a large radius and integrates the RW equation inward ("inward_integration"). Every call is counted;
# each inward integration is also recorded (M*omega, x_ext, start radius, right-hand-side evaluations, wall time).
# Nothing is printed unless asked for, so results and default behaviour are unchanged:
#   set_basis_tracking(verbose = True)  or environment variable GBF_BASIS_VERBOSE=1  -> one line per inward integration
#   basis_tracking_summary()            -> text summary (counts, M*omega range, cost, calls per decade of M*omega)
#   write_basis_tracking_csv(path)      -> one row per inward integration
#   reset_basis_tracking()              -> zero the counters
# The PINN caches the basis per (x_ext, M*omega) (extraction_basis, lru_cache), so these counts are the number of
# basis COMPUTATIONS; extraction_basis.cache_info() in the PINN gives the number of requests served from the cache.
_BASIS_TRACKING = {"verbose": os.environ.get("GBF_BASIS_VERBOSE", "0") == "1",
                   "counts": {"series": 0, "inward_integration": 0, "reference_gbf_ode": 0, "exact_u_at_x": 0},
                   "integrations": []}


def set_basis_tracking(verbose = True):
    """Switch per-call printing of inward basis integrations on or off (counting is always on)."""
    _BASIS_TRACKING["verbose"] = bool(verbose)


def reset_basis_tracking():
    """Zero all counters and forget the recorded inward integrations."""
    for key in _BASIS_TRACKING["counts"]:
        _BASIS_TRACKING["counts"][key] = 0
    _BASIS_TRACKING["integrations"].clear()


def basis_tracking_counts():
    """Copy of the raw counters: series / inward_integration basis calls and the other RW integrations."""
    return dict(_BASIS_TRACKING["counts"])


def basis_tracking_summary():
    """Human-readable summary of how often the numerical u1 integration was used, and where in M*omega."""
    c = _BASIS_TRACKING["counts"]
    recs = _BASIS_TRACKING["integrations"]
    n_basis = c["series"] + c["inward_integration"]
    lines = [f"Far-zone basis computations: {n_basis} (series {c['series']}, inward integration {c['inward_integration']})"]
    if recs:
        Mw = np.array([r["Momega"] for r in recs])
        nfev = np.array([r["rhs_evaluations"] for r in recs])
        secs = np.array([r["seconds"] for r in recs])
        xs = sorted({round(r["x_ext"], 12) for r in recs})
        lines.append(f"  inward integrations: M*omega in [{Mw.min():.4g}, {Mw.max():.4g}], x_ext values {xs}")
        lines.append(f"  cost: {nfev.sum()} RHS evaluations in total (median {int(np.median(nfev))} per call), "
                     f"{secs.sum():.2f} s in total (max {secs.max():.3f} s per call)")
        decades = np.floor(np.log10(Mw)).astype(int)
        per_decade = ", ".join(f"[1e{d}, 1e{d + 1}): {int(np.sum(decades == d))}" for d in sorted(set(decades.tolist())))
        lines.append(f"  inward integrations per decade of M*omega: {per_decade}")
    lines.append(f"Other RW integrations: reference_gbf_ode {c['reference_gbf_ode']}, exact_u_at_x {c['exact_u_at_x']}")
    return "\n".join(lines)


def write_basis_tracking_csv(path):
    """Write one row per inward basis integration (in call order) and return the number of rows written."""
    recs = _BASIS_TRACKING["integrations"]
    with open(path, "w", newline = "") as f:
        writer = csv.writer(f)
        writer.writerow(["call_index", "Momega", "x_ext", "omega_r_ext", "rho_match", "rhs_evaluations", "seconds"])
        for i, r in enumerate(recs):
            writer.writerow([i, f"{r['Momega']:.10e}", f"{r['x_ext']:.10f}", f"{r['omega_r_ext']:.6e}",
                             f"{r['rho_match']:.6e}", r["rhs_evaluations"], f"{r['seconds']:.4f}"])
    return len(recs)


# Extraction-point quantities used by the PINN 

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
        _BASIS_TRACKING["counts"]["series"] += 1
        return complex(u1), complex(-du1_dy), {"method": "series", "series_terms": n_terms, "series_tail": tail}
    t_start = time.perf_counter()

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

    record = {"Momega": Momega, "x_ext": x_ext, "omega_r_ext": k*rho_ext, "rho_match": rho_m,
              "rhs_evaluations": int(sol.nfev), "seconds": time.perf_counter() - t_start}
    _BASIS_TRACKING["counts"]["inward_integration"] += 1
    _BASIS_TRACKING["integrations"].append(record)
    if _BASIS_TRACKING["verbose"]:
        print(f"[far_zone_basis] inward integration #{_BASIS_TRACKING['counts']['inward_integration']}: "
              f"M*omega = {Momega:.6g}, x_ext = {x_ext:g} (omega r = {k*rho_ext:.3g}), from rho = {rho_m:.4g}, "
              f"{int(sol.nfev)} RHS evaluations, {record['seconds']:.3f} s", flush = True)
    return complex(u1_e), complex(du1_dx), {"method": "inward_integration", "series_terms": n_terms,
                                            "series_tail": tail, "rho_match": rho_m,
                                            "rhs_evaluations": int(sol.nfev)}


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

    With G = x y^2/(2 i k) (= g(x)/(2 i omega) in the PINN's units where g(x) is dx/dr_*) and primes denoting d/dx:
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

def scaled_flux_error(alpha, beta):
    """(|alpha|^2 - |beta|^2 - 1)/(|alpha|^2 + |beta|^2): flux-identity error relative to the size of the terms."""
    a2, b2 = abs(alpha)**2, abs(beta)**2
    return (a2 - b2 - 1.0)/(a2 + b2)


def exact_u_at_x(x_ext, Momega, mode, rhostar_start = -40.0, rtol = 1e-12, atol = 1e-30):
    """Exact horizon-normalised u(x_ext) and du/dx(x_ext): what a perfectly trained PINN would output."""
    _BASIS_TRACKING["counts"]["exact_u_at_x"] += 1
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

