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
import bisect
import csv
import json
from datetime import datetime
import math
from scipy.special import lambertw

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "text.usetex": False
})

#Argument parser
#mode is value of l, omega is value of omega and check is included to check system initialises correctly
parser = argparse.ArgumentParser(description = "Train PINN for specific mode l")
parser.add_argument('--mass', type = float, required = True, help = "Black hole mass")
parser.add_argument('--mode', type = int, required = True, help = 'The value of l (mode)')
parser.add_argument('--x_extract', type = float, required = True, help = "GBF extraction coordinate")
parser.add_argument('--omega_start', type = float, default = 0.3, help = 'Upper bound of frequency range')
parser.add_argument('--omega_final', type = float, default = 0.03, help = 'Lower bound of frequency range')
parser.add_argument('--query_steps', type = int, default = 200, help = 'Number of frequencies used for post-training numerical comparison')
parser.add_argument('--adam_steps', type = int, default = 36000, help = "Number of Adam Epochs")
parser.add_argument('--lbfgs_steps', type = int, default = 1000, help = 'Number of outer L-BFGS step calls')
parser.add_argument('--curriculum_steps', type = int, default = 10, help = 'Number of steps taken in the frequency curriculum')
parser.add_argument('--curriculum_start', type = float, default = 0.5, help = 'Fraction of omega_max from which the frequency curriculum starts')
parser.add_argument('--lowfreq_weight', type = float, default = 1e-3, help = 'Weight of the asymptotic low=frequency GBF loss. Set to 0 to disable')
parser.add_argument('--lowfreq_max_M_omega', type =  float, default = 0.1, help = 'Largest  M*omega for which the asymptotic GBF loss will be applied')
parser.add_argument('--lowfreq_points', type = int, default = 64, help = 'Number of fixed log-spaced frequencies for the low-frequency GBF loss')

args = parser.parse_args()
mass = args.mass
mode = args.mode
x_max = args.x_extract

if args.omega_start <= args.omega_final:
    raise ValueError("omega_final must be strictly less than omega_start")

if mass <= 0:
    raise ValueError("mass must be positive.")

if mode < 0:
    raise ValueError("mode must be non-negative.")

if mode < 2:
    raise ValueError("mode must be >= 2 as this PINN is setup for spin-2 perturbations")

if args.omega_start <= 0 or args.omega_final <= 0:
    raise ValueError("Both frequencies must be positive.")

if args.adam_steps < 2:
    raise ValueError("adam_steps must be at least 2.")

if args.lbfgs_steps < 1:
    raise ValueError("lbfgs_steps must be at least 1.")

if args.query_steps < 1:
    raise ValueError("query_steps must be at least 1.")

if args.lowfreq_weight < 0:
    raise ValueError("lowfreq_weight must be non-negative.")

if args.lowfreq_max_M_omega <= 0:
    raise ValueError('lowfreq_max_M_omega must be > 0.')

if args.lowfreq_points < 2:
    raise ValueError("lowfreq_points must be at least 2.")

if not (1e-6 < x_max <= 1.0 - 1e-3):
    raise ValueError("x_extract must be in (1e-6, 0.999] for the current ansatz clamp.")

#Frequency schedule from high to low for continuation
omega_min = float(args.omega_final)
omega_max = float(args.omega_start)
diagnostic_omegas = [omega_min, 0.5*(omega_min + omega_max), float(omega_max)]
monitor_omega = diagnostic_omegas[1]
x_diagnostic = np.linspace(1e-6, x_max, 5000)

# mode = 2
# omega = 0.3

#Global data type
DTYPE = t.float64
NP_DTYPE = np.float32 if DTYPE == t.float32 else np.float64

#GPU capabilities
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
t.set_num_threads(1)
print(f"Using device: {device}", flush = True)

#Set up domain and BH mass
epsilon = 1e-8
# rstar_max = r_to_rstar(x_to_r(x_max, mass), mass)
# rstar_max_tensor = t.tensor(rstar_max, requires_grad = True, dtype = DTYPE, device = device).view(-1, 1)

#Useful quantities
# O = 4*mass*omega
# L = mode*(mode + 1)

#Old activation functions
# class cornell_adaptive_tanh(nn.Module):
#     def __init__(self, num_features):
#         super().__init__()
#         self.beta = nn.Parameter(t.zeros(1, num_features))

#     def forward(self, x):
#         return (1 + self.beta*x)*t.tanh(x)

# class jagtap_adaptive_tanh(nn.Module):
#     def __init__(self, num_features, n = 10.0):
#         super().__init__()
#         self.a = nn.Parameter(t.ones(1, num_features)/n)
#         self.n = n

#     def forward(self, x):
#         return t.tanh(self.n*self.a*x)

# class adaptive_sine(nn.Module):
#     def __init__(self, num_features)
#         super().__init__()
#         self.a = nn.Parameter(t.ones(1, num_features))

#     def forward(self, x):
#         return t.sin(self.a*x)

#Activation Function
#Adaptive tanh function of the form (1 + beta*x)*tanh(n*a*x) where beta and a are trainable parameters
class keerie_adaptive_tanh(nn.Module):
    def __init__(self, num_features, n = 10.0):
        super().__init__()
        self.beta = nn.Parameter(t.zeros(1, num_features))
        self.a = nn.Parameter(t.ones(1, num_features)/n)
        self.n = n

    def forward(self, x):
        return (1 + self.beta*x)*t.tanh(self.n*self.a*x)

#PINN architecture
class Model(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_hidden_layers=2):
        super().__init__() 
        self.input_layer = nn.Linear(in_channels, hidden_channels)
        self.act_input = keerie_adaptive_tanh(hidden_channels)

        self.hidden_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_channels, hidden_channels))
            self.activations.append(keerie_adaptive_tanh(hidden_channels))

        self.output_layer = nn.Linear(hidden_channels, out_channels) 

    def forward(self, x: t.tensor):
        x = self.input_layer(x)
        x = self.act_input(x)

        for layer, act in zip(self.hidden_layers, self.activations):
            x = layer(x)
            x = act(x)

        x = self.output_layer(x) 
        return x 

#Define various functions
#Derivative functions
def grads(y, x):
        dy = t.autograd.grad(y, x, t.ones_like(y), create_graph = True)[0]
        d2y = t.autograd.grad(dy, x, t.ones_like(dy), create_graph = True)[0]
        return dy, d2y

def first_grad(y, x):
    dy = t.autograd.grad(y, x, t.ones_like(y), create_graph = True)[0]
    return dy

def eval_grad(y, x):
    return t.autograd.grad(y, x, t.ones_like(y), create_graph=False, retain_graph = True)[0]

#ODE Coefficients
#ODE of the form A(x)u'' + B(x)u' + C(x)u = 0 where primes denote derivatives wrt x
def A(x):
    A = x*(1 - x)**2
    return A

def B(x, M, omega):
    real = (1 - x)*(1 - 3*x)
    imag = -4*M*omega
    return real, imag
        
def C(x, l):
    return -l*(l + 1) + 3*(1 - x)

#dx/dr_star
def g(x, M):
    return x*(1 - x)**2/(2*M)

#Coefficients obtained via Taylor expansion of u_1 at x = 0 (regular singular point)
def taylor_coeffs(mass, mode, omega):
    Lambda = mode*(mode + 1)

    if not t.is_tensor(omega):
        omega = t.as_tensor(omega, dtype = DTYPE, device = device)

    Omega = 4*mass*omega
    c1 = (Lambda - 3)/(1 - 1j*Omega)
    c2 = ((Lambda + 1)*c1 + 3)/(4 - 1j*2*Omega)
    c3 = (Lambda + 9)*c2/(6 + 3*(1 - 1j*Omega))
    c4 = ((Lambda + 21)*c3 - 5*c2)/(12 + 4*(1 - 1j*Omega))
    return c1.real, c1.imag, c2.real, c2.imag, c3.real, c3.imag, c4.real, c4.imag

def static_solution_coeffs(mass, mode):
    L = mode*(mode + 1)
    a = {mode + 1: 1.0} #Normalised coefficient of maximal power in the static solution polynomial - order l + 1 polynomial. Choose a_{l + 1} = 1 and then build other coeffs via recurrence relation
    for j in range(mode + 1, 3, -1): #Maximum power is always l + 1 so minimum power is 3 as l>=2
        a[j - 1] = (2*mass*(j - 3)*(j + 1))/((j - 1)*(j - 2) - L)*a[j]
    norm = sum(a.values())
    return {k: v/norm for k, v in a.items()}

static_coeffs = static_solution_coeffs(mass, mode)
C_L = float(np.prod(np.arange(1, 2*mode + 2, 2)))**(1/(mode + 1)) #((2l+1)!!)^(1/(l+1))

#Flux loss annealing
def annealing(epoch, total_epochs):
    # lambda_initial = 10.0
    # lambda_final = 1.0

    # progress = epoch/(total_epochs - 1)
    # lambda_flux = lambda_final + 0.5*(lambda_initial - lambda_final)*(1 + np.cos(np.pi*progress))
    lambda_flux = 1.0
    return lambda_flux

#Omega training input helper functions
def scale_omega(omega):
    return 2*(t.log(omega) - np.log(omega_min))/(np.log(omega_max) - np.log(omega_min)) - 1

def prepare_omega_tensor(omega, reference_tensor):

    if t.is_tensor(omega):
        omega_tensor = omega.to(device = reference_tensor.device, dtype = reference_tensor.dtype)

        if omega_tensor.ndim == 0:
            omega_tensor = omega_tensor.expand_as(reference_tensor)

        elif omega_tensor.shape != reference_tensor.shape:
            if omega_tensor.numel() == reference_tensor.numel():
                omega_tensor = omega_tensor.reshape_as(reference_tensor)

            else:
                raise ValueError(f"omega shape {omega_tensor.shape} does not match x shape {reference_tensor.shape}")

    else:
        omega_tensor = t.full_like(reference_tensor, float(omega))

    return omega_tensor

print(f"Training frequencies: 60% uniform + 40% Beta(0.6, 2.0) on [{omega_min:.6f}, {omega_max:.6f}]", flush = True)

beta_dist = dist.Beta(t.tensor([0.6], dtype = DTYPE, device = device), t.tensor([0.1], dtype = DTYPE, device = device))
omega_beta_dist = dist.Beta(t.tensor([0.6], dtype = DTYPE, device = device), t.tensor([2.0], dtype = DTYPE, device = device))

def sample_x_points(n_points, x_max, dtype, device):
    n_uniform = int(0.6*n_points)
    n_edges = n_points - n_uniform

    x_uniform = x_max*t.rand((n_uniform, 1), dtype = dtype, device = device)
    x_edges = x_max*beta_dist.sample((n_edges,)).view(-1, 1).to(device = device, dtype = dtype)

    return t.cat([x_uniform, x_edges], dim = 0)

def sample_omega_points(n_points, omega_lower, dtype, device):
    n_uniform = int(0.6 * n_points)
    n_beta = n_points - n_uniform

    omega_uniform = omega_lower + (omega_max - omega_lower)*t.rand((n_uniform, 1), dtype=dtype, device=device)

    beta_samples = omega_beta_dist.sample((n_beta,)).reshape(-1, 1)
    omega_beta = omega_lower + (omega_max - omega_lower)*beta_samples

    omega_samples = t.cat([omega_uniform, omega_beta], dim=0)

    permutation = t.randperm(n_points, device=device)
    return omega_samples[permutation]

def sample_joint_batch(n_points, x_max, dtype, device, rar_x = None, rar_omega = None, omega_lower = None):

    if omega_lower is None:
        omega_lower = omega_min

    x_base = sample_x_points(n_points = n_points, x_max = x_max, dtype = dtype, device = device)

    omega_base = sample_omega_points(n_points = n_points, omega_lower = omega_lower, dtype = dtype, device = device)

    if rar_x is not None and rar_omega is not None and rar_x.numel() > 0:
        x_tensor = t.cat([x_base, rar_x.to(device = device, dtype = dtype)], dim = 0)

        omega_tensor = t.cat([omega_base, rar_omega.to(device = device, dtype = dtype)], dim = 0)

    else:
        x_tensor = x_base
        omega_tensor = omega_base

    x_tensor.requires_grad_(True)

    return x_tensor, omega_tensor

def ode_residual_score(model, x_tensor, mass, mode,  omega, chunk_size = 2000):
    """Evaluate |R_re|^2 + |R_im|^2 on candidate points.
    The candidate points are processed  in chunks to limit memory usage as second derivatives are involved.
    """

    omega_tensor = prepare_omega_tensor(omega, x_tensor)

    scores = []
    for x_chunk, omega_chunk in zip(x_tensor.split(chunk_size), omega_tensor.split(chunk_size)):
        x_chunk = x_chunk.detach().clone().requires_grad_(True)

        u_re, u_im, *_ = ansatz(model, x_chunk, mass, mode, omega_chunk)

        du_re, d2u_re = grads(u_re, x_chunk)
        du_im, d2u_im = grads(u_im, x_chunk)

        A_ = A(x_chunk)
        B_re, B_im = B(x_chunk, mass, omega_chunk)
        C_ = C(x_chunk, mode)

        res_re = A_*d2u_re + B_re*du_re - B_im*du_im + C_*u_re
        res_im = A_*d2u_im + B_im*du_re + B_re*du_im + C_*u_im
        score = (res_re**2 + res_im**2).flatten().detach()

        scores.append(score)

    return t.cat(scores, dim = 0)

def residual_fields(model, x_tensor, mass, mode, omega, chunk_size = 2000):

    omega_tensor = prepare_omega_tensor(omega, x_tensor)

    ode_squared_chunks = []
    flux_chunks = []

    for x_chunk, omega_chunk in zip(x_tensor.split(chunk_size), omega_tensor.split(chunk_size)):

        x_chunk = x_chunk.detach().clone().requires_grad_(True)
        omega_chunk = omega_chunk.detach()

        u_re, u_im, *_ = ansatz(model, x_chunk, mass, mode, omega_chunk)

        du_re, d2u_re = grads(u_re, x_chunk)
        du_im, d2u_im = grads(u_im, x_chunk)

        A_ = A(x_chunk)
        B_re, B_im = B(x_chunk, mass, omega_chunk)
        C_ = C(x_chunk, mode)

        res_re = A_*d2u_re + B_re*du_re - B_im*du_im + C_*u_re
        res_im = A_*d2u_im + B_re*du_im + B_im*du_re +C_*u_im

        ode_squared = res_re**2 + res_im**2

        J = g(x_chunk, mass)*(u_re*du_im - u_im*du_re) - omega_chunk*(u_re**2 + u_im**2 - 1)

        ode_squared_chunks.append(ode_squared.flatten().detach())
        flux_chunks.append(J.flatten().detach())

    return t.cat(ode_squared_chunks, dim = 0), t.cat(flux_chunks, dim = 0)

def select_rar_joint_points(model, n_candidates, n_add, x_max, mass, mode, device, dtype, existing_rar_x = None, existing_rar_omega = None, min_joint_distance = 0.01, omega_lower = None):

    if omega_lower is None:
        omega_lower = omega_min

    empty = t.empty((0, 1), dtype = dtype, device = device)

    if n_add <= 0:
        return empty, empty, t.empty((0,), dtype = dtype, device = device)

    x_candidates = sample_x_points(n_candidates, x_max, dtype = dtype, device = device)

    omega_candidates = omega_lower + (omega_max - omega_lower)*t.rand((n_candidates, 1), dtype = dtype, device = device)

    scores = ode_residual_score(model, x_candidates, mass, mode, omega_candidates)

    candidate_x_np = x_candidates.flatten().detach().cpu().numpy()
    candidate_omega_np = omega_candidates.flatten().detach().cpu().numpy()

    log_omega_span = np.log(omega_max) - np.log(omega_min)
    candidate_points = np.column_stack((candidate_x_np/x_max, (np.log(candidate_omega_np) - np.log(omega_min))/log_omega_span))

    if existing_rar_x is not None and existing_rar_omega is not None and existing_rar_x.numel() > 0:
        old_x = existing_rar_x.flatten().detach().cpu().numpy()
        old_omega = existing_rar_omega.flatten().detach().cpu().numpy()

        occupied_points = np.column_stack((old_x/x_max, (np.log(old_omega) - np.log(omega_min))/log_omega_span))

    else:
        occupied_points = np.empty((0, 2), dtype = float)

    ranked_indices = t.argsort(scores, descending = True).cpu().numpy()
    accepted_indices = []

    for idx in ranked_indices:
        point = candidate_points[idx]

        if occupied_points.shape[0] == 0:
            nearest_distance = np.inf

        else:
            distances = np.linalg.norm(occupied_points - point, axis = 1)
            nearest_distance = float(np.min(distances))

        if nearest_distance >= min_joint_distance:
            accepted_indices.append(int(idx))
            occupied_points = np.vstack((occupied_points, point))

        if len(accepted_indices) >= n_add:
            break

    selected = t.as_tensor(accepted_indices, dtype = t.long, device = x_candidates.device)

    new_x = x_candidates[selected].detach()
    new_omega = omega_candidates[selected].detach()
    new_scores = scores[selected].detach()

    return new_x, new_omega, new_scores

def smoothstep(z):
    z = float(np.clip(z, 0.0, 1.0))
    return z*z*(3.0 - 2.0*z) #slight increasing curve between z = 0 and z = 1

#Ansatz for the wave-function u
#Ansatz is of the form u(x) = 1 + c1*x + c2*x**2 + (P + exp(2i*omega*r_star)*Q) 
#where P and Q are complex with components corresponding to the four channel neural network output
def ansatz(model, x_tensor, mass, mode, omega):

    omega_tensor = prepare_omega_tensor(omega, x_tensor)

    c1_re, c1_im, c2_re, c2_im, c3_re, c3_im, c4_re, c4_im = taylor_coeffs(mass, mode, omega_tensor)

    omega_scaled = scale_omega(omega_tensor)

    NN_input = t.cat([x_tensor, omega_scaled], dim = 1)
    NN = model(NN_input)

    P_re, P_im, Q_re, Q_im = NN[:, 0:1], NN[:, 1:2], NN[:, 2:3], NN[:, 3:4]
    x_safe = x_tensor.clamp(min = 1e-12, max = 1 - 1e-3)
    rstar = 2*mass/(1 - x_safe) + 2*mass*t.log(x_safe/(1 - x_safe))
    cs, sn = t.cos(2*omega_tensor*rstar), t.sin(2*omega_tensor*rstar)

    u_re = 1 + c1_re*x_tensor + c2_re*x_tensor**2 + c3_re*x_tensor**3 + c4_re*x_tensor**4 + x_tensor**5*(P_re + Q_re*cs - Q_im*sn)
    u_im = c1_im*x_tensor + c2_im*x_tensor**2 + c3_im*x_tensor**3 + c4_im*x_tensor**4 + x_tensor**5*(P_im + Q_im*cs + Q_re*sn)
    return u_re, u_im, P_re, P_im, Q_re, Q_im

#Coefficient of low-frequency limit of GBF (in Finnian Gray paper \Gamma \approx C_{l,s} (M\omega)^{2l+2}) with C_{l,2} given by function below 
def low_frequency_GBF_coeff(mode):

    if mode < 2:
        raise ValueError("l >= 2 is requierd for spin-2 perturbations")
    
    numerator = 2**(2*mode + 2)*math.factorial(mode - 2)*math.factorial(mode)*math.factorial(mode + 2)
    denominator = math.factorial(2*mode)*math.factorial(2*mode + 1)
    return (numerator/denominator)**2

def low_frequency_limit_GBF(mass, mode, omega):
    dimensionless_omega = mass*omega
    C = low_frequency_GBF_coeff(mode)
    return C*(dimensionless_omega)**(2*mode + 2)

def differentiable_lowfreq_gbf(model, mass, mode,  omega_values, x_extract):

    omega_values =  omega_values.to(device = device, dtype = DTYPE).reshape(-1, 1)
    x_values = t.full_like(omega_values, float(x_extract), requires_grad = True)
    u_re, u_im, *_ = ansatz(model, x_values, mass, mode, omega_values)
    du_re = first_grad(u_re, x_values)
    du_im = first_grad(u_im, x_values)

    u = t.complex(u_re.squeeze(-1), u_im.squeeze(-1))
    du = t.complex(du_re.squeeze(-1), du_im.squeeze(-1))
    om = omega_values.squeeze(-1)

    L = mode*(mode + 1)
    Omega = 4.0*mass*om

    a1 = -1j*L/Omega
    a2 = -(3.0 + (2.0 - L)*a1)/(1j*2.0*Omega)

    y = 1.0 - float(x_extract)
    x = float(x_extract)
    u1 = 1.0 + a1*y +  a2*y**2
    du1  = -(a1 + 2.0*a2*y)

    D = 1j*Omega/(y**2*x) + t.conj(du1/u1)
    alpha = (u - du/D)/(u1 - du1/D)
    GBF = 1.0/(alpha.real.square() + alpha.imag.square())
    return GBF.reshape(-1, 1)

lowfreq_omega_max = min(omega_max, args.lowfreq_max_M_omega/mass)

if args.lowfreq_weight > 0 and lowfreq_omega_max > omega_min:
    lowfreq_omega = t.logspace(math.log10(omega_min), math.log10(lowfreq_omega_max), steps = args.lowfreq_points, dtype = DTYPE, device = device).reshape(-1, 1)
    lowfreq_gbf_target = low_frequency_limit_GBF(mass, mode, lowfreq_omega)
    print(f"Low-frequency GBF loss enabled: weight = {args.lowfreq_weight:g}, omega in [{omega_min:6g}, {lowfreq_omega_max}].", flush = True)
    print(f"M*omega <= {mass*lowfreq_omega_max:6g}, points = {args.lowfreq_points}", flush = True)

else:
    lowfreq_omega = t.empty((0, 1), dtype = DTYPE, device = device)
    lowfreq_gbf_target = t.empty((0, 1), dtype = DTYPE, device = device)
    if args.lowfreq_weight > 0:
        print("WARNING: Low-frequency GBF loss disabled because the training interval does not overlap the requested M*omega limit.")

def compute_lowfreq_gbf_loss(model, mass, mode):

    if args.lowfreq_weight <= 0 or lowfreq_omega.numel() == 0:
        return t.zeros((), dtype = DTYPE, device = device)

    gbf_pred = differentiable_lowfreq_gbf(model, mass, mode, lowfreq_omega, x_max)

    tiny = t.finfo(DTYPE).tiny
    log_pred = t.log(gbf_pred.clamp_min(tiny))
    log_target = t.log(lowfreq_gbf_target.clamp_min(tiny))

    return t.mean((log_pred - log_target)**2)

#Current loss function composed of ODE residual and Flux conservation requirement
def compute_loss(model, x_tensor, mass, mode, omega, flux_weight):

    #ODE residual
    u_re, u_im, P_re, P_im, Q_re, Q_im = ansatz(model, x_tensor, mass, mode, omega)
    
    du_re, d2u_re = grads(u_re, x_tensor)
    du_im, d2u_im = grads(u_im, x_tensor)

    A_ = A(x_tensor)
    B_re, B_im = B(x_tensor, mass, omega)
    C_ = C(x_tensor, mode)

    res_ode_re = (A_*d2u_re + B_re*du_re - B_im*du_im + C_*u_re)
    res_ode_im = (A_*d2u_im + B_im*du_re + B_re*du_im + C_*u_im)

    loss_ode_re = t.mean(res_ode_re**2)
    loss_ode_im = t.mean(res_ode_im**2)
    loss_ode = loss_ode_re + loss_ode_im

    #Flux conservation- included to prevent the network from setting u = 0
    J = g(x_tensor, mass)*(u_re*du_im - u_im*du_re) - omega*(u_re**2 + u_im**2 - 1) # should = 0 analytically
    loss_flux = t.mean(J**2)

    active = lowfreq_omega[:, 0] >= omega_low

    if not active.any():
        loss_lowfreq = t.zeros((), dtype = DTYPE, device = device)
    else:
        omega_active = lowfreq_omega[active]
        target_active = lowfreq_gbf_target[active]

        gbf_pred = differentiable_lowfreq_gbf(model, mass, mode, omega_active, x_max)
        loss_lowfreq = t.mean((t.log(gbf_pred.clamp_min(t.finfo(DTYPE).tiny)) - t.log(target_active.clamp_min(t.finfo(DTYPE).tiny)))**2)

    #Loss annealing to be included
    total_loss = loss_ode + flux_weight*loss_flux + args.lowfreq_weight*loss_lowfreq

    return u_re, u_im, J, total_loss, loss_flux, loss_lowfreq, loss_ode, loss_ode_re, loss_ode_im, res_ode_re, res_ode_im, P_re, P_im, Q_re, Q_im

#Calculates GBF via alpha = (u - u'/D)/(u1 - u1'/D) with D the log derivative of the u2 branch
#u1 is obtained via an asymptotic expansion and u is the wavefunction the network approximates
#See paper for full derivation (needs to be made more rigorous)
def extraction(model, x_extraction, mass, mode, omega):

    L = mode*(mode + 1)
    Omega = 4*mass*omega
    
    x_extraction_tensor = t.tensor(x_extraction, requires_grad = True, dtype = DTYPE, device = device).view(-1, 1) 

    u_max_re, u_max_im, *_ = ansatz(model, x_extraction_tensor, mass, mode, omega)

    #Eval grad used to prevent creation of a graph within the extraction function
    du_max_re  = eval_grad(u_max_re, x_extraction_tensor)
    du_max_im  = eval_grad(u_max_im, x_extraction_tensor)

    u_max = complex(u_max_re.item(), u_max_im.item())
    du_max = complex(du_max_re.item(), du_max_im.item())

    #Compute more terms in expansion??
    a1 = -1j*L/Omega
    a2 = -(3 + (2 - L)*a1)/(1j*2*Omega)

    y_extraction = 1 - x_extraction
    rstar_extraction = 2*mass/y_extraction + 2*mass*np.log(x_extraction/y_extraction)
    u1 = 1 + a1*y_extraction + a2*y_extraction**2
    du1 = -(a1 + 2*a2*y_extraction)

    D = 1j*Omega/(y_extraction**2*x_extraction) + np.conj(du1/u1)
    numerator = u_max - du_max/D
    denominator = u1 - du1/D
    alpha = numerator/denominator
    beta = (u_max - alpha*u1)/(np.exp(1j*2*omega*rstar_extraction)*np.conj(u1))

    #Not technically a probability but instead related to the Wronskian and flux conservation at the extraction point
    prob = np.abs(alpha)**2 - np.abs(beta)**2
    gbf = 1/np.abs(alpha)**2

    return alpha, beta, prob, gbf

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
        #Asymptotic expansion of W(exp(z))
        w_large = z_large - np.log(z_large) + np.log(z_large)/z_large

        for _ in range(8):
            residual = w_large + np.log(w_large) - z_large
            w_large -= residual*w_large/(w_large + 1.0)
        w[large] = w_large

    return (1.0 + w).reshape(original_shape)

def RW_potential(u, mode):
    #Dimensionless potential- see appendix B of Gray and Visser paper 
    return (1.0 - 1.0/u)*(mode*(mode + 1)/u**2 -3/u**3)

def negative_integrand_values(ustar_nodes, Momega, mode):
    """Return samples of -A(u*) from eq. (4.13) of Gray & Visser.

    A(u*) = -i V(u*)/(4Momega) [[1, exp(-4iMomegau*)], [-exp(4iMomegau*), -1]].

    Sign is reversed because the Helton-Stuckwisch recursion used in eq. (3.14) has the opposite product-order converntion.
    Applying this to -A approximates the inverse of the desired transfer matrix (eq. 3.11) containing the Bogoliubov coefficients.
    """
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
    """Calculates the GBF for dimensionless frequency x = M*omega using a fifth order path-ordered exponential/product integral approach.
        Parameters: 
            x: float, dimensionless frequency x = M*omega
            mode: int, multipole mode l (>=2 here)
            ustar_min, ustar_max: float, finite approximation of tortoise coordinate limits- paper uses [-50, 350]
            N: int, number of product-integral subintervals (not grid points). Each interval samples A at five equally spaced nodes as in eq. (3.13)

        Returns:
            T: float, transmission probability i.e. GBF
            alpha: complex, Bogoliubov coefficient alpha(x)
            beta: complex, Bogoliubov coefficient beta(x)
        
        The Helton-Stuckwisch approximation computes the inverse transfer matrix H = [[alpha*, -beta*], [-beta, alpha]] so alpha = conj(H[0, 0]) and beta = -H[1, 0]
    """
    x = float(x)
    mode = int(mode)
    N = int(N)
    ustar_min = float(ustar_min)
    ustar_max = float(ustar_max)

    if not np.isfinite(x) or x <= 0.0:
        raise ValueError("x must be finite, positive, dimensionless frequency M*omega")

    if mode < 2:
        raise ValueError("mode must be >= 2 for the spin-2 potential used here")

    if N < 1:
        raise ValueError("N must be a positive number of subintervals")

    if not (np.isfinite(ustar_min) and np.isfinite(ustar_max) and ustar_max > ustar_min):
        raise ValueError("Require finite bounds with ustar_max > ustar_min")

    h = (ustar_max - ustar_min)/N
    #N intervals, with A_j evaluated at u*_{k - 1} + j h/4, j = 0,..., 4
    #Thus 4*N + 1 nodes and no extra endpoint step.
    ustar_nodes = ustar_min + (h/4.0)*np.arange(4*N + 1)
    A_nodes = negative_integrand_values(ustar_nodes, x, mode)

    I = np.eye(2, dtype = np.complex128)
    H = I.copy() #accumulates the inverse transfer matrix by right multiplication

    for k in range(N):
        A0, A1, A2, A3, A4 = A_nodes[4*k:4*k + 5]

        #Helton-Stuckwisch fifth-order product integral recursion (Eq. 3.14)
        K1 = h*A4
        K2 = h*(A3 @ (4.0*I + K1))
        K3 = h*(A2 @ (8.0*(I - K1) + 3.0*K2))
        K4 = h*(A1 @ (32.0*I + 18.0*K1 + 3.0*h*(A2 @ (6.0*I - K1 + K2)) - 3.0*K2))
        K5 = h*(A0 @ (28.0*(I - K1) - 3*h*(A2 @ (16.0*I + 4.0*K1 + K2)) + K4 + 18.0*K2))

        step = I + (28.0*K1 + 32.0*K2 + 6.0*K3 + 4.0*K4 + K5)/360.0
        H = H @ step

    alpha = np.conjugate(H[0, 0])
    beta = - H[1, 0]
    T = float(1.0/(np.abs(alpha)**2))
    return T, complex(alpha), complex(beta)

def product_integral_convergence(x, mode, N_values = (500, 1000, 2000, 4000, 8000), ustar_min = -50.0, ustar_max = 350.0):

    rows = []
    previous_T = None

    for n in N_values:
        T, alpha, beta = product_integral_GBF(x, mode, ustar_min = ustar_min, ustar_max = ustar_max, N = int(n))
        wronskian = np.abs(alpha)**2 - np.abs(beta)**2
        relative_change = np.nan if previous_T is None else np.abs(T - previous_T)/max(abs(T), np.finfo(float).tiny)

        rows.append({"N": int(n), "GBF": T, "alpha": alpha, "beta": beta, "wronskian": float(wronskian), "wronskian_error": float(abs(wronskian - 1.0)), "relative_change_from_previous_N": float(relative_change)})
        previous_T = T

    return rows

def query_gbf(model, mass, mode, omega_query, x_extract):

    if np.isscalar(omega_query):
        alpha, beta, prob, gbf = extraction(model, x_extract, mass, mode, float(omega_query))
        return alpha, beta, prob, gbf

    else:
        results = []

        for omega in np.asarray(omega_query).flatten():
            alpha, beta, prob, gbf = extraction(model, x_extract, mass, mode, float(omega))
            results.append({'omega': float(omega), 'alpha': alpha, 'beta': beta, 'probability': prob, 'gbf': gbf})
    
    return results

def query_wavefunction(model, mass, mode, omega_query, x_values):

    if not t.is_tensor(x_values):
        x_tensor = t.as_tensor(x_values, dtype = DTYPE, device = device).reshape(-1, 1)
        x_tensor.requires_grad_(True)

    else:
        x_tensor = x_values
        x_tensor.requires_grad_(True)

    u_re, u_im, P_re, P_im, Q_re, Q_im = ansatz(model, x_tensor, mass, mode, float(omega_query))

    return (x_tensor.detach().cpu().numpy().flatten(), u_re.detach().cpu().numpy().flatten(), u_im.detach().cpu().numpy().flatten(),
            P_re.detach().cpu().numpy().flatten(), P_im.detach().cpu().numpy().flatten(), Q_re.detach().cpu().numpy().flatten(),
            Q_im.detach().cpu().numpy().flatten())

def evaluate_fixed_frequency(model, mass, mode, omega, x_values):

    if not t.is_tensor(x_values):
        x_tensor = t.as_tensor(x_values, dtype = DTYPE, device = device).reshape(-1, 1)
        x_tensor.requires_grad_(True)

    else:
        x_tensor = x_values
        x_tensor.requires_grad_(True)

    u_re, u_im, J, total_loss, loss_flux, loss_lowfreq, loss_ode, loss_ode_re, loss_ode_im, res_ode_re, res_ode_im, P_re, P_im, Q_re, Q_im = compute_loss(model, x_tensor, mass, mode, float(omega), flux_weight = 0.0)

    return {'x': x_tensor.detach().cpu().numpy().flatten(), 'u_re': u_re.detach().cpu().numpy().flatten(), 'u_im': u_im.detach().cpu().numpy().flatten(), 'flux': J.detach().cpu().numpy().flatten(),
            'res_re': res_ode_re.detach().cpu().numpy().flatten(), 'res_im': res_ode_im.detach().cpu().numpy().flatten()}

def save_training_diagnostics(model, epoch_number, omega):

    plt.figure(figsize = [7, 5])
    plt.plot(hist_epochs, hist_total, label = 'Total')
    plt.plot(hist_epochs, hist_ode, label = 'ODE')
    plt.plot(hist_epochs, hist_flux, label = 'Flux')
    plt.plot(hist_epochs, hist_ode_re, label = 'Real ODE')
    plt.plot(hist_epochs, hist_ode_im, label = 'Imag ODE')
    plt.plot(hist_epochs, hist_lowfreq, label = 'Low frequency')
    plt.yscale('log')
    plt.xlabel('Epoch', fontsize = 16)
    plt.ylabel('Loss', fontsize = 16)
    plt.title(f'Two-input PINN loss, l = {mode}', fontsize = 16)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(diagnostic_dir, 'latest_loss.png'), format = 'png')
    plt.close()

    diagnostic_dir_omega = os.path.join(diagnostic_dir, f'omega{omega:.4f}')
    os.makedirs(diagnostic_dir_omega, exist_ok = True)

    diagnostic = evaluate_fixed_frequency(model, mass, mode, omega, x_diagnostic)

    flux_dir_omega = os.path.join(diagnostic_dir_omega, "Flux")
    residual_dir_omega = os.path.join(diagnostic_dir_omega, "Residuals")
    os.makedirs(flux_dir_omega, exist_ok = True)
    os.makedirs(residual_dir_omega, exist_ok = True)

    plt.figure(figsize = [7, 5])
    plt.plot(diagnostic['x'], diagnostic['res_re'], label = r"$\Re(R_{ODE})$")
    plt.plot(diagnostic['x'], diagnostic['res_im'], label = r'$\Im(R_{ODE})$')
    plt.xlabel('x', fontsize = 16)
    plt.ylabel('ODE Residual', fontsize = 16)
    plt.title(f'ODE Residual, l = {mode}, omega = {omega:.4f}')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(residual_dir_omega, f"Epoch_{epoch_number}.png"), format = 'png')
    plt.close()

    plt.figure(figsize = [7, 5])
    plt.plot(diagnostic['x'], diagnostic['flux'], label = r'$J(x)$')
    plt.axhline(0.0, linestyle = '--', label = 'Target')
    plt.xlabel('x', fontsize = 16)
    plt.ylabel('Flux Residual', fontsize = 16)
    plt.title(f'Flux Residual, l = {mode}, omega = {omega:.4f}')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(flux_dir_omega, f"Epoch_{epoch_number}.png"), format = 'png')
    plt.close()

    flux_rms = np.sqrt(np.mean(diagnostic['flux']**2))
    flux_max = np.max(np.abs(diagnostic['flux']))
    ode_rms = np.sqrt(np.mean(diagnostic['res_im']**2 + diagnostic['res_re']**2))

    print(f"Diagnostic omega= {omega:.4f} | Flux RMS = {flux_rms:.4e} | Max |Flux| = {flux_max:.4e} | ODE RMS = {ode_rms:.4e}", flush = True)

    wavefunction_dir = os.path.join(diagnostic_dir_omega, "Wavefunction")
    pq_dir = os.path.join(diagnostic_dir_omega, "PQ")

    os.makedirs(wavefunction_dir, exist_ok = True)
    os.makedirs(pq_dir, exist_ok = True)

    x_values, u_re, u_im, P_re, P_im, Q_re, Q_im = query_wavefunction(model, mass, mode, omega, x_diagnostic)

    plt.figure(figsize = [7,5])
    plt.plot(x_values, u_re, label = 'Re(u)')
    plt.plot(x_values, u_im, label = 'Im(u)')
    plt.xlabel(r'$x$', fontsize = 16)
    plt.ylabel(r"$u(x, \omega)$", fontsize = 16)
    plt.title(f"Wavefunction, l = {mode}, omega = {omega:.4f}"
            "\n"
            f"Epoch: {epoch_number}")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(wavefunction_dir, f"Epoch_{epoch_number}.png"), format = 'png')
    plt.close()

    plt.figure(figsize = [7, 5])
    plt.plot(x_values, P_re, label = 'Re(P)')
    plt.plot(x_values, P_im, label = 'Im(P)')
    plt.plot(x_values, Q_re, label = 'Re(Q)')
    plt.plot(x_values, Q_im, label = 'Im(Q)')
    plt.xlabel('x', fontsize = 16)
    plt.ylabel('P & Q', fontsize = 16)
    plt.title(f"P & Q, l = {mode}, omega = {omega:.4f}"
            "\n"
            f"Epoch: {epoch_number}")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pq_dir, f"Epoch_{epoch_number}.png"), format = 'png')
    plt.close()

def load_numerical_gbf(csv_path, mass, mode, x_extract):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Numerical GBF CSV not found: {csv_path}")

    numerical_results = {}

    with open(csv_path, 'r', newline =  '') as f:
        reader = csv.DictReader(f)

        required_columns = {"mass", "mode", "x_extract", "omega", "GBF", "success"}
        missing_columns = required_columns - set(reader.fieldnames or [])

        if missing_columns:
            raise ValueError(f"Numerical CSV is missing required columns: {sorted(missing_columns)}")

        for row in reader:

            row_mass = float(row["mass"])
            row_mode = int(row["mode"])
            row_x_extract = float(row["x_extract"])
    
            if not np.isclose(row_mass, mass, rtol = 0.0, atol = 1e-12):
                continue
    
            if row_mode != mode:
                continue

            if not np.isclose(row_x_extract, x_extract, rtol = 0.0, atol = 1e-12):
                continue

            if row["success"].strip().lower() not in {"true", "1"}:
                continue
            
            omega = float(row['omega'])
            gbf = float(row['GBF'])

            if not np.isfinite(gbf):
                continue

            numerical_results[round(omega, 8)] = gbf

    return numerical_results

def compare_pinn_to_numerical(pinn_results, numerical_results, output_csv):

    comparison_omegas = []
    pinn_gbfs = []
    numerical_gbfs = []
    absolute_differences = []
    relative_differences = []

    for result in pinn_results:

        omega = float(result['omega'])
        key = round(omega, 8)

        if key not in numerical_results:
            raise ValueError(f"No numerical result found for omega = {omega:.10f}")

        pinn_gbf = float(result['gbf'])
        numerical_gbf  = float(numerical_results[key])

        abs_diff = abs(pinn_gbf - numerical_gbf)

        if numerical_gbf != 0.0:
            rel_diff = abs_diff/abs(numerical_gbf)
        else:
            rel_diff = np.nan

        comparison_omegas.append(omega)
        pinn_gbfs.append(pinn_gbf)
        numerical_gbfs.append(numerical_gbf)
        absolute_differences.append(abs_diff)
        relative_differences.append(rel_diff)

    with open(output_csv, 'w', newline = '') as f:
        writer = csv.writer(f)

        writer.writerow(['omega', 'PINN_GBF', 'Numerical_GBF', 'Absolute_Difference', 'Relative_Difference'])

        for values in zip(comparison_omegas, pinn_gbfs, numerical_gbfs, absolute_differences, relative_differences):
            writer.writerow([f'{values[0]:.10f}', f'{values[1]:.12e}', f'{values[2]:.12e}', f'{values[3]:.12e}', f'{values[4]:.12e}' if np.isfinite(values[4]) else 'nan'])

    return np.asarray(comparison_omegas), np.asarray(pinn_gbfs), np.asarray(numerical_gbfs), np.asarray(absolute_differences), np.array(relative_differences)

def serialise_query_results(results):
    return [{'omega': float(result['omega']), 'alpha_re': float(result['alpha'].real), 'alpha_im': float(result['alpha'].imag), 'beta_re': float(result['beta'].real),
            'beta_im': float(result['beta'].imag), 'probability': float(result['probability']), 'gbf': float(result['gbf'])} for result in results]

def atomic_torch_save(obj, path):

    temp_path = path + '.tmp'
    t.save(obj, temp_path)
    os.replace(temp_path, path)

def get_cpu_model_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

def save_simulation_comparison_plots(omegas, pinn_gbfs, numerical_gbfs, absolute_differences, relative_differences, title_suffix):

    plt.figure(figsize=[14, 6])

    plt.subplot(1, 2, 1)
    plt.suptitle('Simulation vs. PINN')
    plt.plot(omegas, numerical_gbfs, label='Simulation')
    plt.plot(omegas, pinn_gbfs, label='Conditional PINN')
    plt.xlabel(r'$\omega$', fontsize=16)
    plt.ylabel(r'$\Gamma(\omega)$', fontsize=16)
    plt.title(f'GBF Comparison - {title_suffix}')
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(omegas, numerical_gbfs, label='Simulation')
    plt.plot(omegas, pinn_gbfs, label='Conditional PINN')
    plt.xlabel(r'$\omega$', fontsize=16)
    plt.ylabel(r'$\Gamma(\omega)$', fontsize=16)
    plt.yscale('log')
    plt.title(f'Log-scale GBF Comparison - {title_suffix}')
    plt.grid()
    plt.legend()    

    plt.tight_layout()
    plt.savefig(os.path.join(comparisons_dir, f'SimvsPINN_GBF.png'), format='png')
    plt.close()

    plt.figure(figsize=[14, 6])

    plt.subplot(1, 2, 1)
    plt.suptitle('Simulation vs. PINN')
    plt.plot(omegas, absolute_differences)
    plt.xlabel(r'$\omega$', fontsize=16)
    plt.ylabel('Absolute Difference', fontsize=16)
    plt.title(f'Absolute Difference — {title_suffix}')
    plt.yscale('log')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(omegas, relative_differences)
    plt.xlabel(r'$\omega$', fontsize=16)
    plt.ylabel('Relative Difference', fontsize=16)
    plt.title(f'Relative Difference — {title_suffix}')
    plt.yscale('log')
    plt.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(comparisons_dir, f'SimvsPINN_differences.png'), format='png')
    plt.close()

def plot_joint_residual_heatmap(model, epoch, mass, mode, x_max, rar_x = None, rar_omega = None, nx = 120, n_omega = 120, chunk_size = 2000):

    x_axis = np.linspace(1e-6, x_max, nx)
    omega_axis = np.linspace(omega_min, omega_max, n_omega)

    X, W = np.meshgrid(x_axis, omega_axis, indexing = 'xy')

    x_grid = t.as_tensor(X.reshape(-1, 1), dtype = DTYPE, device = device)
    omega_grid = t.as_tensor(W.reshape(-1, 1), dtype = DTYPE, device = device)

    ode_squared, flux_values = residual_fields(model = model, x_tensor = x_grid, mass = mass, mode = mode, omega = omega_grid, chunk_size = chunk_size)

    ode_magnitude = t.sqrt(t.clamp(ode_squared, min = 0.0)).cpu().numpy()
    ode_magnitude = ode_magnitude.reshape(n_omega, nx)

    flux_grid = flux_values.cpu().numpy().reshape(n_omega, nx)

    log_ode = np.log10(np.maximum(ode_magnitude, 1e-30))
    log_flux = np.log10(np.maximum(np.abs(flux_grid), 1e-30))

    fig, axes = plt.subplots(1, 2, figsize = (15, 6), sharey = True)

    h0 = axes[0].pcolormesh(X, W, log_ode, shading = 'auto', cmap = 'autumn')
    h1 = axes[1].pcolormesh(X, W, log_flux, shading = 'auto', cmap = 'summer')

    fig.colorbar(h0, ax = axes[0]).set_label(r"$\log_{10}|Res_{ODE}|$")
    fig.colorbar(h1, ax = axes[1]).set_label(r"$\log_{10}|J|$")

    if rar_x is not None and rar_omega is not None and rar_x.numel() > 0:
        rar_x_np = rar_x.detach().cpu().numpy().flatten()
        rar_omega_np = rar_omega.detach().cpu().numpy().flatten()

        for ax in axes:
            ax.scatter(rar_x_np, rar_omega_np, s = 10, facecolors = "none", edgecolors = "black", linewidths = 0.5, alpha = 0.7, label = "Retained RAR points", zorder = 3)
            ax.legend(loc = "best")

    axes[0].set_title("ODE residual magnitude")
    axes[1].set_title("Flux residual magnitude")

    for ax in axes:
        ax.set_xlabel(r"$x$", fontsize = 16)
        ax.set_xlim(x_axis[0], x_axis[-1])
        ax.set_ylim(omega_min, omega_max)

    axes[0].set_ylabel(r"$\omega$")

    fig.suptitle(rf"Residuals over $(x, \omega)$ Domain"
                "\n"
                f"l = {mode}, Epoch = {epoch}")

    fig.tight_layout()

    png_path = os.path.join(rar_dir, f"residual_heatmap_epoch_{epoch}.png")
    fig.savefig(png_path, dpi = 200, bbox_inches = "tight")
    plt.close(fig)

    npz_path = os.path.join(rar_dir, f"residual_heatmap_epoch_{epoch}.npz")

    rar_x_save = rar_x.detach().cpu().numpy().flatten() if rar_x is not None else np.empty(0)
    rar_omega_save = rar_omega.detach().cpu().numpy().flatten() if rar_omega is not None else np.empty(0)

    np.savez_compressed(npz_path, x = x_axis, omega = omega_axis, residual_magnitude = ode_magnitude, log10_residual = log_ode, flux_residual = flux_grid, log10_flux_residual = log_flux, 
                        rar_x = rar_x_save, rar_omega = rar_omega_save)

    print(f"Saved RAR residual heatmap: {png_path}", flush = True)


#Setup PINN logistics
#Seed included for reproducibility
t.manual_seed(0)

if t.cuda.is_available():
    t.cuda.manual_seed_all(0)

model = Model(2, 4, 64, num_hidden_layers = 3).to(device = device, dtype = DTYPE)
    
#Make various directories for saving results
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"{run_id}_M{mass:g}_x{x_max:g}_w{omega_min:g}-{omega_max:g}_adam{args.adam_steps}_lbfgs{args.lbfgs_steps}"

base_path = os.path.join("./GBFWXData", f"l{mode}", run_name)
diagnostic_dir = os.path.join(base_path, 'diagnostics')
results_dir = os.path.join(base_path, 'results')
comparisons_dir = os.path.join(base_path, 'comparisons')
final_plots_dir = os.path.join(diagnostic_dir, "final")
rar_dir = os.path.join(diagnostic_dir, "RAR")

os.makedirs(base_path, exist_ok=True)
os.makedirs(diagnostic_dir, exist_ok = True)
os.makedirs(results_dir, exist_ok = True)
os.makedirs(comparisons_dir, exist_ok = True)
os.makedirs(final_plots_dir, exist_ok = True)
os.makedirs(rar_dir, exist_ok = True)

loss_history_dir = os.path.join(diagnostic_dir, "loss_history.csv")

with open(loss_history_dir, 'w', newline = '') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'stage', 'total_loss', 'flux_loss', 'low-freq_loss', 'ode_loss', 'ode_real_loss', 'ode_imag_loss', 'flux_weight', 'grad_norm_preclip', 'gradient_clipped', 'learning_rate_used'])

adam_iterations = args.adam_steps 
lbfgs_iterations = args.lbfgs_steps

learning_rate = 1e-3
lr_min = 1e-6
decay_start = int(0.3*adam_iterations)
optimiser = optim.Adam(model.parameters(), lr = learning_rate)

hist_epochs, hist_total, hist_flux, hist_lowfreq, hist_ode, hist_ode_re, hist_ode_im, hist_weight = [], [], [], [], [], [], [], []
hist_grad_norm, hist_clipped = [], []
diagnostic_model_states = []
hist_learning_rate = []
hist_curriculum_stage = []
hist_omega_lower = []
GBF, probability, alphas, betas, extraction_epochs = [], [], [], [], []

#Base number of paired (x, omega) samples in each Adam epoch.
N_points = 10000

#Joint residual-based adaptive refinement
RAR_INTERVAL = 1000
RAR_CANDIDATES = 5000
RAR_ADD = 250
RAR_MAX = 2000
RAR_MIN_JOINT_DISTANCE = 0.01
RAR_RESTART_AFTER = 5000
rar_full_epoch = None
RAR_HEATMAP_INTERVAL = 5000
RAR_HEATMAP_NX = 120
RAR_HEATMAP_NOMEGA = 120

MAX_GRAD_NORM = 10.0
LBFGS_RESAMPLE_INTERVAL = 200

rar_x = t.empty((0, 1), dtype = DTYPE, device = device)
rar_omega = t.empty((0, 1), dtype = DTYPE, device = device)

omega_low = args.curriculum_start*omega_max
omega_prev = omega_low
n_stages = min(args.curriculum_steps, adam_iterations)
stage_lows = np.geomspace(omega_low, omega_min, n_stages)

stage_edges = np.linspace(0, adam_iterations, n_stages + 1, dtype = int)

run_config = {"arguments": vars(args),
    "physical_parameters": {"mass": mass, "mode": mode, "x_extract": x_max, "omega_min": omega_min, "omega_max": omega_max},
    "reproducibility": {"torch_seed": 0, "dtype": str(DTYPE), "device": str(device)},
    "model": {"in_channels": 2, "out_channels": 4, "hidden_channels": 64, "hidden_layers": 3, "activation": "keerie_adaptive_tanh", "activation_n": 10.0, "ansatz_x_power": 5},
    "sampling": {"base_points_per_adam_step": N_points, "x_uniform_fraction": 0.6, "x_beta_alpha": 0.6,
        "x_beta_beta": 0.1, "omega_uniform_fraction": 0.6, "omega_beta_alpha": 0.6, "omega_beta_beta": 2.0, "omega_scaling": "fixed global logarithmic scaling"},
    "curriculum": {"n_stages": int(n_stages), "stage_lower_bounds": stage_lows.tolist(), "stage_epoch_edges": stage_edges.tolist(), "initial_lower_bound": float(args.curriculum_start*omega_max)},
    "adam": {"steps": adam_iterations, "learning_rate_initial": learning_rate, "learning_rate_minimum": lr_min, "learning_rate_schedule": "exponential in log space, restarted per stage", "max_gradient_norm": MAX_GRAD_NORM},
    "lbfgs": {"outer_steps_requested": lbfgs_iterations, "max_iter": 20, "history_size": 50, "line_search_fn": "strong_wolfe", "resample_interval_outer_steps": LBFGS_RESAMPLE_INTERVAL},
    "loss": {"flux_weight": 1.0},
    "rar": {"interval": RAR_INTERVAL, "candidate_count": RAR_CANDIDATES, "points_added_per_refinement": RAR_ADD, "maximum_points": RAR_MAX, "minimum_joint_distance": RAR_MIN_JOINT_DISTANCE, "restart_after": RAR_RESTART_AFTER},
    "query_steps": args.query_steps,
    "diagnostic_omegas": diagnostic_omegas}

config_path = os.path.join(base_path, "config.json")
with open(config_path, "w") as f:
    json.dump(run_config, f, indent=2)

print(f"Saved run configuration to {config_path}", flush=True)

lr_progress = np.array([0.0, 0.10, 0.60, 1.0])
lr_values = np.array([1e-3, 2e-4, 2e-5, 1e-6])

#Adam loop
for epoch in range(adam_iterations):

    stage = min(int(np.searchsorted(stage_edges[1:], epoch, side = "right")), n_stages - 1)
    stage_start = int(stage_edges[stage])
    stage_end = int(stage_edges[stage + 1])

    if epoch == stage_start:
        omega_low = float(stage_lows[stage])

        rar_x = t.empty((0, 1), dtype = DTYPE, device = device)
        rar_omega = t.empty((0, 1), dtype = DTYPE, device = device)
        rar_full_epoch = None

        print(f"Stage {stage + 1}/{n_stages}: omega range [{omega_low:.6f}, {omega_max:.6f}]", flush = True)

        new_rar_x, new_rar_omega, new_rar_scores = select_rar_joint_points(model = model, n_candidates = RAR_CANDIDATES, n_add = min(RAR_ADD, RAR_MAX), x_max = x_max, mass = mass, mode = mode, device = device, dtype = DTYPE, existing_rar_x = rar_x,
                    existing_rar_omega = rar_omega, min_joint_distance = RAR_MIN_JOINT_DISTANCE, omega_lower = omega_low)

        rar_x = t.cat([rar_x, new_rar_x], dim = 0)
        rar_omega  = t.cat([rar_omega, new_rar_omega], dim = 0)

    progress = epoch/max(adam_iterations - 1, 1)
    omega_initial = args.curriculum_start*omega_max
    expansion = smoothstep((progress - 0.1)/0.5)

    omega_low = np.exp((1.0 - expansion)*np.log(omega_initial) + expansion*np.log(omega_min))

    learning_rate_used = 10.0**np.interp(progress, lr_progress, np.log10(lr_values))

    # stage_progress = (epoch - stage_start)/max(1, stage_end - stage_start - 1)
    # stage_progress = float(np.clip(stage_progress, 0.0, 1.0))

    # learning_rate_used = learning_rate*(lr_min/learning_rate)**stage_progress

    for group in optimiser.param_groups:
        group["lr"] = learning_rate_used

    x_tensor, omega_tensor = sample_joint_batch(n_points = N_points, x_max = x_max, dtype = DTYPE, device = device, rar_x = rar_x, rar_omega = rar_omega, omega_lower = omega_low)

    optimiser.zero_grad(set_to_none = True)

    flux_weight = annealing(epoch, adam_iterations)
    hist_weight.append(flux_weight)
    (Re_u_nn, Im_u_nn, flux_res, loss, loss_f, loss_lowfreq, loss_o, 
    loss_ode_real, loss_ode_imag, res_ode_re, res_ode_im, P_re, P_im, Q_re, Q_im) = compute_loss(model, x_tensor, mass, mode, omega_tensor, flux_weight)

    loss.backward()

    grad_norm = t.nn.utils.clip_grad_norm_(model.parameters(), max_norm = MAX_GRAD_NORM, error_if_nonfinite = True)
    grad_norm_value = grad_norm.item()
    gradient_clipped = grad_norm_value > MAX_GRAD_NORM
    
    hist_grad_norm.append(grad_norm_value)
    hist_clipped.append(gradient_clipped)

    learning_rate_used = optimiser.param_groups[0]["lr"]

    optimiser.step()

    with open(loss_history_dir, 'a', newline = '') as f:
        writer = csv.writer(f)

        writer.writerow([epoch + 1, 'Adam', loss.item(), loss_f.item(), loss_lowfreq.item(), loss_o.item(), loss_ode_real.item(), loss_ode_imag.item(), flux_weight, grad_norm_value, int(gradient_clipped), learning_rate_used])

    hist_epochs.append(epoch + 1)
    hist_total.append(loss.item())
    hist_flux.append(loss_f.item())
    hist_lowfreq.append(loss_lowfreq.item())
    hist_ode.append(loss_o.item())
    hist_ode_re.append(loss_ode_real.item())
    hist_ode_im.append(loss_ode_imag.item())
    hist_learning_rate.append(float(learning_rate_used))
    hist_curriculum_stage.append(int(stage))
    hist_omega_lower.append(float(omega_low))

    if (epoch + 1) % 100 == 0 or epoch == 0 or epoch == (adam_iterations - 1):
            extraction_epochs.append(epoch + 1)
            alpha, beta, prob, gbf = extraction(model, x_max, mass, mode, monitor_omega)
            alphas.append(alpha)
            betas.append(beta)
            probability.append(prob)
            GBF.append(gbf)

            if (epoch + 1) % 500 == 0 or epoch == 0:
                print(f"""l = {mode} | Adam Epoch: {epoch + 1} / {adam_iterations}. 
                        Batch omegas min|max|mean: {omega_tensor.min().item():.4f} | {omega_tensor.max().item():.4f} | {omega_tensor.mean().item():.4f},
                        Diagnostic omegas = {[f"{om:.4f}" for om in diagnostic_omegas]},
                        Total scaled loss: {loss.item():.4e}, 
                        Flux loss: {loss_f.item():.4e},
                        ODE loss: {loss_o.item():.4e},
                        Monitor GBF: {gbf},
                        Monitor |alpha|^2 - |beta|^2: {prob}.""", flush = True)
                print("-"*60, flush = True)

    if (epoch + 1) % 1000 == 0:
        print("-"*60)

        diagnostic_model_states.append({'iteration': epoch + 1, 'stage': 'Adam', 'state_dict': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}})

        for omega_diag in diagnostic_omegas:
            save_training_diagnostics(model, epoch_number = epoch + 1, omega = omega_diag)
        print("-"*60)

#Residual-based adapative refinement using the complex ODE residual

    current_epoch = epoch + 1

    rar_restart_now = rar_full_epoch is not None and current_epoch < adam_iterations and current_epoch - rar_full_epoch >= RAR_RESTART_AFTER

    if rar_restart_now:
        
        print(f"RAR set filled at epoch {rar_full_epoch}. Restarting at epoch {current_epoch}.")
        
        rar_x = t.empty((0, 1), dtype = DTYPE, device = device)
        rar_omega = t.empty((0, 1), dtype = DTYPE, device = device)
        rar_full_epoch = None

    should_refine = (current_epoch % RAR_INTERVAL == 0 or rar_restart_now) and current_epoch < adam_iterations and rar_x.shape[0] < RAR_MAX

    if should_refine:
        print("-"*60)
        print(f"Joint RAR triggered at Adam epoch {epoch + 1}...", flush  = True)

        n_add = min(RAR_ADD, RAR_MAX - rar_x.shape[0])

        new_rar_x, new_rar_omega, new_rar_scores = select_rar_joint_points(model = model, n_candidates = RAR_CANDIDATES, n_add = n_add, x_max = x_max, mass = mass,
                        mode = mode, device = device, dtype = DTYPE, existing_rar_x = rar_x, existing_rar_omega = rar_omega, min_joint_distance = RAR_MIN_JOINT_DISTANCE, omega_lower = omega_low)

        actual_added = new_rar_x.shape[0]

        if actual_added > 0:
            rar_x = t.cat([rar_x, new_rar_x], dim = 0)
            rar_omega = t.cat([rar_omega, new_rar_omega], dim = 0)

            print(f"Added {actual_added} joint RAR points. Total retained = {rar_x.shape[0]}", flush = True)
            print(f"RAR x range = [{new_rar_x.min().item():.6f}, {new_rar_x.max().item():.6f}]", flush = True)
            print(f"RAR omega range = [{new_rar_omega.min().item():.6f}, {new_rar_omega.max().item():.6f}]", flush = True)

            print(f"Maximum selected ODE residual = {new_rar_scores.max().item():.4e}", flush = True)

        else:
            print("No additional RAR points met the separation constraint.")

        if rar_x.shape[0] >= RAR_MAX and rar_full_epoch is None:
            rar_full_epoch = current_epoch
            print(f"RAR set is full. Next restart at Adam epoch {rar_full_epoch + RAR_RESTART_AFTER}.", flush = True)

        print("-"*60)

    if (epoch + 1) % RAR_HEATMAP_INTERVAL == 0:
        plot_joint_residual_heatmap(model = model, epoch = epoch + 1, mass = mass, mode = mode, x_max = x_max, rar_x = rar_x, rar_omega = rar_omega,
                                nx = RAR_HEATMAP_NX, n_omega = RAR_HEATMAP_NOMEGA, chunk_size = 2000)

print("Adam training complete. Switching to L-BFGS:", flush = True)
print("="*60)

adam_checkpoint_path = os.path.join(base_path, "after_adam.pth")

atomic_torch_save({
    "checkpoint_version": 1,
    "model_state_dict": get_cpu_model_state(model),
    "config": run_config,
    "training_status": {"stage_completed": "Adam", "adam_steps_completed": adam_iterations},
    "training_history": {"epochs": [int(v) for v in hist_epochs],
        "total_loss": [float(v) for v in hist_total],
        "ode_loss": [float(v) for v in hist_ode],
        "flux_loss": [float(v) for v in hist_flux],
        "lowfreq_loss": [float(v) for v in hist_lowfreq],
        "learning_rate": hist_learning_rate,
        "curriculum_stage": hist_curriculum_stage,
        "omega_lower": hist_omega_lower,
        "grad_norm_preclip": hist_grad_norm,
        "gradient_clipped": [bool(v) for v in hist_clipped]},
    "rar_points": {"x": rar_x.detach().cpu(), "omega": rar_omega.detach().cpu()}}, adam_checkpoint_path)

print(f"Saved post-Adam checkpoint to {adam_checkpoint_path}", flush=True)

epochs = np.arange(1, len(hist_grad_norm) + 1)

fig, ax = plt.subplots(figsize = (8, 5))
ax.plot(epochs, hist_grad_norm, label = "Gradient Norm History")
ax.axhline(MAX_GRAD_NORM, linestyle = '--', label = 'Clipping threshold')
ax.set_yscale('log')
ax.set_xlabel('Adam Epoch', fontsize = 16)
ax.set_ylabel("Pre-clip Gradient Norm", fontsize = 16)
ax.grid(True)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(diagnostic_dir, "gradient_norm.png"), format = 'png', dpi = 200)
plt.close(fig)

window = 500
clipped_array = np.asarray(hist_clipped, dtype = float)

if len(clipped_array) >= window:
    clip_fraction = np.convolve(clipped_array, np.ones(window)/window, mode = 'valid')

    fig, ax = plt.subplots(figsize = (8, 5))
    ax.plot(np.arange(window, len(clipped_array) + 1), clip_fraction)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Adam Epoch')
    ax.set_ylabel('Fraction clipped in last 500 epochs')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(diagnostic_dir, "gradient_clipping_fraction.png"), format = 'png', dpi = 200)
    plt.close(fig)

lbfgs_steps_completed = 0
flux_weight = annealing(adam_iterations - 1, adam_iterations)

lbfgs_success = True
x_tensor_lbfgs, omega_tensor_lbfgs = sample_joint_batch(n_points = N_points, x_max = x_max, dtype = DTYPE, device = device, rar_x = rar_x, rar_omega = rar_omega, omega_lower = omega_min)
lbfgs_optimiser = optim.LBFGS(model.parameters(), lr = 1.0, max_iter = 20, history_size = 50, line_search_fn = "strong_wolfe")

for epoch in range(lbfgs_iterations):
    info = {'total': 0, 'flux': 0, 'low_freq': 0, 'ode': 0, 'loss_re': 0, 'loss_im': 0, 'res_re': 0, 'res_im': 0}

    if epoch > 0 and epoch % LBFGS_RESAMPLE_INTERVAL == 0:

        x_tensor_lbfgs, omega_tensor_lbfgs = sample_joint_batch(n_points = N_points, x_max = x_max, dtype = DTYPE, device = device, rar_x = rar_x, rar_omega = rar_omega, omega_lower = omega_min)
        lbfgs_optimiser = optim.LBFGS(model.parameters(), lr = 1.0, max_iter = 20, history_size = 50, line_search_fn = "strong_wolfe")

        print(f"Refreshed L-BFGS batch  and restarted optimiser history at outer step {epoch}.", flush = True)
    
    def closure():
        lbfgs_optimiser.zero_grad(set_to_none = True)
        (Re_u_nn, Im_u_nn, flux_res,  loss, loss_f, loss_lowfreq, loss_o, loss_ode_re, loss_ode_im,
        res_ode_re, res_ode_im, P_re, P_im, Q_re, Q_im) = compute_loss(model, x_tensor_lbfgs, mass, mode, omega_tensor_lbfgs, flux_weight)
        loss.backward()

        info.update({'total': loss.item(), 'flux': loss_f.item(), 'low_freq': loss_lowfreq.item(), 'ode': loss_o.item(), 'loss_re': loss_ode_re.item(), 'loss_im': loss_ode_im.item()})

        return loss

    lbfgs_optimiser.step(closure)

    with_grad = compute_loss(model, x_tensor_lbfgs, mass, mode, omega_tensor_lbfgs, flux_weight)
    _, _, _, loss_now, lf_now, llfreq_now, lo_now, lre, lim, *_ = with_grad
    info.update({'total': loss_now.item(), 'flux': lf_now.item(), 'low_freq': llfreq_now.item(), 'ode': lo_now.item(), 'loss_re': lre.item(), 'loss_im': lim.item()})

    global_epoch = adam_iterations + epoch
    
    hist_epochs.append(global_epoch + 1)
    hist_total.append(loss_now.item())
    hist_flux.append(lf_now.item())
    hist_lowfreq.append(llfreq_now.item())
    hist_ode.append(lo_now.item())
    hist_ode_re.append(lre.item())
    hist_ode_im.append(lim.item())
    hist_weight.append(flux_weight)

    with open(loss_history_dir, 'a', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow([global_epoch + 1, "L-BFGS", loss_now.item(), lf_now.item(), llfreq_now.item(), lo_now.item(), lre.item(), lim.item(), flux_weight,"", "", ""])

    if not np.isfinite(info['total']):
        print(f"L-BFGS diverged at epoch {epoch}; stopping.", flush=True)
        lbfgs_success = False
        break

    lbfgs_steps_completed = epoch + 1

    if (epoch + 1) % 40 == 0 or epoch == (lbfgs_iterations - 1):
        #Printing and plotting
        extraction_epochs.append(global_epoch + 1)
        alpha, beta, prob, gbf = extraction(model, x_max, mass, mode, monitor_omega)
        alphas.append(alpha)
        betas.append(beta)
        probability.append(prob)
        GBF.append(gbf)
        
        print(f"""l = {mode} | L-BFGS Epoch: {epoch + 1} / {lbfgs_iterations}. 
                    Fixed-batch omega min|max = {omega_tensor_lbfgs.min().item():.4f} | {omega_tensor_lbfgs.max().item():.4f},
                    Diagnostic omegas = {[f"{om:.4f}" for om in diagnostic_omegas]},
                    Total scaled loss: {info['total']:.4e}, 
                    Flux loss: {info['flux']:.4e},
                    ODE loss: {info['ode']:.4e},
                    Monitor GBF: {gbf},
                    Monitor |alpha|^2 - |beta|^2: {prob}.""", flush = True)
        print("-"*60, flush = True)

        diagnostic_model_states.append({'iteration': global_epoch + 1, 'stage': 'L-BFGS', 'state_dict': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}})

        plot_joint_residual_heatmap(model = model, epoch = global_epoch + 1, mass = mass, mode = mode, x_max = x_max, rar_x = rar_x, rar_omega = rar_omega, 
                                    nx = RAR_HEATMAP_NX, n_omega = RAR_HEATMAP_NOMEGA, chunk_size = 2000)

        for omega_diag in diagnostic_omegas:
            save_training_diagnostics(model, epoch_number = global_epoch + 1, omega = omega_diag)


checkpoint_path = os.path.join(base_path, "trained_model.pth")

final_model_checkpoint = {
    "checkpoint_version": 1, 
    "model_state_dict": get_cpu_model_state(model), 
    "model_architecture": run_config["model"], 
    "config": run_config,
    "training_status": {"adam_steps_requested": adam_iterations,
        "adam_steps_completed": len([e for e in hist_epochs if e <= adam_iterations]),
        "lbfgs_steps_requested": lbfgs_iterations,
        "lbfgs_steps_completed": lbfgs_steps_completed,
        "lbfgs_finite_loss": bool(lbfgs_success),
        "last_recorded_loss": float(hist_total[-1])},
    "training_history": {"epochs": [int(v) for v in hist_epochs],
        "total_loss": [float(v) for v in hist_total],
        "ode_loss": [float(v) for v in hist_ode],
        "flux_loss": [float(v) for v in hist_flux],
        "lowfreq_loss": [float(v) for v in hist_lowfreq],
        "ode_real_loss": [float(v) for v in hist_ode_re],
        "ode_imag_loss": [float(v) for v in hist_ode_im],
        "flux_weight": [float(v) for v in hist_weight]},
    "rar_points": {"x": rar_x.detach().cpu(),
        "omega": rar_omega.detach().cpu()},
        "adam_learning_rate": hist_learning_rate,
        "adam_curriculum_stage": hist_curriculum_stage,
        "adam_omega_lower": hist_omega_lower,
        "adam_grad_norm_preclip": hist_grad_norm,
        "adam_gradient_clipped": [bool(v) for v in hist_clipped],

    "diagnostic_model_states": diagnostic_model_states}

atomic_torch_save(final_model_checkpoint, checkpoint_path)
print(f"Saved trained model checkpoint to {checkpoint_path}", flush=True)

print("Conditional PINN training complete.", flush = True)

omega_query = np.logspace(np.log10(omega_min), np.log10(omega_max), args.query_steps)
dense_query_results = query_gbf(model, mass, mode, omega_query, x_max)
dense_query_omegas = np.array([result['omega'] for result in dense_query_results])
dense_query_gbfs = np.array([result['gbf'] for result in dense_query_results])
query_gbf_path = os.path.join(results_dir, "PINN_query_frequencies.csv")
query_wronskian = np.array([result["probability"] for result in dense_query_results])

wronskian_error = np.abs(query_wronskian - 1.0)

print(f"Maximum Wronskian-identity error: {np.max(wronskian_error):.6e}", flush = True)

plt.figure(figsize = (7, 5))
plt.plot(dense_query_omegas, query_wronskian, label = r"$|\alpha|^2 - |\beta|^2$")
plt.axhline(1.0,  linestyle = '--', label = "Target")
plt.xlabel(r"$\omega$", fontsize = 16)
plt.ylabel("Wronskian")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(comparisons_dir, "Wronskian.png"), dpi = 200, format = 'png')
plt.close()

with open(query_gbf_path, 'w', newline = '') as f:
    writer = csv.writer(f)

    writer.writerow(['omega', 'alpha_re', 'alpha_im', 'beta_re',  'beta_im', 'prob', 'PINN_GBF'])

    for result in dense_query_results:
            writer.writerow([f"{result['omega']:.10f}", f"{result['alpha'].real:.12e}", f"{result['alpha'].imag:.12e}", f"{result['beta'].real:.12e}", f"{result['beta'].imag:.12e}",
                    f"{result['probability']:.12e}", f"{result['gbf']:.12e}"])


plt.figure(figsize = [7, 5])
plt.plot(dense_query_omegas, dense_query_gbfs, label = 'Two-input PINN')
plt.xlabel(r'$\omega$', fontsize = 16)
plt.ylabel(r"$\Gamma(\omega)$", fontsize = 16)
plt.title(f"Two-input PINN GBF, l = {mode}", fontsize = 16)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(base_path, "WXPINN_GBF.png"), format = 'png')
plt.close()

print("Preparing final post-training diagnostics...")
print("="*60)

for omega_plot in diagnostic_omegas:

    final_plot_omega_dir = os.path.join(final_plots_dir, f"omega{omega_plot:.4f}")
    os.makedirs(final_plot_omega_dir, exist_ok = True)

    x_values, u_re, u_im, P_re, P_im, Q_re, Q_im = query_wavefunction(model, mass, mode,  omega_plot, x_diagnostic)

    x_plot = x_values
    r_plot = 2*mass/(1 - x_plot)

    plt.figure(figsize=[14, 6])

    plt.subplot(1, 2, 1)
    plt.suptitle("Direct Neural Network Output\n"
                f"l = {mode}, omega = {omega_plot:.4f}")
    plt.plot(x_plot, P_re, label='Re(P)')
    plt.plot(x_plot, P_im, label='Im(P)')
    plt.plot(x_plot, Q_re, label='Re(Q)')
    plt.plot(x_plot, Q_im, label='Im(Q)')
    plt.xlabel('x', fontsize = 16)
    plt.ylabel('P and Q', fontsize = 16)
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(r_plot, P_re, label='Re(P)')
    plt.plot(r_plot, P_im, label='Im(P)')
    plt.plot(r_plot, Q_re, label='Re(Q)')
    plt.plot(r_plot, Q_im, label='Im(Q)')
    plt.xlabel('r', fontsize = 16)
    plt.ylabel('P and Q', fontsize = 16)
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(final_plot_omega_dir, 'PQ.png'), format='png')
    plt.close()

    plt.figure(figsize=[14, 6])

    plt.subplot(1, 2, 1)
    plt.suptitle("Wave function u built via ansatz of P and Q\n"
        f"l = {mode}, omega = {omega_plot:.4f}")
    plt.plot(x_plot, u_re, label=r'$\Re(u_{NN})$')
    plt.plot(x_plot, u_im, label=r'$\Im(u_{NN})$')
    plt.xlabel('x', fontsize = 16)
    plt.ylabel(r'$u(x)$', fontsize = 16)
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(r_plot, u_re, label=r'$\Re(u_{NN})$')
    plt.plot(r_plot, u_im, label=r'$\Im(u_{NN})$')
    plt.xlabel('r', fontsize = 16)
    plt.ylabel('u(r)', fontsize = 16)
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(final_plot_omega_dir, 'u.png'), format='png')
    plt.close()

numerical_csv_path = f'./Numerical/l{mode}/Output/numericalGBF.csv'

print(f"Loading numerical GBF data from: {numerical_csv_path}")

numerical_results = load_numerical_gbf(numerical_csv_path, mass, mode, x_max)

print(f"Loaded {len(numerical_results)} numerical results.")

query_comparison_csv = os.path.join(comparisons_dir, "PINN_vs_Numerical_query.csv")
query_comparison_omegas, query_pinn_gbfs, query_numerical_gbfs, query_absolute_differences, query_relative_differences = compare_pinn_to_numerical(dense_query_results,
                    numerical_results, query_comparison_csv) 
save_simulation_comparison_plots(query_comparison_omegas, query_pinn_gbfs, query_numerical_gbfs, query_absolute_differences, query_relative_differences, "Dense Query Frequencies")

plt.figure(figsize=[7, 5])

plt.plot(query_comparison_omegas, query_numerical_gbfs, label='Numerical')
plt.plot(query_comparison_omegas, query_pinn_gbfs, 'x-', label='Two-input PINN')
plt.xlabel(r'$\omega$', fontsize = 16)
plt.ylabel(r'$\Gamma(\omega)$', fontsize = 16)
plt.title(f'Grey-Body Factor, l = {mode}')
plt.grid()
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(comparisons_dir, 'Final_GBF_Comparison_query.png'), format='png')
plt.close()

final_alpha, final_beta, final_prob, final_gbf = extraction(model, x_max, mass, mode, monitor_omega)

T = 1/final_alpha
R = final_beta/final_alpha

final_loss_data = compute_loss(model, x_tensor_lbfgs, mass, mode, omega_tensor_lbfgs, flux_weight)
_, _, _, final_total_loss, final_flux_loss, final_lowfreq_loss, final_ode_loss, _, _, *_ = final_loss_data

query_relative_finite = query_relative_differences[np.isfinite(query_relative_differences)]

result_file_path = os.path.join(results_dir, 'result.txt')

with open(result_file_path, 'w') as f:
    f.write("Two-input PINN\n")
    f.write(f"mass = {mass}\n")
    f.write(f"l = {mode}\n")
    f.write(f"x_extract = {x_max}\n")

    f.write(f"omega_min = {omega_min:.10f}\n")
    f.write(f"omega_max = {omega_max:.10f}\n")
    f.write("training_omega_sampling = 60% uniform + 40% skewed beta distribution \n")
    f.write(f"Adam_iterations = {adam_iterations}\n")
    f.write(f"Adam_base_points_per_epoch = {N_points}\n")
    f.write(f"rar_points_final = {rar_x.shape[0]}\n")
    f.write(f"lbfgs_iterations_requested = {lbfgs_iterations}\n")
    f.write(f"query_frequencies = {len(omega_query)}\n")
    f.write("diagnostic_frequencies = " + ", ".join(f"{om:.10f}" for om in diagnostic_omegas) + "\n")

    f.write(f"final_total_loss = {final_total_loss.item():.12e}\n")
    f.write(f"final_ODE_loss = {final_ode_loss.item():.12e}\n")
    f.write(f"final_flux_loss = {final_flux_loss.item():.12e}\n")
    f.write(f"final_low_freq_loss = {final_lowfreq_loss.item():.12e}\n")

    f.write(f"monitor_frequency = {monitor_omega:.10f}\n")
    f.write(f"alpha_re = {final_alpha.real:.12e}\n")
    f.write(f"alpha_im = {final_alpha.imag:.12e}\n")
    f.write(f"beta_re = {final_beta.real:.12e}\n")
    f.write(f"beta_im = {final_beta.imag:.12e}\n")

    f.write(f"T_re = {T.real:.12e}\n")
    f.write(f"T_im = {T.imag:.12e}\n")
    f.write(f"R_re = {R.real:.12e}\n")
    f.write(f"R_im = {R.imag:.12e}\n")

    f.write(f"mid_frequency_prob = {final_prob:.12e}\n")
    f.write(f"monitor_frequency_GBF = {final_gbf:.12e}\n")

    f.write(f"query_mean_relative_error = {np.mean(query_relative_finite):.12e}\n")
    f.write(f"query_max_relative_error = {np.max(query_relative_finite):.12e}\n")

    f.write(f"lbfgs_success = {lbfgs_success}\n")

checkpoint = {'model_state_dict': model.state_dict(),
            'model_architecture': {'in_channels': 2, 'out_channels': 4, 'hidden_channels': 64, 'hidden_layers': 3},
            'activation_config': {'type': 'keerie_adaptive_tanh', 'n': 10.0},
            'omega_input_scaling': {'omega_min': omega_min, 'omega_max': omega_max},
            'rar_config': {'interval': RAR_INTERVAL, 'candidates': RAR_CANDIDATES, 'add_per_refinement': RAR_ADD, 'max_points': RAR_MAX, 'min_joint_distance': RAR_MIN_JOINT_DISTANCE,
                        "restart_after": RAR_RESTART_AFTER},
            'ansatz_config': {'nn_output_scale': 1.0, 'x_power': 5, 'x_safe_min': 1e-12, 'x_safe_max': 1 - 1e-3},
            'mass': mass,
            'mode': mode,
            'x_max': x_max,

            'omega_range': {'min': omega_min, 'max': omega_max},
            'training_omega_sampling': "60% uniform + 40% skewed beta distribution",
            'adam_base_points_per_epoch': N_points,
            'query_steps': args.query_steps,
            'diagnostic_omegas': diagnostic_omegas,
            'monitor_omega': monitor_omega,
            'x_diagnostic': x_diagnostic.tolist(),

            'adam_iterations': adam_iterations,
            'lbfgs_iterations': lbfgs_iterations,

            'flux_weight_initial': 1.0,
            'flux_weight_final': 1.0,

            'rar_points': {'x': rar_x.detach().cpu(), 'omega': rar_omega.detach().cpu()},

            'training_history': {'epochs': [int(value) for value in hist_epochs], 'total_loss': [float(value) for value in hist_total], 'flux_loss': [float(value) for value in hist_flux],
                        'ode_loss': [float(value) for value in hist_ode], 'ode_real_loss': [float(value) for value in hist_ode_re], 'ode_imag_loss': [float(value) for value in hist_ode_im],
                        'flux_weight': [float(value) for value in hist_weight]},

            'extraction_history': {'epochs': [int(value) for value in extraction_epochs], 'alpha_re': [float(value.real) for value in alphas], 'alpha_im': [float(value.imag) for value in alphas],
                                    'beta_re': [float(value.real) for value in betas], 'beta_im': [float(value.imag) for value in betas], 'probability': [float(value) for value in probability],
                                    'gbf': [float(value) for value in GBF]},

            'dense_query_results': serialise_query_results(dense_query_results),

            'numerical_results': [{'omega': float(omega), 'gbf': float(gbf)} for omega, gbf in sorted(numerical_results.items())],

            'query_comparison': {'omega': query_comparison_omegas.tolist(), 'pinn_gbf': query_pinn_gbfs.tolist(), 'numerical_gbf': query_numerical_gbfs.tolist(),
                                'absolute_difference': query_absolute_differences.tolist(), 'relative_difference': query_relative_differences.tolist()},

            'diagnostic_model_states': diagnostic_model_states,
            'lbfgs_success': bool(lbfgs_success),
            'training_complete': bool(lbfgs_success)}

checkpoint["model_state_dict"] = get_cpu_model_state(model)

checkpoint["config"] = run_config
checkpoint["training_history"].update({
    "adam_learning_rate": hist_learning_rate,
    "adam_curriculum_stage": hist_curriculum_stage,
    "adam_omega_lower": hist_omega_lower,
    "adam_grad_norm_preclip": hist_grad_norm,
    "adam_gradient_clipped": [bool(v) for v in hist_clipped],
})

checkpoint_path = os.path.join(base_path, f'pinn_conditional_GBFWX_l{mode}.pth')
atomic_torch_save(checkpoint, checkpoint_path)

print(f"Checkpoint saved to {checkpoint_path}", flush = True)

print("Calculating GBF via product integral approach...")
print("-"*60)

Momega_values = mass*query_comparison_omegas

PI_T_list = []
PI_unitarity_list = []

for x in Momega_values:
    T_PI, alpha_PI, beta_PI = product_integral_GBF(x, mode = mode, ustar_max = 350.0, N = 20000)
    PI_T_list.append(T_PI)
    unitarity = np.abs(alpha_PI)**2 - np.abs(beta_PI)**2
    PI_unitarity_list.append(unitarity)

PI_T = np.asarray(PI_T_list)
PI_unitarity = np.asarray(PI_unitarity_list)
abs_diff = np.abs(query_pinn_gbfs - PI_T)
rel_diff = []

for i, T in enumerate(PI_T):
    if T != 0.0:
        relative_difference = abs_diff[i]/T
    else:
        relative_difference = np.nan
    rel_diff.append(relative_difference)

rel_diff = np.asarray(rel_diff)

plt.figure(figsize = [14, 6])

plt.subplot(1, 2, 1)
plt.suptitle('Product Integral vs. PINN')
plt.plot(query_comparison_omegas, PI_T, label = 'Product Integral')
plt.plot(query_comparison_omegas, query_pinn_gbfs, label = 'Conditional PINN')
plt.xlabel(r'$\omega$', fontsize = 16)
plt.ylabel(r'$\Gamma(\omega)$', fontsize = 16)
plt.title(r'GBF Comparison - Dense Query Frequencies', fontsize = 16)
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(query_comparison_omegas, PI_T_list, label = 'Product Integral')
plt.plot(query_comparison_omegas, query_pinn_gbfs, label = 'Conditional PINN')
plt.yscale('log')
plt.xlabel(r'$\omega$', fontsize = 16)
plt.ylabel(r'$\Gamma(\omega)$', fontsize = 16)
plt.title(r'Log-scale GBF Comparison - Dense Query Frequencies', fontsize = 16)
plt.grid()
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(comparisons_dir, 'PIvsPINN_GBF.png'), format = 'png', dpi = 200)
plt.close()

plt.figure(figsize = [14, 6])

plt.subplot(1, 2, 1)
plt.suptitle('Product Integral vs. PINN')
plt.plot(query_comparison_omegas, abs_diff)
plt.yscale('log')
plt.xlabel(r'$\omega$', fontsize = 16)
plt.ylabel(r'Absolute Difference', fontsize = 16)
plt.title(r'Absolute Difference - Dense Query Frequencies', fontsize = 16)
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(query_comparison_omegas, rel_diff)
plt.yscale('log')
plt.xlabel(r'$\omega$', fontsize = 16)
plt.ylabel(r'Relative Difference', fontsize = 16)
plt.title(r'Relative Difference - Dense Query Frequencies', fontsize = 16)
plt.grid()

plt.tight_layout()
plt.savefig(os.path.join(comparisons_dir, 'PIvsPINN_differences.png'), format = 'png', dpi = 200)
plt.close()

omega_common = query_comparison_omegas
Momega_common = mass*omega_common

asymptotic_gbf = low_frequency_limit_GBF(mass, mode, omega_common)
lowfreq_mask = Momega_common <= args.lowfreq_max_M_omega

plt.figure(figsize = (7, 6))
plt.plot(omega_common, query_numerical_gbfs, label = 'Simulation')
plt.plot(omega_common, query_pinn_gbfs, label = 'Conditional PINN')
plt.plot(omega_common, PI_T, label = 'Product Integral')
plt.plot(omega_common[lowfreq_mask], asymptotic_gbf[lowfreq_mask], "--", label = "Low-frequency limit")
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\omega$", fontsize = 16)
plt.ylabel(r"$\Gamma(\omega)$", fontsize = 16)
plt.title(f"GBF Comparison, M = {mass}, l = {mode}", fontsize = 16)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(comparisons_dir, "Full_GBF_comparison.png"), format = 'png', dpi = 200)
plt.close(fig)

convergence_results = product_integral_convergence(
    x=0.2,              # M*omega
    mode=2,
    N_values=(1000, 2000, 4000, 8000),
    ustar_min=-50.0,
    ustar_max=350.0,
)

for row in convergence_results:
    print(
        f"N={row['N']:5d} | "
        f"GBF={row['GBF']:.12e} | "
        f"relative change={row['relative_change_from_previous_N']:.3e} | "
        f"Wronskian error={row['wronskian_error']:.3e}"
    )