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
parser.add_argument('--omega_start', type = float, default = 0.3, help = 'Initial (higher) frequency')
parser.add_argument('--omega_final', type = float, default = 0.03, help = 'Final (lower) frequency')
parser.add_argument('--num_steps', type = int, default = 10, help = "Number of training frequency steps")
parser.add_argument('--query_steps', type = int, default = 200, help = 'Number of frequencies used for dense post-training queries')

args = parser.parse_args()
mass = args.mass
mode = args.mode
x_max = args.x_extract

if args.omega_start < args.omega_final:
    raise ValueError("omega_final must be less than omega_start")

if mass <= 0:
    raise ValueError("mass must be positive.")

if mode < 0:
    raise ValueError("mode must be non-negative.")

if args.omega_start <= 0 or args.omega_final <= 0:
    raise ValueError("Both frequencies must be positive.")

if args.omega_start <= args.omega_final:
    raise ValueError("omega_start must be strictly greater than omega_final.")

if args.num_steps < 1:
    raise ValueError("num_steps must be at least 1.")

if args.query_steps < 1:
    raise ValueError("query_steps must be at least 1.")

if not (1e-6 < x_max <= 1.0 - 1e-3):
    raise ValueError("x_extract must be in (1e-6, 0.999] for the current ansatz clamp.")

#Frequency schedule from high to low for continuation
omega_schedule = np.linspace(args.omega_start, args.omega_final, args.num_steps)
omega_min = float(args.omega_final)
omega_max = float(args.omega_start)
diagnostic_omegas = [float(omega_schedule[0]), float(omega_schedule[len(omega_schedule)//2]), float(omega_schedule[-1])]
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
    return c1.real, c1.imag, c2.real, c2.imag

#Flux loss annealing
def annealing(epoch, total_epochs):
    lambda_initial = 10.0
    lambda_final = 1.0

    progress = epoch/(total_epochs - 1)
    lambda_flux = lambda_final + 0.5*(lambda_initial - lambda_final)*(1 + np.cos(np.pi*progress))
    return lambda_flux

#Omega training input helper functions
def scale_omega(omega):
    return 2.0*(omega - omega_min)/(omega_max - omega_min) - 1.0

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

print(f"Initialising training for omega = {omega_schedule}")

beta_dist = dist.Beta(t.tensor([0.6], dtype = DTYPE, device = device), t.tensor([0.1], dtype = DTYPE, device = device))

def sample_x_points(n_points, x_max, dtype, device):
    n_uniform = int(0.6*n_points)
    n_edges = n_points - n_uniform

    x_uniform = x_max*t.rand((n_uniform, 1), dtype = dtype, device = device)
    x_edges = x_max*beta_dist.sample((n_edges,)).view(-1, 1).to(device = device, dtype = dtype)

    return t.cat([x_uniform, x_edges], dim = 0)

def sample_frequency_batch(omega_values, n_points_per_frequency, x_max, dtype, device, rar_points_by_omega = None):

    x_batches = []
    omega_batches = []

    for omega in omega_values:
        x_base = sample_x_points(n_points_per_frequency, x_max, dtype, device)

        key = round(float(omega), 4)

        if (rar_points_by_omega is not None and key in rar_points_by_omega and rar_points_by_omega[key].numel() > 0):
            x_slice = t.cat([x_base, rar_points_by_omega[key]], dim = 0)

        else:
            x_slice = x_base

        omega_slice = t.full_like(x_slice, float(omega))

        x_batches.append(x_slice)
        omega_batches.append(omega_slice)

    x_tensor = t.cat(x_batches, dim = 0)
    omega_tensor = t.cat(omega_batches, dim = 0)

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

def select_rar_points(model, n_candidates, n_add, x_max, mass, mode, omega, device, dtype, existing_rar_points = None, min_dx = 1e-3):
    """Sample candidate points, rank them by ODE residual and retain the highest-residual points subject to a minimum separation in x."""

    x_candidates = sample_x_points(n_candidates, x_max, dtype, device)
    scores = ode_residual_score(model, x_candidates, mass, mode, omega)

    n_add = min(n_add, x_candidates.shape[0])

    ranked_indices = t.argsort(scores, descending =  True)
    candidate_x = x_candidates.flatten().detach().cpu().numpy()

    if existing_rar_points is not None and existing_rar_points.numel() > 0:
        occupied_x = existing_rar_points.flatten().detach().cpu().numpy().tolist()
        occupied_x.sort()
    else:
        occupied_x = []

    accepted_indices = []

    for idx in ranked_indices.detach().cpu().numpy():
        x_value = float(candidate_x[idx])

        insertion_index = bisect.bisect_left(occupied_x, x_value)

        if insertion_index > 0:
            dx_left = x_value - occupied_x[insertion_index  - 1]
        else:
            dx_left = np.inf

        if insertion_index < len(occupied_x):
            dx_right = occupied_x[insertion_index] - x_value
        else:
            dx_right = np.inf

        nearest_dx = min(dx_left, dx_right)

        if nearest_dx >= min_dx:
            accepted_indices.append(idx)
            bisect.insort(occupied_x, x_value)

        if len(accepted_indices) >= n_add:
            break

    accepted_indices = t.tensor(accepted_indices, dtype = t.long, device = x_candidates.device)

    new_points = x_candidates[accepted_indices].detach()
    new_scores = scores[accepted_indices].detach()

    return new_points, new_scores

#Ansatz for the wave-function u
#Ansatz is of the form u(x) = 1 + c1*x + c2*x**2 + 100*(P + exp(2i*omega*r_star)*Q) 
#where P and Q are complex with components corresponding to the four channel neural network output
def ansatz(model, x_tensor, mass, mode, omega):

    omega_tensor = prepare_omega_tensor(omega, x_tensor)

    c1_re, c1_im, c2_re, c2_im = taylor_coeffs(mass, mode, omega_tensor)

    omega_scaled = scale_omega(omega_tensor)

    NN_input = t.cat([x_tensor, omega_scaled], dim = 1)
    NN = model(NN_input)

    P_re, P_im, Q_re, Q_im = NN[:, 0:1], NN[:, 1:2], NN[:, 2:3], NN[:, 3:4]
    x_safe = x_tensor.clamp(min = 1e-12, max = 1 - 1e-3)
    rstar = 2*mass/(1 - x_safe) + 2*mass*t.log(x_safe/(1 - x_safe))
    cs, sn = t.cos(2*omega_tensor*rstar), t.sin(2*omega_tensor*rstar)

    u_re = 1 + c1_re*x_tensor + c2_re*x_tensor**2 + x_tensor**3*(P_re + Q_re*cs - Q_im*sn)
    u_im = c1_im*x_tensor + c2_im*x_tensor**2 + x_tensor**3*(P_im + Q_im*cs + Q_re*sn)
    return u_re, u_im, P_re, P_im, Q_re, Q_im

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

    #Loss annealing to be included
    total_loss = loss_ode + flux_weight*loss_flux

    return u_re, u_im, J, total_loss, loss_flux, loss_ode, loss_ode_re, loss_ode_im, res_ode_re, res_ode_im, P_re, P_im, Q_re, Q_im

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

    u_re, u_im, J, total_loss, loss_flux, loss_ode, loss_ode_re, loss_ode_im, res_ode_re, res_ode_im, P_re, P_im, Q_re, Q_im = compute_loss(model, x_tensor, mass, mode, float(omega), flux_weight = 0.0)

    return {'x': x_tensor.detach().cpu().numpy().flatten(), 'u_re': u_re.detach().cpu().numpy().flatten(), 'u_im': u_im.detach().cpu().numpy().flatten(), 'flux': J.detach().cpu().numpy().flatten(),
            'res_re': res_ode_re.detach().cpu().numpy().flatten(), 'res_im': res_ode_im.detach().cpu().numpy().flatten()}

def save_training_diagnostics(model, epoch_number, omega):

    plt.figure(figsize = [7, 5])
    plt.plot(hist_epochs, hist_total, label = 'Total')
    plt.plot(hist_epochs, hist_ode, label = 'ODE')
    plt.plot(hist_epochs, hist_flux, label = 'Flux')
    plt.plot(hist_epochs, hist_ode_re, label = 'Real ODE')
    plt.plot(hist_epochs, hist_ode_im, label = 'Imag ODE')
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

def save_comparison_plots(omegas, pinn_gbfs, numerical_gbfs, absolute_differences, relative_differences, title_suffix, prefix):

    plt.figure(figsize=[14, 6])

    plt.subplot(1, 2, 1)
    plt.plot(omegas, numerical_gbfs, 'o-', label='Numerical')
    plt.plot(omegas, pinn_gbfs, 'x-', label='Conditional PINN')
    plt.xlabel(r'$\omega$', fontsize=16)
    plt.ylabel(r'$\Gamma(\omega)$', fontsize=16)
    plt.title(f'GBF Comparison — {title_suffix}')
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(omegas, numerical_gbfs, 'o-', label='Numerical')
    plt.plot(omegas, pinn_gbfs, 'x-', label='Conditional PINN')
    plt.xlabel(r'$\omega$', fontsize=16)
    plt.ylabel(r'$\Gamma(\omega)$', fontsize=16)
    plt.yscale('log')
    plt.title(f'GBF Comparison — {title_suffix}')
    plt.grid()
    plt.legend()    

    plt.tight_layout()
    plt.savefig(os.path.join(comparisons_dir, f'{prefix}_GBF.png'), format='png')
    plt.close()

    plt.figure(figsize=[14, 6])

    plt.subplot(1, 2, 1)
    plt.plot(omegas, absolute_differences, 'o-')
    plt.xlabel(r'$\omega$', fontsize=16)
    plt.ylabel('Absolute Difference', fontsize=16)
    plt.title(f'Absolute Difference — {title_suffix}')
    plt.yscale('log')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(omegas, relative_differences, 'o-')
    plt.xlabel(r'$\omega$', fontsize=16)
    plt.ylabel('Relative Difference', fontsize=16)
    plt.title(f'Relative Difference — {title_suffix}')
    plt.yscale('log')
    plt.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(comparisons_dir, f'{prefix}_Differences.png'), format='png')
    plt.close()

#Setup PINN logistics
#Seed included for reproducibility
t.manual_seed(0)
model = Model(2, 4, 32, num_hidden_layers = 3).to(device = device, dtype = DTYPE)
    
#Make various directories for saving results
base_path = f'./GBFWXData/l{mode}'
checkpoint_dir = os.path.join(base_path, 'checkpoints')
diagnostic_dir = os.path.join(base_path, 'diagnostics')
results_dir = os.path.join(base_path, 'results')
comparisons_dir = os.path.join(base_path, 'comparisons')
final_plots_dir = os.path.join(diagnostic_dir, "final")

os.makedirs(base_path, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok = True)
os.makedirs(diagnostic_dir, exist_ok = True)
os.makedirs(results_dir, exist_ok = True)
os.makedirs(comparisons_dir, exist_ok = True)
os.makedirs(final_plots_dir, exist_ok = True)

loss_history_dir = os.path.join(diagnostic_dir, "loss_history.csv")

with open(loss_history_dir, 'w', newline = '') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'stage', 'total_loss', 'flux_loss', 'ode_loss', 'ode_real_loss', 'ode_imag_loss', 'flux_weight'])

# print("="*60, flush = True)
# print(f"WS step {step_idx + 1} / {len(omega_schedule)} | l = {mode}, omega = {omega:.4f}", flush = True)
# print("="*60, flush = True)

learning_rate = 1e-3
optimiser = optim.Adam(model.parameters(), lr = learning_rate)

adam_iterations = 18000 
lbfgs_iterations = 1000 

# if is_first_frequency_of_run:
#     print("First frequency of this run- using larger training budget.")
# else:
#     print("Continuation frequency- using standard training budget.")
# print("-"*60)

hist_epochs, hist_total, hist_flux, hist_ode, hist_ode_re, hist_ode_im, hist_weight = [], [], [], [], [], [], []
diagnostic_model_states = []
GBF, probability, alphas, betas, extraction_epochs = [], [], [], [], []

# RAR parameters
RAR_INTERVAL = 1000       # refine every N Adam epochs
RAR_CANDIDATES = 5000     # candidate points tested at each refinement
RAR_ADD = 250             # worst residual points added per refinement
RAR_MAX = 2000            # maximum number of retained RAR points
RAR_MIN_DX = 1e-4
FREQUENCIES_PER_BATCH = min(6, len(omega_schedule))
N_points = 9000*FREQUENCIES_PER_BATCH
N_POINTS_PER_FREQUENCY = N_points//FREQUENCIES_PER_BATCH

# RAR points are frequency-specific
rar_points_by_omega = {round(float(omega), 4): t.empty((0, 1), dtype = DTYPE, device = device) for omega in omega_schedule}

#Adam loop
for epoch in range(adam_iterations):

    frequency_indices = t.randperm(len(omega_schedule), device = device)[:FREQUENCIES_PER_BATCH]

    batch_omegas = [float(omega_schedule[i]) for i in frequency_indices.cpu().numpy()]

    x_tensor, omega_tensor = sample_frequency_batch(omega_values = batch_omegas, n_points_per_frequency = N_POINTS_PER_FREQUENCY, x_max = x_max, dtype = DTYPE, device = device,
                            rar_points_by_omega = rar_points_by_omega)

    optimiser.zero_grad(set_to_none = True)

    flux_weight = annealing(epoch, adam_iterations)
    hist_weight.append(flux_weight)
    (Re_u_nn, Im_u_nn, flux_res, loss, loss_f, loss_o, 
    loss_ode_real, loss_ode_imag, res_ode_re, res_ode_im, P_re, P_im, Q_re, Q_im) = compute_loss(model, x_tensor, mass, mode, omega_tensor, flux_weight)

    loss.backward()
    optimiser.step()

    with open(loss_history_dir, 'a', newline = '') as f:
        writer = csv.writer(f)

        writer.writerow([epoch + 1, 'Adam', loss.item(), loss_f.item(), loss_o.item(), loss_ode_real.item(), loss_ode_imag.item(), flux_weight])

    hist_epochs.append(epoch + 1)
    hist_total.append(loss.item())
    hist_flux.append(loss_f.item())
    hist_ode.append(loss_o.item())
    hist_ode_re.append(loss_ode_real.item())
    hist_ode_im.append(loss_ode_imag.item())

    if (epoch + 1) % 100 == 0 or epoch == 0 or epoch == (adam_iterations - 1):
            extraction_epochs.append(epoch + 1)
            alpha, beta, prob, gbf = extraction(model, x_max, mass, mode, monitor_omega)
            alphas.append(alpha)
            betas.append(beta)
            probability.append(prob)
            GBF.append(gbf)

            if (epoch + 1) % 500 == 0 or epoch == 0:
                print(f"""l = {mode} | Adam Epoch: {epoch + 1} / {adam_iterations}. 
                        Batch omegas = {batch_omegas},
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

    if ((epoch + 1) % RAR_INTERVAL == 0 and (epoch + 1) < adam_iterations):

        print("-"*60)
        print(f"RAR triggered at Adam epoch {epoch + 1}: Refining the frequencies in this batch...")

        for omega in batch_omegas:

            key = round(float(omega), 4)

            current_rar_points = rar_points_by_omega[key]
            if current_rar_points.shape[0] >= RAR_MAX:
                continue

            n_add = min(RAR_ADD, RAR_MAX - current_rar_points.shape[0])

            new_rar_points, new_rar_scores = select_rar_points(model = model, n_candidates = RAR_CANDIDATES, n_add = n_add, x_max = x_max,
                mass = mass, mode = mode, omega = omega, device = device, dtype = DTYPE, existing_rar_points = current_rar_points, min_dx = RAR_MIN_DX)

            rar_points_by_omega[key] = t.cat([current_rar_points, new_rar_points], dim = 0)

            actual_added = new_rar_points.shape[0]

            print(f"omega = {omega:.4f}: added {actual_added} RAR points (requested {n_add}). Total RAR points = {rar_points_by_omega[key].shape[0]}")

            if actual_added > 0:
                print(f"Maximum selected ODE residual = {new_rar_scores.max().item():.4e}")
                print(f"RAR x range = [{new_rar_points.min().item():.6f}, {new_rar_points.max().item():.6f}]")

            print("-"*60)

print("Adam training complete. Switching to L-BFGS:", flush = True)
print("="*60)
lbfgs_optimiser = optim.LBFGS(model.parameters(), lr = 1.0, max_iter = 20,  history_size = 50, line_search_fn = "strong_wolfe")


lbfgs_frequencies = min(5, len(omega_schedule))
lbfgs_points_per_frequency = N_points//lbfgs_frequencies

lbfgs_frequency_indices = np.linspace(0, len(omega_schedule) - 1, lbfgs_frequencies, dtype = int)
lbfgs_omegas = [float(omega_schedule[i]) for i in lbfgs_frequency_indices]

x_tensor_lbfgs, omega_tensor_lbfgs = sample_frequency_batch(omega_values = lbfgs_omegas, n_points_per_frequency = lbfgs_points_per_frequency, x_max = x_max,
                                        dtype = DTYPE, device = device, rar_points_by_omega = rar_points_by_omega)

flux_weight = annealing(adam_iterations - 1, adam_iterations)

lbfgs_success = True

for epoch in range(lbfgs_iterations):
    info = {'total': 0, 'flux': 0, 'ode': 0, 'loss_re': 0, 'loss_im': 0, 'res_re': 0, 'res_im': 0}

    def closure():
        lbfgs_optimiser.zero_grad(set_to_none = True)
        (Re_u_nn, Im_u_nn, flux_res, loss, loss_f, loss_o, loss_ode_re, loss_ode_im,
        res_ode_re, res_ode_im, P_re, P_im, Q_re, Q_im) = compute_loss(model, x_tensor_lbfgs, mass, mode, omega_tensor_lbfgs, flux_weight)
        loss.backward()

        info.update({'total': loss.item(), 'flux': loss_f.item(), 'ode': loss_o.item(), 'loss_re': loss_ode_re.item(), 'loss_im': loss_ode_im.item()})

        return loss

    lbfgs_optimiser.step(closure)

    with_grad = compute_loss(model, x_tensor_lbfgs, mass, mode, omega_tensor_lbfgs, flux_weight)
    _, _, _, loss_now, lf_now, lo_now, lre, lim, *_ = with_grad
    info.update({'total': loss_now.item(), 'flux': lf_now.item(), 'ode': lo_now.item(), 'loss_re': lre.item(), 'loss_im': lim.item()})

    global_epoch = adam_iterations + epoch
    
    hist_epochs.append(global_epoch + 1)
    hist_total.append(loss_now.item())
    hist_flux.append(lf_now.item())
    hist_ode.append(lo_now.item())
    hist_ode_re.append(lre.item())
    hist_ode_im.append(lim.item())
    hist_weight.append(flux_weight)

    with open(loss_history_dir, 'a', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow([(global_epoch + 1), 'L-BFGS', loss_now.item(), lf_now.item(), lo_now.item(), lre.item(), lim.item(), flux_weight])

    if not np.isfinite(info['total']):
        print(f"L-BFGS diverged at epoch {epoch}; stopping.", flush=True)
        lbfgs_success = False
        break

    if (epoch + 1) % 40 == 0 or epoch == (lbfgs_iterations - 1):
        #Printing and plotting
        extraction_epochs.append(global_epoch + 1)
        alpha, beta, prob, gbf = extraction(model, x_max, mass, mode, monitor_omega)
        alphas.append(alpha)
        betas.append(beta)
        probability.append(prob)
        GBF.append(gbf)
        
        print(f"""l = {mode} | L-BFGS Epoch: {epoch + 1} / {lbfgs_iterations}. 
                    L-BFGS omegas = {lbfgs_omegas},
                    Diagnostic omegas = {[f"{om:.4f}" for om in diagnostic_omegas]},
                    Total scaled loss: {info['total']:.4e}, 
                    Flux loss: {info['flux']:.4e},
                    ODE loss: {info['ode']:.4e},
                    Monitor GBF: {gbf},
                    Monitor |alpha|^2 - |beta|^2: {prob}.""", flush = True)
        print("-"*60, flush = True)

        diagnostic_model_states.append({'iteration': global_epoch + 1, 'stage': 'L-BFGS', 'state_dict': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}})
        for omega_diag in diagnostic_omegas:
            save_training_diagnostics(model, epoch_number = global_epoch + 1, omega = omega_diag)

print("Conditional PINN training complete.", flush = True)


training_query_results = query_gbf(model, mass, mode, omega_schedule, x_max)
training_omegas = np.array([result['omega'] for result in training_query_results])
training_gbfs = np.array([result['gbf'] for result in training_query_results])
training_gbf_path = os.path.join(results_dir, "PINN_training_frequencies.csv")

with open(training_gbf_path, 'w', newline = '') as f:
    writer = csv.writer(f)

    writer.writerow(['omega', 'alpha_re', 'alpha_im', 'beta_re',  'beta_im', 'prob', 'PINN_GBF'])

    for result in training_query_results:
        writer.writerow([f"{result['omega']:.10f}", f"{result['alpha'].real:.12e}", f"{result['alpha'].imag:.12e}", f"{result['beta'].real:.12e}", f"{result['beta'].imag:.12e}",
                f"{result['probability']:.12e}", f"{result['gbf']:.12e}"])

GBF_global = {}
for result in training_query_results:
    key = round(result['omega'], 4)
    GBF_global[key] = result['gbf']

omega_query = np.linspace(omega_min, omega_max, args.query_steps)
dense_query_results = query_gbf(model, mass, mode, omega_query, x_max)
dense_query_omegas = np.array([result['omega'] for result in dense_query_results])
dense_query_gbfs = np.array([result['gbf'] for result in dense_query_results])
query_gbf_path = os.path.join(results_dir, "PINN_query_frequencies.csv")

with open(query_gbf_path, 'w', newline = '') as f:
    writer = csv.writer(f)

    writer.writerow(['omega', 'alpha_re', 'alpha_im', 'beta_re',  'beta_im', 'prob', 'PINN_GBF'])

    for result in dense_query_results:
            writer.writerow([f"{result['omega']:.10f}", f"{result['alpha'].real:.12e}", f"{result['alpha'].imag:.12e}", f"{result['beta'].real:12e}", f"{result['beta'].imag:.12e}",
                    f"{result['probability']:.12e}", f"{result['gbf']:.12e}"])


plt.figure(figsize = [7, 5])
plt.plot(dense_query_omegas, dense_query_gbfs, label = 'Two-input PINN')
plt.xlabel(r'$\omega$', fontsize = 16)
plt.ylabel(r"$\Gamma(\omega)$", fontsize = 16)
plt.title(f"Two-input PINN GBF, l = {mode}", fontsize = 16)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(f"./GBFWXData/l{mode}/WXPINN_GBF.png", format = 'png')
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

training_comparison_csv = os.path.join(comparisons_dir, "PINN_vs_Numerical_training.csv")
training_comparison_omegas, training_pinn_gbfs, training_numerical_gbfs, training_absolute_differences, training_relative_differences = compare_pinn_to_numerical(training_query_results,
                    numerical_results, training_comparison_csv)
save_comparison_plots(training_comparison_omegas, training_pinn_gbfs, training_numerical_gbfs, training_absolute_differences, training_relative_differences, 
                    "Training Frequencies", "training")

query_comparison_csv = os.path.join(comparisons_dir, "PINN_vs_Numerical_query.csv")
query_comparison_omegas, query_pinn_gbfs, query_numerical_gbfs, query_absolute_differences, query_relative_differences = compare_pinn_to_numerical(dense_query_results,
                    numerical_results, query_comparison_csv) 
save_comparison_plots(query_comparison_omegas, query_pinn_gbfs, query_numerical_gbfs, query_absolute_differences, query_relative_differences, "Dense Query Frequencies", "query")

plt.figure(figsize=[7, 5])

plt.plot(training_comparison_omegas, training_numerical_gbfs, 'o-', label='Numerical')
plt.plot(training_comparison_omegas, training_pinn_gbfs, 'x-', label='Two-input PINN')
plt.xlabel(r'$\omega$', fontsize = 16)
plt.ylabel(r'$\Gamma(\omega)$', fontsize = 16)
plt.title(f'Grey-Body Factor, l = {mode}')
plt.grid()
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(comparisons_dir, 'Final_GBF_Comparison_training.png'), format='png')
plt.close()

plt.figure(figsize=[7, 5])

plt.plot(query_comparison_omegas, query_numerical_gbfs, 'o-', label='Numerical')
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

_, _, _, final_total_loss, final_flux_loss, final_ode_loss, _, _, *_ = final_loss_data

training_relative_finite = training_relative_differences[np.isfinite(training_relative_differences)]
query_relative_finite = query_relative_differences[np.isfinite(query_relative_differences)]

result_file_path = os.path.join(results_dir, 'result.txt')

with open(result_file_path, 'w') as f:
    f.write("Two-input PINN\n")
    f.write(f"mass = {mass}\n")
    f.write(f"l = {mode}\n")
    f.write(f"x_extract = {x_max}\n")

    f.write(f"omega_min = {omega_min:.10f}\n")
    f.write(f"omega_max = {omega_max:.10f}\n")
    f.write(f"training_frequencies = {len(omega_schedule)}\n")
    f.write(f"query_frequencies = {len(omega_query)}\n")
    f.write("diagnostic_frequencies = " + ", ".join(f"{om:.10f}" for om in diagnostic_omegas) + "\n")

    f.write(f"final_total_loss = {final_total_loss.item():.12e}\n")
    f.write(f"final_ODE_loss = {final_ode_loss.item():.12e}\n")
    f.write(f"final_flux_loss = {final_flux_loss.item():.12e}\n")

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

    f.write(f"training_mean_relative_error = {np.mean(training_relative_finite):.12e}\n")
    f.write(f"training_max_relative_error = {np.max(training_relative_finite):.12e}\n")
    f.write(f"query_mean_relative_error = {np.mean(query_relative_finite):.12e}\n")
    f.write(f"query_max_relative_error = {np.max(query_relative_finite):.12e}\n")

    f.write(f"lbfgs_success = {lbfgs_success}\n")

checkpoint = {'model_state_dict': model.state_dict(),
            'model_architecture': {'in_channels': 2, 'out_channels': 4, 'hidden_channels': 32, 'hidden_layers': 3},
            'activation_config': {'type': 'keerie_adaptive_tanh', 'n': 10.0},
            'omega_input_scaling': {'omega_min': omega_min, 'omega_max': omega_max},
            'rar_config': {'interval': RAR_INTERVAL, 'candidates': RAR_CANDIDATES, 'add_per_refinement': RAR_ADD, 'max_points': RAR_MAX},
            'ansatz_config': {'nn_output_scale': 1.0, 'x_power': 3, 'x_safe_min': 1e-12, 'x_safe_max': 1 - 1e-3},
            'mass': mass,
            'mode': mode,
            'x_max': x_max,

            'omega_schedule': omega_schedule.tolist(),
            'query_steps': args.query_steps,
            'diagnostic_omegas': diagnostic_omegas,
            'monitor_omega': monitor_omega,
            'x_diagnostic': x_diagnostic.tolist(),

            'adam_iterations': adam_iterations,
            'lbfgs_iterations': lbfgs_iterations,

            'flux_weight_initial': 10.0,
            'flux_weight_final': 1.0,

            'rar_points_by_omega': {key: value.detach().cpu() for key, value in rar_points_by_omega.items()},

            'training_history': {'epochs': [int(value) for value in hist_epochs], 'total_loss': [float(value) for value in hist_total], 'flux_loss': [float(value) for value in hist_flux],
                        'ode_loss': [float(value) for value in hist_ode], 'ode_real_loss': [float(value) for value in hist_ode_re], 'ode_imag_loss': [float(value) for value in hist_ode_im],
                        'flux_weight': [float(value) for value in hist_weight]},

            'extraction_history': {'epochs': [int(value) for value in extraction_epochs], 'alpha_re': [float(value.real) for value in alphas], 'alpha_im': [float(value.imag) for value in alphas],
                                    'beta_re': [float(value.real) for value in betas], 'beta_im': [float(value.imag) for value in betas], 'probability': [float(value) for value in probability],
                                    'gbf': [float(value) for value in GBF]},

            'training_query_results': serialise_query_results(training_query_results),
            'dense_query_results': serialise_query_results(dense_query_results),

            'numerical_results': [{'omega': float(omega), 'gbf': float(gbf)} for omega, gbf in sorted(numerical_results.items())],

            'training_comparison': {'omega': training_comparison_omegas.tolist(), 'pinn_gbf': training_pinn_gbfs.tolist(), 'numerical_gbf': training_numerical_gbfs.tolist(),
                                'absolute_difference': training_absolute_differences.tolist(), 'relative_difference': training_relative_differences.tolist()},

            'query_comparison': {'omega': query_comparison_omegas.tolist(), 'pinn_gbf': query_pinn_gbfs.tolist(), 'numerical_gbf': query_numerical_gbfs.tolist(),
                                'absolute_difference': query_absolute_differences.tolist(), 'relative_difference': query_relative_differences.tolist()},

            'diagnostic_model_states': diagnostic_model_states,
            'lbfgs_success': bool(lbfgs_success),
            'training_complete': True}

checkpoint_path = os.path.join(checkpoint_dir, f'pinn_conditional_GBFWX_l{mode}.pth')

t.save(checkpoint, checkpoint_path)

print(f"Checkpoint saved to {checkpoint_path}", flush = True)