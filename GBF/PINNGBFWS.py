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

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "text.usetex": False
})

#Argument parser
#mode is value of l, omega is value of omega and check is included to check system initialises correctly
parser = argparse.ArgumentParser(description = "Train PINN for specific mode l")
parser.add_argument('--mode', type = int, required = True, help = 'The value of l (mode)')
parser.add_argument('--omega_start', type = float, default = 0.3, help = 'Initial (higher) frequency')
parser.add_argument('--omega_final', type = float, default = 0.03, help = 'Final (lower) frequency')
parser.add_argument('--num_steps', type = int, default = 10, help = "Number of warm-start steps")
parser.add_argument('--resume', action = 'store_true', help = 'Resume from the latest warm-start checkpoint')

args = parser.parse_args()
mode = args.mode

if args.omega_start < args.omega_final:
    raise ValueError("omega_final must be less than omega_start")

#Frequency schedule from high to low for continuation
omega_schedule = np.linspace(args.omega_start, args.omega_final, args.num_steps)

print(f"Initialising training for omega = {omega_schedule}")

print("Not safe to train yet!")
# raise
# mode = 2
# omega = 0.3

#Global data type
DTYPE = t.float64
NP_DTYPE = np.float32 if DTYPE == t.float32 else np.float64

#GPU capabilities
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
t.set_num_threads(4)
print(f"Using device: {device}", flush = True)

#Set up domain and BH mass
epsilon = 1e-8
mass = 0.5
x_max = 0.95 
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
    imag = -4*M*omega*t.ones_like(x)
    return real, imag
        
def C(x, l):
    return -l*(l + 1) + 3*(1 - x)

#dx/dr_star
def g(x, M):
    return x*(1 - x)**2/(2*M)

#Coefficients obtained via Taylor expansion of u_1 at x = 0 (regular singular point)
def taylor_coeffs(mass, mode, omega):
    Lambda = mode*(mode + 1)
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

#Ansatz for the wave-function u
#Ansatz is of the form u(x) = 1 + c1*x + c2*x**2 + 100*(P + exp(2i*omega*r_star)*Q) 
#where P and Q are complex with components corresponding to the four channel neural network output
def ansatz(model, x_tensor, mass, mode, omega):
    c1_re, c1_im, c2_re, c2_im = taylor_coeffs(mass, mode, omega)
    NN = model(x_tensor)
    P_re, P_im, Q_re, Q_im = NN[:, 0:1], NN[:, 1:2], NN[:, 2:3], NN[:, 3:4]
    x_safe = x_tensor.clamp(min = 1e-12, max = 1 - 1e-3)
    rstar = 2*mass/(1 - x_safe) + 2*mass*t.log(x_safe/(1 - x_safe))
    cs, sn = t.cos(2*omega*rstar), t.sin(2*omega*rstar)

    u_re = 1 + c1_re*x_tensor + c2_re*x_tensor**2 + 100.0*x_tensor**3*(P_re + Q_re*cs - Q_im*sn)
    u_im = c1_im*x_tensor + c2_im*x_tensor**2 + 100.0*x_tensor**3*(P_im + Q_im*cs + Q_re*sn)
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

#Setup PINN logistics
#Seed included for reproducibility
t.manual_seed(0)
model = Model(1, 4, 32, num_hidden_layers = 3).to(device = device, dtype = DTYPE)
GBF_global = {}

resume_path = os.path.join(f"./GBFWSData/l{mode}", "latest_warm_start_checkpoint.pth")
start_step = 0

if args.resume and os.path.exists(resume_path):
    print("Previous model exists...")
    print(f"Loading warm start checkpoint {resume_path}", flush = True)
    print('-'*60)

    checkpoint = t.load(resume_path, map_location = device, weights_only = False)

    if checkpoint['mode'] != mode:
        raise ValueError(f"Checkpoint is for l = {checkpoint['mode']}, current run requested l={mode}.")

    if not np.isclose(checkpoint['mass'], mass):
        raise ValueError("Checkpoint mass does not match current run.")

    if not np.isclose(checkpoint['x_max'], x_max):
        raise ValueError("Checkpoint x_max does not match current run.")

    model.load_state_dict(checkpoint['model_state_dict'])
    loaded_GBF_global = checkpoint.get("GBF_global", {})

    if isinstance(loaded_GBF_global, dict):
        GBF_global = loaded_GBF_global

    else:
        old_schedule = checkpoint.get("omega_schedule", [])

        if len(old_schedule) != len(loaded_GBF_global):
            raise ValueError("Old checkpoint contains list-based GBF_global but its omega_schedule is incompatible.")

        GBF_global = {round(float(omega), 4): float(gbf) for omega, gbf in zip(old_schedule, loaded_GBF_global)}

    completed_omegas = set(GBF_global.keys())

    remaining_steps = [i for i, omega in enumerate(omega_schedule) if round(float(omega), 4) not in completed_omegas]

    if len(remaining_steps) == 0:
        start_step = len(omega_schedule)
        print("No frequencies remaining in the schedule.")
    else:
        start_step = remaining_steps[0]
        print(f"Next omega = {omega_schedule[start_step]:.4f}")

    print(f"Resuming from  omega = {checkpoint['omega']:.4f}", flush = True)

    # if start_step < len(omega_schedule):
    #     print(f"Next omega = {omega_schedule[start_step]:.4f}")

for step_idx in range(start_step, len(omega_schedule)):
    omega = float(omega_schedule[step_idx])
    is_first_frequency_of_run = (step_idx == start_step)
    
    #Make various directories for saving results
    base_path = f'./GBFWSData/l{mode}/omega{omega:.4f}'
    out_dir = os.path.join(base_path, 'NNOutput')
    loss_dir = os.path.join(base_path, 'Loss')
    flux_dir = os.path.join(base_path, 'Flux')
    final_plots_dir = os.path.join(base_path, 'FinalPlots')
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(out_dir, exist_ok = True)
    os.makedirs(loss_dir, exist_ok = True)
    os.makedirs(flux_dir, exist_ok = True)
    os.makedirs(final_plots_dir, exist_ok = True)

    print("="*60, flush = True)
    print(f"WS step {step_idx + 1} / {len(omega_schedule)} | l = {mode}, omega = {omega:.4f}", flush = True)
    print("="*60, flush = True)

    learning_rate = 1e-3
    optimiser = optim.Adam(model.parameters(), lr = learning_rate)

    adam_iterations = 18000 if is_first_frequency_of_run else 10000
    lbfgs_iterations = 1000 if is_first_frequency_of_run else 800

    if is_first_frequency_of_run:
        print("First frequency of this run- using larger training budget.")
    else:
        print("Continuation frequency- using standard training budget.")
    print("-"*60)

    hist_total, hist_flux, hist_ode, hist_ode_re, hist_ode_im, hist_weight = [], [], [], [], [], []
    GBF, probability, alphas, betas, extraction_epochs = [], [], [], [], []
    N_points = 10000

    #Adam loop
    for epoch in range(adam_iterations):
        optimiser.zero_grad(set_to_none = True)
        N_uniform = int(0.6*N_points)
        x_uniform = x_max*t.rand((N_uniform, 1), dtype = DTYPE, device = device)

        N_edges = N_points - N_uniform
        r_h, r_far =2*mass, 2*mass/(1 - x_max)
        r_samp = r_h + (r_far - r_h)*t.rand((N_edges, 1), dtype = DTYPE, device = device)
        x_edges = 1 - 2*mass/r_samp

        x_tensor = t.cat([x_uniform, x_edges], dim = 0).requires_grad_(True)

        flux_weight = annealing(epoch, adam_iterations)
        hist_weight.append(flux_weight)
        (Re_u_nn, Im_u_nn, flux_res, loss, loss_f, loss_o, 
        loss_ode_real, loss_ode_imag, res_ode_re, res_ode_im, P_re, P_im, Q_re, Q_im) = compute_loss(model, x_tensor, mass, mode, omega, flux_weight)

        loss.backward()
        optimiser.step()

        hist_total.append(loss.item())
        hist_flux.append(loss_f.item())
        hist_ode.append(loss_o.item())
        hist_ode_re.append(loss_ode_real.item())
        hist_ode_im.append(loss_ode_imag.item())

        if (epoch + 1) % 100 == 0 or epoch == 0 or epoch == (adam_iterations - 1):
                extraction_epochs.append(epoch)
                alpha, beta, prob, gbf = extraction(model, x_max, mass, mode, omega)
                alphas.append(alpha)
                betas.append(beta)
                probability.append(prob)
                GBF.append(gbf)
                if (epoch + 1) % 500 ==0 or epoch == 0:
                    print(f"""l = {mode}, omega = {omega:.4f} | Adam Epoch: {epoch + 1} / {adam_iterations}. Total scaled loss: {loss.item():.4e}, 
                            Flux loss: {loss_f.item():.4e},
                            ODE loss: {loss_o.item():.4e},
                            Real component of ODE loss: {loss_ode_real.item():.4e}, 
                            Imaginary component of ODE loss: {loss_ode_imag.item():.4e},
                            Current value of alpha: {alpha.real:.5f} + {alpha.imag:.5f}i,
                            Current value of beta: {beta.real:.5f} + {beta.imag:.5f}i,
                            Current value of |alpha|^2 - |beta|^2: {prob},
                            Current value of GBF: {gbf}.""", flush = True)
                    print("-"*60, flush = True)

        if (epoch + 1) % 1000 == 0:
                x_np = x_tensor.cpu().detach().numpy().flatten()
                idx = np.argsort(x_np)
        
                res_re_plot = res_ode_re.cpu().detach().numpy().flatten()
                res_im_plot = res_ode_im.cpu().detach().numpy().flatten()
                u_re_plot = Re_u_nn.cpu().detach().numpy().flatten()
                u_im_plot = Im_u_nn.cpu().detach().numpy().flatten()
                flux_plot = flux_res.cpu().detach().numpy().flatten()
        
                plt.figure()
                plt.plot(x_np[idx], res_re_plot[idx], color = 'blue', label = r'$\Re (Res_{ODE})$')
                plt.plot(x_np[idx], res_im_plot[idx], color = 'green', label = r'$\Im (Res_{ODE})$')
                plt.plot(x_np[idx], u_re_plot[idx], color = 'orange', label = r'$\Re (u_{NN})$')
                plt.plot(x_np[idx], u_im_plot[idx], color = 'red', label = r'$\Im (u_{NN})$')
                plt.xlabel('x', fontsize = 25)
                plt.ylabel('Output', fontsize = 25)
                plt.title(f"l = {mode}, omega = {omega:.4f}", fontsize = 20)
                plt.grid()
                plt.legend(fontsize = 15, loc = 'best')
                plt.tight_layout()
                plt.savefig(f'{out_dir}/Output_Epoch_{epoch + 1}.png', format = 'png')
                plt.close()
        
                plt.figure()
                plt.plot(hist_total, label = 'Total')
                plt.plot(hist_flux, label = 'Flux')
                plt.plot(hist_ode, label = 'ODE')
                plt.plot(hist_ode_re, label = 'Real component of ODE')
                plt.plot(hist_ode_im, label = 'Imaginary component of ODE')
                plt.yscale('log')
                plt.ylabel('Loss', fontsize = 25)
                plt.xlabel('Epoch', fontsize = 25)
                plt.title(f"l = {mode}, omega = {omega:.4f}", fontsize = 20)
                plt.legend(fontsize = 15, loc = 'best')
                plt.grid()
                plt.tight_layout()
                plt.savefig(f'{loss_dir}/Loss_Epoch_{epoch + 1}.png', format = 'png')
                plt.close()
        
                plt.figure()
                plt.plot(x_np[idx], flux_plot[idx], color = 'purple', label = 'Flux Residual')
                plt.axhline(0.0, color = 'cyan', linestyle = '--', label = 'Target')
                plt.xlabel('x', fontsize = 25)
                plt.ylabel(r'Flux Residual', fontsize = 25)
                plt.title(f"l = {mode}, omega = {omega:.4f}", fontsize = 20)
                # plt.yscale()
                plt.grid()
                plt.legend(fontsize = 15, loc = 'best')
                plt.tight_layout()
                plt.savefig(f'{flux_dir}/Flux_Residual_Epoch_{epoch + 1}.png', format = 'png')
                plt.close()
        
                #Save current checkpoint in case of a crash or blow up 
                t.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                            'GBF': GBF, 'probability': probability},
                os.path.join(base_path, 'checkpoint_latest.pth'))

    plt.figure()
    plt.plot(hist_weight)
    plt.xlabel('Epoch')
    plt.ylabel(r'$\lambda_{\mathrm{flux}}$')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{base_path}/FluxWeights.png', format = 'png')
    plt.close()

    print("Adam training complete. Switching to L-BFGS:", flush = True)
    print("="*60)
    lbfgs_optimiser = optim.LBFGS(model.parameters(), lr = 1.0, max_iter = 20,  history_size = 50, line_search_fn = "strong_wolfe")

    N_uniform = int(0.6*N_points)
    x_uniform = x_max*t.rand((N_uniform, 1), dtype = DTYPE, device = device)

    N_edges = N_points - N_uniform
    r_h, r_far = 2*mass, 2*mass/(1 - x_max)
    r_samp = r_h + (r_far - r_h)*t.rand((N_edges, 1), dtype = DTYPE, device = device)
    x_edges = 1 - 2*mass/r_samp

    x_tensor_lbfgs = t.cat([x_uniform, x_edges], dim = 0)
    x_tensor_lbfgs.requires_grad_(True)

    flux_weight = annealing(adam_iterations - 1, adam_iterations)

    for epoch in range(lbfgs_iterations):
        info = {'total': 0, 'flux': 0, 'ode': 0, 'loss_re': 0, 'loss_im': 0, 'res_re': 0, 'res_im': 0}
        plot_data = {}

        def closure():
            lbfgs_optimiser.zero_grad(set_to_none = True)
            (Re_u_nn, Im_u_nn, flux_res, loss, loss_f, loss_o, loss_ode_re, loss_ode_im,
            res_ode_re, res_ode_im, P_re, P_im, Q_re, Q_im) = compute_loss(model, x_tensor_lbfgs, mass, mode, omega, flux_weight)
            loss.backward()

            info.update({'total': loss.item(), 'flux': loss_f.item(), 'ode': loss_o.item(), 'loss_re': loss_ode_re.item(), 'loss_im': loss_ode_im.item()})

            plot_data['x'] = x_tensor_lbfgs.cpu().detach().numpy()
            plot_data['re_u'] = Re_u_nn.cpu().detach().numpy()
            plot_data['im_u'] = Im_u_nn.cpu().detach().numpy()
            plot_data['flux'] = flux_res.cpu().detach().numpy()
            plot_data['res_re'] = res_ode_re.cpu().detach().numpy()
            plot_data['res_im'] = res_ode_im.cpu().detach().numpy()
            plot_data['P_re'] = P_re.cpu().detach().numpy()
            plot_data['P_im'] = P_im.cpu().detach().numpy()
            plot_data['Q_re'] = Q_re.cpu().detach().numpy()
            plot_data['Q_im'] = Q_im.cpu().detach().numpy()

            return loss

        lbfgs_optimiser.step(closure)

        with_grad = compute_loss(model, x_tensor_lbfgs, mass, mode, omega, flux_weight)
        _, _, _, loss_now, lf_now, lo_now, lre, lim, *_ = with_grad
        info.update({'total': loss_now.item(), 'flux': lf_now.item(), 'ode': lo_now.item(), 'loss_re': lre.item(), 'loss_im': lim.item()})

        if not np.isfinite(info['total']):
            print(f"L-BFGS diverged at epoch {epoch}; stopping.", flush=True)
            break

        if (epoch + 1) % 40 == 0 or epoch == (lbfgs_iterations - 1):
                #Printing and plotting
                extraction_epochs.append(epoch + adam_iterations)
                alpha, beta, prob, gbf = extraction(model, x_max, mass, mode, omega)
                alphas.append(alpha)
                betas.append(beta)
                probability.append(prob)
                GBF.append(gbf)
                
                print(f"""l = {mode}, omega = {omega:.4f} | L-BFGS Epoch: {epoch + 1} / {lbfgs_iterations}. Total scaled loss: {info['total']:.4e}, 
                            Flux loss: {info['flux']:.4e},
                            ODE loss: {info['ode']:.4e},
                            Real component of loss: {info['loss_re']:.4e}, 
                            Imaginary component of loss: {info['loss_im']:.4e}, 
                            Current value of alpha: {alpha.real:.5f} + {alpha.imag:.5f}i,
                            Current value of beta: {beta.real:.5f} + {beta.imag:.5f}i,
                            Current value of |alpha|^2 - |beta|^2: {prob},
                            Current value of GBF: {gbf}.""", flush = True)
                print("-"*60, flush = True)

                x_plot = plot_data['x'].flatten()
                idx = np.argsort(x_plot)
        
                plt.figure()
                plt.plot(x_plot[idx], plot_data['res_re'].flatten()[idx], color = 'blue', label = r'$\Re (Res_{ODE})$')
                plt.plot(x_plot[idx], plot_data['res_im'].flatten()[idx], color = 'green', label = r'$\Im (Res_{ODE})$')
                plt.plot(x_plot[idx], plot_data['re_u'].flatten()[idx], color = 'orange', label = r'$\Re (u_{NN})$')
                plt.plot(x_plot[idx], plot_data['im_u'].flatten()[idx], color = 'red', label = r'$\Im (u_{NN})$')
                plt.xlabel('x', fontsize = 25)
                plt.ylabel('Output', fontsize = 25)
                plt.title(f"l = {mode}, omega = {omega:.4f}", fontsize = 20)
                plt.grid()
                plt.legend(fontsize = 25, loc = 'best')
                plt.tight_layout()
                plt.savefig(f'{out_dir}/Output_Epoch_{epoch + 1 + adam_iterations}.png', format = 'png')
                plt.close()
            
                plt.figure()
                plt.plot(hist_total, label = 'Total')
                plt.plot(hist_flux, label = 'Flux')
                plt.plot(hist_ode, label = 'ODE')
                plt.plot(hist_ode_re, label = 'Real component')
                plt.plot(hist_ode_im, label = 'Imaginary component')
                plt.yscale('log')
                plt.ylabel('Loss', fontsize = 25)
                plt.xlabel('Epoch', fontsize = 25)
                plt.title(f"l = {mode}, omega = {omega:.4f}", fontsize = 20)
                plt.legend(fontsize = 15, loc = 'best')
                plt.grid()
                plt.tight_layout()
                plt.savefig(f'{loss_dir}/Loss_Epoch_{epoch + 1 + adam_iterations}.png', format = 'png')
                plt.close()
        
                plt.figure()
                plt.plot(x_plot[idx], plot_data['flux'].flatten()[idx], color = 'purple', label = 'Flux Residual')
                plt.axhline(0.0, color = 'cyan', linestyle = '--', label = 'Target')
                plt.xlabel('x', fontsize = 25)
                plt.ylabel(r'Flux Residual', fontsize = 25)
                plt.title(f"l = {mode}, omega = {omega:.4f}", fontsize = 20)
                # plt.yscale('symlog')
                plt.grid()
                plt.legend(fontsize = 15, loc = 'best')
                plt.tight_layout()
                plt.savefig(f'{flux_dir}/Flux_Residual_Epoch_{epoch + 1 + adam_iterations}.png', format = 'png')
                plt.close()

    print(f"Training complete for l = {mode}, omega = {omega:.4f}. Plotting results...)")
    print("="*60)

    r_plot = 2*mass/(1 - x_plot)
    #Final plots after training
    plt.figure(figsize = [20, 10])
    plt.subplot(1, 2, 1)
    plt.suptitle("Direct Neural Network Output"
        "\n"
        f"l = {mode}, omega = {omega:.4f}")
    plt.plot(x_plot[idx], plot_data['P_re'].flatten()[idx], label = 'Re(P)')
    plt.plot(x_plot[idx], plot_data['P_im'].flatten()[idx], label = 'Im(P)')
    plt.plot(x_plot[idx], plot_data['Q_re'].flatten()[idx], label = 'Re(Q)')
    plt.plot(x_plot[idx], plot_data['Q_im'].flatten()[idx], label = 'Im(Q)')
    plt.xlabel('x', fontsize = 20)
    plt.ylabel('P and Q', fontsize = 20)
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    plt.plot(r_plot[idx], plot_data['P_re'].flatten()[idx], label = 'Re(P)')
    plt.plot(r_plot[idx], plot_data['P_im'].flatten()[idx], label = 'Im(P)')
    plt.plot(r_plot[idx], plot_data['Q_re'].flatten()[idx], label = 'Re(Q)')
    plt.plot(r_plot[idx], plot_data['Q_im'].flatten()[idx], label = 'Im(Q)')
    plt.xlabel('r', fontsize = 20)
    plt.ylabel('P and Q', fontsize = 20)
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(f'{final_plots_dir}/PQ.png', format = 'png')
    plt.close()

    plt.figure(figsize  = [20, 10])
    plt.subplot(1, 2, 1)
    plt.suptitle("Wave function u built via ansatz of P and Q"
        "\n"
        f"l = {mode}, omega = {omega:.4f}")
    plt.plot(x_plot[idx], plot_data['re_u'].flatten()[idx], color = 'orange', label = r'$\Re (u_{NN})$')
    plt.plot(x_plot[idx], plot_data['im_u'].flatten()[idx], color = 'red', label = r'$\Im (u_{NN})$')
    plt.xlabel('x', fontsize = 20)
    plt.ylabel(r'$u(x)$', fontsize = 20)
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    plt.plot(r_plot[idx], plot_data['re_u'].flatten()[idx], color = 'orange', label = r'$\Re (u_{NN})$')
    plt.plot(r_plot[idx], plot_data['im_u'].flatten()[idx], color = 'red', label = r'$\Im (u_{NN})$')
    plt.xlabel('r', fontsize = 20)
    plt.ylabel('u(r)', fontsize = 20)
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(f'{final_plots_dir}/u.png', format = 'png')
    plt.close()

    plt.figure(figsize = [20, 10])
    plt.subplot(1, 2, 1)
    plt.suptitle("ODE Residual"
        "\n"
        f"l = {mode}, omega = {omega:.4f}")
    plt.plot(x_plot[idx], plot_data['res_re'].flatten()[idx], label = 'Re(res)')
    plt.plot(x_plot[idx], plot_data['res_im'].flatten()[idx], label = 'Im(res)')
    plt.xlabel('x', fontsize = 20)
    plt.ylabel('Residual', fontsize = 20)
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    plt.plot(r_plot[idx], plot_data['res_re'].flatten()[idx], label = 'Re(res)')
    plt.plot(r_plot[idx], plot_data['res_im'].flatten()[idx], label = 'Im(res)')
    plt.xlabel('r', fontsize = 20)
    plt.ylabel('Residual', fontsize = 20)
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(f'{final_plots_dir}/Residuals.png', format = 'png')
    plt.close()

    alphas = np.array(alphas)
    alpha_real_array, alpha_imag_array = alphas.real, alphas.imag
    betas = np.array(betas)
    beta_real_array, beta_imag_array = betas.real, betas.imag

    plt.figure(figsize = [7, 7])
    plt.plot(alphas.real, alphas.imag, 'r--', alpha = 0.9)
    plt.scatter(alphas[0].real, alphas[0].imag, color='blue', label = f'Initial: {alphas[0].real:.3f} + {alphas[0].imag:.3f}i')
    plt.scatter(alphas[-1].real, alphas[-1].imag, color='green', label = f'Final: {alphas[-1].real:.3f} + {alphas[-1].imag:.3f}i')
    plt.xlabel(r'$\Re(\alpha)$', fontsize = 18)
    plt.ylabel(r'$\Im(\alpha)$', fontsize = 18)
    plt.title(f'l = {mode}, omega = {omega:.4f}', fontsize = 18)
    plt.tight_layout()
    plt.grid()
    plt.legend()
    plt.savefig(f'{final_plots_dir}/alpha_convergence.png', format = 'png')
    plt.close()

    plt.figure(figsize = [7, 7])
    plt.plot(betas.real, betas.imag, 'g--', alpha = 0.9)
    plt.scatter(betas[0].real, betas[0].imag, color='blue', label = f'Initial: {betas[0].real:.3f} + {betas[0].imag:.3f}i')
    plt.scatter(betas[-1].real, betas[-1].imag, color='green', label = f'Final: {betas[-1].real:.3f} + {betas[-1].imag:.3f}i')
    plt.xlabel(r'$\Re(\beta)$', fontsize = 18)
    plt.ylabel(r'$\Im(\beta)$', fontsize = 18)
    plt.title(f'l = {mode}, omega = {omega:.4f}', fontsize = 18)
    plt.tight_layout()
    plt.grid()
    plt.legend()
    plt.savefig(f'{final_plots_dir}/beta_convergence.png', format = 'png')
    plt.close()

    fig, ax1 = plt.subplots(figsize = [7, 4.5])

    ax1.plot(extraction_epochs, GBF, color = 'blue', label = r'$\Gamma$')
    ax1.scatter(extraction_epochs[-1], GBF[-1], marker = 'o', s = 30, color = 'lime', label = r"Final $\Gamma$")
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch', fontsize = 14)
    ax1.set_ylabel(r'$\Gamma$', fontsize = 16)
    ax1.tick_params(axis = 'y')
    # ax1.axhline(7.0011982e-05, color = 'blue', linestyle = ':', linewidth = 1,
    #             label = r'$\Gamma_{\rm ref}$')
    ax1.grid(alpha = 0.3)

    ax2 = ax1.twinx()
    ax2.plot(extraction_epochs, probability, color = 'red', label = r'$|\alpha|^2 - |\beta|^2$')
    ax2.scatter(extraction_epochs[-1], probability[-1], marker = 'o', s = 30, color = 'magenta', label = r'Final $|\alpha|^2 - |\beta|^2$')
    ax2.set_ylabel(r'$|\alpha|^2 - |\beta|^2$', fontsize = 16)
    ax2.tick_params(axis = 'y')
    ax2.axhline(1.0, color = 'red', linestyle = ':', linewidth = 1, label = r'Target $|\alpha|^2 - |\beta|^2$')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize = 11, loc = 'best')
    plt.title(f'l = {mode}, omega = {omega:.4f}', fontsize = 16)
    plt.tight_layout()
    plt.savefig(f'{final_plots_dir}/GBFProb.png', format = 'png')
    plt.close()

    print(f"Plots saved to {final_plots_dir}")
    print("="*60)
    print("Saving results and trained model...")
    print("="*60)

    final_alpha, final_beta, final_prob, final_gbf = extraction(model, x_max, mass, mode, omega)
    GBF_global[round(omega, 4)] = final_gbf
    T = 1/final_alpha
    R = final_beta/final_alpha

    result_file_path = os.path.join(base_path, 'result.txt')
    with open(result_file_path, 'w') as f:
        f.write(f"l = {int(mode)}\n")
        f.write(f"omega = {omega:.4f}\n")
        f.write(f"final ODE loss = {info['ode']:.4e}\n")
        f.write(f"final flux loss = {info['flux']:.4e}\n")
        f.write(f"alpha_re = {final_alpha.real:.10f}\n")
        f.write(f"alpha_im = {final_alpha.imag:.10f}\n")
        f.write(f"beta_re = {final_beta.real:.10f}\n")
        f.write(f"beta_im = {final_beta.imag:.10f}\n")
        f.write(f"T_re = {T.real:.10f}\n")
        f.write(f"T_im = {T.imag:.10f}\n")
        f.write(f"R_re = {R.real:.10f}\n")
        f.write(f"R_im = {R.imag:.10f}\n")
        f.write(f"Prob = {final_prob:.10f}\n")
        f.write(f"GBF = {final_gbf:.10e}\n")

    checkpoint = {'model_state_dict': model.state_dict(),
            'model_architecture': {'in_channels': 1, 'out_channels': 4, 'hidden_channels': 32,  'hidden_layers': 3},
            'mode': mode,
            'omega': omega,
            'step_idx': step_idx,
            'next_step_idx': step_idx + 1,
            'omega_schedule': omega_schedule.tolist(),
            'mass': mass,
            'x_max': x_max,
            'dtype': str(DTYPE),
            'adam_iterations': adam_iterations,
            'lbfgs_iterations': lbfgs_iterations,
            'flux_weight_initial': 10.0,
            'flux_weight_final': 1.0,
            'final_alpha': final_alpha,
            'final_beta': final_beta,
            'final_prob': final_prob,
            'final_gbf': final_gbf,
            'GBF_global': GBF_global}
        
    checkpoint_path = os.path.join(base_path, f'pinn_checkpoint_GBFWS_l{mode}_omega{omega:.4f}.pth')
    t.save(checkpoint, checkpoint_path)

    #Save and update the most recent warm-start checkpoint
    resume_path = os.path.join(f'./GBFWSData/l{mode}', 'latest_warm_start_checkpoint.pth')
    t.save(checkpoint, resume_path)
    print(f"Checkpoint saved to {checkpoint_path}", flush=True)

print(f'WS training complete.', flush = True)
print(f"l = {mode} mode training completed successfully.", flush = True)
print("="*60)

plot_data_global = sorted(GBF_global.items())
plot_omegas = [omega for omega, gbf in plot_data_global]
plot_GBFs = [gbf for omega, gbf in plot_data_global]

plt.figure(figsize = [6,4])
plt.plot(plot_omegas, plot_GBFs, 'o-', color = 'red', label = 'Grey-body Factor')
plt.xlabel(r'$\omega$', fontsize = 16)
plt.ylabel(r'$\Gamma \left(\omega \right)$', fontsize = 16)
plt.title(f'The Grey-Body Factor for l = {mode}'
        "\n"
        "Obtained via PINN")
plt.grid()
plt.legend()
plt.savefig(f"./GBFWSData/l{mode}/GreyBodyFactor.png", format = 'png')
plt.close()

print(f"Full Grey-Body Factor figure saved to ./GBFWSData/l{mode}/GreyBodyFactor.png")