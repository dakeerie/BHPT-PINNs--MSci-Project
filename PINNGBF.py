import torch as t
import numpy as np 
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
import torch.distributions as dist
import os
import argparse
from Functions import *
from matplotlib import rc

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "text.usetex": False
})

parser = argparse.ArgumentParser(description = "Train PINN for specific mode l")
parser.add_argument('--mode', type = int, required = True, help = 'The value of l (mode)')
parser.add_argument('--omega', type = float, required = True, help = 'Incident wave frequency')
parser.add_argument('--check', action = 'store_true', help = 'Run a quick sanity check and exit without training')

args = parser.parse_args()
mode = args.mode
omega = args.omega

DTYPE = t.float64
NP_DTYPE = np.float32 if DTYPE == t.float32 else np.float64

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
t.set_num_threads(4)
print(f"Using device: {device}", flush = True)

#Save utilities
base_path = f'./GBFData/l{mode}/omega{omega}'

out_dir = os.path.join(base_path, 'NNOutput')
loss_dir = os.path.join(base_path, 'Loss')
flux_dir = os.path.join(base_path, 'Flux')
os.makedirs(base_path, exist_ok=True)
os.makedirs(out_dir, exist_ok = True)
os.makedirs(loss_dir, exist_ok = True)
os.makedirs(flux_dir, exist_ok = True)

# mode = 2
# omega = 0.3
epsilon = 1e-8
mass = 0.5
x_max = 0.95 
rstar_max = r_to_rstar(x_to_r(x_max, mass), mass)
rstar_max_tensor = t.tensor(rstar_max, requires_grad = True, dtype = DTYPE, device = device).view(-1, 1)

#Useful quantities
# O = 4*mass*omega
# L = mode*(mode + 1)

#PINN architecture
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

class keerie_adaptive_tanh(nn.Module):
    def __init__(self, num_features, n = 10.0):
        super().__init__()
        self.beta = nn.Parameter(t.zeros(1, num_features))
        self.a = nn.Parameter(t.ones(1, num_features)/n)
        self.n = n

    def forward(self, x):
        return (1 + self.beta*x)*t.tanh(self.n*self.a*x)

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

#Set up

#Define functions
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
def taylor_coeffs(mass, omega, mode):
    Lambda = mode*(mode + 1)
    Omega = 4*mass*omega
    c1 = (Lambda - 3)/(1 - 1j*Omega)
    c2 = ((Lambda + 1)*c1 + 3)/(4 - 1j*2*Omega)
    return c1.real, c1.imag, c2.real, c2.imag

c1_re, c1_im, c2_re, c2_im = taylor_coeffs(mass, omega, mode)
# def annealing(epoch, total_epochs):
#     if epoch <= 0.1*total_epochs:
#         BC = 100.0
#         AMPLITUDE = 100.0
#         UMAX = 10.0
#         UMAX_DERIV = 10.0
#         ODE = 0.1
#         WRON = 0.1

#     elif 0.1*total_epochs < epoch < 0.6*total_epochs:
#         BC = 100.0
#         AMPLITUDE = 100.0
#         UMAX = (50.0 - 10.0)/(0.6 - 0.1)*(epoch/total_epochs) + 2
#         UMAX_DERIV = (50.0 - 10.0)/(0.6 - 0.1)*(epoch/total_epochs) + 2
#         ODE = (10.0 - 0.1)/(0.6 - 0.1)*(epoch/total_epochs) - 1.88
#         WRON = 0.1

#     elif epoch >= 0.6*total_epochs:
#         BC = 100.0
#         AMPLITUDE = 100.0
#         UMAX = 50.0
#         UMAX_DERIV = 50.0
#         ODE = 10
#         WRON = (100.0 - 0.1)/(1.0 - 0.6)*(epoch/total_epochs) - 149.75
    
#     return [BC, AMPLITUDE, UMAX, UMAX_DERIV, ODE, WRON]

def ansatz(model, x_tensor, mass, omega):
    NN = model(x_tensor)
    P_re, P_im, Q_re, Q_im = NN[:, 0:1], NN[:, 1:2], NN[:, 2:3], NN[:, 3:4]
    x_safe = x_tensor.clamp(min = 1e-12, max = 1 - 1e-3)
    rstar = 2*mass/(1 - x_safe) + 2*mass*t.log(x_safe/(1 - x_safe))
    cs, sn = t.cos(2*omega*rstar), t.sin(2*omega*rstar)

    u_re = 1 + c1_re*x_tensor + c2_re*x_tensor**2 + 100.0*x_tensor**3*(P_re + Q_re*cs - Q_im*sn)
    u_im = c1_im*x_tensor + c2_im*x_tensor**2 + 100.0*x_tensor**3*(P_im + Q_im*cs + Q_re*sn)
    return u_re, u_im, P_re, P_im, Q_re, Q_im

def compute_loss(model, x_tensor, mass, mode, omega):
    """weights: [weight_horizon, weight_amplitude, weight_umax, 
    weight_umax_deriv, weight_ODE, weight_wronskian]
    returns: Re(w_nn), Im(w_nn), total loss, horizon loss, amplitude loss, 
    u boundary loss, u' boundary loss, ODE loss, Re(ODE loss), Im(ODE loss), wronskian loss"""

    #Physics loss
    u_re, u_im, *_ = ansatz(model, x_tensor, mass, omega)
    
    du_re, d2u_re = grads(u_re, x_tensor)
    du_im, d2u_im = grads(u_im, x_tensor)

    A_ = A(x_tensor)
    B_re, B_im = B(x_tensor, mass, omega)
    C_ = C(x_tensor, mode)

    res_ode_re = (A_*d2u_re + B_re*du_re - B_im*du_im + C_*u_re)
    res_ode_im = (A_*d2u_im + B_im*du_re + B_re*du_im + C_*u_im)

    # det_d2u_re, det_d2u_im = d2u_re.detach(), d2u_im.detach()
    # det_du_re, det_du_im = du_re.detach(), du_im.detach()
    # det_u_re, det_u_im = u_re.detach(), u_im.detach()

    # normalise = A_**2*(det_d2u_re**2 + det_d2u_im**2) + (B_re**2 + B_im**2)*(det_du_re**2 + det_du_im**2) + C_**2*(det_u_re**2 + det_u_im**2)

    
    # loss_ode_re = t.mean(res_ode_re**2/(normalise + epsilon))
    # loss_ode_im = t.mean(res_ode_im**2/(normalise + epsilon))
    loss_ode_re = t.mean(res_ode_re**2)
    loss_ode_im = t.mean(res_ode_im**2)
    loss_ode = loss_ode_re + loss_ode_im

    #Flux conservation
    J = g(x_tensor, mass)*(u_re*du_im - u_im*du_re) - omega*(u_re**2 + u_im**2 - 1) # should = 0 analytically
    # det_scale = (omega*(det_u_re**2 + det_u_im**2 + 1.0))**2
    # loss_flux = t.mean(J**2/(det_scale + epsilon))
    loss_flux = t.mean(J**2)

    total_loss = loss_ode + 10*loss_flux

    # #Wronskian/Probability flux conservation loss
    # loss_wronskian = (1 - (1 + model.beta_re**2 + model.beta_im**2)/(model.alpha_re**2 + model.alpha_im**2 + 1e-8))**2

    return u_re, u_im, J, total_loss, loss_flux, loss_ode, loss_ode_re, loss_ode_im, res_ode_re, res_ode_im

def extraction(model, x_extraction, mass, mode, omega):

    L = mode*(mode + 1)
    Omega = 4*mass*omega
    
    x_extraction_tensor = t.tensor(x_extraction, requires_grad = True, dtype = DTYPE, device = device).view(-1, 1) 

    u_max_re, u_max_im, *_ = ansatz(model, x_extraction_tensor, mass, omega)

    du_max_re  = eval_grad(u_max_re, x_extraction_tensor)
    du_max_im  = eval_grad(u_max_im, x_extraction_tensor)

    u_max = complex(u_max_re.item(), u_max_im.item())
    du_max = complex(du_max_re.item(), du_max_im.item())

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
        
    prob = np.abs(alpha)**2 - np.abs(beta)**2
    gbf = 1/np.abs(alpha)**2

    return alpha, beta, prob, gbf

print('-'*30, flush = True)
print(f'Starting training for l = {mode} with omega = {omega}', flush = True)
print('-'*30, flush = True)

hist_total = []
hist_flux = []
hist_ode = []
hist_ode_re = []
hist_ode_im = []
GBF = []
probability = []
alphas = []
betas = []
extraction_epochs = []

N_points = 10000
learning_rate = 1e-3
t.manual_seed(0)
model = Model(1, 4, 32, num_hidden_layers = 3).to(device = device, dtype = DTYPE)

adam_parameters = model.parameters()

optimiser = optim.Adam(adam_parameters, lr = learning_rate)

if args.check:
    print('\n' + '='*60, flush = True)
    print('Running preliminary test:', flush = True)
    print('='*60, flush = True)

    print('\n Ansatz finite everywhere?', flush = True)
    xs = t.tensor([[1e-10], [1e-6], [1e-3], [0.5], [x_max]], dtype = DTYPE, device = device, requires_grad = True)
    u_re, u_im, P_re, P_im, Q_re, Q_im = ansatz(model, xs, mass, omega)
    print(f"{'x':>12}{'u_re':>16}{'u_im':>16}", flush = True)
    for i, xv in enumerate(xs.flatten().tolist()):
        print(f"{xv:>12.1e}{u_re[i, 0].item():>16.8f}{u_im[i, 0].item():>16.8f}", flush = True)
    assert t.isfinite(u_re).all() and t.isfinite(u_im).all(), 'Ansatz gave non-finite u'

    print('\n Horizon constraint check:', flush = True)
    for h in (1e-5, 1e-6, 1e-7):
        xh = t.tensor([[h]], dtype = DTYPE, device = device)
        a, b, *_ = ansatz(model, xh, mass, omega)
        print(f"    eps = {h:.0e}: (u_re - 1)/eps = {(a.item() - 1)/h:>12.6f}"
              f"                             u_im/eps = {b.item()/h:>12.6f}", flush = True)
    print(f"    target c1 = {c1_re:>12.6f}                                                                    {c1_im:>12.6f}", flush = True)

    print("\n Autograd chain check via finite difference:", flush = True)
    x0, dh = 0.5, 1e-6
    xg = t.tensor([[x0]], dtype = DTYPE, device = device, requires_grad = True)
    ur, ui, *_ = ansatz(model, xg, mass, omega)
    dur, _ = grads(ur, xg); dui, _ = grads(ui, xg)
    with t.no_grad():
        xp = t.tensor([[x0 + dh]], dtype = DTYPE, device = device)
        xm = t.tensor([[x0 - dh]], dtype = DTYPE, device = device)
        up_re, up_im, *_ = ansatz(model, xp, mass, omega)
        um_re, um_im, *_ = ansatz(model, xm, mass, omega)
    fd_re = (up_re.item() - um_re.item())/(2*dh)
    fd_im = (up_im.item() - um_im.item())/(2*dh)
    print(f'     du/dx at x = {x0}: autograd {dur.item():>12.6f} {dui.item():>12.6f}', flush = True)
    print(f'                        finite diff {fd_re:>12.6f} {fd_im:>12.6f}', flush = True)

    print("\n Extraction returns finite complex scalars?", flush = True)
    al, be, pr, gb = extraction(model, x_max, mass, mode, omega)
    print(f" Extration at x_max = {x_max}", flush = True)
    print(f"    alpha = {al.real:.6f} + {al.imag:.6f}i, |alpha| = {abs(al):.6f}", flush = True)
    print(f"    beta = {be.real:.6f} + {be.imag:.6f}, |beta| = {abs(be):.6f}", flush = True)
    print(f"prob = {pr:.6f}   GBF = {gb:.6e}", flush = True)
    assert np.isfinite([al.real, al.imag, be.real, be.imag, pr, gb]).all()

    print("\n Loss is finite and backward runs?", flush = True)
    xt = x_max*t.rand((2000, 1), dtype = DTYPE, device = device)
    xt.requires_grad_(True)
    _, _, _, loss_c, lf, lo, *_ = compute_loss(model, xt, mass, mode, omega)
    loss_c.backward()
    gnorm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
    print(f" loss = {loss_c.item():.6e} (ode {lo.item():.4e}, flux {lf.item():.4e})", flush = True)
    print(f' grad norm  = {gnorm:.6e}', flush = True)
    assert np.isfinite(loss_c.item()) and gnorm > 0, 'loss or gradient is bad'

    print("\n Five Adam steps:", flush = True)
    for k in range(5):
        optimiser.zero_grad(set_to_none = True)
        xt = x_max*t.rand((2000, 1), dtype = DTYPE, device = device)
        xt.requires_grad_(True)
        _, _, _, l5, *_ = compute_loss(model, xt, mass, mode, omega)
        l5.backward(); optimiser.step()
        print(f"Step {k}: loss = {l5.item():.6e}", flush = True)
    print("\n" + "="*60, flush = True)
    print("Preliminary test passed!", flush = True)
    print('='*60, flush = True)
    raise SystemExit(0)
        
        
Adam_iterations = 18000
for epoch in range(Adam_iterations):
    optimiser.zero_grad(set_to_none = True)
    N_uniform = int(0.6*N_points)
    x_uniform = x_max*t.rand((N_uniform, 1), dtype = DTYPE, device = device)

    N_edges = N_points - N_uniform
    r_h, r_far = 2*mass, 2*mass/(1 - x_max)
    r_samp = r_h + (r_far - r_h)*t.rand((N_edges, 1), dtype = DTYPE, device = device)
    x_edges = 1 - 2*mass/r_samp

    x_tensor = t.cat([x_uniform, x_edges], dim = 0)
    x_tensor.requires_grad_(True)

    # loss_weights = annealing(epoch, Adam_iterations)
    Re_u_nn, Im_u_nn, flux_res, loss, loss_f, loss_o, loss_ode_real, loss_ode_imag, res_ode_re, res_ode_im = compute_loss(model, x_tensor, mass, mode, omega)

    loss.backward()
    optimiser.step()
    
    hist_total.append(loss.item())
    hist_flux.append(loss_f.item())
    hist_ode.append(loss_o.item())
    hist_ode_re.append(loss_ode_real.item())
    hist_ode_im.append(loss_ode_imag.item())

    if (epoch + 1) % 100 == 0 or epoch == 0 or epoch == (Adam_iterations - 1):
        extraction_epochs.append(epoch)
        alpha, beta, prob, gbf = extraction(model, x_max, mass, mode, omega)
        alphas.append(alpha)
        betas.append(beta)
        probability.append(prob)
        GBF.append(gbf)
        if (epoch + 1) % 500 ==0 or epoch == 0:
            print(f"""Epoch: {epoch + 1} / {Adam_iterations}. Total scaled loss: {loss.item():.4e}, 
                    Flux loss: {loss_f.item():.4e},
                    ODE loss: {loss_o.item():.4e},
                    Real component of ODE loss: {loss_ode_real.item():.4e}, 
                    Imaginary component of ODE loss: {loss_ode_imag.item():.4e},
                    Current value of alpha: {alpha.real:.5f} + {alpha.imag:.5f}i,
                    Current value of beta: {beta.real:.5f} + {beta.imag:.5f}i,
                    Current value of |alpha|^2 - |beta|^2: {prob},
                    Current value of GBF: {gbf}.
                    """, flush = True)
            print("-"*30, flush = True)

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
        plt.title(f"l = {mode}, omega = {omega}", fontsize = 20)
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
        plt.title(f"l = {mode}, omega = {omega}", fontsize = 20)
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
        plt.title(f"l = {mode}, omega = {omega}", fontsize = 20)
        # plt.yscale()
        plt.grid()
        plt.legend(fontsize = 15, loc = 'best')
        plt.tight_layout()
        plt.savefig(f'{flux_dir}/Flux_Residual_Epoch_{epoch + 1}.png', format = 'png')
        plt.close()


        t.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                    'GBF': GBF, 'probability': probability},
        os.path.join(base_path, 'checkpoint_latest.pth'))

print("Adam training complete. Switching to L-BFGS:", flush = True)


lbfgs_optimiser = t.optim.LBFGS(model.parameters(), lr = 1.0, max_iter = 20,
            history_size = 50, line_search_fn = 'strong_wolfe')

lbfgs_iterations = 1000
# lbfgs_weights = annealing(Adam_iterations, Adam_iterations)
# N_points = 2*N_points
N_uniform = int(0.6*N_points)
x_uniform = x_max*t.rand((N_uniform, 1), dtype = DTYPE, device = device)

N_edges = N_points - N_uniform
r_h, r_far = 2*mass, 2*mass/(1 - x_max)
r_samp = r_h + (r_far - r_h)*t.rand((N_edges, 1), dtype = DTYPE, device = device)
x_edges = 1 - 2*mass/r_samp

x_tensor_lbfgs = t.cat([x_uniform, x_edges], dim = 0)
x_tensor_lbfgs.requires_grad_(True)

for epoch in range(lbfgs_iterations):
    info = {'total': 0, 'flux': 0, 'ode': 0, 'loss_re': 0, 'loss_im': 0, 'res_re': 0, 'res_im': 0}
    plot_data = {}

    def closure():
        lbfgs_optimiser.zero_grad(set_to_none = True)

        Re_u_nn, Im_u_nn, flux_res, loss, loss_f, loss_o, loss_ode_re, loss_ode_im, res_ode_re, res_ode_im = compute_loss(model, x_tensor_lbfgs, mass, mode, omega)       
        loss.backward()

        info.update({'total': loss.item(), 'flux': loss_f.item(), 'ode': loss_o.item(), 'loss_re': loss_ode_re.item(), 'loss_im': loss_ode_im.item()})

        plot_data['x'] = x_tensor_lbfgs.cpu().detach().numpy()
        plot_data['re_w'] = Re_u_nn.cpu().detach().numpy()
        plot_data['im_w'] = Im_u_nn.cpu().detach().numpy()
        plot_data['flux'] = flux_res.cpu().detach().numpy()
        plot_data['res_re'] = res_ode_re.cpu().detach().numpy()
        plot_data['res_im'] = res_ode_im.cpu().detach().numpy()

        return loss

    lbfgs_optimiser.step(closure)

    with_grad = compute_loss(model, x_tensor_lbfgs, mass, mode, omega)
    _, _, _, loss_now, lf_now, lo_now, lre, lim, *_ = with_grad
    info.update({'total': loss_now.item(), 'flux': lf_now.item(), 'ode': lo_now.item(), 'loss_re': lre.item(), 'loss_im': lim.item()})

    # loss_h.append(info['h'])
    # loss_norm.append(info['norm'])
    # loss_ODE.append(info['ode'])
    if not np.isfinite(info['total']):
        print(f"L-BFGS diverged at epoch {epoch}; stopping.", flush=True)
        break
    
    hist_total.append(info['total'])
    hist_flux.append(info['flux'])
    hist_ode.append(info['ode'])
    hist_ode_re.append(info['loss_re'])
    hist_ode_im.append(info['loss_im'])

    if (epoch + 1) % 20 == 0 or epoch == (lbfgs_iterations - 1):

        extraction_epochs.append(epoch + Adam_iterations)
        alpha, beta, prob, gbf = extraction(model, x_max, mass, mode, omega)
        alphas.append(alpha)
        betas.append(beta)
        probability.append(prob)
        GBF.append(gbf)
        
        print(f"""L-BFGS Epoch: {epoch + 1} / {lbfgs_iterations}. Total scaled loss: {info['total']:.4e}, 
                    Flux loss: {info['flux']:.4e},
                    ODE loss: {info['ode']:.4e},
                    Real component of loss: {info['loss_re']:.4e}, 
                    Imaginary component of loss: {info['loss_im']:.4e}, 
                    Current value of alpha: {alpha.real:.5f} + {alpha.imag:.5f}i,
                    Current value of beta: {beta.real:.5f} + {beta.imag:.5f}i,
                    Current value of |alpha|^2 - |beta|^2: {prob},
                    Current value of GBF: {gbf}.
                    """, flush = True)
        print("-"*30, flush = True)
        
        x_plot = plot_data['x'].flatten()
        idx = np.argsort(x_plot)

        plt.figure()
        plt.plot(x_plot[idx], plot_data['res_re'].flatten()[idx], color = 'blue', label = r'$\Re (Res_{ODE})$')
        plt.plot(x_plot[idx], plot_data['res_im'].flatten()[idx], color = 'green', label = r'$\Im (Res_{ODE})$')
        plt.plot(x_plot[idx], plot_data['re_w'].flatten()[idx], color = 'orange', label = r'$\Re (u_{NN})$')
        plt.plot(x_plot[idx], plot_data['im_w'].flatten()[idx], color = 'red', label = r'$\Im (u_{NN})$')
        plt.xlabel('x', fontsize = 25)
        plt.ylabel('Residual', fontsize = 25)
        plt.title(f"l = {mode}, omega = {omega}", fontsize = 20)
        plt.grid()
        plt.legend(fontsize = 25, loc = 'best')
        plt.tight_layout()
        plt.savefig(f'{out_dir}/Output_Epoch_{epoch + 1 + Adam_iterations}.png', format = 'png')
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
        plt.title(f"l = {mode}, omega = {omega}", fontsize = 20)
        plt.legend(fontsize = 15, loc = 'best')
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'{loss_dir}/Loss_Epoch_{epoch + 1 + Adam_iterations}.png', format = 'png')
        plt.close()

        plt.figure()
        plt.plot(x_plot[idx], plot_data['flux'].flatten()[idx], color = 'purple', label = 'Flux Residual')
        plt.axhline(0.0, color = 'cyan', linestyle = '--', label = 'Target')
        plt.xlabel('x', fontsize = 25)
        plt.ylabel(r'Flux Residual', fontsize = 25)
        plt.title(f"l = {mode}, omega = {omega}", fontsize = 20)
        # plt.yscale('symlog')
        plt.grid()
        plt.legend(fontsize = 15, loc = 'best')
        plt.tight_layout()
        plt.savefig(f'{flux_dir}/Flux_Residual_Epoch_{epoch + 1 + Adam_iterations}.png', format = 'png')
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
plt.title(f'l = {mode}, omega = {omega}', fontsize = 18)
plt.tight_layout()
plt.grid()
plt.legend()
plt.savefig(f'{base_path}/alpha_convergence.png', format = 'png')
plt.close()

plt.figure(figsize = [7, 7])
plt.plot(betas.real, betas.imag, 'g--', alpha = 0.9)
plt.scatter(betas[0].real, betas[0].imag, color='blue', label = f'Initial: {betas[0].real:.3f} + {betas[0].imag:.3f}i')
plt.scatter(betas[-1].real, betas[-1].imag, color='green', label = f'Final: {betas[-1].real:.3f} + {betas[-1].imag:.3f}i')
plt.xlabel(r'$\Re(\beta)$', fontsize = 18)
plt.ylabel(r'$\Im(\beta)$', fontsize = 18)
plt.title(f'l = {mode}, omega = {omega}', fontsize = 18)
plt.tight_layout()
plt.grid()
plt.legend()
plt.savefig(f'{base_path}/beta_convergence.png', format = 'png')
plt.close()

fig, ax1 = plt.subplots(figsize = [7, 4.5])

ax1.plot(extraction_epochs, GBF, color = 'blue', label = r'$\Gamma$')
ax1.set_yscale('log')
ax1.set_xlabel('Epoch', fontsize = 14)
ax1.set_ylabel(r'$\Gamma$', fontsize = 16)
ax1.tick_params(axis = 'y')
# ax1.axhline(7.0011982e-05, color = 'blue', linestyle = ':', linewidth = 1,
#             label = r'$\Gamma_{\rm ref}$')
ax1.grid(alpha = 0.3)

ax2 = ax1.twinx()
ax2.plot(extraction_epochs, probability, color = 'red',
         label = r'$|\alpha|^2 - |\beta|^2$')
ax2.set_ylabel(r'$|\alpha|^2 - |\beta|^2$', fontsize = 16)
ax2.tick_params(axis = 'y')
ax2.axhline(1.0, color = 'red', linestyle = ':', linewidth = 1, label = r'Target $|\alpha|^2 - |\beta|^2$')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize = 11, loc = 'best')
plt.title(f'l = {mode}, omega = {omega}', fontsize = 16)
plt.tight_layout()
plt.savefig(f'{base_path}/GBFProb.png', format = 'png')
plt.close()

results = {
    'model_state_dict': model.state_dict(),
    'alpha_history': {'re': alpha_real_array, 'im': alpha_imag_array},
    'beta_history': {'re': beta_real_array, 'im': beta_imag_array},
    'loss_history': {'total': hist_total, 'flux': hist_flux, 'ode': hist_ode, 'real': hist_ode_re, 'imag': hist_ode_im},
    'final_alpha': [alpha_real_array[-1], alpha_imag_array[-1]],
    'final_beta': [beta_real_array[-1], beta_imag_array[-1]],
    'GBF_history': GBF,
    'Probability': probability
    }

checkpoint_path = os.path.join(base_path, f'pinn_checkpoint_gbf_l{mode}_omega{omega}.pth') 
t.save(results, checkpoint_path)

final_alpha, final_beta, final_prob, final_gbf = extraction(model, x_max, mass, mode, omega)
T = 1/final_alpha
R = final_beta/final_alpha

result_file_path = os.path.join(base_path, 'result.txt')
with open(result_file_path, 'w') as f:
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


print(f'Training complete. Checkpoint saved to {checkpoint_path}', flush = True)
print(f"l = {mode} mode with omega = {omega} completed successfully.", flush = True)
print(f"""Final values:
                    Total scaled loss: {info['total']:.4e},
                    Flux loss: {info['flux']:.4e},
                    ODE loss: {info['ode']:.4e},
                    Real component of loss: {info['loss_re']:.4e}, 
                    Imaginary component of loss: {info['loss_im']:.4e},
                    Final value of alpha: {final_alpha.real:.5f} + {final_alpha.imag:.5f}i,
                    Final value of beta: {final_beta.real:.5f} + {final_beta.imag:.5f}i,
                    Final value of T: {T.real:.5f} + {T.imag:.5f}i,
                    Final value of |T|^2: {np.abs(T)**2:.5f}
                    Final value of R: {R.real:.5f} + {R.imag:.5f}i,
                    Final value of |R|^2: {np.abs(R)**2:.5f},
                    Final value of |T|^2 + |R|^2: {(np.abs(R)**2 + np.abs(T)**2):.5f}.
                    The grey body factor for l = {mode}, omega = {omega} is {np.abs(T)**2:.5f}
                    """, flush = True)