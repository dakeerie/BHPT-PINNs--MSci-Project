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
    "font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "text.usetex": False
})

j = complex(0, 1)

parser = argparse.ArgumentParser(description = "Train PINN for specific mode l")
parser.add_argument('--mode', type = int, required = True, help = 'The value of l (mode)')
parser.add_argument('--delta', type = float, default = 0.1, help = 'Range of random perturbation away from true QNM')

args = parser.parse_args()
mode = args.mode
delta = args.delta

omega_dict = {
    2: complex(0.74734, -0.17792),
    3: complex(1.19889, -0.18541),
    4: complex(1.61835, -0.18832),
    5: complex(2.02458, -0.18974),
    6: complex(2.42402, -0.19053)
}

if mode not in omega_dict:
    raise ValueError(f"Mode l={mode} is not supported. Choose from {list(omega_dict.keys())}.")

base_path = f'./QNMs/Output/l{mode}/delta_{delta}'

out_dir = os.path.join(base_path, 'NNOutput')
loss_dir = os.path.join(base_path, 'Loss')
os.makedirs(out_dir, exist_ok = True)
os.makedirs(loss_dir, exist_ok = True)

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#PINN architecture
class adaptive_tanh(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.beta = nn.Parameter(t.zeros(1, num_features))

    def forward(self, x):
        return (1 + self.beta*x)*t.tanh(x)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_hidden_layers=2, rho_init = None):
        super().__init__() 
        self.input_layer = nn.Linear(in_channels, hidden_channels)
        self.act_input = adaptive_tanh(hidden_channels)

        self.hidden_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_channels, hidden_channels))
            self.activations.append(adaptive_tanh(hidden_channels))

        self.output_layer = nn.Linear(hidden_channels, out_channels) 

        self.rho_re = nn.Parameter(t.tensor(rho_init.real, dtype = DTYPE), requires_grad = False)
        self.rho_im = nn.Parameter(t.tensor(rho_init.imag, dtype = DTYPE), requires_grad = False)

    def forward(self, x: t.tensor):
        x = self.input_layer(x)
        x = self.act_input(x)

        for layer, act in zip(self.hidden_layers, self.activations):
            x = layer(x)
            x = act(x)

        x = self.output_layer(x) 
        return x 

#Set up
DTYPE = t.float32
NP_DTYPE = np.float32 if DTYPE == t.float32 else np.float64
mass = 0.5

#Define functions
def grads(y, x):
        dy = t.autograd.grad(y, x, t.ones_like(y), create_graph = True)[0]
        d2y = t.autograd.grad(dy, x, t.ones_like(dy), create_graph = True)[0]
        return dy, d2y

def first_grad(y, x):
    dy = t.autograd.grad(y, x, t.ones_like(y), create_graph = True)[0]
    return dy

def A(x):
    A = x*(1 - x)**2
    return A

def B(x, rho_re, rho_im, M):
        B_re = (4*rho_re + 3)*x**2 - 4*(2*rho_re + 1)*x + 2*rho_re + 1
        B_im = (4*rho_im)*x**2 - 4*(2*rho_im)*x + 2*rho_im
        return B_re, B_im

def C(x, rho_re, rho_im, M, l):
    C_re = -l*(l + 1) - 3*(x - 1) + 4*rho_re*x - 8*(rho_re**2 - rho_im**2) + \
    4*(rho_re**2 - rho_im**2)*x - 4*rho_re
    C_im = 4*rho_im*x - 16*rho_re*rho_im + 8*rho_re*rho_im*x - 4*rho_im
    return C_re, C_im

def annealing(epoch, total_epochs):
    if epoch <= 0.2*total_epochs:
        BC = 100.0
        NT = 100.0
        ODE = 0.01

    elif 0.2*total_epochs < epoch < 0.6*total_epochs:
        BC = (10.0 - 100.0)/(0.6 - 0.2)*(epoch/total_epochs) + 145
        NT = (10.0 - 100.0)/(0.6 - 0.2)*(epoch/total_epochs) + 145
        ODE = (1.0 - 0.1)/(0.6 - 0.2)*(epoch/total_epochs) - 0.35
    
    elif 0.6*total_epochs <= epoch < 0.8*total_epochs:
        BC = 10.0
        NT = 10.0
        ODE = 1.0

    elif epoch >= 0.8*total_epochs:
        BC = 10.0
        NT = 10.0
        ODE = (5.0 - 1.0)/(1.0 - 0.8)*(epoch/total_epochs) - 15
    
    return [BC, NT, BC, ODE]

def compute_loss(model, x_tensor, weights, mass, mode):
    """weights: [weight_horizon, weight_NT, weight_infinity, weight_ODE]
    returns: Re(w_nn), Im(w_nn) total loss, horizon loss, non-trivial loss, infinity loss, ODE loss, Re(ODE loss), Im(ODE loss)"""

    #Boundary tensors
    x_horizon = t.tensor(0., requires_grad = True, dtype = DTYPE, device = device).view(-1, 1)
    x_infinity = t.tensor(1., requires_grad = True, dtype = DTYPE, device = device).view(-1, 1) 

    #Horizon loss
    w_h = model(x_horizon)
    w_h_re, w_h_im = w_h[:, 0:1], w_h[:, 1:2]
    dw_h_re = first_grad(w_h_re, x_horizon)
    dw_h_im = first_grad(w_h_im, x_horizon)

    B_h_re, B_h_im = B(x_horizon, model.rho_re, model.rho_im, mass)
    C_h_re, C_h_im = C(x_horizon, model.rho_re, model.rho_im, mass, mode)

    loss_horizon_re = (B_h_re*dw_h_re - B_h_im*dw_h_im +
                        C_h_re*w_h_re - C_h_im*w_h_im)
    loss_horizon_im = (B_h_im*dw_h_re + B_h_re*dw_h_im + 
                    C_h_im*w_h_re + C_h_re*w_h_im)
    loss_horizon = t.mean(loss_horizon_re**2 + loss_horizon_im**2)

    #Non-trivial loss
    loss_nontrivial = t.mean((1.0 - w_h_re)**2 + (0.0 - w_h_im)**2)

    #Infinity loss
    w_inf = model(x_infinity)
    w_inf_re, w_inf_im = w_inf[:, 0:1], w_inf[:, 1:2]
    dw_inf_re = first_grad(w_inf_re, x_infinity)
    dw_inf_im = first_grad(w_inf_im, x_infinity)

    B_inf_re, B_inf_im = B(x_infinity, model.rho_re, model.rho_im, mass)
    C_inf_re, C_inf_im = C(x_infinity, model.rho_re, model.rho_im, mass, mode)

    loss_infinity_re = (B_inf_re*dw_inf_re - B_inf_im*dw_inf_im + 
                        C_inf_re*w_inf_re - C_inf_im*w_inf_im)
    loss_infinity_im = (B_inf_im*dw_inf_re + B_inf_re*dw_inf_im + 
                        C_inf_im*w_inf_re + C_inf_re*w_inf_im)
    loss_infinity = t.mean(loss_infinity_re**2 + loss_infinity_im**2)

    #Physics loss
    w_nn = model(x_tensor)
    w_nn_re, w_nn_im = w_nn[:, 0:1], w_nn[:, 1:2]
    dw_nn_re, d2w_nn_re = grads(w_nn_re, x_tensor)
    dw_nn_im, d2w_nn_im = grads(w_nn_im, x_tensor)

    A_ = A(x_tensor)
    B_re, B_im = B(x_tensor, model.rho_re, model.rho_im, mass)
    C_re, C_im = C(x_tensor, model.rho_re, model.rho_im, mass, mode)

    loss_ode_re = (A_*d2w_nn_re + B_re*dw_nn_re - B_im*dw_nn_im + C_re*w_nn_re - C_im*w_nn_im)
    loss_ode_im = (A_*d2w_nn_im + B_im*dw_nn_re + B_re*dw_nn_im + C_im*w_nn_re + C_re*w_nn_im)
    loss_ode = t.mean(loss_ode_re**2 + loss_ode_im**2)

    total_loss = (weights[0]*loss_horizon + weights[1]*loss_nontrivial + 
                weights[2]*loss_infinity + weights[3]*loss_ode)
    
    return w_nn_re, w_nn_im, total_loss, loss_horizon, loss_nontrivial, loss_infinity, loss_ode, loss_ode_re, loss_ode_im

omega = omega_dict[mode]
rho = -1j*omega
rho_perturbed = rho + complex(np.random.uniform(-delta, delta), np.random.uniform(-delta, delta))

print('-'*30)
print(f'Starting training for l = {mode} with delta = {delta}')
initial_dist = np.abs(rho_perturbed - rho)
print(f"Initial rho: {rho_perturbed.real:.5f} + {rho_perturbed.imag:.5f}i")
print(f"Target rho: {rho.real:.5f} + {rho.imag:.5f}i")
print(f"Initial distance from target: {initial_dist:.5f}")
print('-'*30)

loss_h = []
loss_nt = []
loss_inf = []
loss_ODE = []
loss_total = []
rho_real_array = []
rho_imag_array = []

N_points = 10000
learning_rate = 1e-3
model = Model(1, 2, 32, num_hidden_layers = 3, rho_init = rho_perturbed).to(device = device, dtype = DTYPE)

adam_parameters = [p for p in model.parameters() if p is not model.rho_re and p is not model.rho_im]

optimiser = optim.Adam(adam_parameters, lr = learning_rate)

Adam_iterations = 18000

for epoch in range(Adam_iterations):
    optimiser.zero_grad()
    N_uniform = int(0.65*N_points)
    x_uniform = t.rand((N_uniform, 1), dtype = DTYPE, device = device)

    N_edges = N_points - N_uniform
    beta_dist = dist.Beta(t.tensor([0.5]), t.tensor([0.5]))
    x_edges = beta_dist.sample((N_edges,)).view(-1, 1).to(device = device, dtype = DTYPE)

    x_tensor = t.cat([x_uniform, x_edges], dim = 0)
    x_tensor.requires_grad_(True)

    if epoch == int(0.5*Adam_iterations):
        print("Unfreezing rho...")
        model.rho_re.requires_grad = True
        model.rho_im.requires_grad = True
        optimiser.add_param_group({'params': [model.rho_re, model.rho_im], 'lr': 1e-4})

    loss_weights = annealing(epoch, Adam_iterations)
    Re_w_nn, Im_w_nn, loss, l_h, l_nt, l_inf, l_ode, l_ode_re, l_ode_im = compute_loss(model, x_tensor, loss_weights, mass, mode)

    loss.backward()
    optimiser.step()

    rho_real_array.append(model.rho_re.item())
    rho_imag_array.append(model.rho_im.item())

    with t.no_grad():
        loss_h.append(l_h.item())
        loss_nt.append(l_nt.item())
        loss_inf.append(l_inf.item())
        loss_ODE.append(l_ode.item())
        loss_total.append(loss.item())

    if (epoch + 1) % 500 == 0 or epoch == 0 or epoch == (Adam_iterations - 1):
        print(f"""Epoch: {epoch + 1} / {Adam_iterations}. Total scaled loss: {loss.item():.4e}, 
                    ODE(0) loss: {l_h.item():.4e}, 
                    ODE(1) loss: {l_inf.item():.4e}, 
                    Non-trivial solution loss: {l_nt.item():.4e},
                    Physics loss: {l_ode.item():.4e},
                    Initial value of rho: {rho_perturbed.real:.5f} + {rho_perturbed.imag:.5f}i,
                    Current value of rho: {model.rho_re.item():.5f} + {model.rho_im.item():.5f}i,
                    Target rho: {rho.real} + {rho.imag}""")
        print("-"*30)

    if (epoch + 1) % 1000 == 0:
        x_np = x_tensor.cpu().detach().numpy().flatten()
        idx = np.argsort(x_np)

        res_re_plot = l_ode_re.cpu().detach().numpy().flatten()
        res_im_plot = l_ode_im.cpu().detach().numpy().flatten()
        w_re_plot = Re_w_nn.cpu().detach().numpy().flatten()
        w_im_plot = Im_w_nn.cpu().detach().numpy().flatten()

        plt.figure()
        plt.plot(x_np[idx], res_re_plot[idx], color = 'blue', label = r'$\Re (Res_{ODE})$')
        plt.plot(x_np[idx], res_im_plot[idx], color = 'green', label = r'$\Im (Res_{ODE})$')
        plt.plot(x_np[idx], w_re_plot[idx], color = 'orange', label = r'$\Re (w_{NN})$')
        plt.plot(x_np[idx], w_im_plot[idx], color = 'red', label = r'$\Im (w_{NN})$')
        plt.xlabel('x', fontsize = 25)
        plt.ylabel('Output', fontsize = 25)
        plt.grid()
        plt.legend(fontsize = 25)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/Output_Epoch_{epoch + 1}.png', format = 'png')
        plt.close()

        plt.figure()
        plt.plot(loss_total, label = 'Total')
        plt.plot(loss_h, label = 'Horizon')
        plt.plot(loss_inf, label = 'Infinity')
        plt.plot(loss_ODE, label ='ODE')
        plt.plot(loss_nt, label = 'Non-trivial solution')
        plt.yscale('log')
        plt.ylabel('Loss', fontsize = 25)
        plt.xlabel('Epoch', fontsize = 25)
        plt.legend(fontsize = 25)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'{loss_dir}/Loss_Epoch_{epoch + 1}.png', format = 'png')
        plt.close()

print("Adam training complete. Switching to L-BFGS:")

model.rho_re.requires_grad = True
model.rho_im.requires_grad = True

lbfgs_optimiser = t.optim.LBFGS(model.parameters(), lr = 1.0, max_iter = 20,
            history_size = 50, line_search_fn = 'strong_wolfe')

lbfgs_iterations = 1000
lbfgs_weights = [10.0, 10.0, 10.0, 1.0]

for epoch in range(lbfgs_iterations):
    N_uniform = int(0.65*N_points)
    x_uniform = t.rand((N_uniform, 1), dtype = DTYPE, device = device)

    N_edges = N_points - N_uniform
    beta_dist = dist.Beta(t.tensor([0.5]), t.tensor([0.5]))
    x_edges = beta_dist.sample((N_edges,)).view(-1, 1).to(device = device, dtype = DTYPE)

    x_tensor = t.cat([x_uniform, x_edges], dim = 0)
    x_tensor.requires_grad_(True)

    info = {'total': 0, 'h': 0, 'nt': 0, 'inf': 0, 'ode':0}
    plot_data = {}

    def closure():
        lbfgs_optimiser.zero_grad()

        Re_w_nn, Im_w_nn, loss, l_h, l_nt, l_inf, l_ode, l_ode_re, l_ode_im = compute_loss(model, x_tensor, lbfgs_weights, mass, mode)       
        loss.backward()

        info.update({'total': loss.item(), 'h': l_h.item(), 'nt': l_nt.item(), 'inf': l_inf.item(), 'ode': l_ode.item()})

        plot_data['x'] = x_tensor.cpu().detach().numpy()
        plot_data['re_w'] = Re_w_nn.cpu().detach().numpy()
        plot_data['im_w'] = Im_w_nn.cpu().detach().numpy()
        plot_data['res_re'] = l_ode_re.cpu().detach().numpy()
        plot_data['res_im'] = l_ode_im.cpu().detach().numpy()

        return loss

    lbfgs_optimiser.step(closure)

    loss_h.append(info['h'])
    loss_nt.append(info['nt'])
    loss_inf.append(info['inf'])
    loss_ODE.append(info['ode'])
    loss_total.append(info['total'])

    rho_real_array.append(model.rho_re.item())
    rho_imag_array.append(model.rho_im.item())

    if (epoch + 1) % 100 == 0:
        print(f"""L-BFGS Epoch: {epoch + 1} / {lbfgs_iterations}. Total scaled loss: {info['total']:.4e}, 
                    ODE(0) loss: {info['h']:.4e}, 
                    ODE(1) loss: {info['inf']:.4e}, 
                    Non-trivial solution loss: {info['nt']:.4e},
                    Physics loss: {info['ode']:.4e},
                    Initial value of rho: {rho_perturbed.real:.5f} + {rho_perturbed.imag:.5f}i,
                    Current value of rho: {model.rho_re.item():.5f} + {model.rho_im.item():.5f}i,
                    Target rho: {rho.real} + {rho.imag}""")
        print("-"*30)

    if epoch == (lbfgs_iterations - 1):
        print(f"""Training complete for l = {mode}. Total scaled loss: {info['total']:.4e}, 
                    ODE(0) loss: {info['h']:.4e}, 
                    ODE(1) loss: {info['inf']:.4e}, 
                    Non-trivial solution loss: {info['nt']:.4e},
                    Physics loss: {info['ode']:.4e},
                    Current value of rho: {model.rho_re.item():.5f} + {model.rho_im.item():.5f}i
                    Target rho: {rho.real} + {rho.imag}""")

    if (epoch + 1) % 100 == 0 or epoch == (lbfgs_iterations - 1):
        x_plot = plot_data['x'].flatten()
        idx = np.argsort(x_plot)

        plt.figure()
        plt.plot(x_plot[idx], plot_data['res_re'].flatten()[idx], color = 'blue', label = r'$\Re (Res_{ODE})$')
        plt.plot(x_plot[idx], plot_data['res_im'].flatten()[idx], color = 'green', label = r'$\Im (Res_{ODE})$')
        plt.plot(x_plot[idx], plot_data['re_w'].flatten()[idx], color = 'orange', label = r'$\Re (w_{NN})$')
        plt.plot(x_plot[idx], plot_data['im_w'].flatten()[idx], color = 'red', label = r'$\Im (w_{NN})$')
        plt.xlabel('x', fontsize = 25)
        plt.ylabel('Residual', fontsize = 25)
        plt.grid()
        plt.legend(fontsize = 25)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/Output_Epoch_{epoch + 1 + Adam_iterations}.png', format = 'png')
        plt.close()
    
        plt.figure()
        plt.plot(loss_total, label = 'Total')
        plt.plot(loss_h, label = 'Horizon')
        plt.plot(loss_inf, label = 'Infinity')
        plt.plot(loss_ODE, label ='ODE')
        plt.plot(loss_nt, label = 'Non-trivial solution')
        plt.yscale('log')
        plt.ylabel('Loss', fontsize = 25)
        plt.xlabel('Epoch', fontsize = 25)
        plt.legend(fontsize = 25)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'{loss_dir}/Loss_Epoch_{epoch + 1 + Adam_iterations}.png', format = 'png')
        plt.close()
    

rho_real_array = np.array(rho_real_array)
rho_imag_array = np.array(rho_imag_array)

plt.figure(figsize = [7, 7])
plt.plot(rho_real_array, rho_imag_array, 'r--', alpha = 0.9)
plt.scatter(rho.real, rho.imag, color='gold', marker='x', s=100, label = f'Target: {rho.real} + {rho.imag}i')
plt.scatter(rho_perturbed.real, rho_perturbed.imag, color='blue', label = f'Initial: {rho_perturbed.real:.5f} + {rho_perturbed.imag:.5f}i')
plt.scatter(model.rho_re.item(), model.rho_im.item(), color='green', label = f'Final: {model.rho_re.item():.5f} + {model.rho_im.item():.5f}i')
plt.xlabel(r'$\Re(\rho)$', fontsize = 18)
plt.ylabel(r'$\Im(\rho)$', fontsize = 18)
plt.title(f'l = {mode}, delta = {delta}', fontsize = 18)
plt.tight_layout()
plt.grid()
plt.legend()
plt.savefig(f'{base_path}/rho_convergence.png', format = 'png')
plt.close()

results = {
    'model_state_dict': model.state_dict(),
    'rho_history': {'re': rho_real_array, 'im': rho_imag_array},
    'loss_history': {'total': loss_total, 'h': loss_h, 'nt': loss_nt, 'inf': loss_inf, 'ode': loss_ODE},
    'final_rho': [model.rho_re.item(), model.rho_im.item()]
    }

checkpoint_path = os.path.join(base_path,f'pinn_checkpoint_qnm_l{mode}_delta{delta}.pth') 
t.save(results, checkpoint_path)
print(f'Training complete. Checkpoint saved to {checkpoint_path}')
print(f"l = {mode} mode with delta = {delta} completed successfully.")