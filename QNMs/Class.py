import numpy as np
from scipy.special import lambertw
import matplotlib.pyplot as plt
import imageio
import os
from pathlib import Path
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": "" # Clear the helvet and sansmath packages
})

j = 1j 

class SchwarzschildPerturbation:
    """
    Class to simulate perturbations of a Schwarzschild black hole by solving 
    the RZW equation using a Leapfrog solver.
    """

    def __init__(self, M=1.0, l=2, potential='axial', rstar_min=-400, rstar_max=700, dx=0.1, dt=None):
        self.M = M
        self.l = l
        self.potential = potential
        self.dx = dx
        self.dt = dt if dt is not None else 0.5*dx

        if self.dt > self.dx:
            raise ValueError("CFL condition violated: dt must be <= dx")
        
        self.rstar = np.arange(rstar_min*2*M, (rstar_max + dx)*2*M, dx)
        self.r = self._rstar_to_r(self.rstar)
        self.V = self._compute_potential()

        self.Psi = None
        self.times = None
        self.Nt = None

    def _rstar_to_r(self, rs):
        rs = np.array(rs)
        return (2*self.M*(1 + lambertw(np.exp(rs/(2*self.M) - 1)))).real
    
    def _r_to_rstar(self, r):
        r = np.array(r)
        if np.any(r <= 2*self.M):
            raise ValueError("All r must be > 2M")
        return r + 2*self.M*np.log(r/(2*self.M) - 1)
    
    def _compute_potential(self):
        if self.potential == 'axial':
            return (1 - 2*self.M/self.r)*(self.l*(self.l + 1)/self.r**2 - 6*self.M/self.r**3)
        elif self.potential == 'polar':
            n = 0.5*(self.l - 1)*(self.l + 2)
            num = 2*(1 - 2*self.M/self.r)*(
                9*self.M**3 +
                9*n*self.M**2*self.r + 
                3*n**2*self.M*self.r**2 + 
                n**2*(1 + n)*self.r**3
            )
            den = self.r**3*(3*self.M + n*self.r)**2
            return num / den
        elif self.potential == 'zero':
            return np.zeros_like(self.r)
        else:
            raise ValueError("Potential argument must be either 'axial', 'polar' or 'zero'.")
        
    def _FD_x_derivative(self, Psi):
        dPsi = np.zeros_like(Psi)
        dPsi[1:-1] = (Psi[2:] - Psi[:-2])/(2*self.dx)
        dPsi[0] = (Psi[1] - Psi[0])/self.dx
        dPsi[-1] = (Psi[-1] - Psi[-2])/self.dx
        return dPsi
        
    def solve(self, Psi0, Nt, dPsi0 = None):
        """ Solve the wave equation using leapfrog integration. """

        if len(Psi0) != len(self.rstar):
            raise ValueError(f"Psi0 length ({len(Psi0)}) must match grid size ({len(self.rstar)})")

        self.Nt = Nt
        self.times = np.arange(Nt)*self.dt

        if dPsi0 is None:
            dPsi0 = self._FD_x_derivative(Psi0)

        psi_arr = np.zeros((len(self.rstar), Nt), dtype=complex)
        psi_arr[:, 0] = Psi0
        lap = (Psi0[2:] - 2*Psi0[1:-1] + Psi0[:-2]) / self.dx**2
        
        psi_arr[1:-1, 1] = (psi_arr[1:-1, 0] + self.dt*dPsi0[1:-1] 
                            + 0.5*self.dt**2*(lap - self.V[1:-1]*Psi0[1:-1]))
        
        psi_arr[0, 1] = psi_arr[0, 0] + self.dt/self.dx*(psi_arr[1, 0] - psi_arr[0, 0])
        psi_arr[-1, 1] = psi_arr[-1, 0] - self.dt/self.dx*(psi_arr[-1, 0] - psi_arr[-2, 0])

        for q in range(1, Nt - 1):
            psi_arr[1:-1, q + 1] = (2*psi_arr[1:-1, q] - psi_arr[1:-1, q - 1]
                + (self.dt / self.dx)**2*(psi_arr[2:, q] + psi_arr[:-2, q] - 2*psi_arr[1:-1, q])
                - self.V[1:-1]*psi_arr[1:-1, q]*self.dt**2)
            
            psi_arr[0, q + 1] = self.dt/self.dx*(psi_arr[1, q] - psi_arr[0, q]) + psi_arr[0, q]
            psi_arr[-1, q + 1] = -self.dt/self.dx*(psi_arr[-1, q] - psi_arr[-2, q]) + psi_arr[-1, q]

        self.Psi = psi_arr.T
        return self

    def make_gif(self, filename = None, component = 'real', fps = 10, frame_interval = 100,
                ylim = None, show_potential = True, omega = None, fig_title = None, xlimit = None):
        """
        Create an animated GIF in the 'SS_Perturbation_gifs' folder.
        
        Parameters:
        -----------
        filename : str, optional
            Custom filename. If None, generates one based on Mass, l, parity, and omega.
        component : str
            Which component to plot: 'real', 'imag', or 'abs'
        fps : int
            Frames per second
        frame_interval : int
            Plot every Nth timestep
        omega : float, optional
            Used for filename generation and title
        """
        if self.Psi is None:
            raise ValueError("Must run solve() before making GIF")
        
        output_dir = Path("SS_Perturbation_gifs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            omega_str = f"_w{omega:.2f}" if omega is not None else ""
            # Naming convention: Mass, Mode, Parity
            name_str = f"BH_M{self.M:.1f}_l{self.l}_{self.potential}{omega_str}_{component}.gif"
            output_path = output_dir / name_str
        else:

            if not filename.endswith('.gif'):
                filename += '.gif'
            output_path = output_dir / filename
        
        temp_dir = output_dir / 'temp_frames'
        temp_dir.mkdir(parents=True, exist_ok=True)

        if component == 'real':
            wave_data = self.Psi.real
            lbl = r'$\Re(\Psi)$'
        elif component == 'imag':
            wave_data = self.Psi.imag
            lbl = r'$\Im(\Psi)$'
        elif component == 'abs':
            wave_data = np.abs(self.Psi)
            lbl = r'$|{\Psi}|$'
        else:
            raise ValueError("component must be 'real', 'imag', or 'abs'")
        
        if ylim is None:
            ylim = (wave_data.min()*1.1, max(wave_data.max(), self.V.max())*1.1)
        
        frames = []
        frame_indices = range(0, self.Nt, frame_interval)

        if fig_title is None:
            omega_display = rf", $\omega={omega:.2f}$" if omega is not None else ""
            fig_title = f"{self.potential.capitalize()} Perturbation: $M={self.M}$, $l={self.l}${omega_display}"

        print(f"Generating frames for {output_path.name}...")
        
        for i, t_idx in enumerate(frame_indices):
            fig, ax = plt.subplots(figsize=(8, 5))
            
            ax.plot(self.rstar/(2*self.M), 2*self.M*wave_data[t_idx, :], label=lbl)
            
            if show_potential:
                ax.plot(self.rstar/(2*self.M), 4*self.M**2*self.V,
                        label = 'Potential', zorder=0)
            
            ax.set_title(fig_title, fontsize=25)
            ax.tick_params(labelsize = 18)
            ax.set_ylim(ylim)
            ax.set_xlim(xlimit)
            ax.set_xlabel(r'$r_*/2M$', fontsize = 25)
            ax.set_ylabel(r'$2M \Psi$ \& $4M^2 V$', fontsize = 25)
            ax.grid(True, alpha = 0.6)
            ax.legend(loc='upper right', fontsize = 23)
            
            frame_path = temp_dir / f'frame_{i:04d}.png'
            plt.savefig(frame_path, dpi = 100, bbox_inches='tight', facecolor='white')
            plt.close()

            frames.append(imageio.v2.imread(frame_path))

        print(f"Compiling GIF at {output_path}...")
        imageio.mimsave(output_path, frames, fps=fps, loop=0)

        for file_path in temp_dir.glob('*.png'):
            file_path.unlink()
        temp_dir.rmdir()
        
        print("Done!")
        return output_path
        
    def plot_snapshot(self, t_indices=None, component='real', show_potential=True):
        if self.Psi is None:
            raise ValueError("Must run solve() before plotting")
        
        if t_indices is None:
            t_indices = np.linspace(0, self.Nt - 1, 5, dtype=int)
        
        if component == 'real':
            wave_data = self.Psi.real
        elif component == 'imag':
            wave_data = self.Psi.imag
        elif component == 'abs':
            wave_data = np.abs(self.Psi)
        else:
            raise ValueError("Component must be 'real', 'imag' or 'abs'")
        
        plt.figure(figsize=(10, 6))
        for t in t_indices:
            plt.plot(self.rstar, wave_data[t, :], label=f't = {self.times[t]:.1f} M')
        
        if show_potential:
            plt.plot(self.rstar, self.V,
                    label=rf'{self.potential.capitalize()} potential')
        
        plt.xlabel(r'$r_*$', fontsize = 16)
        plt.ylabel(r'$\Psi$', fontsize = 16)
        plt.legend(fontsize = 12)
        plt.grid(True)
        plt.tight_layout()
        plt.show()



def gaussian_initial_profile(rstar, rstar0, width, omega):
    return (1 / np.sqrt(2*np.pi*width**2)*np.exp(-(rstar - rstar0)**2/(2*width**2))*np.exp(-1j*omega*rstar))

if __name__ == "__main__":
    # Example 1: Axial l=2
    axial_sim = SchwarzschildPerturbation(M=1.0, l=2, potential='axial', rstar_min = -400, rstar_max = 700, dx = 0.1)
    width = 10*axial_sim.M
    rstar0 = 110*axial_sim.M
    omega = 0.5
    Psi0 = 1/(np.sqrt(2*np.pi*width**2))*np.exp(-(axial_sim.rstar - rstar0)**2/(2*width**2))*np.exp(-j*omega*axial_sim.rstar)
    dPsi0 = (-j*omega - (axial_sim.rstar - rstar0)/width**2)*Psi0
    
    axial_sim.solve(Psi0, Nt = 8000, dPsi0 = dPsi0)
    
    axial_sim.make_gif(component='real', frame_interval = 100, omega = omega)

#     # Example 2: Polar l=3
#     sim_polar = SchwarzschildPerturbation(M = 1.0, l = 3, parity = 'polar', rstar_min = -400, rstar_max = 700, dx = 0.1)
#     Psi0_polar = gaussian_initial_profile(sim_polar.rstar, rstar0 = 100*sim_polar.M, width = 10*sim_polar.M, omega = 0.8)
    
#     sim_polar.solve(Psi0_polar, Nt = 8000)

#     sim_polar.make_gif(component='real', frame_interval = 50, omega = 0.8)