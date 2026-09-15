import numpy as np
import matplotlib.pyplot as plt

L = 1
dx = 0.05
c = 1.0
dt = 1.0*dx/c #CFL condition- information can't travel along grid faster than it can in the
#physical world

x = np.arange(0, L*(1+dx), dx)
print(x)
npts = len(x)
nsteps = 200

y = np.zeros((npts, 3)) #past present and future
y[:,0] = np.sin(2*np.pi*x/L) #includes boundary conditions already
y[1:-1, 1] = y[1:-1, 0] + 0.5* c**2*(dt/dx)**2*(y[2:, 0] + y[:-2, 0] - 2*y[1:-1, 0])

for k in range(0, nsteps):
    y[1:-1, 2] = 2*y[1:-1, 1] - y[1:-1, 0] + c**2*(dt/dx)**2*(y[2:,1] + y[:-2, 1] - 2*y[1:-1, 1])
    y[1:-1, 0] = y[1:-1, 1]
    y[1:-1, 1] = y[1:-1, 2]

    plt.figure(1)
    plt.clf()
    plt.plot(x, y[:,2], 'b-o')
    plt.ylim(-1.2, 1.2)
    plt.title('t = %0.2f' % (k*dt))
    plt.pause(0.01)