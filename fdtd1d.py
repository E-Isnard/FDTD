import numpy as np
import matplotlib.pyplot as plt

"""
1D FDTD with constant eps and mu

H has only one component along y and E only along z

H_y and E_z only depend on x.

We use a normalized version of E: Ẽ_z = sqrt(eps/mu)*E

dt/(eps*dx)*sqrt(eps/mu) = dt/(sqrt(eps*mu)*dx) = dt*c/dx = 1/2

(dt/(mu*dx))*sqrt(mu/eps) = dt*c/dx = 1/2
"""

# eps_0 = 8.85418782E-12 
# mu_0 = 4*np.pi*1E-7
L = 1
Tmax = 10
eps = 10
mu = 10
c = (eps*mu)**(-1/2)
dx=0.01
dt = dx/(2*c)

t = np.arange(0,Tmax,dt)
x = np.arange(0,L,dx)

nt = len(t)
nx = len(x)

print(f"nt={nt}")
print(f"nx={nx}")

H = np.zeros((nt,nx))
E = np.zeros((nt,nx))
kSource = int(nx/2)
source = 1*np.exp(-(t-Tmax/2)**2)
# E[0,iSource] = 1
#E[0] = np.exp(-100*(x-L/2)**2)
# E[0] = np.sin(2*np.pi*x)
E[0,kSource]=source[0]
for i in range(nt-1):

    H[i+1,:-1] = H[i,:-1]+1/2*(E[i,1:]-E[i,:-1])
    H[i+1,-1] = H[i+1,-2]

    E[i+1,1:] = E[i,1:]+1/2*(H[i+1,1:]-H[i+1,:-1])
    E[i+1,0]=E[i+1,1]
    E[i+1,kSource] = source[i+1]

for i in range(nt):
    y = H[i]
    y2 = E[i]
    if i==0:
        line, = plt.plot(x,y)
        line2, = plt.plot(x,y2)
        plt.ylim([-3,3])
        plt.legend(["$H_y$","$E_z$"])
        plt.title(f"Step n°1 ($\\eps={eps}$ and $\\mu={mu}$)")
    else:
        line.set_ydata(y)
        line2.set_ydata(y2)
        plt.title(f"Etape n°{i+1} ($\\eps={eps}$ and $\\mu={mu}$)")
    plt.pause(0.1)

plt.show()

# plt.plot(t,H[:,0])
# plt.show()
