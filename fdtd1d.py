import numpy as np
import matplotlib.pyplot as plt

"""
1D FDTD with constant eps and mu

H has only one component along y and E only along z

H_y and E_z only depend on x.
"""

L = 1
Tmax = 100
eps = 10
mu = 10
c = (eps*mu)**(-1/2)
dx=0.01
courant = 0.5
dt = (dx*courant)/c 

t = np.arange(0,Tmax,dt)
x = np.arange(0,L,dx)

nt = len(t)
nx = len(x)

print(f"nt={nt}")
print(f"nx={nx}")

H = np.zeros((nt,nx))
E = np.zeros((nt,nx))
H[0] = np.exp(-(x-L/2)**2/0.01)
E[0] = np.exp(-(x-L/3)**2/0.01)

for i in range(nt-1):

    H[i+1,:-1] = H[i,:-1]+dt/(dx*mu)*(E[i,1:]-E[i,:-1])
    H[i+1,-1]=H[i,-1]+dt/(dx*mu)*(-E[i,-1])

    E[i+1,0]=E[i,0]+dt*(dx*mu)*(H[i+1,0])
    E[i+1,1:] = E[i,1:]+dt*(dx*eps)*(H[i+1,1:]-H[i+1,:-1])

for i in range(nt):
    y = H[i]
    y2 = E[i]
    if i==0:
        line, = plt.plot(x,y)
        line2, = plt.plot(x,y2)
        plt.ylim([-20,20])
        plt.legend(["$H_y$","$E_z$"])
    else:
        line.set_ydata(y)
        line2.set_ydata(y2)
    plt.pause(0.01)

plt.show()

