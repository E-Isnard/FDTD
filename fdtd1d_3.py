import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

"""
1D FDTD with time and space dependant electric permitivity

H has only one component along y and E only along z

H_y and E_z only depend on x.

We use a normalized version of E: Ẽ_z = sqrt(eps/mu)*E

Time unit: 1ns
Length Unit: 1mm

eps
"""

eps_0_SI = 8.85418782E-12 
mu_0_SI = 4*np.pi*1E-7
L = 400E-3
Tmax = 6E-9
dx=1.5E-3
x = np.arange(0,L,dx)
nx = len(x)

eps=eps_0_SI
mu = mu_0_SI
ws = 1E10*2*np.pi
wm = 1E9*2*np.pi

c = (eps*mu)**(-1/2)
print(f"c={c}")
dt = dx/(2*c)
print(f"dt={dt}")
t = np.arange(0,Tmax,dt)
epsR = 3*(1+0.67*np.sin(wm*t))
nt = len(t)

print(f"nt={nt}")
print(f"nx={nx}")
eta1 = np.sqrt(mu/eps)
eta2 = np.sqrt(mu/(eps*epsR))
gamma = (eta2-eta1)/(eta2+eta1)
tau = 2*eta2/(eta2+eta1)
print(f"Γ={np.round(gamma,2)}")
print(f"τ={np.round(tau,2)}")

H = np.zeros((nt,nx))
E = np.zeros((nt,nx))
#Position of the dielectric
i1 = int(nx/2)-40
i2 = i1+62
kSource = 10
source = np.sin(ws*t)
# source = 1*np.exp(-(t-5)**2)
b = 0.5*np.ones((nx,))
a = np.ones((nx,))

# E[0,iSource] = 1
#E[0] = np.exp(-100*(x-L/2)**2)
# E[0] = np.sin(2*np.pi*x)
E[0,kSource]=source[0]

for i in range(nt-1):

    a[i1:i2]=epsR[i+1]/epsR[i]
    b[i1:i2]=0.5/epsR[i]

    H[i+1,:-1] = H[i,:-1]+1/2*(E[i,1:]-E[i,:-1])
    E[i+1,1:] = a[1:]*E[i,1:]+b[1:]*(H[i+1,1:]-H[i+1,:-1])

    #ABC 2 (from "Electromagnetism simulation using the fdtd method with python" textbook)
    # E[i+1,0]= E[i-1,1]
    # E[i+1,-1] = E[i-1,2]

    #ABC 2 (from Liu phd thesis)
    E[i+1,0] = E[i,1]-1/3*(E[i+1,1]-E[i,0])
    E[i+1,-1] = E[i,-2]-1/3*(E[i+1,-2]-E[i,-1])

    #Soft source
    E[i+1,kSource] += source[i]

for i in range(nt):
    y = H[i]
    y2 = E[i]
    if i==0:
        line, = plt.plot(x,y)
        line2, = plt.plot(x,y2)
        plt.ylim([-10,10])
        plt.legend(["$H_y$","$E_z$"])
        plt.fill_betweenx([-10,10],x1=x[i1],x2=x[i2],color="grey",alpha=0.8)
    else:
        line.set_ydata(y)
        line2.set_ydata(y2)
    plt.pause(0.001)



plt.show()

plt.plot(t,E[:,i2+2])
plt.title("Normalized Electric Field $E_z(x_0,t)$ with $x_0=x_{out}+2\Delta x$")
plt.xlabel("time (s)")
plt.ylabel("$E_z(x_0,t)$")

plt.show()