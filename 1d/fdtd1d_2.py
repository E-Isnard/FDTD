"""
1D FDTD with time independent mu and epsilon,however they are space dependant.

H has only one component along y and E only along z

H_y and E_z only depend on x.

We use a normalized version of E: Ẽ_z = sqrt(eps/mu)*E

"""
import numpy as np
import matplotlib.pyplot as plt



# eps_0 = 8.85418782E-12 
# mu_0 = 4*np.pi*1E-7
L = 1
Tmax = 30
dx=2.5E-3
x = np.arange(0,L,dx)
nx = len(x)
epsR=3
eps=10

mu = 10

c = (eps*mu)**(-1/2)
dt = dx/(2*c)

t = np.arange(0,Tmax,dt)

nt = len(t)

print(f"nt={nt}")
print(f"nx={nx}")
eta1 = np.sqrt(mu/eps)
eta2 = np.sqrt(mu/(eps*epsR))
gamma = (eta2-eta1)/(eta2+eta1)
tau = 2*eta2/(eta2+eta1)
print(f"gamma={np.round(gamma,2)}")
print(f"tau={np.round(tau,2)}")

H = np.zeros((nt,nx))
E = np.zeros((nt,nx))
#Position of the dielectric
i1 = int(nx/2)-40
i2 = i1+20
kSource = 10
source = 1*np.exp(-(t-5)**2)
source = np.sin(2*np.pi*t)
b = 0.5*np.ones((nx,))
b[i1:i2]/=epsR

# E[0,iSource] = 1
#E[0] = np.exp(-100*(x-L/2)**2)
# E[0] = np.sin(2*np.pi*x)
E[0,kSource]=source[0]

for i in range(nt-1):

    H[i+1,1:] = H[i,1:]+1/2*(E[i,1:]-E[i,:-1])
    H[i+1,0] = E[i,0]
    E[i+1,:-1] = E[i,:-1]+b[:-1]*(H[i+1,1:]-H[i+1,:-1])
    E[i+1,-1] = -H[i+1,-1]
    #ABC 2 (from "Electromagnetism simulation using the fdtd method with python" textbook)
    # E[i+1,0]= E[i-1,1]
    # E[i+1,-1] = E[i-1,2]

    #ABC 2 (from Liu phd thesis)
    # E[i+1,0] = E[i,1]-1/3*(E[i+1,1]-E[i,0])
    # E[i+1,-1] = E[i,-2]-1/3*(E[i+1,-2]-E[i,-1])
    # H[i+1,0] = H[i,1]-1/3*(H[i+1,1]-H[i,0])
    # H[i+1,-1] = H[i,-2]-1/3*(H[i+1,-2]-H[i,-1])

    #Hard source
    E[i+1,kSource] = source[i+1]

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

# plt.plot(t,H[:,0])
# plt.show()
