import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

"""
1D FDTD with time and space dependant electric permitivity

H has only one component along y and E only along z

H_y and E_z only depend on x.

We use a normalized version of E: Ẽ_z = sqrt(eps/mu)*E

"""

xmax = 400E-3
Tmax = 6E-9
dx=1.5E-3

#EM constants
eps_0_SI = 8.85418782E-12 
mu_0_SI = 4*np.pi*1E-7
eps_0=eps_0_SI
mu_0 = mu_0_SI
eps_r0 = 3

#Pulsation for source and epsR
ws = 1E10*2*np.pi
wm = 1E9*2*np.pi

#Length of the dielectric stab
L=3E-3

#Source and its position
ks = 10
source_func = lambda t :np.sin(ws*t)

#Modulation depth
m_depth = 0.67

def FDTD_1D(Tmax,dx,xmax,eps_r0,L,source_func,ks,m_depth):
    
    c = (eps_0*mu_0)**(-1/2)
    dt = dx/(2*c)
    t = np.arange(0,Tmax,dt)
    nt = len(t)

    source = source_func(t)

    x = np.arange(0,xmax,dx)
    nx = len(x)
    epsR = eps_r0*(1+m_depth*np.sin(wm*t))
    
    #Position of the dielectric
    shift = int(L/dx)
    k1 = int(nx/2)
    k2 = k1+shift
   
    H = np.zeros((nt,nx))
    E = np.zeros((nt,nx))

    b = 0.5*np.ones((nx,))
    a = np.ones((nx,))

    E[0,ks]=source[0]

    for i in range(nt-1):

        a[(k1+1):(k2-1)]=epsR[i+1]/epsR[i]
        b[(k1+1):(k2-1)]=0.5/epsR[i]
        a[k1] = a[k2] = (1+epsR[i+1])/(1+epsR[i])
        b[k1] = b[k2] = 1/(1+epsR[i])
    

        H[i+1,:-1] = H[i,:-1]+1/2*(E[i,1:]-E[i,:-1])
        E[i+1,1:] = a[1:]*E[i,1:]+b[1:]*(H[i+1,1:]-H[i+1,:-1])

        #ABC 2 (from Liu phd thesis)
        E[i+1,0] = E[i,1]-1/3*(E[i+1,1]-E[i,0])
        E[i+1,-1] = E[i,-2]-1/3*(E[i+1,-2]-E[i,-1])

        #Soft source
        E[i+1,ks] += source[i]
    return (t,x,k1,k2,E,H)



def anim_E_H(t,E,H,k1,k2,y_low,y_high,anim=True,interval=1E-3):
    for i in range(len(t)):
        y = H[i]
        y2 = E[i]
        if i==0:
            line, = plt.plot(x,y)
            line2, = plt.plot(x,y2)
            plt.ylim([y_low,y_high])
            plt.legend(["$H_y$","$E_z$"])
            plt.fill_betweenx([y_low,y_high],x1=x[k1],x2=x[k2],color="grey",alpha=0.8)
        else:
            line.set_ydata(y)
            line2.set_ydata(y2)
        if anim:
            plt.pause(interval)
    plt.show()


def plot_E(E,ko):
    
    plt.plot(t,E[:,ko])
    # plt.plot(t,E_ext)
    plt.title("Normalized Electric Field $\\tilde{E}_z(x_0,t)$ with $x_0=x_{out}+2\Delta x$")
    plt.xlabel("time (s)")
    plt.ylabel("$\\tilde{E}_z(x_0,t)$")

    plt.show()

def w(Tmax,dx,xmax,eps_r0,L,ks,m_depth,ws,ko):
    source1_func = lambda t:np.sin(ws*t)
    source2_func = lambda t:np.cos(ws*t)
    t,_,_,_,E1,_ = FDTD_1D(Tmax,dx,xmax,eps_r0,L,source1_func,ks,m_depth)
    _,_,_,_,E2,_ = FDTD_1D(Tmax,dx,xmax,eps_r0,L,source2_func,ks,m_depth)
    nt = len(t)
    phi = np.arctan2(E1[:,ko],E2[:,ko])
    dt = Tmax/nt
    w_vec = ((phi[:-4]-8*phi[1:-3]+8*phi[3:-1]-phi[4:])/(12*dt))
    # w_vec = (phi[1:]-phi[:-1])/dt
    return (phi,w_vec,E1,E2)


t,x,k1,k2,E,H = FDTD_1D(Tmax,dx,xmax,eps_r0,L,source_func,ks,0.67)

# coeff = (2*np.tan(wm*t/2)+m_depth)/np.sqrt(4-m_depth**2)
# coeff2 = wm*L*np.sqrt(4-m_depth**2)
# coeff3 = np.arctan(coeff)-coeff2
# w_ext = (1/np.cos(wm*t/2)**2)/(1+coeff**2)*ws/np.cos(coeff3**2)**2 * 1/(1+(np.sqrt(4-m_depth**2)/2*np.tan(coeff3)-m_depth/2)**2)

phi,w_vec,E1,E2 = w(Tmax,dx,xmax,eps_r0,L,ks,m_depth,ws,k2+2)
plt.plot(t,phi)
plt.show()
plt.plot(t[2:-2],w_vec)
plt.show()
plot_E(E,k2)
# anim_E_H(t,E,H,k1,k2,-2,2)



