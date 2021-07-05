"""
2D FDTD with time and space depandant electric permitivity

We use TMz, which means that E has only one component along z and 
that H has components along x and y. They both depend only on x and y.

We use a normalized version of E: Ẽ_z = √(ɛ0/µ0)*E
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import perf_counter
from scipy.signal import hilbert
from scipy.ndimage import median_filter
from fillbetween3d import fill_between_3d

def progress(i, n):
    i += 1
    k = int(i/n*20)
    print(
        f'\rProgression:[{k*"#"}{(20-k)*" "}] [{i/n*100:.0f} %]', end='' if i != n else "\n", flush=True)

def abc_order_1(i,Ez):
    Ez[i+1, 0, :] = Ez[i, 1, :]+k_abc*(Ez[i+1, 1, :]-Ez[i, 0, :])
    Ez[i+1, -1, :] = Ez[i, -2, :]+k_abc*(Ez[i+1, -2, :]-Ez[i, -1, :])
    Ez[i+1, :, 0] = Ez[i, :, 1]+k_abc*(Ez[i+1, :, 1]-Ez[i, :, 0])
    Ez[i+1, :, -1] = Ez[i, :, -2]+k_abc*(Ez[i+1, :, -2]-Ez[i, :, -1])

# Units
t_unit = 1E-9  # ns
xy_unit = 1E-3  # mm

# EM constants
eps_0_SI = 8.85418782E-12
mu_0_SI = 4*np.pi*1E-7
eps_0 = eps_0_SI
mu_0 = mu_0_SI
c = (eps_0*mu_0)**(-1/2)

dxy = 1.5E-3
xymax = 300E-3
# courant_number = c*dt/dx
courant_number = 0.95/np.sqrt(2)

dt = courant_number*dxy/c
Tmax = 1500*dt
x = np.arange(0, xymax, dxy)
y = np.arange(0, xymax, dxy)
t = np.arange(0, Tmax, dt)

nxy = len(x)
nt = len(t)

Hx = np.zeros((nt, nxy, nxy))
Hy = np.zeros((nt, nxy, nxy))
Ez = np.zeros((nt, nxy, nxy))

ca = np.ones((nxy,nxy))
cb = courant_number*np.ones((nxy,nxy))

j1=0
j2 = nxy-1
k1 = 40
shift = int(93E-3/dxy)
k2 = k1+shift
epsR = 4*(1+0.67*np.sin(1e9*2*np.pi*t))

X, Y = np.meshgrid(x, y)
mid = int(nxy/2)
t0 = 20
spread = 6
k_abc = (courant_number-1)/(courant_number+1)
k2_abc = 2/(courant_number+1)
k3_abc = courant_number**2/(2*(courant_number+1))
k4_abc = -1/(2*dxy*(courant_number+1))
s = perf_counter()
for i in range(nt-1):
    source = np.sin(2*np.pi*t[i]*1e10)*courant_number
    # source = np.exp(-0.5 * ((t0 - i) / spread) ** 2)
    ca[j1:j2,k1:k2] = epsR[i+1]/epsR[i]
    cb[j1:j2,k1:k2] = courant_number/epsR[i]

    Hx[i+1,:,1:] = Hx[i,:,1:] - courant_number*(Ez[i,:,1:]-Ez[i,:,:-1])
    Hy[i+1,1:,:] = Hy[i,1:,:] + courant_number*(Ez[i,1:,:]-Ez[i,:-1,:])
    #Silver-Muller
#     Hy[i+1,0,:] = Ez[i,0,:]
#     Hx[i+1,:,0] = -Ez[i,:,0]

    Ez[i+1,:-1,:-1] = ca[:-1,:-1]*Ez[i,:-1,:-1]+cb[:-1,:-1]*(Hy[i+1,1:,:-1]-Hy[i+1,:-1,:-1]-Hx[i+1,:-1,1:]+Hx[i+1,:-1,:-1])
    #Silver-Muller
#     Ez[i+1,-1,:] = -Hy[i+1,-1,:]
#     Ez[i+1,:,-1] = Hx[i+1,:,-1]
  
    abc_order_1(i,Ez)

    Ez[i+1, :, 20] += source
    # Ez[i+1, mid, mid+10] += source

    progress(i, nt-1)


print(f"Calculations took {(perf_counter()-s):.2f} s")
print(
    f"Memory taken by Hx,Hy and Ez: {((Hx.nbytes+Hy.nbytes+Ez.nbytes)/(1024**2)):.2f} MB")


def anim3d():

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d", xlabel="X", ylabel="Y", zlabel="Z")

    x2 = x

    # Sets of [x, y, 0] lines (with constant y) for defining the bases of the polygons.
    set01 = [x2, k1*dxy*np.ones(len(x2)), np.zeros(len(x2))]

    # Sets of the form [x, y, fi] (with constant y) representing each function 3D line.
    set1  = [x2, k1*dxy*np.ones(len(x2)), 10*np.ones(len(x2))]

    def anim(i):
        ax.clear()
        ax.set_zlim(-1, 1)
        plot = ax.plot_surface(X, Y, Ez[i])
        ax.plot(*set1, lw=2, zorder=20, c="C1")
        fill_between_3d(ax, *set01, *set1, mode = 1, c="C1")
        plt.title(i)
        return plot,

    a = animation.FuncAnimation(
        fig, anim, interval=1000/60, frames=int((nt-1)*0.75), blit=False, repeat=False)
    print("anim3d:")
    # a.save("animation3d.gif", fps=60, progress_callback=progress)
    plt.show()



def animContour():
    vmin = 0
    vmax = 1
    cmap = "jet"
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()
    plt_tmp = ax.contourf(X, Y, Ez[-1], vmin=vmin,
                          vmax=vmax, levels=100, cmap=cmap)
    plt.colorbar(plt_tmp)
    ax.contourf(X, Y, Ez[0], vmin=vmin, vmax=vmax, levels=100, cmap=cmap)
    
    x1 = x[k1]
    x2 = x[k2]
    y1 = y[j1]
    y2 = y[j2]
    coord = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    coord.append(coord[0]) #repeat the first point to create a 'closed loop'

    xs, ys = zip(*coord) #create lists of x and y values


    def anim2(i):
        ax.clear()
        plt.plot(xs,ys,color="black",lw=4) 
        plot = ax.contourf(X, Y, Ez[i], vmin=vmin,
                           vmax=vmax, levels=100, cmap=cmap)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        return plot,

    a2 = animation.FuncAnimation(
        fig, anim2, interval=1000/60, frames=int((nt-1)*1), blit=False, repeat=False)
    print("contour:")
#     a2.save("contour.mp4", fps=20, progress_callback=progress)
    plt.show()


# anim3d()
animContour()
E_out = Ez[:,mid,k2+10]
plt.figure(figsize=(10,10))
plt.xlabel("X")
plt.plot(t,E_out)
plt.show()