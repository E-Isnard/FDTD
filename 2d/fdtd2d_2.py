"""
2D FDTD with time and space depandant electric permitivity

We use TMz, which means that E has only one component along z and 
that H has components along x and y. They both depend only on x and y.

We use a normalized version of E: Ẽ_z = sqrt(eps/mu)*E
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit
import time


@njit
def func():

    # Units
    t_unit = 1E-9  # ns
    xy_unit = 1E-3  # mm

    dxy = 0.01
    xymax = 150*dxy
    # courant_number = c*dt/dx
    courant_number = 1/2

    # EM constants
    eps_0_SI = 8.85418782E-12
    mu_0_SI = 4*np.pi*1E-7
    eps_0 = eps_0_SI
    mu_0 = mu_0_SI
    c = (eps_0*mu_0)**(-1/2)
    dt = courant_number*dxy/c
    print(dt)
    Tmax = 200*dt

    x = np.arange(0, xymax, dxy)
    y = np.arange(0, xymax, dxy)
    t = np.arange(0, Tmax, dt)

    nxy = len(x)
    nt = len(t)

    Hx = np.zeros((nt, nxy, nxy))
    Hy = np.zeros((nt, nxy, nxy))
    Ez = np.zeros((nt, nxy, nxy))
    spread = 6
    t0 = 20
    mid = int(nxy/2)

    for i in range(nt-1):
        pulse = np.exp(-0.5 * ((t0 - i) / spread) ** 2)
        for j in range(1, nxy):
            for k in range(1, nxy):
                Ez[i+1, j, k] = Ez[i, j, k]+courant_number * \
                    (Hy[i, j, k]-Hy[i, j-1, k]-Hx[i, j, k]+Hx[i, j, k-1])

        for j in range(nxy-1):
            for k in range(nxy-1):
                Hx[i+1, j, k] = Hx[i, j, k]+courant_number * \
                    (Ez[i+1, j, k]-Ez[i+1, j, k+1])
                Hy[i+1, j, k] = Hy[i, j, k]+courant_number * \
                    (Ez[i+1, j+1, k]-Ez[i+1, j, k])
        Ez[i+1, mid, mid] = pulse
        
    return (t,x,y,Ez,Hx,Hy)
    # print(
    #     f'\rProgression:[{k*"#"}{(20-k)*" "}] [{i+1}/{nt-1}]', end='', flush=True)
t,x,y,Ez,Hx,Hy = func()
nt = len(t)
X, Y = np.meshgrid(x, y)
fig = plt.figure(figsize=(30, 30))
ax = plt.axes(projection="3d")

ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(-10, 10)


def anim(i):
    ax.clear()
    plot = ax.plot_surface(X, Y, Ez[i])
    ax.set_zlim(-0.1, 0.1)
    return plot,

time.sleep(2)
a = animation.FuncAnimation(
    fig, anim, interval=1000/60, frames=nt-1, blit=False, repeat=False)
# a.save("oui.gif")
plt.show()
