# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ov463DtbC5WwOvmFGXrREQp07p8A_jgI
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit

# Units

t_unit = 1E-9  # ns
xy_unit = 1E-3  # mm

xymax = 100E-3/xy_unit
Tmax = 20E-9/t_unit
dxy = 1E-3/xy_unit
# courant_number = c*dt/dx
courant_number = 1/2

# EM constants
eps_0_SI = 8.85418782E-12
mu_0_SI = 4*np.pi*1E-7
eps_0 = 1
mu_0 = 1
c = (eps_0*mu_0)**(-1/2)
dt = courant_number*dxy/c

x = np.arange(0, xymax, dxy)
y = np.arange(0, xymax, dxy)
t = np.arange(0, Tmax, dt)
X, Y = np.meshgrid(x, y)


nxy = len(x)
nt = len(t)
print("nt =", nt)
print("nxy =", nxy)


@njit(cache=True)
def main():
    Hx = np.zeros((nt, nxy, nxy))
    Hy = np.zeros((nt, nxy, nxy))
    Ez = np.zeros((nt, nxy, nxy))
    med = int(nxy/2)
    Ez[0] = 10*np.exp(-0.01*((X-40)**2+(Y-40)**2))
    for i in range(nt-1):
        for j in range(1, nxy):
            for k in range(1, nxy):
                Hx[i+1, j, k] = Hx[i, j, k]-courant_number * \
                    (Ez[i, j, k]-Ez[i, j, k-1])
                Hy[i+1, j, k] = Hy[i, j, k]+courant_number * \
                    (Ez[i, j, k]-Ez[i, j-1, k])

        for j in range(nxy-1):
            for k in range(nxy-1):
                Ez[i+1, j, k] = Ez[i, j, k]+courant_number * \
                    (Hy[i+1, j+1, k]-Hy[i+1, i, j] -
                     Hx[i+1, j, k+1]+Hx[i+1, j, k])

    return (t, x, y, Ez, Hx, Hy)


t, x, y, Ez, Hx, Hy = main()

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(30, 30))
ax = plt.axes(projection="3d")


ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(-10, 10)


def anim(i):
    ax.clear()
    plot = ax.plot_surface(X, Y, Ez[i], cmap="Spectral")
    ax.set_zlim(-100, 100)
    return plot,


a = animation.FuncAnimation(
    fig, anim, interval=1000/10, frames=39, blit=False, repeat=False)
plt.show()

# plt.savefig("oui.png")
