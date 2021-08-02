from numba.core.decorators import njit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import lagrange
from scipy.integrate import quad
from scipy.special import legendre
from time import perf_counter


def progress(i, n):
    i += 1
    k = int(i/n*20)
    print(
        f'\rProgression:[{k*"#"}{(20-k)*" "}] [{(i/n*100):.0f} %]', end='' if i != n else "\n", flush=True)


def elem_lagrange(xi, xj):
    l1 = lagrange([xi, xj], [1, 0])
    l2 = lagrange([xi, xj], [0, 1])
    return l1, l2


def block_matrix(A, B, C, D):
    d = 2
    M = np.zeros((2*d, 2*d))
    M[:d, :d] = A
    M[:d, d:] = B
    M[d:, :d] = C
    M[d:, d:] = D
    return M


L = 1
ne = 120
nn = ne+1
T = 1
nt = 1200
h = L/ne
dt = T/nt

print(dt/h)


Mk = h*np.array([[1/3, 1/6], [1/6, 1/3]])
invMk = 1/h*np.array([[4, -2], [-2, 4]])
Kk = np.array([[-1/2, 1/2], [-1/2, 1/2]])
S1k = np.array([[-1/2, 0], [0, 1/2]])
S2k = np.array([[0, 1/2], [0, 0]])
S3k = np.array([[0, 0], [1/2, 0]])
S4k = np.array([[0, 0], [0, 1]])
S5k = np.array([[0, 0], [1, 0]])

Mk_bar = block_matrix(Mk, 0, 0, Mk)
invMk_bar = block_matrix(invMk,0,0,invMk)
Kk_bar = block_matrix(0, Kk, Kk, 0)
S1k_bar = block_matrix(0, S1k, S1k, 0)
S2k_bar = block_matrix(0, S2k, S2k, 0)
S3_bar = block_matrix(0, S3k, S3k, 0)

u = np.zeros((nt, ne, 2))


def u0(x):
    return np.sin(2*np.pi*x)


x = np.linspace(0, L, nn)
t = np.linspace(0, T, nt)

b0k = np.zeros((2,))
# L2 proj of u0 on Vh
# for k in range(ne):
#     xk = k*h
#     xkp1 = xk+h
#     l1, l2 = elem_lagrange(xk, xkp1)
#     b0k[0] = quad(lambda x: l1(x)*u0(x), xk, xkp1)[0]
#     b0k[1] = quad(lambda x: l2(x)*u0(x), xk, xkp1)[0]
#     u[0, k, :] = invMk@b0k

@njit(cache=True)
def f(u,l,k):
    p = k+1 if k+1 < ne else 0
    return invMk@((Kk-S1k)@u[l, k]-S2k@u[l, p]+S3k@u[l, k-1])


u[0, :, 0] = u0(x[:-1])
u[0, :, 1] = u0(x[1:])

s = perf_counter()
for l in range(nt-1):
    for k in range(ne):
        
        k1 = f(u,l,k)
        # k2 = f(u+k1*dt/2,l,k)
        # k3 = f(u+k2*dt/2,l,k)
        # k4 = f(u+k3*dt,l,k)
        # du = dt/6*(k1+2*k2+2*k3+k4)

        u[l+1, k] = u[l, k]+k1*dt
    progress(l, nt-1)
print(f"Time taken: {perf_counter()-s} s")

def plot(u, i):
    plt.clf()
    plt.ylim(-3/2, 3/2)
    for k in range(ne):
        xk = (k)*h
        xkp1 = (k+1)*h
        Ik = np.array([xk, xkp1])
        plt.plot(Ik, [u[i, k, 0], u[i, k, 1]])
    plt.plot(x, u0(x-t[i]), c="tab:blue")


fig = plt.figure()

plot(u, 0)

anim = True
if anim:
    def animation(i):
        plot(u, i)

    anim = FuncAnimation(fig, animation, nt, repeat=False, interval=100)

# anim.save("anim.mp4",progress_callback=progress)
plt.show()
