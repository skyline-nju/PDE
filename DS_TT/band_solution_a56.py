import numpy as np
import matplotlib.pyplot as plt
from RK4_ODE2nd import RK4_ODE2, RK4_ODE2nd_ge

"""_summary_
Phenomenological hydrodynamic equations up to a6 term.

So far, the trajectory seems to be very unstable once considering a5 and a6 term.
"""

def locate_zero(x):
    idx = []
    for i in range(x.size):
        if x[i-1] < 0 and x[i] >= 0:
            idx.append(i)
    return np.array(idx)


def remove_early_periods(zero_idx, z, m, m_dot, periods_removed=2):
    idx0 = zero_idx[periods_removed]
    z = z[idx0:]
    m = m[idx0:]
    m_dot = m_dot[idx0:]
    zero_idx = zero_idx[periods_removed:]
    z -= z[0]
    zero_idx -= zero_idx[0]
    return zero_idx, z, m, m_dot


class PhenoHydros:
    def __init__(self, c, rho_g, D=1, v0=1, lamb=1, xi=1, a4=-1, a5=0, a6=0, phi_g=1):
        self.c = c
        self.rho_g = rho_g
        self.D = D
        self.v0 = v0
        self.lamb = lamb
        self.xi = xi
        self.a4 = a4
        self.a5 = a5
        self.a6 = a6
        self.phi_g = phi_g
    
    def H_deriv(self, m):
        return -(self.phi_g - self.rho_g) * m + self.v0 / self.c * m**2 + self.a4 * m**3 + self.a5 * m**4 - self.a6 * m**5 

    def H(self, m):
        return -(self.phi_g - self.rho_g) * m**2/2 + self.v0 / self.c * m**3/3 + self.a4 * m**4/4 + self.a5 * m**5/5 - self.a6 * m**6/6

    def friction(self, m):
        return self.c - self.lamb * self.v0 / self.c - self.xi * m

    def f(self, m, m_dot):
        return -(self.friction(m) * m_dot + self.H_deriv(m)) / self.D
    
    def intg(self, h, z0, z1, m_0, m_dot_0):
        n = int((z1-z0)/h)
        z_arr = np.linspace(z0, z1, n, endpoint=False)
        m_arr = np.zeros_like(z_arr)
        m_dot_arr = np.zeros_like(z_arr)
        m_arr[0] = m_0
        m_dot_arr[0] = m_dot_0

        for i in range(1, n):
            m_arr[i], m_dot_arr[i] = RK4_ODE2(m_arr[i-1], m_dot_arr[i-1], h, self.f)
        return z_arr, m_arr, m_dot_arr
    

    def get_w(self, w0, w_dot_0, z0_arr, m_arr, m_dot_arr, sigma=0):
        ''' w = v1 - U0

            sigma = -0.0237 for limit circle with rho_g = 0.835 and c = 1.14

            For hete with rho_g = 0.840055880329 and c=1.12
            sigma = -0.10868   
        '''
        def func(i, x, x_dot):
            U0 = m_arr[i]
            R0 = self.rho_g + self.v0/self.c * U0
            # a21 = -(R0 - self.phi_g - self.a4 * U0**2 + 0.2) / self.D
            a21 = -(R0 - self.phi_g - self.a4 * U0**2 - sigma) / self.D

            # a22 = -(self.c - self.xi * U0 - self.lamb * self.v0 / self.c) / self.D
            a22 = -(self.c - self.xi * U0) / self.D

            return a21 * x + a22 * x_dot
        
        h = z0_arr[2] - z0_arr[0]
        t_arr = np.arange(z0_arr.size // 2) * h
        w_arr = np.zeros_like(t_arr)
        w_dot_arr = np.zeros_like(t_arr)

        w_arr[0] = w0
        w_dot_arr[0] = w_dot_0

        for i in range(0, t_arr.size-1):
            w_arr[i+1], w_dot_arr[i+1] = RK4_ODE2nd_ge(w_arr[i], w_dot_arr[i], h, 2 * i, func)
        
        return t_arr, w_arr, w_dot_arr

    
    def get_phi0(self, x0, x_dot_0, z0_arr, m_arr, m_dot_arr):
        def func(i, x, x_dot):
            U0 = m_arr[i]
            U0_dot = m_dot_arr[i]
            R0 = self.rho_g + self.v0/self.c * U0
            a21 = -(2 * R0 - self.rho_g - self.xi * U0_dot - self.phi_g - 3 * self.a4 * U0**2) / self.D
            a22 = -(self.c - self.lamb * self.v0 / self.c - self.xi * U0) / self.D
            return a21 * x + a22 * x_dot

        h = z0_arr[2] - z0_arr[0]
        t_arr = np.arange(z0_arr.size // 2) * h
        x_arr = np.zeros_like(t_arr)
        x_dot_arr = np.zeros_like(t_arr)

        x_arr[0] = x0
        x_dot_arr[0] = x_dot_0

        for i in range(0, t_arr.size-1):
            x_arr[i+1], x_dot_arr[i+1] = RK4_ODE2nd_ge(x_arr[i], x_dot_arr[i], h, 2 * i, func)
        
        return t_arr, x_arr, x_dot_arr


if __name__ == "__main__":
    # rho_g = 0.840055880329
    # c=1.12
    # m_0 = 1e-8
    # ph = PhenoHydros(c, rho_g, D=1)
    # h = 0.005
    # z0 = 0
    # z1 = 1000
    # m_dot_0 = 0
    # z, m, m_dot = ph.intg(h, z0, z1, m_0, m_dot_0)
    # plt.plot(m, m_dot)
    # plt.show()
    # plt.close()


    rho_g = 0.92
    c0 = 1.38
    a4 = -2.1
    a5 = 2.45
    a6 = 1

    xi = 3.1133

    ph = PhenoHydros(c0, rho_g, xi=xi, a4=a4, a5=a5, a6=a6)

    h = 0.005
    z0 = 0
    z1 = 1000

    m_0 = 0.05
    m_dot_0 = 0
    # m_0 = 0.2
    # m_dot_0 = 0.1
    z, m, m_dot = ph.intg(h, z0, z1, m_0, m_dot_0)

    plt.plot(m, m_dot)
    plt.show()
    plt.close()