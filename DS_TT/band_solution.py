import numpy as np
import matplotlib.pyplot as plt
from RK4_ODE2nd import RK4_ODE2, RK4_ODE2nd_ge

"""Nice parameters:
    Homoclinic orbits
    c=1.12, rho_g=0.840055880329, m=1e-8, m_dot=0, other params = 1

    c=1.12, rho_g=0.8479083791452791, m=1e-7, m_dot=0, D=0.01, other params = 1

    c=1.14, rho_g=0.8341205982775, m=1e-8, m_dot=0, other params = 1

    c=1.14, rho_g=0.833679247417284, m=1e-8, m_dot=0, D=10, other params = 1

    Heteroclinic trajectory
    c=1.1547, rho_g=0.8333333022524737, m=1e-8, m_dot=0, other params = 1
    m_0 in [0.2, 0.1, 0.02, 1e-8]:


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
    def __init__(self, c, rho_g, D=1, v0=1, lamb=1, xi=1, a4=1, phi_g=1):
        self.c = c
        self.rho_g = rho_g
        self.D = D
        self.v0 = v0
        self.lamb = lamb
        self.xi = xi
        self.a4 = a4
        self.phi_g = phi_g
    
    def H_deriv(self, m):
        return -(self.phi_g - self.rho_g) * m + self.v0 / self.c * m**2 - self.a4 * m**3

    def H(self, m):
        return -(self.phi_g - self.rho_g) * m**2/2 + self.v0 / self.c * m**3/3 - self.a4 * m**4/4

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
    # rho_g = 0.835
    # c = 1.14
    # m_0 = 0.04442609229

    rho_g = 0.840055880329
    c=1.12
    m_0 = 1e-8

    # rho_g=0.8333333022524737
    # c=1.1547
    # m_0=1e-8

    ph = PhenoHydros(c, rho_g, D=1)

    h = 0.005
    z0 = 0
    z1 = 1000
    m_dot_0 = 0

    # for m_0 in [0.2, 0.1, 0.02, 1e-8]:
    # for m_0 in [1e-8]:
    z, m, m_dot = ph.intg(h, z0, z1, m_0, m_dot_0)
    # line, = plt.plot(m_0, m_dot_0, "o")
    #     # plt.plot(m, m_dot, c=line.get_c())

    # # z, m, m_dot = ph.intg(h, z0, z1, 0.0, 0)
    # plt.plot(m, m_dot)
    # # plt.plot(0, 0, "x") 

    # plt.plot(z, m, z, m_dot)
    # # # plt.xlim(-0.01, 0.01)
    # # # plt.ylim(-0.01, 0.01)

    # # # plt.plot(m[zero_idx[-1]], m_dot[zero_idx[-1]], "s")

    # plt.show()
    # plt.close()

    zero_idx = locate_zero(m_dot)
    zero_idx, z, m, m_dot = remove_early_periods(zero_idx, z, m, m_dot, periods_removed=2)
        
    # plt.plot(z, m, z, m_dot)
    # for i in zero_idx:
    #     plt.axvline(z[i], linestyle="--", c="tab:grey")
    # plt.plot(m, m_dot)
    # plt.show()
    # plt.close()


    # m = np.linspace(-0.2, 0.65, 1000)
    # H = np.array([ph.H(i) for i in m])
    # plt.axhline(0)
    # plt.plot(m, H)
    # plt.show()
    # plt.close()

    # w0 = 1
    # w_dot_0 = 0

    # t, w, w_dot = ph.get_w(w0, w_dot_0, z, m, m_dot)
    

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    # ax1.plot(w0, w_dot_0, "o")
    # ax1.plot(w, w_dot)

    # for i in zero_idx:
    #     ax2.axvline(z[i], linestyle="--", c="tab:grey")

    # ax2_r = ax2.twinx()
    # ax2_r.plot(z, m, "--", c="tab:pink")
    # ax2.axhline(0, linestyle="dotted", c="k")
    # ax2.plot(t, w, t, w_dot)
    # plt.show()
    # plt.close()
        
    # w0_arr = [1, 2, -1, -1]
    # w_dot_0_arr = [0, 1, 0, -1]
    w0_arr = [0]
    w_dot_0_arr=[1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)

    # sigma_hete = [-0.08700319, -0.0990942924, -0.119391]
    sigma_homo = [-0.146244589, -0.271045093]
    n_periods = 1
    print(zero_idx[n_periods])
    print(z[zero_idx[n_periods]])
    z, m, m_dot = z[:zero_idx[n_periods]], m[:zero_idx[n_periods]], m_dot[:zero_idx[n_periods]]
    # for j in range(len(w0_arr)):
    for j, sigma in enumerate(sigma_homo):
        t, w, w_dot = ph.get_w(w0_arr[0], w_dot_0_arr[0], z, m, m_dot, sigma_homo[j])
        line, = ax1.plot(w0_arr[0], w_dot_0_arr[0], "o")
        ax1.plot(w, w_dot, c=line.get_c(), label=r"$\hat{w}_{%d}$" % (j+1))
        # ax1.plot(w, w_dot, c=line.get_c(), label=r"$w(0)=%g,\dot{w}(0)=%g$" % (w0_arr[j], w_dot_0_arr[j]))
        ax2.plot(t, w, c=line.get_c())
    ax1.set_xlabel(r"$w$", fontsize="x-large")
    ax1.set_ylabel(r"$\dot{w}$", fontsize="x-large")
    ax1.legend()
    ax2.axhline(0, linestyle="--", c="tab:grey")
    ax2.set_ylabel(r"w", fontsize="x-large")
    ax2.set_xlabel(r"z", fontsize="x-large")
    ax1.axvline(0, linestyle="--", c="tab:grey")
    ax1.axhline(0, linestyle="--", c="tab:grey")

    # ax2_right = ax2.twinx()
    # ax2_right.plot(z, m)
    for i in range(n_periods):
        ax2.axvline(z[zero_idx[i]], linestyle="dashed", c="tab:red")
    ax2.axvline(z[-1], linestyle="dashed", c="tab:red")
    # plt.suptitle(r"$\sigma = 0$")
    
    plt.show()
    plt.close()


    w0 = 0
    w_dot_0 = 0.1
    sigma_arr = np.linspace(-0.4, 0.01, num=1000)

    n_periods = 1
    z, m, m_dot = z[:zero_idx[n_periods]+1], m[:zero_idx[n_periods]+1], m_dot[:zero_idx[n_periods]+1]

    w_right = np.zeros_like(sigma_arr)
    for i, sigma in enumerate(sigma_arr):
        t, w, w_dot = ph.get_w(w0, w_dot_0, z, m, m_dot, sigma)
        w_right[i] = w[-1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)
    ax1.plot(sigma_arr, w_right)
    ax1.axhline(0, linestyle="--", c="tab:grey")
    ax2.plot(sigma_arr, w_right)
    ax2.axhline(0, linestyle="--", c="tab:grey")
    ax2.set_xlim(-0.3, -0.12)
    sigma_home = [-0.146244589, -0.271045093]

    # sigma_hete = [-0.08700319, -0.0990942924, -0.119391, -0.1466410549]
    for sigma in sigma_home:
        ax2.axvline(sigma, linestyle=":", c="tab:red")
        ax1.axvline(sigma, linestyle=":", c="tab:red")

    
    ax1.set_xlabel(r"$\sigma$", fontsize="x-large")
    ax1.set_ylabel(r"$w(z=L_x)$", fontsize="x-large")
    ax2.set_xlabel(r"$\sigma$", fontsize="x-large")
    ax2.set_ylabel(r"$w(z=L_x)$", fontsize="x-large")
    plt.show()
    plt.close()




