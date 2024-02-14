import numpy as np
import matplotlib.pyplot as plt

import fqs

def get_BCDE(c0, rho_g, phi_g, v0, a4, a5, a6):
    B = -a5 / a6
    C = -a4 / a6
    D = -v0 / (a6 * c0)
    E = (phi_g - rho_g) / a6
    return B, C, D, E


phi_g = v0 = lambdaa = 1
xi = 1
a6 = 1
a4 = -0.5
a5 = 1

nc = 300
n_rho_g = 300
c_1D = np.linspace(1, 3, nc)
rho_g_1D = np.linspace(0.5, 1, n_rho_g)

a5_arr = np.arange(20) * 0.05
a4_arr = -np.arange(20) * 0.05
for a5 in a5_arr:
    for a4 in a4_arr:
        print(a4, a5)
        state = np.zeros((nc, n_rho_g))

        for i, c0 in enumerate(c_1D):
            for j, rho_g in enumerate(rho_g_1D):
                B, C, D, E = get_BCDE(c0, rho_g, phi_g, v0, a4, a5, a6)
                xs = fqs.single_quartic(1, B, C, D, E)
                if xs[0].imag == 0 and xs[1].imag == 0 and xs[2].imag == 0 and xs[3].imag == 0:
                    xs_real = np.array([i.real for i in xs])
                    if xs_real.min() > 0 and xs_real.max() < 1:
                        state[j, i] = 1
        if np.sum(state) > 0.:
            plt.imshow(state, origin="lower", extent=[c_1D[0], c_1D[-1], rho_g_1D[0], rho_g_1D[-1]])
            plt.axis("auto")
            plt.title(r"$a_4=%g, a_5=%g$" % (a4, a5))
            plt.colorbar()
            plt.show()
            plt.close()
