import numpy as np
import matplotlib.pyplot as plt
from band_solution import PhenoHydros, locate_zero
from scipy.integrate import cumulative_trapezoid


def get_mu(M_arr, z_arr, D, c, v0, lamb, xi, z_shfit=0):
    f1 = (c / D - lamb * v0 / (c * D)) * (z_arr-z_shfit) 
    return np.exp(f1 - xi / D * M_arr) / D


def band1():
    rho_g = 0.840055880329
    c = 1.12
    D=1
    v0=1
    lamb=1
    xi=1
    a4=1
    phi_g=1
    band = PhenoHydros(c, rho_g, D, v0, lamb, xi, a4, phi_g)

    h = 0.01
    z0 = 0
    z1 = 500
    m_0 = 1e-8
    m_dot_0 = 0
    z, m, m_dot = band.intg(h, z0, z1, m_0, m_dot_0)
    M = cumulative_trapezoid(m, x=z, initial=0)
    
    mu = get_mu(M, z, D, c, v0, lamb, xi, z_shfit=0)

    f0 = mu * m_dot * (D * m_dot - lamb * v0 / c * m + v0/c * m * M)
    I0 = cumulative_trapezoid(f0, x=z, initial=0)

    f1 = mu * m_dot * (v0/c**2 * m * (m - 0) - (1+lamb * v0/c**2) * m_dot)
    I1 = cumulative_trapezoid(f1, x=z, initial=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), constrained_layout=True, sharex=True)
    ax1.plot(z, m, label=r"$U_0(z)$")
    ax1.plot(z, m_dot, label=r"$\partial_z U_0(z)$")
    ax1.set_ylabel(r"$U_0, \partial_z U_0$")
    ax1.axhline(0, linestyle="dashed", c="tab:grey")
    ax1.legend()

    # ax2.plot(z, I0, label=r"$G_0(z)$")
    # ax2.plot(z, I1, label=r"$G_1(z)$")
    # ax2.axhline(0, linestyle="dashed", c="tab:grey")
    # ax2.set_ylabel(r"$G_{0, 1}$")
    # ax2.legend()
    # ax2.set_xlabel(r"$z$")

    idx_zero_m_dot = locate_zero(m_dot)
    for i in idx_zero_m_dot:
        ax1.axvline(z[i], linestyle="dotted", c="k")
    ax2.axhline(0, linestyle="dashed", c="tab:grey")
    ax2.set_ylabel(r"$\sigma_1$")
    ax2.set_xlabel(r"$z$")
    for i in idx_zero_m_dot[1:]:
        ax2.plot(z[i], I0[i]/I1[i], "o", c="tab:blue")
        # ax2.plot(z[i], I2[i]/I1[i], "s", c="tab:orange", fillstyle="none")
        ax2.axvline(z[i], linestyle="dotted", c="k")
    
    ax1.set_xlim(z0, z1)
    ax1.set_title("(a)")
    ax2.set_title("(b)")
    plt.show()
    # plt.savefig("SH_homo.pdf")
    plt.close()


def band2():
    rho_g = 0.8341205982775
    c = 1.14
    D=1
    v0=1
    lamb=1
    xi=1
    a4=1
    phi_g=1
    band = PhenoHydros(c, rho_g, D, v0, lamb, xi, a4, phi_g)

    h = 0.001
    z0 = 0
    z1 = 120
    m_0 = 1e-8
    m_dot_0 = 0
    z, m, m_dot = band.intg(h, z0, z1, m_0, m_dot_0)

    M = cumulative_trapezoid(m, x=z, initial=0)
    mu = get_mu(M, z, D, c, v0, lamb, xi, z_shfit=0)

    f0 = mu * m_dot * (D * m_dot - lamb * v0 / c * m + v0/c * m * M)
    I0 = cumulative_trapezoid(f0, x=z, initial=0)

    f1 = mu * m_dot * (v0/c**2 * m * (m - 0) - (1+lamb * v0/c**2) * m_dot)
    I1 = cumulative_trapezoid(f1, x=z, initial=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), constrained_layout=True, sharex=True)
    ax1.plot(z, m, label=r"$U_0(z)$")
    ax1.plot(z, m_dot, label=r"$\partial_z U_0(z)$")
    ax1.set_ylabel(r"$U_0, \partial_z U_0$")
    ax1.axhline(0, linestyle="dashed", c="tab:grey")
    ax1.legend()

    ax2.plot(z, I0, label=r"$G_0(z)$")
    ax2.plot(z, I1, label=r"$G_1(z)$")
    ax2.axhline(0, linestyle="dashed", c="tab:grey")
    ax2.set_ylabel(r"$G_{0, 1}$")
    ax2.legend()
    ax2.set_xlabel(r"$z$")

    ax1.set_xlim(z0, z1)
    ax1.set_title("(a)")
    ax2.set_title("(b)")
    plt.show()
    # plt.savefig("SH_homo.pdf")
    plt.close()



def band3():
    rho_g = 0.833679247417285
    c = 1.14
    D=10
    v0=1
    lamb=1
    xi=1
    a4=1
    phi_g=1
    band = PhenoHydros(c, rho_g, D, v0, lamb, xi, a4, phi_g)

    h = 0.01
    z0 = 0
    z1 = 2000
    m_0 = 1e-8
    m_dot_0 = 0
    z, m, m_dot = band.intg(h, z0, z1, m_0, m_dot_0)

    # mask = z >= 350
    # z = z[mask] - 350
    # m = m[mask]
    # m_dot = m_dot[mask]
    plt.plot(z, m, z, m_dot)


    # m_int = cumulative_trapezoid(m_dot, x=z, initial=0)
    M = cumulative_trapezoid(m, x=z, initial=0)
    # plt.plot(z, M, "--")

    mu = get_mu(M, z, D, c, v0, lamb, xi, z_shfit=0)

    # ax_right = plt.gca().twinx()
    # ax_right.plot(z, mu * m_dot, ":")

    plt.show()
    plt.close()


    # R0 = rho_g + v0/c * m
    # m_ddot = np.gradient(m_dot, z)
    # m_tdot = np.gradient(m_ddot, z)
    # L_U0_dot = (R0 + v0/c * m - xi * m_dot - (phi_g + 3 * a4 * m**2)) * m_dot \
    #     + (c - xi * m - lamb * v0 / c) * m_ddot + D * m_tdot


    # y = c * m_dot - xi * m * m_dot + D * m_ddot - lamb * v0 / c * m_dot \
    #     - (phi_g + a4 * m**2) * m + R0 * m

    # y_dot = np.gradient(y, z)
    # plt.plot(z, y)
    # plt.show()
    # plt.close()

    f0 = mu * m_dot * (D * m_dot - lamb * v0 / c * m + v0/c * m * M)
    f1 = mu * m_dot * (v0/c**2 * m * (m -m[0]) - (1+lamb * v0/c**2) * m_dot)

    I0 = cumulative_trapezoid(f0, x=z, initial=0)
    I1 = cumulative_trapezoid(f1, x=z, initial=0)

    # plt.plot(z, f0, z, f1)
    # plt.axhline(0)
    # ax_right = plt.gca().twinx()
    # ax_right.plot(z, I0, ":", z, I1, "--")
    # ax_right.axhline(0)
    plt.plot(z, I0/I1/z, "--")
    plt.show()
    plt.close()


def shape_band():
    rho_g = 0.8479083791452791
    c = 1.12
    D=0.01
    v0=1
    lamb=1
    xi=1
    a4=1
    phi_g=1
    band = PhenoHydros(c, rho_g, D, v0, lamb, xi, a4, phi_g)

    h = 0.001
    z0 = 0
    z1 = 100
    m_0 = 1e-7
    m_dot_0 = 0
    z, m, m_dot = band.intg(h, z0, z1, m_0, m_dot_0)

    m -= m.min()
    plt.plot(z, m, z, m_dot)
    

    # m_int = cumulative_trapezoid(m_dot, x=z, initial=0)
    M = cumulative_trapezoid(m, x=z, initial=0)
    plt.plot(z, M, "--")

    mu = get_mu(M, z, D, c, v0, lamb, xi, z_shfit=20)

    ax_right = plt.gca().twinx()
    ax_right.plot(z, mu * m_dot, ":")

    plt.show()
    plt.close()

    # # mu = 1
    # f0 = mu * m_dot * (D * m_dot - lamb * v0 / c * m + v0/c * m * M)
    # f1 = mu * m_dot * (v0/c**2 * m **2 - (1+lamb * v0/c**2) * m_dot)

    # I0 = cumulative_trapezoid(f0, x=z, initial=0)
    # I1 = cumulative_trapezoid(f1, x=z, initial=0)

    # plt.plot(z, f0, z, f1)
    # ax_right = plt.gca().twinx()
    # # ax_right.plot(z, I0, ":", z, I1, "--")
    # ax_right.plot(z, I0/I1, ".")
    # plt.show()
    # plt.close()

def fat_band():
    rho_g = 0.8333333022525
    c = 1.1547
    D=1
    v0=1
    lamb=1
    xi=1
    a4=1
    phi_g=1
    band = PhenoHydros(c, rho_g, D, v0, lamb, xi, a4, phi_g)

    h = 0.001
    z0 = 0
    z1 = 740
    m_0 = 1e-8
    # m_0 = 0.1
    m_dot_0 = 0
    z, m, m_dot = band.intg(h, z0, z1, m_0, m_dot_0)
    idx_zero_m_dot = locate_zero(m_dot)

    z1 -= z[idx_zero_m_dot[0]]
    z = z[idx_zero_m_dot[0]:] - z[idx_zero_m_dot[0]]
    m = m[idx_zero_m_dot[0]:]
    m_dot = m_dot[idx_zero_m_dot[0]:]
    idx_zero_m_dot -= idx_zero_m_dot[0]
    


    M = cumulative_trapezoid(m, x=z, initial=0)

    mu = get_mu(M, z, D, c, v0, lamb, xi, z_shfit=0)

    # R0 = rho_g + v0/c * m
    # m_ddot = np.gradient(m_dot, z)
    # m_tdot = np.gradient(m_ddot, z)
    # L_U0_dot = (R0 + v0/c * m - xi * m_dot - (phi_g + 3 * a4 * m**2)) * m_dot \
    #     + (c - xi * m - lamb * v0 / c) * m_ddot + D * m_tdot


    # y = c * m_dot - xi * m * m_dot + D * m_ddot - lamb * v0 / c * m_dot \
    #     - (phi_g + a4 * m**2) * m + R0 * m

    # y_dot = np.gradient(y, z)

    # mu = 1
    f0 = mu * m_dot * (D * m_dot - lamb * v0 / c * m + v0/c * m * M)
    f2 = mu * m_dot * (D * m_dot - lamb * v0 / c * m + v0/c * m * 0)

    I0 = cumulative_trapezoid(f0, x=z, initial=0)
    I2 = cumulative_trapezoid(f2, x=z, initial=0)


    f1 = mu * m_dot * (v0/c**2 * m * (m - 0) - (1+lamb * v0/c**2) * m_dot)
    I1 = cumulative_trapezoid(f1, x=z, initial=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), constrained_layout=True, sharex=True)
    ax1.plot(z, m, label=r"$U_0(z)$")
    ax1.plot(z, m_dot, label=r"$\partial_z U_0(z)$")
    ax1.set_ylabel(r"$U_0, \partial_z U_0$")
    ax1.axhline(0, linestyle="dashed", c="tab:grey")
    ax1.legend()

    # ax2.plot(z, I0, label=r"$G_0(z)$")
    # ax2.plot(z, I1, label=r"$G_1(z)$")
    # ax2.axhline(0, linestyle="dashed", c="tab:grey")
    # ax2.set_ylabel(r"$G_{0, 1}$")
    # ax2.legend()
    # ax2.set_xlabel(r"$z$")

    for i in idx_zero_m_dot:
        ax1.axvline(z[i], linestyle="dotted", c="k")
    ax2.axhline(0, linestyle="dashed", c="tab:grey")
    ax2.set_ylabel(r"$\sigma_1$")
    ax2.set_xlabel(r"$z$")
    ax2_right = ax2.twinx()
    ax2_right.plot(z, mu)
    for i in idx_zero_m_dot[1:]:
        ax2.plot(z[i], I0[i]/I1[i], "o", c="tab:blue")
        # ax2.plot(z[i], I2[i]/I1[i], "s", c="tab:orange", fillstyle="none")
        ax2.axvline(z[i], linestyle="dotted", c="k")
    ax1.set_xlim(z0, z1)
    ax1.set_title("(a)")
    ax2.set_title("(b)")
    plt.show()
    # plt.savefig("SH_hete_multi.pdf")
    plt.close()



    # plt.plot(z, L_U0_dot)
    # plt.plot(z, y_dot)
    # plt.show()
    # plt.close()



    # plt.plot(z, I0/I1, ".")

    # f1 = mu * m_dot * (v0/c**2 * m * (m - m[0]) - (1+lamb * v0/c**2) * m_dot)
    # I1 = cumulative_trapezoid(f1, x=z, initial=0)
    # plt.axhline(0, linestyle="dashed", c="k")

    # plt.plot(z, I0/I1, ".")
    # plt.ylim(-200, 100)
    # ax_r = plt.gca().twinx()
    # ax_r.plot(z, m, "--")
    # plt.show()
    # plt.close()


def limit_circle1():
    rho_g = 0.835
    c = 1.14
    D=1
    v0=1
    lamb=1
    xi=1
    a4=1
    phi_g=1
    band = PhenoHydros(c, rho_g, D, v0, lamb, xi, a4, phi_g)

    h = 0.01
    z0 = 0
    z1 = 176
    m_0 = 0.04442608367
    # m_0 = 0.4665
    m_dot_0 = 0
    z, m, m_dot = band.intg(h, z0, z1, m_0, m_dot_0)

    idx_zero_m_dot = locate_zero(m_dot)

    M = cumulative_trapezoid(m, x=z, initial=0)
    
    mu = get_mu(M, z, D, c, v0, lamb, xi, z_shfit=0)

    f0 = mu * m_dot * (D * m_dot - lamb * v0 / c * m + v0/c * m * M)
    I0 = cumulative_trapezoid(f0, x=z, initial=0)

    f1 = mu * m_dot * (v0/c**2 * m * (m - 0) - (1+lamb * v0/c**2) * m_dot)
    I1 = cumulative_trapezoid(f1, x=z, initial=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), constrained_layout=True, sharex=True)
    ax1.plot(z, m, label=r"$U_0(z)$")
    ax1.plot(z, m_dot, label=r"$\partial_z U_0(z)$")
    ax1.set_ylabel(r"$U_0, \partial_z U_0$")
    ax1.axhline(0, linestyle="dashed", c="tab:grey")
    for i in idx_zero_m_dot:
        ax1.axvline(z[i], linestyle="dotted", c="k")
    ax1.legend()


    ax2.axhline(0, linestyle="dashed", c="tab:grey")
    ax2.set_ylabel(r"$\sigma_1$")
    ax2.set_xlabel(r"$z$")
    ax2_right = ax2.twinx()
    ax2_right.plot(z, mu)
    for i in idx_zero_m_dot[:]:
        ax2.plot(z[i], I0[i]/I1[i], "o", c="tab:blue")

        ax2.axvline(z[i], linestyle="dotted", c="k")
    ax1.set_xlim(z0, z1)
    ax1.set_title("(a)")
    ax2.set_title("(b)")
    plt.show()
    # plt.savefig("SH_limit_cycle.pdf")
    plt.close()

    


def limit_circle2():
    rho_g = 0.8345
    c = 1.14
    D=1
    v0=1
    lamb=1
    xi=1
    a4=1
    phi_g=1
    band = PhenoHydros(c, rho_g, D, v0, lamb, xi, a4, phi_g)

    h = 0.01
    z0 = 0
    z1 = 170 * 5
    # m_0 = 0.045
    m_0 = 0.02394485
    m_dot_0 = 0
    z, m, m_dot = band.intg(h, z0, z1, m_0, m_dot_0)

    # mask = z >= 100
    # z = z[mask] - 100
    # m = m[mask]
    # m_dot = m_dot[mask]
    plt.plot(z, m, z, m_dot)


    # m_int = cumulative_trapezoid(m_dot, x=z, initial=0)
    M = cumulative_trapezoid(m, x=z, initial=0)

    # ax_right = plt.gca().twinx()
    # ax_right.plot(z, mu * m_dot, ":")

    plt.show()
    plt.close()

    plt.plot(z, M, "--")
    mu = get_mu(M, z, D, c, v0, lamb, xi)
    ax_r = plt.gca().twinx()
    ax_r.plot(z, mu)

    plt.show()
    plt.close()


    R0 = rho_g + v0/c * m
    m_ddot = np.gradient(m_dot, z)
    m_tdot = np.gradient(m_ddot, z)
    L_U0_dot = (R0 + v0/c * m - xi * m_dot - (phi_g + 3 * a4 * m**2)) * m_dot \
        + (c - xi * m - lamb * v0 / c) * m_ddot + D * m_tdot

    plt.plot(z, L_U0_dot)
    plt.show()
    plt.close()

    f0 = mu * m_dot * (D * m_dot - lamb * v0 / c * m + v0/c * m * M)
    f1 = mu * m_dot * (v0/c**2 * (m - m[0]) * m - (1+lamb * v0/c**2) * m_dot)

    I0 = cumulative_trapezoid(f0, x=z, initial=0)
    I1 = cumulative_trapezoid(f1, x=z, initial=0)

    plt.plot(z, f0, z, f1)
    # # ax_right = plt.gca().twinx()
    # # ax_right.plot(z, I0, ":", z, I1, "--")

    # # ax_right.plot(z, I0/I1, "--")
    # plt.plot(z, I0/I1, ":")
    plt.show()
    plt.close()

    # I3 = cumulative_trapezoid(mu * m_dot * m, x=z, initial=0)
    # plt.plot(z, I3)
    # plt.show()
    # plt.close()


if __name__ == "__main__":
    # shape_band()
    # band1()
    fat_band()
    # limit_circle1()
