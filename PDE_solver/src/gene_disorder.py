"""
    Read and generate disorder.
    2018/12/16
"""

import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy import interpolate


def read_RT(Nx, Ny, Lx, Ly):
    fin = "../data/disorder_realization/RT_Nx%d_Ny%d_Lx%d_Ly%d.bin" % (Nx, Ny,
                                                                       Lx, Ly)
    with open(fin, "rb") as f:
        buf = f.read()
        data = struct.unpack("%dd" % (Nx * Ny), buf)
        rand_torques = np.array(data).reshape(Nx, Ny)
    print("mean torques =", np.mean(rand_torques))
    return rand_torques


def read_RF(Nx, Ny, Lx, Ly):
    fin = "../data/disorder_realization/RF_Nx%d_Ny%d_Lx%d_Ly%d.bin" % (Nx, Ny,
                                                                       Lx, Ly)
    with open(fin, "rb") as f:
        buf = f.read()
        data = struct.unpack("%dd" % (Nx * Ny * 2), buf)
        RFx = np.array(data[:Nx * Ny]).reshape(Nx, Ny)
        RFy = np.array(data[Nx * Ny:]).reshape(Nx, Ny)
    return RFx, RFy


def create_RT(L, block_size=3):
    np.random.seed(1)
    gamma = np.linspace(-1, 1, L * L) * np.pi
    np.random.shuffle(gamma)
    gamma = gamma.reshape(L, L)
    gamma_ext = np.zeros((L + 8, L + 8))
    gamma_ext[0:4, 4:L + 4] = gamma[L - 4:L, :]
    gamma_ext[4:L + 4, 4:L + 4] = gamma
    gamma_ext[L + 4:L + 8, 4:L + 4] = gamma[0:4, :]
    gamma_ext[:, 0:4] = gamma_ext[:, L:L + 4]
    gamma_ext[:, L + 4:L + 8] = gamma_ext[:, 4:8]

    x = y = np.linspace(-4, L + 4, L + 8, endpoint=False) * block_size + 1
    x_fit = y_fit = np.linspace(0, L * block_size, L * block_size, False)

    gamma_fit = np.zeros((L * block_size, L * block_size))
    f = interpolate.interp2d(x, y, gamma_ext, "linear")
    # f = interpolate.RectBivariateSpline(x, y, gamma_ext)
    gamma_fit = f(x_fit, y_fit)
    print("gamma_fit: min=", gamma_fit.min(), "max=", gamma_fit.max())
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 5.5))
    box = [0, L, 0, L]
    im1 = ax1.imshow(gamma, origin="lower", extent=box)
    im2 = ax2.imshow(gamma_fit, origin="lower", extent=box)
    cb1 = plt.colorbar(im1, ax=ax1, orientation="horizontal")
    cb2 = plt.colorbar(im2, ax=ax2, orientation="horizontal")
    cb1.set_label("random torque", fontsize="x-large")
    cb2.set_label("random torque", fontsize="x-large")
    ax1.set_title(r"${\rm d}x=1$", fontsize="xx-large")
    ax2.set_title(r"${\rm d}x=1/3$", fontsize="xx-large")
    plt.tight_layout()
    plt.show()
    plt.close()
    print("mean of gamma = ", np.mean(gamma), "mean of gamma_fit = ",
          np.mean(gamma_fit))
    gamma_x = np.mean(np.cos(gamma))
    gamma_y = np.mean(np.sin(gamma))
    gamma_fit_x = np.mean(np.cos(gamma_fit))
    gamma_fit_y = np.mean(np.sin(gamma_fit))
    print("mean of module = ", np.sqrt(gamma_x**2 + gamma_y**2),
          "mean of module of fitting field = ",
          np.sqrt(gamma_fit_x**2 + gamma_fit_y**2))

    fout = "../data/disorder_realization/RT_Nx%d_Ny%d_Lx%d_Ly%d.bin" % (
        L * block_size, L * block_size, L, L)
    with open(fout, "wb") as f:
        gamma_out = gamma_fit.flatten()
        gamma_out.tofile(f)


if __name__ == "__main__":
    L = 64
    block_size = 2
    create_RT(L, block_size)

    # Nx = Ny = L * block_size
    # rand_torques = read_RT(Nx, Ny, L, L)
    # plt.imshow(rand_torques, origin="lower")
    # plt.show()
    # plt.close()
