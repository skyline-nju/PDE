import numpy as np


def get_freq(Nx, Ny, spacing):
    ky_1d = np.zeros(Ny)
    kx_1d = np.zeros(Nx // 2 + 1)
    for i in range(Ny):
        if i < Ny // 2:
            ky_1d[i] = i / (Ny * spacing) * np.pi * 2
        else:
            ky_1d[i] = (i - Ny) / (Ny * spacing) * np.pi * 2
    for i in range(Nx // 2 + 1):
        kx_1d[i] = i / (Nx * spacing) * np.pi * 2
    kx, ky = np.meshgrid(kx_1d, ky_1d)
    k2 = kx ** 2 + ky ** 2
    return kx, ky, k2


def get_freq_pad(Nx, Ny, spacing):
    Ky = int(Ny / 2 * 3)
    Kx_old = Nx // 2 + 1
    Kx = int(Kx_old / 2 * 3)
    ky_1d_pad = np.zeros(Ky)
    kx_1d_pad = np.zeros(Kx)

    Ny_pad = Ky
    Nx_pad = (Kx - 1) * 2

    for i in range(Ky):
        if i < Ky // 2:
            ky_1d_pad = i / (Ny_pad * spacing) * np.pi * 2
        else:
            ky_1d_pad = (i-Ny_pad) / (Ny_pad * spacing) * np.pi * 2
    for i in range(Kx):
        kx_1d_pad[i] = i / (Nx_pad * spacing) * np.pi * 2
    kxx_pad, kyy_pad = np.meshgrid(kx_1d_pad, ky_1d_pad)
    kx_pad = np.array([kxx_pad, kxx_pad])
    ky_pad = np.array([kyy_pad, kyy_pad])
    k2_pad = kx_pad ** 2 + ky_pad ** 2
    return kx_pad, ky_pad, k2_pad


def zeros_pad(u):
    Kx = u.shape[-1]
    Ky = u.shape[-2]

    Kx_pad = int(Kx / 2 * 3)
    Ky_pad = int(Ky / 2 * 3)
    
    u_pad = np.zeros((u.shape[0], Ky_pad, Kx_pad), np.complex128)
    u_pad[:, :Ky//2, :Kx] = u[:, :Ky//2, :]
    u_pad[:, Ky_pad - Ky//2:, :Kx] = u[:, Ky//2:, :]
    return u_pad


def dealias_product(u_pad, v_pad, Kx, Ky):
    prod_hat_pad = np.fft.rfft2(u_pad * v_pad)
    Ky_pad = prod_hat_pad.shape[-2]
    Kx_pad = prod_hat_pad.shape[-1]
    prod_hat = np.zeros((u_pad.shape[0], Ky, Kx), np.complex128)

    prod_hat[:, :Ky//2, :] = prod_hat_pad[:, :Ky//2, :Kx]
    prod_hat[:, Ky//2:, :] = prod_hat_pad[:, Ky_pad-Ky//2:, :Kx]

    Nx_Ny = (2 * Kx - 1) * Ky
    Nx_pad_Ny_pad = (2 * Kx_pad - 1) * Ky_pad
    prod_hat *= (Nx_pad_Ny_pad / Nx_Ny)

    return prod_hat