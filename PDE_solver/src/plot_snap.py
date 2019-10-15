"""
    Read binary file.
"""

import numpy as np
import struct
import os


class Snapshot:
    def __init__(self, filename):
        s = os.path.basename(filename).replace(".bin", "").split("_")
        self.rho0 = float(s[3].replace("r", ""))
        self.eta = float(s[1].replace("eta", ""))
        self.eps = float(s[2].replace("zeta", ""))
        self.Lx = int(s[-5].replace("Lx", ""))
        self.Ly = int(s[-4].replace("Ly", ""))
        self.Nx = int(s[-3].replace("Nx", ""))
        self.Ny = int(s[-2].replace("Ny", ""))
        self.frame_size = 4 * (1 + self.Nx * self.Ny * 3)
        self.open_file(filename)
        print("frame size =", self.frame_size)

    def open_file(self, filename):
        self.f = open(filename, "rb")
        self.f.seek(0, 2)
        self.file_size = self.f.tell()
        self.f.seek(0)
        print("open ", filename, "with size =", self.file_size)

    def one_frame(self):
        buf = self.f.read(4)
        t = struct.unpack("f", buf)[0]
        n_buf = self.Nx * self.Ny * 4
        buf = self.f.read(n_buf)
        rho = np.array(struct.unpack("%df" % (n_buf // 4), buf)).reshape(
            self.Ny, self.Nx).T
        buf = self.f.read(n_buf)
        px = np.array(struct.unpack("%df" % (n_buf // 4), buf)).reshape(
            self.Ny, self.Nx).T
        buf = self.f.read(n_buf)
        py = np.array(struct.unpack("%df" % (n_buf // 4), buf)).reshape(
            self.Ny, self.Nx).T
        frame = [t, rho, px, py]
        return frame

    def gene_frames(self, beg=0, end=None, sep=1):
        self.f.seek(beg * self.frame_size)
        if end is None:
            max_size = self.file_size
        else:
            max_size = end * self.frame_size
        count = 0
        while max_size - self.f.tell() >= self.frame_size:
            if count % sep == 0:
                yield self.one_frame()
            else:
                self.f.seek(self.frame_size, 1)
            count += 1


def plotRF(eta, zeta, rho0, dt, L, block_size=3, t_pause=None, save_fig=False):
    import matplotlib.pyplot as plt
    if t_pause is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(12, 5))
        plt.ion()
    f0 = r"../data/RF_eta%g_zeta%g_r%g_Lx%d_Ly%d_Nx%d_Ny%d_dt%g.bin" % (
        eta, zeta, rho0, L, L, L * block_size, L * block_size, dt)
    if save_fig:
        snap_dir = f0.replace(".bin", "")
        if not os.path.exists(snap_dir):
            os.mkdir(snap_dir)

    snap = Snapshot(f0)
    frames = snap.gene_frames()
    for frame in frames:
        t, rho, px, py = frame
        print("t =", t, "rho_mean =", np.mean(rho), "rho_min =", rho.min())
        # fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 5))
        rho1 = np.cbrt(rho)
        if t_pause is None:
            fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 5))
        im1 = ax1.imshow(rho1, origin="lower", extent=[0, L, 0, L])
        ax1.set_xlabel(r"$x$", fontsize="x-large")
        ax1.set_ylabel(r"$y$", fontsize="x-large")
        cb1 = plt.colorbar(im1, ax=ax1, orientation="horizontal")
        cb1.set_label(r"$\rho ^ {1/3}$", fontsize="x-large")

        rho2 = rho.copy()
        rho2[rho > 0] = 0
        im2 = ax2.imshow(rho2, origin="lower", extent=[0, L, 0, L])
        ax2.set_xlabel(r"$x$", fontsize="x-large")
        ax2.set_ylabel(r"$y$", fontsize="x-large")
        cb2 = plt.colorbar(im2, ax=ax2, orientation="horizontal")
        cb2.set_label(
            r"$\rho^* (=\rho\ {\rm if}\ \rho < 0\ {\rm else}\ =0) $",
            fontsize="x-large")

        ori = np.arctan2(py, px) / np.pi * 360
        ori[ori < 0] += 360
        im3 = ax3.imshow(
            ori,
            origin="lower",
            extent=[0, L, 0, L],
            vmin=0,
            vmax=360,
            cmap="hsv")
        ax3.set_xlabel(r"$x$", fontsize="x-large")
        ax3.set_ylabel(r"$y$", fontsize="x-large")
        cb3 = plt.colorbar(im3, ax=ax3, orientation="horizontal")
        cb3.set_label("orientation", fontsize="x-large")
        phi = np.sqrt(px.mean()**2 + py.mean()**2)

        title = "additive RF: " +\
            r"$L=%d, \eta=%g, \epsilon=%g, \rho_0=%g, t=%d, \phi=%.4f$" % (
                L, eta, zeta, rho0, t, phi)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.suptitle(title, fontsize="xx-large", y=0.995)
        # plt.show()
        if t_pause is not None:
            if save_fig:
                plt.savefig("%s%s%04d.png" % (snap_dir, os.path.sep, t))
            plt.pause(t_pause)
            ax1.clear()
            ax2.clear()
            ax3.clear()
            cb1.remove()
            cb2.remove()
            cb3.remove()
        else:
            plt.show()
            plt.close()


if __name__ == "__main__":
    eta = 0.1
    zeta = 0.
    rho0 = 1
    L = 64
    block_size = 2
    dt = 0.05
    plotRF(eta, zeta, rho0, dt, L, block_size, 0.1, False)
