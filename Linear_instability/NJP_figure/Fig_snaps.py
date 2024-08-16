import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import image as mpimg
import os
import sys


def CCB_phase_separation():
    fig = plt.figure(figsize=(12, 8.7), layout="constrained")
    subfigs = fig.subfigures(3, 1, wspace=0.001, hspace=0.001, height_ratios=[160, 160, 640])

    axes0 = subfigs[0].subplots(1, 3, sharex=True, sharey=True)
    axes1 = subfigs[1].subplots(1, 3, sharex=True, sharey=True)
    axes2 = subfigs[2].subplots(1, 2, sharex=True, sharey=True)

    axes = [i for i in axes0]
    for i in axes1:
        axes.append(i)
    for i in axes2:
        axes.append(i)

    fnames = ["G_CCB.png", "G_LA_CCB.png", "LA_CCB.png",
              "G_LB_CCB.png", "LB_CCB_0.png", "LB_CCB_1.png",
              "G_LB_CCB_L640.png", "LB_CCB_L640.png"]
    titles = ["(a) G+CCB", "(b) G+LA+CCB", "(c) LA+CCB", "(d) G+LB+CCB",
              "(e) LB+CCB: initial configuration", "(f) LB+CCB: stready state",
              "(g) G+LB+CCB", "(h) LB+CCB"]

    for i, ax in enumerate(axes):
        fname = f"fig/snap/{fnames[i]}"
        im = mpimg.imread(fname)
        ax.imshow(im)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        # ax.set_title(titles[i], fontsize="x-large")

    ax_in = axes1[2].inset_axes([1-0.25, 1-0.25, 0.25, 0.25])
    fname = f"fig/snap/LB_CCB_L120_40.png"
    im = mpimg.imread(fname)
    ax_in.imshow(im)
    ax_in.set_xticklabels([])
    ax_in.set_yticklabels([])
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    ax_in.set_aspect("equal")

    fs = "xx-large"
    bbox=dict(edgecolor="w", facecolor="w", boxstyle="Square, pad=0.08")
    axes0[0].text(0.01, 0.83, "(a) G+CCB", fontsize=fs, transform=axes0[0].transAxes, backgroundcolor="w", bbox=bbox)
    axes0[1].text(0.01, 0.83, "(b) G+LA+CCB", fontsize=fs, transform=axes0[1].transAxes, backgroundcolor="w", bbox=bbox)
    axes0[2].text(0.01, 0.83, "(c) LA+CCB", fontsize=fs, transform=axes0[2].transAxes, backgroundcolor="w", bbox=bbox)
    axes1[0].text(0.01, 0.83, "(d) G+LB+CCB", fontsize=fs, transform=axes1[0].transAxes, backgroundcolor="w", bbox=bbox)
    axes1[1].text(0.01, 0.83, r"(e) LB+CCB: $t=0$", fontsize=fs, transform=axes1[1].transAxes, backgroundcolor="w", bbox=bbox)
    axes1[2].text(0.01, 0.8, r"(f) LB+CCB: $t=10^5$", fontsize=fs, transform=axes1[2].transAxes, backgroundcolor="w", bbox=bbox)
    axes2[0].text(0.01, 0.955, "(g) G+LB+CCB", fontsize=fs, transform=axes2[0].transAxes, backgroundcolor="w", bbox=bbox)
    axes2[1].text(0.01, 0.955, "(h) LB+CCB", fontsize=fs, transform=axes2[1].transAxes, backgroundcolor="w", bbox=bbox)

    
    plt.show()
    # plt.savefig("fig/snaps_PS_CCB.pdf")
    plt.close()


def Eq_phase_separation():
    fig, axes = plt.subplots(2, 3, figsize=(10, 2.8), sharex=True, sharey=True, constrained_layout=True)

    fnames = ["G_LA.png", "G_LAB.png", "G_LA_LAB.png", "G_LA_LB.png", "LA_LB.png", "LA_LB_LAB.png"]
    titles = ["(a) G+LA", "(b) G+LAB", "(c) G+LA+LAB", "(d) G+LA+LB",
              "(e) LA+LB", "(f) LA+LB+LAB"]

    for i, ax in enumerate(axes.flat):
        fname = f"fig/snap/{fnames[i]}"
        im = mpimg.imread(fname)
        ax.imshow(im)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_title(titles[i], fontsize="x-large")
    
    plt.show()
    # plt.savefig("fig/snaps_PS_Eq.pdf")
    plt.close()




if __name__ == "__main__":
    # CCB_phase_separation()
    # Eq_phase_separation()


    fig, axes = plt.subplots(1, 2, figsize=(8, 4.35), sharex=True, sharey=True, constrained_layout=True)
    fnames = ["wo_rep.png", "w_rep.png"]
    titles = ["(a) Without repulsion", "(b) With repulsion"]
    
    for i, ax in enumerate(axes):
        fname = f"fig/snap/{fnames[i]}"
        im = mpimg.imread(fname)
        ax.imshow(im)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_title(titles[i], fontsize="x-large")
    # plt.show()
    plt.savefig("fig/snaps_wo_w_rep.pdf")
    plt.close()