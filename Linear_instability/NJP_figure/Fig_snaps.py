import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import image as mpimg
import os
import sys


def CCB_phase_separation():
    fig, axes = plt.subplots(2, 3, figsize=(10, 2.8), sharex=True, sharey=True, constrained_layout=True)

    fnames = ["G_CCB.png", "G_LA_CCB.png", "LA_CCB.png", "G_LB_CCB.png", "LB_CCB_0.png", "LB_CCB_1.png"]
    titles = ["(a) G+CCB", "(b) G+LA+CCB", "(c) LA+CCB", "(d) G+LB+CCB",
              "(e) LB+CCB: initial configuration", "(f) LB+CCB: stready state"]

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
    
    # plt.show()
    plt.savefig("fig/snaps_PS_CCB.pdf")
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
    
    # plt.show()
    plt.savefig("fig/snaps_PS_Eq.pdf")
    plt.close()

if __name__ == "__main__":
    # CCB_phase_separation()
    Eq_phase_separation()
