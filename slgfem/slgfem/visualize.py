# ~~~
# This file is part of the paper:
#
#           " A super-localized generalized finite element method "
#
#   https://github.com/TiKeil/SL-GFEM.git
#
# Copyright 2022 all developers. All rights reserved.
# License: Licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Contributors team: Philip Freese, Moritz Hauck, Tim Keil, and Daniel Peterseim
# ~~~

import numpy as np
import matplotlib
from matplotlib import cm, pyplot as plt

from gridlod import util


def d3sol(N, s, string=''):
    '''
    3d solution
    '''
    fig = plt.figure(string)
    ax = fig.add_subplot(111, projection='3d')

    xp = util.pCoordinates(N)
    X = xp[0:, 1:].flatten()
    Y = xp[0:, :1].flatten()
    X = np.unique(X)
    Y = np.unique(Y)

    X, Y = np.meshgrid(X, Y)

    uLodFine = s.reshape(N + 1)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, uLodFine, cmap=cm.jet)
    ax.set_zticks([])
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title(string)


def drawCoefficient(N, a, logNorm=True, colorbar_font_size=None, cmap=None, limits=[None, None]):
    if a.ndim == 3:
        a = np.linalg.norm(a, axis=(1, 2), ord=2)

    aCube = a.reshape(N, order='F')
    aCube = np.ascontiguousarray(aCube.T)

    plt.clf()

    cbar_scale = matplotlib.colors.LogNorm() if logNorm else None
    plt.imshow(aCube,
               origin='lower',
               interpolation='none', cmap=cmap,
               vmax=limits[0], vmin=limits[1],
               norm=cbar_scale
               )

    plt.xticks([])
    plt.yticks([])
    if colorbar_font_size:
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=colorbar_font_size)
    plt.tight_layout()