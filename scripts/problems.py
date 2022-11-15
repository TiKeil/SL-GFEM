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

"""
Trivial problem
"""

def trivial_problem(N_fine, localized_construction=False, constant_rhs=True):
    # this is the simplest test problem with constant diffusion and right hand side
    NWorldFine = np.array([N_fine, N_fine])
    NpFine = np.prod(NWorldFine + 1)

    if localized_construction:
        aFine_constructor = lambda patch: np.ones(patch.NPatchFine).flatten()
        aFine = None
    else:
        aFine_constructor = None
        aFine = np.ones(NWorldFine).flatten()

    if constant_rhs:
        f_fine = np.ones(NpFine)
    else:
        xp = util.pCoordinates(NWorldFine)
        # f_fine = np.sin(20 * xp[:, 0]) * np.sin(10 * xp[:, 1])
        f_fine = (xp[:, 0] + np.cos(3 * np.pi * xp[:, 0])) * xp[:, 1]**3

    return aFine, f_fine, aFine_constructor

"""
Crack problem without random field
"""

from gridlod import util

def crack_functions_in_pymor(contrast):
    # NOTE: importing pymor functions outside results in an Exception in the MPI case !!
    from pymor.analyticalproblems.functions import ConstantFunction, ExpressionFunction
    bounding_box = [[0, 0], [1, 1]]

    A, D = tuple(bounding_box[0]), (bounding_box[1][0], bounding_box[0][1])

    # thickness of the boundaries
    a = D[0] - A[0]
    b = a / 40.
    B, C = (A[0] + b, A[1]), (D[0] - b, A[1])
    c = bounding_box[1][1]
    E, F, G, H = (D[0], c), (C[0], c), (B[0], c), (A[0], c)

    d = c / 40.

    high_conductivity = contrast

    const_function = ConstantFunction(1, 2)

    X = '(x[0] >= A) * (x[0] < B)'
    Y = '(x[1] >= C) * (x[1] <= D)'

    horizontal_channel_1 = ExpressionFunction(f'{high_conductivity} * {X} * {Y}', 2,
                                              values={'A': A[0], 'B': D[0] - 2*b,
                                                      'C': A[1] + c/2 + 3*d, 'D': A[1] + c/2 + 4*d},
                                              name='horizontal_channel_1')
    horizontal_channel_2 = ExpressionFunction(f'{high_conductivity} * {X} * {Y}', 2,
                                              values={'A': A[0] + b, 'B': D[0],
                                                      'C': A[1] + c/2, 'D': A[1] + c/2 + d},
                                              name='horizontal_channel_3')

    vertical_channel_1 = ExpressionFunction(f'{high_conductivity} * {X} * {Y}', 2,
                                            values={'A': A[1] + c/2 + 3*d, 'B': A[1] + c/2 + 4*d,
                                                    'C': A[0], 'D': D[0] - 2*b},
                                            name='vertical_channel_1')
    vertical_channel_2 = ExpressionFunction(f'{high_conductivity} * {X} * {Y}', 2,
                                            values={'A': A[1] + c / 2, 'B': A[1] + c / 2 + d,
                                                    'C': A[0] + b, 'D': D[0]},
                                            name='vertical_channel_3')


    diffusion_function = horizontal_channel_1 + horizontal_channel_2 + \
                            vertical_channel_1 + vertical_channel_2 + const_function

    source_function = ConstantFunction(1, 2)

    return diffusion_function, source_function

def diffusion_function_on_patch(diffusion_function, patch):
    xt = util.computePatchtCoords(patch)
    return diffusion_function(xt)

def crack_problem(N_fine, contrast=1e3, localized_construction=False, constant_rhs=True):
    #### This is inspired by the model problem from https://arxiv.org/pdf/2103.10884.pdf (but with zero dirichlet BC)

    # use pymors way of constructing data functions (without parameters)
    diffusion_function, source_functions = crack_functions_in_pymor(contrast)

    NWorldFine = np.array([N_fine, N_fine])
    if localized_construction:
        aFine_constructor = lambda patch: diffusion_function_on_patch(diffusion_function, patch)
        aFine = None
    else:
        xt = util.tCoordinates(NWorldFine)
        aFine = diffusion_function(xt)
        aFine = np.array([contrast if a > contrast else a for a in aFine])
        aFine_constructor = None

    # right hand side currently defined on fine mesh.
    NpFine = np.prod(NWorldFine + 1)

    if constant_rhs:
        f_fine = np.ones(NpFine)
    else:
        xp = util.pCoordinates(NWorldFine)
        f_fine = (xp[:, 0] + np.cos(3 * np.pi * xp[:, 0])) * xp[:, 1]**3

    return aFine, f_fine, aFine_constructor

"""
Random field problem
"""

from gridlod.world import Patch

def construct_random_field_on_T(Tpatch, range=None, shape=None):
    T = Tpatch.TInd
    NFine = Tpatch.NPatchFine

    from scripts.pymor_random_field import RandomFieldFunction
    # construct local random field
    random_field = RandomFieldFunction(range=range, shape=shape, seed=T)

    xt = util.tCoordinates(NFine)
    aFine_on_T = random_field(xt)
    return aFine_on_T

def local_random_field(Tpatch, range, local_shape):
    return construct_random_field_on_T(Tpatch, range, local_shape)

def construct_random_field_on_patch(patch, range, local_shape):
    # TODO: document and/or simplify this code
    coarse_indices = patch.coarseIndices
    coarse_indices_mod = coarse_indices % patch.world.NWorldCoarse[0]
    mod_old = -1
    j, l = 0, -1
    blocks = []
    for i, (T, Tmod) in enumerate(zip(coarse_indices, coarse_indices_mod)):
        if Tmod < mod_old:
            j += 1
            l = 0
        else:
            l += 1
        Tpatch = Patch(patch.world, 0, T)
        a = local_random_field(Tpatch, range, local_shape).reshape(Tpatch.NPatchFine)
        if l==0:
            blocks.append(([a]))
        else:
            blocks[j].append(a)
        mod_old = Tmod
    aPatchblock = np.block(blocks).ravel()
    return aPatchblock

def random_diffusion_on_patch(range, shape, patch):
    local_shape_refiner = patch.world.NWorldFine[0] // shape[0]
    local_shape_refiner = 1 if local_shape_refiner == 0 else local_shape_refiner
    return construct_random_field_on_patch(patch, range, patch.world.NCoarseElement // local_shape_refiner)

def random_diffusion_problem(N_fine, range=None, shape=None, localized_construction=False,
                             constant_rhs=True):
    # case 2 from matlab
    NWorldFine = np.array([N_fine, N_fine])
    NpFine = np.prod(NWorldFine + 1)
    # right hand side currently defined on fine mesh.
    if constant_rhs:
        f_fine = np.ones(NpFine)
    else:
        xp = util.pCoordinates(NWorldFine)
        f_fine = (xp[:, 0] + np.cos(3 * np.pi * xp[:, 0])) * xp[:, 1]**3

    if localized_construction:
        # NOTE: this way of randomness does not coincide with the one from the global case below
        aFine_constructor = lambda patch: random_diffusion_on_patch(range, shape, patch)
        aFine = None
    else:
        from scripts.pymor_random_field import RandomFieldFunction
        xt = util.tCoordinates(NWorldFine)
        random_field = RandomFieldFunction(range=range, shape=shape, seed=0)
        aFine = random_field(xt)
        aFine_constructor = None

    return aFine, f_fine, aFine_constructor

"""
Crack with random field problem
"""

def crack_with_random_diffusion_on_patch(range, shape, diffusion_function, patch):
    xt = util.computePatchtCoords(patch)
    crack = diffusion_function(xt)

    local_shape_refiner = patch.world.NWorldFine[0] // shape[0]
    local_shape_refiner = 1 if local_shape_refiner == 0 else local_shape_refiner
    random_field = construct_random_field_on_patch(patch, range, patch.world.NCoarseElement//local_shape_refiner)
    return crack + random_field

def crack_with_ms_problem(N_fine, contrast=1e3, range=None, shape=None,
                          localized_construction=False, constant_rhs=True):
    #### This is inspired by the model problem from https://arxiv.org/pdf/2103.10884.pdf (but with zero dirichlet BC)

    NWorldFine = np.array([N_fine, N_fine])

    # use pymors way of constructing data functions (without parameters)
    diffusion_function, source_functions = crack_functions_in_pymor(contrast)

    if localized_construction:
        # NOTE: this way of randomness does not coincide with the one from the global case below
        aFine_constructor = lambda patch: crack_with_random_diffusion_on_patch(range, shape, diffusion_function, patch)
        aFine = None
    else:
        xt = util.tCoordinates(NWorldFine)
        from scripts.pymor_random_field import RandomFieldFunction
        random_field = RandomFieldFunction(range=range, shape=shape, seed=0)
        diffusion_function += random_field
        aFine = diffusion_function(xt)
        aFine = np.array([contrast if a > contrast else a for a in aFine])
        aFine_constructor = None

    # right hand side currently defined on fine mesh.
    NpFine = np.prod(NWorldFine + 1)

    if constant_rhs:
        f_fine = np.ones(NpFine)
    else:
        xp = util.pCoordinates(NWorldFine)
        f_fine = (xp[:, 0] + np.cos(3 * np.pi * xp[:, 0])) * xp[:, 1]**3

    return aFine, f_fine, aFine_constructor