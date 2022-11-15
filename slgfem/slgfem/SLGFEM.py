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
import scipy
import os

from gridlod import coef, util, fem, linalg
from gridlod.world import World, Patch, NodePatch

"""
# SL-GFEM
"""

def sl_gfem(world, ell, locfuncs, p, aFine, f_fine, free_fine, AFine, MFine, pool, minimal_printout=False,
              use_svd=False, aFine_constructor=None, storage_path=None):
    if not minimal_printout:
        print('________ THIS IS SL-GFEM _____________')
        print('\n   Computing V_z on omega_z for all z ...', end='', flush=True)
    else:
        print('  Computing SL-GFEM...')

    if aFine is None:
        assert aFine_constructor is not None

    Ainvs_on_fine_zs = pool.map(compute_AinvcharKs_node_patch_on_node_patch, list(range(world.NpCoarse)),
                                world=world, ell=ell, locfuncs=locfuncs, aFine=aFine,
                                use_svd=use_svd, minimal_printout=minimal_printout,
                                aFine_constructor=aFine_constructor, p=p, storage_path=storage_path)
    if storage_path is not None:
        Ainvs_on_fine_zs = []
        # for the case that all basis functions are stored in the file system
        for z in range(world.NpCoarse):
            Ainvs_on_fine_zs_ = scipy.sparse.load_npz(f'{storage_path}/bases_{z}.npz')
            Ainvs_on_fine_zs.append(Ainvs_on_fine_zs_)
            os.remove(f'{storage_path}/bases_{z}.npz')

    print('                         ...done')
    if not minimal_printout:
        print('\n   Assembling and solving global PUMSLOD system')

    G = scipy.sparse.hstack(Ainvs_on_fine_zs)

    bFine = MFine * f_fine
    bFine_free = bFine[free_fine]
    AFine_free = AFine[free_fine][:, free_fine]

    b_slod = G.T.dot(bFine_free)
    A_slod = (G.T * AFine_free).dot(G)

    if not minimal_printout:
        print(f'   Shape of the coarse system: {A_slod.shape}')
        # print(f'      with condition {np.linalg.cond(A_slod.todense()):.2f}')
        print(f'   Shape of the fine system:   {AFine_free.shape}')
        # print(f'     with condition {np.linalg.cond(AFine_free.todense()):.2f}')
        print('__________________________________________________ \n')
    else:
        print(f'   Spapes of the coarse system vs. fine system: {A_slod.shape}  {AFine_free.shape}\n')
        # print(f'            with conditions {np.linalg.cond(A_slod.todense()):.2f} '
        #       f'and {np.linalg.cond(AFine_free.todense()):.2f}')

    sol = linalg.linSolve(A_slod, b_slod)
    u_h_slod_free = G.dot(sol)

    u_h_slod = np.zeros(world.NpFine)
    u_h_slod[free_fine] = u_h_slod_free

    return u_h_slod, A_slod.shape

def compute_AinvcharKs_node_patch_on_node_patch(z, world=None, ell=None, locfuncs=None, aFine=None,
                                                use_svd=False, minimal_printout=False,
                                                aFine_constructor=None, p=0, storage_path=None):
    if not minimal_printout:
        print('s', end='', flush=True)

    node_patch = NodePatch(world, 1, z)                         # w_z = N_1(z)
    patch_of_node_patch = NodePatch(world, ell + 1, z)          # N_l(w_z) = N_{l+1}(z)

    # NOTE: this assembles on the fine level
    boundaryMap = world.boundaryConditions==0
    fixedFine = util.boundarypIndexMap(world.NWorldFine, boundaryMap=boundaryMap)
    freeFine = np.setdiff1d(np.arange(world.NpFine), fixedFine)

    # dofs that are free in w_z
    # TODO: can we use setdiff1d here?
    free_fine_on_node_patch = freeFine[np.isin(freeFine, node_patch.finepIndices)]
    # slow version
    # free_fine_on_node_patch = [i for i in node_patch.finepIndices if i in free_fine]

    # true free DoFs on w_z
    local_free_on_node_patch = np.where(np.isin(node_patch.finepIndices, free_fine_on_node_patch))[0]
    # slow version
    # local_free_on_node_patch = [i for (i, dof) in enumerate(node_patch.finepIndices)
    #                             if dof in free_fine_on_node_patch]

    # dofs that are free in w_z with respect to N_l(w_z)
    free_fine_on_node_patch_in_node_l_patch = np.where(
        np.isin(patch_of_node_patch.finepIndices, free_fine_on_node_patch))[0]
    # slow version
    # free_fine_on_node_patch_in_node_l_patch = [i for (i, dof) in enumerate(patch_of_node_patch.finepIndices)
    #                                            if (dof in node_patch.finepIndices and
    #                                                dof in free_fine_on_node_patch)]

    # __________________________________________________________________________________________

    # extract PoU
    # NOTE: this assembles on the fine level
    prolongation_nodes = fem.assembleProlongationMatrix(
        world.NWorldCoarse, world.NCoarseElement)[free_fine_on_node_patch, z].toarray()

    # one loop for all coarse T
    if aFine is None:
        aPatch = aFine_constructor(patch_of_node_patch)
    else:
        aPatch = coef.localizeCoefficient(patch_of_node_patch, aFine)
    free = util.interiorpIndexMap(patch_of_node_patch.NPatchFine)

    S_h_loc = fem.assemblePatchMatrix(patch_of_node_patch.NPatchFine, world.ALocFine, aPatch)
    S_free = S_h_loc[free][:, free]
    ps = shifted_legendre_polynomials(p)
    xt = util.pCoordinates(world.NCoarseElement)

    mat = np.empty((len(xt), len(ps) ** 2))
    k = 0
    for i in range(len(ps)):
        for j in range(len(ps)):
            product_p = lambda x, y: ps[i](x) * ps[j](y)
            mat[:, k] = product_p(xt[:, 0], xt[:, 1])
            k += 1

    CGtoDG_matrix = CGtoDG(world.NCoarseElement)
    pbasisDG = CGtoDG_matrix.dot(mat)

    I = np.array([])
    new_world = World(patch_of_node_patch.NPatchCoarse, world.NCoarseElement)
    for T_ in range(patch_of_node_patch.NtCoarse):
        T_patch_in_node_patch = Patch(new_world, 0, T_)
        elements = T_patch_in_node_patch.fineIndices
        blockI = 4 * np.repeat(elements, 4) + np.tile(np.arange(4), len(elements))
        I = np.append(I, np.tile(blockI, pbasisDG.shape[1]))

    J = np.repeat(np.arange(patch_of_node_patch.NtCoarse * pbasisDG.shape[1]), pbasisDG.shape[0])
    V = np.tile(pbasisDG.T.flatten(), patch_of_node_patch.NtCoarse)

    rhsmatrixdg1hp = scipy.sparse.csr_matrix((V, (I, J)),
                                             shape=(patch_of_node_patch.NtFine * 4,
                                                    patch_of_node_patch.NtCoarse * pbasisDG.shape[1]))

    CGtoDG_fine = CGtoDG(patch_of_node_patch.NPatchFine)
    DG_mass_matrix_fine = DGmassMatrix(patch_of_node_patch.NPatchFine)
    rhs = CGtoDG_fine.T.dot(DG_mass_matrix_fine.dot(rhsmatrixdg1hp))
    rhs_free = rhs[free, :]

    res = scipy.sparse.linalg.spsolve(S_free, rhs_free)
    res_full = np.zeros((np.prod(patch_of_node_patch.NPatchFine + 1), rhs.shape[1]))
    res_full[free] = res.toarray()
    Ainvs_node_patch = res_full[free_fine_on_node_patch_in_node_l_patch, :]

    if use_svd:
        ### SVD
        U, S, W = np.linalg.svd(Ainvs_node_patch, full_matrices=False)
        assert isinstance(locfuncs, int)
        return_Ainv = U[:, :locfuncs]

        # multiply with PoU
        return_Ainv = np.multiply(return_Ainv, prolongation_nodes)
        return_Ainv_on_full_fine = np.zeros((world.NpFine, return_Ainv.shape[1]))
        return_Ainv_on_full_fine[free_fine_on_node_patch] = return_Ainv
    else:
        ### EVP as in Paper !
        Ainvs_node_patch_PU = np.multiply(Ainvs_node_patch.T, prolongation_nodes[:, 0])
        Ainvs_node_l_patch = res.toarray()

        if aFine is None:
            aPatch = aFine_constructor(node_patch)
        else:
            aPatch = coef.localizeCoefficient(node_patch, aFine)

        S_h_loc = fem.assemblePatchMatrix(node_patch.NPatchFine, world.ALocFine, aPatch)
        S_node_patch = S_h_loc[local_free_on_node_patch][:, local_free_on_node_patch]

        eig_rhs = np.dot(Ainvs_node_l_patch.T * S_free, Ainvs_node_l_patch)
        eig_lhs = np.dot(Ainvs_node_patch_PU * S_node_patch, Ainvs_node_patch_PU.T)

        lambdas, u = scipy.linalg.eigh(eig_lhs, eig_rhs)
        u_reversed = u.T[::-1].T
        lambdas_reversed = lambdas[::-1]
        if not isinstance(locfuncs, int):
            if locfuncs == 'max':
                locfuncs = np.inf
            else:
                assert 0, 'this string can not be understood for locfuncs'

        us = u_reversed[:, :min(locfuncs, len(u))]
        return_Ainv = np.dot(Ainvs_node_patch, us)
        return_Ainv = np.multiply(return_Ainv, prolongation_nodes)
        # remove almost zero vectors
        return_Ainv = return_Ainv[:, np.logical_not(np.all(np.isclose(return_Ainv, 0, atol=1e-8), axis=0))]
        return_Ainv_on_full_fine = np.zeros((world.NpFine, return_Ainv.shape[1]))
        # additional norming needed for stability.
        norm = np.sqrt(np.abs(np.dot(return_Ainv.T * S_node_patch, return_Ainv)))
        diag_norm = np.diag(norm)
        diag_norm = [1./n for n in diag_norm] # sometimes value are too zero
        return_Ainv_on_full_fine[free_fine_on_node_patch] = diag_norm * return_Ainv

    return_value = return_Ainv_on_full_fine[freeFine]
    sparse_return_value = scipy.sparse.csr_matrix(return_value)
    # storing in the file system for less communication between the ranks
    if storage_path is not None:
        scipy.sparse.save_npz(f'{storage_path}/bases_{z}.npz', sparse_return_value)
        return True
    else:
        return sparse_return_value

def shifted_legendre_polynomials(degree=5):
    # shifted and normalized Legendre polynomials on [0,1]
    p0 = lambda x: 0*x + 1.
    p1 = lambda x: np.sqrt(3) * (2*x - 1)
    p2 = lambda x: np.sqrt(5) * (6*x**2 - 6*x + 1)
    p3 = lambda x: np.sqrt(7) * (20*x**3 - 30*x**2 + 12*x - 1)
    p4 = lambda x: np.sqrt(9) * (70*x**4 - 140*x**3 + 90*x**2 - 20*x + 1)
    p5 = lambda x: np.sqrt(11) * (252*x**5 - 630*x**4 + 560*x**3 - 210*x**2 + 30*x - 1)

    ps = [p0, p1, p2, p3, p4, p5]
    return ps[:degree+1]

def CGtoDG(NPatch):
    from gridlod.fem import localToPatchSparsityPattern
    NtPatch = np.prod(NPatch)
    NpPatch = np.prod(NPatch+1)
    rows, cols = localToPatchSparsityPattern(NPatch)
    row = rows[0:-1:4]
    P = scipy.sparse.csr_matrix((np.ones(NtPatch * 4), (np.arange(NtPatch * 4), row)), shape=(NtPatch * 4, NpPatch))
    return P

def DGmassMatrix(NPatch):
    from gridlod.fem import localMassMatrix
    Mloc = localMassMatrix(NPatch)
    Np = np.prod(NPatch)
    DGmassMatrix = scipy.sparse.kron(scipy.sparse.eye(Np), Mloc)
    return DGmassMatrix