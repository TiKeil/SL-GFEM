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
from copy import deepcopy

from gridlod.world import Patch, World
from gridlod import coef, util, fem, linalg

def slod(world, ell, aFine, f_fine, free_fine, AFine, MFine, pool=None, minimal_printout=False, exact_sampling=False,
         aFine_constructor=None, storage_path=None):
    if not minimal_printout:
        print('________ THIS IS SLOD _____________')
        print('\n   Computing phi_loc for all T ...', end='', flush=True)
    else:
        print('  Computing SLOD...')

    np.random.seed(0)
    Ainvs_on_fine = pool.map(compute_philoc_on_element_patch, list(range(world.NtCoarse)),
                             world=world, ell=ell, free_fine=free_fine, aFine=aFine,
                             exact_sampling=exact_sampling, minimal_printout=minimal_printout,
                             aFine_constructor=aFine_constructor, storage_path=storage_path)
    print('                         ...done')

    if storage_path is not None:
        Ainvs_on_fine = []
        # for the case that all basis functions are stored in the file system
        for T in range(world.NtCoarse):
            if os.path.exists(f'{storage_path}/bases_{T}.npz'):
                Ainvs_on_fine_ = scipy.sparse.load_npz(f'{storage_path}/bases_{T}.npz')
                Ainvs_on_fine.append(Ainvs_on_fine_)
                os.remove(f'{storage_path}/bases_{T}.npz')

    if not minimal_printout:
        print('\n   Assembling and solving global SLOD system')

    Ainvs = [Ainv for Ainv in Ainvs_on_fine if Ainv is not None]
    G = scipy.sparse.hstack(Ainvs)

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

def compute_philoc_on_element_patch(T, world=None, ell=None, free_fine=None, aFine=None, exact_sampling=False,
                                    minimal_printout=False, aFine_constructor=None, storage_path=None):
    if not minimal_printout:
        print(f's', end='', flush=True)
    patch = Patch(world, ell, T)

    m = ell ** (np.log((2 * ell + 1) ** world.d / len(patch.coarseIndices)) / np.log((2 * ell + 1) / (2 * ell)))
    if np.abs(m - np.round(m)) < 1e-10:
        m = int(np.round(m))

        # free DoFs on patch
        free = util.interiorpIndexMap(patch.NPatchFine)

        # find dofs that dirichlet dofs on the patch but NOT in Omega
        inside_dirichlet = []
        outside_dirichlet = []
        all_dirichlet = []
        for i, node in enumerate(patch.finepIndices):
            if i not in free:
                all_dirichlet.append(i)
                if node in free_fine:
                    inside_dirichlet.append(i)
                else:
                    outside_dirichlet.append(i)

        pre_free_dirichlet = deepcopy(inside_dirichlet)
        # find dofs that are dirichlet dofs in Omega but still belong to the dofs for the harmonic functions.
        xp_s_full = util.pCoordinates(world.NWorldFine)
        xp_s = xp_s_full[patch.finepIndices]
        middle_x = xp_s[len(xp_s)//2]
        assert patch.world.NWorldCoarse[0] == patch.world.NWorldCoarse[1]
        max_dist_in_patch = (2 * ell + 1)/2 * 1/patch.world.NWorldCoarse[0]
        for (i, x) in enumerate(xp_s):
            add = False
            if i not in free:
                if abs(x[0] - middle_x[0]) == max_dist_in_patch:
                    add = True
                if abs(x[1] - middle_x[1]) == max_dist_in_patch:
                    add = True
                if add:
                    pre_free_dirichlet.append(i)

        pre_free_dirichlet = np.unique(np.sort(pre_free_dirichlet))
        free_dirichlet = []
        horizontal_dofs = patch.NPatchFine[0] + 1
        for dof in pre_free_dirichlet:
            if dof == 0:
                # corner bottom-left
                if not (1 in pre_free_dirichlet and horizontal_dofs in pre_free_dirichlet):
                    continue
            if dof == horizontal_dofs - 1:
                # corner bottom-right
                if not (dof-1 in pre_free_dirichlet and dof + horizontal_dofs in pre_free_dirichlet):
                    continue
            if dof == (patch.NpFine - horizontal_dofs):
                # corner top-left
                left = dof + 1 in pre_free_dirichlet
                right = dof - horizontal_dofs in pre_free_dirichlet
                if not (left and right):
                    continue
            if dof == (patch.NpFine - 1):
                # corner top-right
                left = dof - 1 in pre_free_dirichlet
                right = dof - horizontal_dofs in pre_free_dirichlet
                if not (left and right):
                    continue
            free_dirichlet.append(dof)

        cp_nodes = free_dirichlet

        if aFine is None:
            aPatch = aFine_constructor(patch)
        else:
            aPatch = coef.localizeCoefficient(patch, aFine)
        A_h_loc = fem.assemblePatchMatrix(patch.NPatchFine, world.ALocFine, aPatch)
        A_free = A_h_loc[free][:, free]

        XX = []

        if exact_sampling:
            range_for_harmonic_sampling = len(free_dirichlet)
            fs = scipy.sparse.eye(range_for_harmonic_sampling)
        else:
            # approximate space of harmonic functions
            oversampling_factor = 5
            range_for_harmonic_sampling = oversampling_factor * patch.NtCoarse # O(l ^ d)
            fs = np.random.rand(len(cp_nodes), range_for_harmonic_sampling)

        Afs = - A_h_loc[free][:, cp_nodes] * fs
        x_free = scipy.sparse.linalg.spsolve(A_free, Afs)
        x_full = np.zeros((len(patch.finepIndices), range_for_harmonic_sampling))
        x_full[free] = x_free
        x_full[cp_nodes] = fs

        for x_full_ in x_full.T:
            # We need to project first from fine nodes on fine element, then fine elements on coarse elements.
            x_proj = Projection_from_fine_nodes_to_coarse_elements(x_full_, patch)

            XX.append(np.array(x_proj))

        U, S, W = np.linalg.svd(np.array(XX).T, full_matrices=False)

        # m many left singular vectors to the smalles singular values
        XX_ = U[:, U.shape[0]-m:].T

        phi_locs = []
        ## Note: this can be done more efficient similar to the higher-order case in the SL-GFEM
        for m_ in range(m):
            x_on_fine = np.zeros(patch.NtFine)
            # we need coarse elements to fine elements
            # this should be done in a matrix
            for T_ in range(len(patch.coarseIndices)):
                new_world = World(patch.NPatchCoarse, world.NCoarseElement)
                T_patch_in_patch = Patch(new_world, 0, T_)
                x_on_fine[T_patch_in_patch.fineIndices] = XX_[m_][T_]

            rhs_xx = np.sum(fem.assemblePatchMatrix(patch.NPatchFine, world.MLocFine, x_on_fine), axis=1)[free]
            phi_loc = scipy.sparse.linalg.spsolve(A_free, rhs_xx)
            phi_loc_full = np.zeros(patch.NpFine)
            phi_loc_full[free] = phi_loc
            phi_locs.append(phi_loc_full)

        phi_locs_on_full = []
        for phi_loc in phi_locs:
            phi_loc_on_full = np.zeros(world.NpFine)
            phi_loc_on_full[patch.finepIndices] = phi_loc
            phi_locs_on_full.append(phi_loc_on_full[free_fine])

        return_value = np.array(phi_locs_on_full).T
        sparse_return_value = scipy.sparse.csr_matrix(return_value)
        if storage_path is not None:
            scipy.sparse.save_npz(f'{storage_path}/bases_{T}.npz', sparse_return_value)
            return True
        else:
            return sparse_return_value
    return None

def Projection_from_fine_nodes_to_coarse_elements(x_full, patch):
    # NOTE: This part of the code is very slow and can not compete with the SL-GFEM!
    # However, this can be done much better by using a respective Matrix vector multiplication.
    fine_ts = patch.fineIndices.reshape(patch.NPatchFine, order='F').T
    x_reshaped = x_full.reshape(patch.NPatchFine + 1, order='F').T
    x_proj_to_ts = []
    for x_coord in range(fine_ts.shape[0]):
        for y_coord in range(fine_ts.shape[1]):
            x_proj_to_ts.append(np.sum([x_reshaped[x_coord, y_coord],
                                        x_reshaped[x_coord + 1, y_coord],
                                        x_reshaped[x_coord, y_coord + 1],
                                        x_reshaped[x_coord + 1, y_coord + 1]]) / 4)
    x_proj_to_fine_ts = np.array(x_proj_to_ts).reshape(patch.NPatchFine).flatten()
    x_proj = []
    for T__ in range(len(patch.coarseIndices)):
        new_world = World(patch.NPatchCoarse, patch.world.NCoarseElement)
        T_patch_in_patch = Patch(new_world, 0, T__)
        xs_in_T = x_proj_to_fine_ts[T_patch_in_patch.fineIndices]
        x_proj.append(np.sum(xs_in_T) / len(xs_in_T))

    return x_proj