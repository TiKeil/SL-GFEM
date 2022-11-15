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

import time, itertools, numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

from gridlod import femsolver
from gridlod.world import World, Patch

from scripts.problems import trivial_problem, crack_problem, random_diffusion_problem, crack_with_ms_problem

from pymor.parallel.mpi import MPIPool
from pymor.parallel.dummy import DummyPool

def test_script(path,
                problem,                                        # problem type (see below)
                use_pool,                                       # using pool for parallel script or not (mpi required)
                methods_to_test = ['SL-GFEM'],                  # methods to test (SL-GFEM / SLOD / PGLOD)
                plotting=True,                                  # plotting additional data
                localized_construction=False,                   # do not construct aFine globally
                constant_rhs=True,                              # constant rhs in global system or sinus
                contrast=None,                                  # contrast for the crack problem
                N_fine=32,                                      # fixed size of the fine mesh n_h x n_h
                varying_parameters=None,                        # varying parameters
                minimal_printout=True,                          # potential verbose printout
                additional_string='',                           # marking experiments
                storage_path=None,                              # for storing the basis functions in the file system
                writing_files=True,                             # you can disable that the script writes files into your
                                                                # file system
                ):
    # either use mpi pool or dummy pool, the usage is the same
    pool = MPIPool() if use_pool else DummyPool()

    print(f"\nPath is {path}") if len(path) > 0 else print(f"\nPath is empty")
    print(f"Storage path is {storage_path}\n") if storage_path is not None else print(f"Storage path is empty\n")
    """
    # Parameters to test
    """
    assert varying_parameters is not None, 'you did not choose any config'

    # varying_parameters = [N_coarse_params, ell_params, n_params, p_params]
    N_coarse_params = varying_parameters[0]             # Coarse elements per dimension         N_coarse x N_coarse
    ell_params = varying_parameters[1]                  # Localization parameter l
    n_params = varying_parameters[2]                    # Number of local functions from SVD of the local problems
    p_params = varying_parameters[3]                    # Degree of polynomials in the SL-GFEM

    """
    # Start preparatory work
    """

    boundary_conditions = np.array([[0, 0], [0, 0]])      # boundary conditions: Dirichlet zero everywhere
    NWorldFine = np.array([N_fine, N_fine])

    # problem definition
    random_field_size = (256, 256)      # corresponds to h = 2**{-8}
    random_field_range = [1, 100]       # iid values on this interval

    print(f'Used model problem: {problem}\n')
    if problem == 'trivial':
        #### A = 1 , f = 1 (if constant_rhs = True)
        aFine, f_fine, aFine_constructor = trivial_problem(N_fine,
                                                           localized_construction=localized_construction,
                                                           constant_rhs=constant_rhs)
    elif problem == 'random':
        #### A = random field on random_field_range , f = 1 (if constant_rhs = True)
        aFine, f_fine, aFine_constructor = random_diffusion_problem(N_fine,
                                                                    range=random_field_range,
                                                                    shape=random_field_size,
                                                                    localized_construction=localized_construction,
                                                                    constant_rhs=constant_rhs)
    elif problem == 'coarse_random':
        #### A = coarse random field on coarse_random_field_size , f = 1 (if constant_rhs = True)
        coarse_random_field_size = (8, 8)
        aFine, f_fine, aFine_constructor = random_diffusion_problem(N_fine,
                                                                    range=random_field_range,
                                                                    shape=coarse_random_field_size,
                                                                    localized_construction=localized_construction,
                                                                    constant_rhs=constant_rhs)
    elif problem == 'crack':
        #### A = 4 channels without random field, f = 1 (if constant_rhs = True)
        assert contrast is not None
        aFine, f_fine, aFine_constructor = crack_problem(N_fine,
                                                         contrast=contrast,
                                                         localized_construction=localized_construction,
                                                         constant_rhs=constant_rhs)
    elif problem == 'crack_with_ms':
        #### A = 4 channels with random field on random_field_range, f = 1 (if constant_rhs = True)
        assert contrast is not None
        aFine, f_fine, aFine_constructor = crack_with_ms_problem(N_fine,
                                                                 contrast=contrast,
                                                                 range=random_field_range,
                                                                 shape=random_field_size,
                                                                 localized_construction=localized_construction,
                                                                 constant_rhs=constant_rhs)
    else:
        # define your own problem !
        assert 0, 'problem description not found'

    if localized_construction:
        assert aFine_constructor is not None

    """
    # Reference solution
    """
    NWorldCoarse = np.array([N_coarse_params[0], N_coarse_params[0]])
    world = World(NWorldCoarse, NWorldFine // NWorldCoarse, boundary_conditions)

    aFine_global = aFine_constructor(Patch(world, np.inf, 0)) if aFine is None else aFine

    from slgfem.visualize import drawCoefficient
    plt.figure()
    drawCoefficient(NWorldFine, aFine_global, colorbar_font_size=16)
    if writing_files:
        plt.savefig(f'{path}data/{problem}/{N_fine}_{N_coarse_params[0]}_coefficient.png', bbox_inches='tight')
    if not constant_rhs:
        plt.figure()
        drawCoefficient(NWorldFine+1, f_fine, colorbar_font_size=14, logNorm=False)
        plt.tight_layout()
        if writing_files:
            plt.savefig(f'{path}data/{problem}/{N_fine}_{N_coarse_params[0]}_rhs.png', )

    if plotting:
        plt.show()
    tic = time.perf_counter()
    print('Computing reference solution with FEM\n')
    u_h, AFine, MFine, H1Fine, free_fine = femsolver.solveFine(world, aFine_global, f_fine,
                                                               None, boundary_conditions, True)

    """
    # Starting experiments
    """

    SL_GFEM_gathering = ['SL-GFEM']
    SLOD_gathering = ['SLOD']
    PGLOD_gathering = ['PGLOD']

    for (N, ell) in itertools.product(N_coarse_params, ell_params):
        for (i, (n, p)) in enumerate(itertools.product(n_params, p_params)):
            if minimal_printout:
                print("________________________________________________________________")
                print(f"       n_h = {N_fine}, n_H = {N}, l = {ell}, n = {n}, p = {p}\n")
            else:
                print(f"Inputs are: \n n_h =      {N_fine}"
                      f"\n n_H =      {N}\n l =        {ell}\n n =        {n}\n p =        {p}\n")

            if not (1 + 2 * ell) < N and 'SLOD' in methods_to_test:
                print(' ... aborting because patches are the entire domain.')
                continue

            """
            # Gridlod parameters
            """

            NWorldCoarse = np.array([N, N])
            NCoarseElement = NWorldFine // NWorldCoarse
            world = World(NWorldCoarse, NCoarseElement, boundary_conditions)

            """
            # SL-GFEM
            """

            if 'SL-GFEM' in methods_to_test:
                local_tic = time.perf_counter()
                from slgfem.SLGFEM import sl_gfem
                use_svd = False  # use_svd = True
                u_h_sl_gfem, funcs_sl_gfem = sl_gfem(world, ell, n, p, aFine, f_fine, free_fine, AFine, MFine,
                                                           pool=pool,
                                                           minimal_printout=minimal_printout, use_svd=use_svd,
                                                           aFine_constructor=aFine_constructor,
                                                           storage_path=storage_path)
                sl_gfem_time = time.perf_counter() - local_tic

            """
            # SLOD
            """

            if 'SLOD' in methods_to_test and i == 0:
                local_tic = time.perf_counter()
                from slgfem.SLOD import slod
                # exact_sampling = True
                exact_sampling = False
                u_h_slod, funcs_slod = slod(world, ell, aFine, f_fine, free_fine, AFine, MFine, pool=pool,
                                            minimal_printout=minimal_printout, exact_sampling=exact_sampling,
                                            aFine_constructor=aFine_constructor,
                                            storage_path=storage_path)
                slod_time = time.perf_counter() - local_tic

            """
            # Standard PG-LOD 
            """

            if 'PGLOD' in methods_to_test and i == 0:
                local_tic = time.perf_counter()
                from slgfem.LOD import PGLOD
                u_h_pglod = PGLOD(world, ell, aFine, f_fine, pool=pool, minimal_printout=minimal_printout,
                                  aFine_constructor=aFine_constructor)
                pglod_time = time.perf_counter() - local_tic

            """
            Timings
            """

            if minimal_printout:
                print()
            if 'SL-GFEM' in methods_to_test:
                print(f'SL-GFEM took  : {sl_gfem_time:.2f} seconds')
            if 'PGLOD' in methods_to_test: # and i == 0:
                print(f'PGLOD took    : {pglod_time:.2f} seconds')
            if 'SLOD' in methods_to_test: # and i == 0:
                print(f'SLOD took     : {slod_time:.2f} seconds')

            """
            # Plotting
            """

            if plotting:
                from slgfem.visualize import d3sol
                d3sol(NWorldFine, u_h, string='FEM')
                if 'SL-GFEM' in methods_to_test:
                    d3sol(NWorldFine, u_h_sl_gfem, string='SL-GFEM')
                    d3sol(NWorldFine, u_h - u_h_sl_gfem, string='Error: FEM - SL-GFEM')
                if 'PGLOD' in methods_to_test and i == 0:
                    d3sol(NWorldFine, u_h_pglod, string='PGLOD')
                    d3sol(NWorldFine, u_h - u_h_pglod, string='Error: FEM - PGLOD')
                if 'SLOD' in methods_to_test and i == 0:
                    d3sol(NWorldFine, u_h_slod, string='SLOD')
                    d3sol(NWorldFine, u_h - u_h_slod, string='Error: FEM - SLOD')
                plt.show()

            """
            # Errors
            """

            print()
            NORM_H1 = H1Fine
            NORM_L2 = MFine
            NORM_ENERGY = AFine
            h1_norm_u_h = np.sqrt(np.dot(u_h, NORM_H1 * u_h))
            e_norm_u_h = np.sqrt(np.dot(u_h, NORM_ENERGY * u_h))
            l2_norm_u_h = np.sqrt(np.dot(u_h, NORM_L2 * u_h))
            if 'SL-GFEM' in methods_to_test:
                e_sl_gfem = u_h - u_h_sl_gfem
                abs_h1_error_sl_gfem = np.sqrt(np.dot(e_sl_gfem, NORM_H1 * e_sl_gfem))
                rel_h1_error_sl_gfem = abs_h1_error_sl_gfem / h1_norm_u_h
                abs_e_error_sl_gfem = np.sqrt(np.dot(e_sl_gfem, NORM_ENERGY * e_sl_gfem))
                rel_e_error_sl_gfem = abs_e_error_sl_gfem / e_norm_u_h
                abs_l2_error_sl_gfem = np.sqrt(np.dot(e_sl_gfem, NORM_L2 * e_sl_gfem))
                rel_l2_error_sl_gfem = abs_l2_error_sl_gfem / l2_norm_u_h
                SL_GFEM_gathering.append((N, ell, n, p, (abs_h1_error_sl_gfem, rel_h1_error_sl_gfem,
                                                           abs_e_error_sl_gfem, rel_e_error_sl_gfem,
                                                           abs_l2_error_sl_gfem, rel_l2_error_sl_gfem),
                                            funcs_sl_gfem))
            if 'PGLOD' in methods_to_test and i == 0:
                e_pglod = u_h - u_h_pglod
                abs_h1_error_pglod = np.sqrt(np.dot(e_pglod, NORM_H1 * e_pglod))
                rel_h1_error_pglod = abs_h1_error_pglod / h1_norm_u_h
                abs_e_error_pglod = np.sqrt(np.dot(e_pglod, NORM_ENERGY * e_pglod))
                rel_e_error_pglod = abs_e_error_pglod / e_norm_u_h
                abs_l2_error_pglod = np.sqrt(np.dot(e_pglod, NORM_L2 * e_pglod))
                rel_l2_error_pglod = abs_l2_error_pglod / l2_norm_u_h
                PGLOD_gathering.append((N, ell, n, p, (abs_h1_error_pglod, rel_h1_error_pglod,
                                                       abs_e_error_pglod, rel_e_error_pglod,
                                                       abs_l2_error_pglod, rel_l2_error_pglod), (world.NpCoarse,)))
            if 'SLOD' in methods_to_test and i == 0:
                e_slod = u_h - u_h_slod
                abs_h1_error_slod = np.sqrt(np.dot(e_slod, NORM_H1 * e_slod))
                rel_h1_error_slod = abs_h1_error_slod / h1_norm_u_h
                abs_e_error_slod = np.sqrt(np.dot(e_slod, NORM_ENERGY * e_slod))
                rel_e_error_slod = abs_e_error_slod / e_norm_u_h
                abs_l2_error_slod = np.sqrt(np.dot(e_slod, NORM_L2 * e_slod))
                rel_l2_error_slod = abs_l2_error_slod / l2_norm_u_h
                SLOD_gathering.append((N, ell, n, p, (abs_h1_error_slod, rel_h1_error_slod,
                                                      abs_e_error_slod, rel_e_error_slod,
                                                      abs_l2_error_slod, rel_l2_error_slod), funcs_slod))
            if 'SL-GFEM' in methods_to_test:
                print(f'rel. h1-error sl-gfem  : {rel_h1_error_sl_gfem:.2e}       abs: {abs_h1_error_sl_gfem:.2e}')
            if 'PGLOD' in methods_to_test and i == 0:
                print(f'rel. h1-error pglod    : {rel_h1_error_pglod:.2e}       abs: {abs_h1_error_pglod:.2e}')
            if 'SLOD' in methods_to_test and i == 0:
                print(f'rel. h1-error slod     : {rel_h1_error_slod:.2e}       abs: {abs_h1_error_slod:.2e}')
            print()
            if 'SL-GFEM' in methods_to_test:
                print(f'rel. en-error sl-gfem  : {rel_e_error_sl_gfem:.2e}       abs: {abs_e_error_sl_gfem:.2e}')
            if 'PGLOD' in methods_to_test and i == 0:
                print(f'rel. en-error pglod    : {rel_e_error_pglod:.2e}       abs: {abs_e_error_pglod:.2e}')
            if 'SLOD' in methods_to_test and i == 0:
                print(f'rel. en-error slod     : {rel_e_error_slod:.2e}       abs: {abs_e_error_slod:.2e}')
            print()
            if 'SL-GFEM' in methods_to_test:
                print(f'rel. l2-error sl-gfem  : {rel_l2_error_sl_gfem:.2e}       abs: {abs_l2_error_sl_gfem:.2e}')
            if 'PGLOD' in methods_to_test and i == 0:
                print(f'rel. l2-error pglod    : {rel_l2_error_pglod:.2e}       abs: {abs_l2_error_pglod:.2e}')
            if 'SLOD' in methods_to_test and i == 0:
                print(f'rel. l2-error slod     : {rel_l2_error_slod:.2e}       abs: {abs_l2_error_slod:.2e}')

    gatherings = [SL_GFEM_gathering, SLOD_gathering, PGLOD_gathering]

    def plot_and_store_error(gatherings, error='abs_h1', type='localization'):
        assert type in ['localization', 'convergence']
        plt.figure()
        actual_gatherings = [g for g in gatherings if len(g)>1]
        for gathering in actual_gatherings:
            name = gathering[0]
            if type == 'localization':
                # localization plot where l is variable and N, n, p are fixed
                for (i, (n_, p_)) in enumerate(itertools.product(n_params, p_params)):
                    y_axis, accepted_ell_params = [], []
                    for (_, l, n, p, (abs_h1, rel_h1, abs_e, rel_e, abs_l2, rel_l2), funcs_) in gathering[1:]:
                        # l vs. err - plot
                        if n == n_ and p == p_:
                            y_axis.append(abs_h1) if error == 'abs_h1' else 0
                            y_axis.append(rel_h1) if error == 'rel_h1' else 0
                            y_axis.append(abs_e) if error == 'abs_e' else 0
                            y_axis.append(rel_e) if error == 'rel_e' else 0
                            y_axis.append(abs_l2) if error == 'abs_l2' else 0
                            y_axis.append(rel_l2) if error == 'rel_l2' else 0
                            funcs = funcs_[0] # number of basis functions
                        if l not in accepted_ell_params: # some ells may not be used
                            accepted_ell_params.append(l)
                    label = f'{name}, $n={n_}$, $p={p_}$, $n_c={funcs}$' if 'SL-GFEM' in name else f'{name}'
                    mfc = 'none' if 'SL-GFEM' in name else 'black'
                    markersize = 2 + i if 'SL-GFEM' in name else 4
                    plt.semilogy(accepted_ell_params, y_axis, 'o-', markersize=markersize, mfc=mfc, label=label)
                    plt.xlabel('$\ell$')
                    if not 'SL-GFEM' in name: # SLOD and PGLOD are not dependent on n or p
                        break
            else:
                # convergence plot where N is variable and l, n, p are fixed
                old_l = 0    # make sure that SLOD does not iterate over n or p
                for (i, (l_, n_, p_)) in enumerate(itertools.product(ell_params, n_params, p_params)):
                    if not 'SL-GFEM' in name:
                        # make sure that SLOD does not iterate over n or p
                        if l_ != old_l:
                            old_l = l_
                        else:
                            continue
                    y_axis, accepted_N_coarse_params = [], []
                    # iterate over the data and search for the corresponding entries
                    for (N, l, n, p, (abs_h1, rel_h1, abs_e, rel_e, abs_l2, rel_l2), funcs_) in gathering[1:]:
                        # l vs. err - plot
                        if n == n_ and p == p_ and l == l_:
                            y_axis.append(abs_h1) if error == 'abs_h1' else 0
                            y_axis.append(rel_h1) if error == 'rel_h1' else 0
                            y_axis.append(abs_e) if error == 'abs_e' else 0
                            y_axis.append(rel_e) if error == 'rel_e' else 0
                            y_axis.append(abs_l2) if error == 'abs_l2' else 0
                            y_axis.append(rel_l2) if error == 'rel_l2' else 0
                            funcs = funcs_[0]  # number of basis functions
                            if N not in accepted_N_coarse_params: # some Ns may not be used for this l
                                accepted_N_coarse_params.append(N)
                    label = f'{name}, n={n_}, p={p_}, l={l_}, $n_c={funcs}$' if 'SL-GFEM' in name else f'{name}, l={l_}'
                    mfc = 'none' if 'SL-GFEM' in name else 'black'
                    markersize = 2 + i if 'SL-GFEM' in name else 4 + l_
                    plt.loglog(accepted_N_coarse_params, y_axis, 'o-', markersize=markersize, mfc=mfc, label=label)
                    plt.xscale('log', base=2)
                    plt.xlabel('$1/H$')
        plt.grid()
        plt.legend()
        if type == 'localization':
            plt.title(f'{error}-error for $n_h$={N_fine} and $n_H$={N_coarse_params[0]}')
            plt.xticks(ell_params)
            if writing_files:
                tikzplotlib.save(
                    f'{path}data/{problem}/{additional_string}{N_fine}_{N_coarse_params[0]}_{type}_{error}.tex')
                plt.savefig(f'{path}data/{problem}/{additional_string}{N_fine}_{N_coarse_params[0]}_{type}_{error}.png')
        else:
            plt.title(f'{error}-error for $n_h$={N_fine}')
            plt.xticks(N_coarse_params)
            ell_string = f'ell_{ell_params[0]}' if len(ell_params) == 1 else 'all_l'
            if writing_files:
                tikzplotlib.save(f'{path}data/{problem}/{additional_string}{N_fine}_{type}_{error}_{ell_string}.tex')
                plt.savefig(f'{path}data/{problem}/{additional_string}{N_fine}_{type}_{error}_{ell_string}.png')
        if not use_pool:
            plt.show()

    if len(N_coarse_params) == 1:
        # plot_and_store_error(gatherings, error='abs_h1', type='localization')
        plot_and_store_error(gatherings, error='rel_h1', type='localization')
        # plot_and_store_error(gatherings, error='abs_e', type='localization')
        plot_and_store_error(gatherings, error='rel_e', type='localization')
        # plot_and_store_error(gatherings, error='abs_l2', type='localization')
        plot_and_store_error(gatherings, error='rel_l2', type='localization')
    else:
        # plot_and_store_error(gatherings, error='abs_h1', type='convergence')
        plot_and_store_error(gatherings, error='rel_h1', type='convergence')
        # plot_and_store_error(gatherings, error='abs_e', type='convergence')
        plot_and_store_error(gatherings, error='rel_e', type='convergence')
        # plot_and_store_error(gatherings, error='abs_l2', type='convergence')
        plot_and_store_error(gatherings, error='rel_l2', type='convergence')

    # store gatherings
    if writing_files:
        np.savez(f'{path}data/{problem}/{additional_string}{N_fine}_gatherings.npz')

    print(f'\nThe script took {time.perf_counter()-tic:.3f} seconds')
    del pool