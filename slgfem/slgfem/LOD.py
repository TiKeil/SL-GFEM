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
import scipy.sparse as sparse

def PGLOD(world, ell, aFine, f_fine, pool, minimal_printout=False, aFine_constructor=None):
    from gridlod import util, pglod, fem

    assert np.max(f_fine) == 1 and np.min(f_fine) == 1, 'This code is not written for non-trivial rhs'
    # TODO: f_fine is currently not used and has to be 1!
    f = np.ones(world.NpCoarse)

    if not minimal_printout:
        print('________ THIS IS PG-LOD _____________')
    else:
        print('  Computing PG-LOD...')

    def computeKmsij(T, minimal_printout=True):
        from gridlod import interp, coef, lod
        from gridlod.world import Patch
        if not minimal_printout:
            print('s', end='', flush=True)
        patch = Patch(world, ell, T)
        IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, world.boundaryConditions)
        if aFine is None:
            aPatch = lambda: aFine_constructor(patch)
        else:
            aPatch = lambda: coef.localizeCoefficient(patch, aFine)

        correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
        csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
        return patch, correctorsList, csi.Kmsij

    if not minimal_printout:
        print('   Solving corrector problems ...', end='', flush=True)
    # Use mapper to distribute computations (mapper could be the 'map' built-in or e.g. an ipyparallel map)
    patchT, correctorsListT, KmsijT = zip(*pool.map(computeKmsij, list(range(world.NtCoarse)),
                                                    minimal_printout=minimal_printout))
    print('                         ...done')

    KFull = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT)
    MFull = fem.assemblePatchMatrix(world.NWorldCoarse, world.MLocCoarse)

    free_coarse = util.interiorpIndexMap(world.NWorldCoarse)

    bFull = MFull * f

    KFree = KFull[free_coarse][:, free_coarse]
    bFree = bFull[free_coarse]

    xFree = sparse.linalg.spsolve(KFree, bFree)

    basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)
    basisCorrectors = pglod.assembleBasisCorrectors(world, patchT, correctorsListT)
    modifiedBasis = basis - basisCorrectors
    xFull = np.zeros(world.NpCoarse)
    xFull[free_coarse] = xFree
    uLodCoarse = basis * xFull
    uLodFine = modifiedBasis * xFull

    print(f'   Shape of the coarse system: {KFree.shape} with condition {np.linalg.cond(KFree.todense()):.2f}')
    if not minimal_printout:
        print('__________________________________________________ \n')
    return uLodFine