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

import sys
from scripts.test_all_methods import test_script

use_pool = True

assert len(sys.argv) == 1 or len(sys.argv) == 3
path = str(sys.argv[1]) if len(sys.argv) == 3 else ''
storage_path = str(sys.argv[2]) if len(sys.argv) == 3 else None

# see test_all_methods for a description of the problems
problem = 'random'

constant_rhs = False
contrast = None

plotting = False
localized_construction = False

"""
# What methods do we test ?
"""

methods_to_test = [
    'SL-GFEM',
]

n_fine = 1024

N_coarse_params = [4, 8, 16, 32]
ell_params = [1, 2]
n_params = [40, 70, 100, 130]
p_params = [0, 1, 2]

varying_parameters = [N_coarse_params, ell_params, n_params, p_params]

minimal_printout = True

test_script(path,
            problem,                                        # problem type (see below)
            use_pool,                                       # using pool for parallel script or not (mpi required)
            methods_to_test = methods_to_test,              # methods to test (PUMSLOD / SLOD / PGLOD)
            plotting=plotting,                              # plotting additional data
            localized_construction=localized_construction,  # do not construct aFine globally
            constant_rhs=constant_rhs,                      # constant rhs in global system or sinus
            contrast=contrast,                              # contrast for the crack problem
            N_fine=n_fine,                                  # fixed size of the fine mesh n_h x n_h
            varying_parameters=varying_parameters,          # varying parameters
            minimal_printout=minimal_printout,              # verbose vs. minimal printout
            additional_string='exp_4_',
            storage_path=storage_path
            )
