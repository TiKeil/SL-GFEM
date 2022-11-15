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

use_pool = False
# use_pool = True     # <---- change this to True to have parallel code. You need to install OPEN_MPI and run mpirun

if len(sys.argv) < 2:
    problem = 'trivial'
    path = ''
    # storage_path = 'mpi_storage'
    storage_path = None
else:
    problem = str(sys.argv[1])
    path = str(sys.argv[2]) if len(sys.argv) > 2 else ''
    storage_path = str(sys.argv[3])

#### live plotting of more data
# plotting = True
plotting = False

#### either construct the coefficient A beforehand or for every patch (minimizes communication in the parallel case)
localized_construction = False
# localized_construction = True

# experiment for higher order rhs
# constant_rhs = False
constant_rhs = True

# contrast for crack problem
contrast = 1e8
# contrast = None

"""
# What methods do we test ?
"""

methods_to_test = [
     'SL-GFEM',
    'PGLOD',
      'SLOD'
]

n_fine = 32

n_H = 4
ell = 1
n = 10
p = 0

# H convergence plot, fixed l
N_coarse_params = [4]
ell_params = [1]
n_params = [2,3,'max']
p_params = [0]

# localization plot in l, fixed H
# N_coarse_params = [4]
# ell_params = [1, 2, 3]
# n_params = [100]
# p_params = [0, 1, 2]


varying_parameters = [N_coarse_params, ell_params, n_params, p_params]
# varying_parameters = [[n_H], [ell], [n], [p]]

# minimal_printout = True
minimal_printout = False

writing_files = False
# writing_files = True

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
            additional_string='minimal',
            storage_path=storage_path,
            writing_files=writing_files
            )