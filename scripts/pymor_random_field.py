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
from pymor.analyticalproblems.functions import Function

class RandomFieldFunction(Function):
    #########
    ### Copied from https://github.com/TiKeil/Trust-region-TSRBLOD-code/blob/main/pdeopt/pdeopt/problems.py
    #########

    """Define a 2D |Function| via a random distribution.
    Parameters
    ----------
    bounding_box
        Lower left and upper right coordinates of the domain of the function.
    seed
        random seed for the distribution
    range
        defines the range of the random field
    """

    dim_domain = 2
    shape_range = ()

    def __init__(self, bounding_box=None, range=None, shape=(100,100), seed=0):
        bounding_box = bounding_box or [[0., 0.], [1., 1.]]
        range = range or [0., 1.]
        assert isinstance(range, list) and len(range) == 2
        a_val, b_val = range[0], range[1]
        np.random.seed(seed)
        self.diffusion_field = np.random.uniform(a_val, b_val, np.prod(shape)).reshape(shape).T[:, ::-1]
        self.__auto_init(locals())
        self.lower_left = np.array(bounding_box[0])
        self.size = np.array(bounding_box[1] - self.lower_left)

    def evaluate(self, x, mu=None):
        indices = np.maximum(np.floor((x - self.lower_left) * np.array(self.diffusion_field.shape) /
                                      self.size).astype(int), 0)
        F = self.diffusion_field[np.minimum(indices[..., 0], self.diffusion_field.shape[0] - 1),
                         np.minimum(indices[..., 1], self.diffusion_field.shape[1] - 1)]
        return F