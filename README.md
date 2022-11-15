[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7324123.svg)](https://doi.org/10.5281/zenodo.7324123)

```
# ~~~
# This file is part of the paper:
#   
#           " A super-localized generalized finite element method "
#
#   https://github.com/TiKeil/SL-GFEM.git
#
# Copyright 2022 all developers. All rights reserved.
# License: Licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Authors:
#   Philip Freese, Moritz Hauck, Tim Keil, Daniel Peterseim
# ~~~
```
# SL-GFEM

In this repository, we provide the code for the numerical experiments in Section 7 of the paper
"A super-localized generalized finite element method" by Philip Freese, Moritz Hauck, Tim Keil, and Daniel Peterseim.
The preprint is available [here](https://arxiv.org/abs/tba).

If you want to have a closer look at the implementation or generate the results by yourself, we provide simple setup
instructions for configuring your own `python` environment.
We note that our setup instructions are written for Ubuntu Linux only and we do not provide setup instructions
for MacOS and Windows. Our setup instructions have successfully been tested on a fresh Ubuntu 20.04.2.0 LTS system.
The actual experiments have been computed on the
[PALMA II HPC cluster](<https://www.uni-muenster.de/IT/en/services/unterstuetzungsleistung/hpc/index.shtml>).
For the concrete configurations we refer to the scripts in `submit_to_cluster`.

# Organization of the repository

Apart from the package requirements in the `requirements.txt` file, we used an external software package:

- [`gridlod`](https://github.com/fredrikhellman/gridlod) is a discretization toolkit for the
Localized Orthogonal Decompostion (LOD) method. 

We added `gridlod` as external software as editable submodules with a fixed commit hash.
For the SL-GFEM, we have introduced a `python` module `slgfem`.
The rest of the code is contained in `scripts`, where you find the definition of the model problems
(in `problems.py`) and the main scripts for the numerical experiments.
The `main_*` files are used to run the respective experiments, numbered from 1-4
(corresponding to section 7.1-7.4 in the paper).

# Setup

On a standard Ubuntu system (with Python and C compilers installed) it will most likely be enough
to just run our setup script. For that, please clone the repository

```
git clone https://github.com/TiKeil/SL-GFEM.git
```

and execute the provided setup script via 

```
cd SL-GFEM
./setup.sh
```

If this does not work for you, and you don't know how to help yourself,
please follow the extended setup instructions below.

## Installation on a fresh system

We also provide setup instructions for a fresh Ubuntu system (20.04.2.0 LTS).
The following steps need to be taken:

```
sudo apt update
sudo apt upgrade
sudo apt install git
sudo apt install build-essential
sudo apt install python3-dev
sudo apt install python3-venv
sudo apt install libopenmpi-dev
sudo apt install libsuitesparse-dev
```

Now you are ready to clone the repository and run the setup script:

```
git clone https://github.com/TiKeil/SL-GFEM.git
cd SL-GFEM
./setup.sh
```

# Running the experiments

You can make sure your that setup is complete by running the minimal test script
after activating the virtual environment

```
source venv/bin/activate
cd scripts/test_scripts
mpirun -n [nprocs] python main_minimal_test.py
```

If this works fine (with error output in the end), your setup is working well.
Note that for many experiments, an HPC cluster is recommend.
In particular, starting the scripts with only a few parallel cores (or even without `mpirun`)
on your local computer may take hours.

Please have a look at the description of the arguments of `scripts/test_all_methods.py`
to try different configurations of the given problem classes. Note that it is also possible to solve
your own multi-scale problems with our code since the problem definitions that are used in
`scripts/problems.py` are very general. 

# Additional information on the diffusion coefficients in Section 7.3

In the paper, we have not provided an expression for the diffusion coefficient that has been used for
Section 7.3. For the exact expressions we refer to `scripts/problems.py` and the problem `crack_with_ms`.

# Questions

If there are any questions of any kind, please contact us via <tim.keil@wwu.de>.
