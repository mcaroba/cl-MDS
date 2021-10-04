#!/bin/bash

cd fortran
python3 -m numpy.f2py --f90flags='-fopenmp' -lgomp -c anchor_pts.f90 -m anchor_selection
