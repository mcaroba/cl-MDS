cd src
f2py3 --f90flags='-fopenmp' -lgomp -c anchor_pts.f90 -m anchor_selection
