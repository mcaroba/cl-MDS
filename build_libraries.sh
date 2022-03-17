#!/bin/bash

rm src/*.so

cd src
python3 -m numpy.f2py --f90flags='-fopenmp' -lgomp -c anchor_pts.f90 -m anchor_selection

cd ../fast-kmedoids
./build_libraries.sh
cp src/*.so ../src/.
