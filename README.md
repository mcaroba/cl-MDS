# cl-MDS
Cluster-based multidimensional scaling (**cl-MDS**) embedding tool for data visualization.

**cl-MDS** is copyright (c) 2018-2022 by Patricia Hernández-León and Miguel A. Caro. It is
distributed under the GNU General Public License version 3. **cl-MDS** is shipped with other
codes it relies on for some of its functionalities, but which are not copyright of the
**cl-MDS** authors. They are shipped for the user's convenience in accordance with their
own licenses/terms of usage. See the LICENSE.md file for detailed information on this
software's license.

-- this file is a work in progress --


## Installation

### Dependencies

- NumPy
- scikit-learn
- SciPy
- A Fortran compiler (successfully tested with `gfortran`)

Optional packages:
- For atomic databases 
    - ASE
    - [quippy](https://github.com/libAtoms/QUIP?tab=readme-ov-file#quip---quantum-mechanics-and-interatomic-potentials): `quippy-ase`

- For plotting features
    - Matplotlib
    - Ovito (atomic databases)

- For examples (apart from the above)
    - UMAP: `umap-learn`

### Conda/pip installation

-- _packaging branch_ --

### Manual installation

Clone the **cl-MDS** repository *recursively*:

    git clone --recursive http://github.com/mcaroba/cl-MDS.git

Execute the build script:

    cd cl-MDS/
    ./build_libraries.sh

Add the root directory to your Python path:

    echo "export PYTHONPATH=$(pwd):\$PYTHONPATH" >> ~/.bashrc
    source ~/.bashrc


## Documentation

Documentation is available in -- _documentation branch_ --

Main features:
- 2-dimensional embedding of high-dimensional data
- Plotting support 
- Atomic-related functionalities: SOAP-based distances, visualization of medoids, ...

Extra features:
- Hierarchical clustering to enhance the embedding
- Customizable sparsification via k-medoids, CUR, random
- Sparsification support for big datasets
- Adding new data to an existing cl-MDS embedding (embedding estimation)

### Tutorial and examples  
Check examples/ folder (we will add more here and there soon).
For atomic databases, check examples/tutorial\_atoms\_file/ folder.

Tutorial Jupyter Notebook in process -- _documentation branch_ --


## Citation

If you use **cl-MDS**, please provide a link to this GitHub repository and cite:

>**Patricia Hernández-León, Miguel A. Caro**. *Cluster-based multidimensional
>scaling embedding tool for data visualization*. [Phys. Scr. 99 066004
>(2024)](https://iopscience.iop.org/article/10.1088/1402-4896/ad432e).

## Ongoing development:

- Packaging:
    - [ ] requirements for conda/mamba env (+ jupyter, + fortran)
    - [ ] update README.md
    - [ ] create __init__.py
    - [ ] create pyproject.toml
    - [ ] test local pip install
-  Documentation:
    - [ ] update current tutorials in Jupyter notebook
    - [ ] write documentation file
- Functionalities:
    - upgrade of existing features:
        - [ ] carving medoids out (i.e., representative atomic env.)
        - [ ] saving medoids
        - [ ] plotting clMDS data (gnuplot)
    - [ ] add plots with matplotlib
    - [ ] alternative hierarchy implementation
- Tests: 
    - [ ]
