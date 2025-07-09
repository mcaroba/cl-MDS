# cl-MDS
Cluster-based multidimensional scaling (**cl-MDS**) embedding tool for data visualization

**cl-MDS** is copyright (c) 2018-2022 by Patricia Hernández-León and Miguel A. Caro. It is
distributed under the GNU General Public License version 3. **cl-MDS** is shipped with other
codes it relies on for some of its functionalities, but which are not copyright of the
**cl-MDS** authors. They are shipped for the user's convenience in accordance with their
own licenses/terms of usage. See the LICENSE.md file for detailed information on this
software's license.

-- this documentation file is a work in progress --

## Installation

### Prerrequisites

- Numpy
- Sklearn
- Scipy
- A Fortran compiler (successfully tested with `gfortran`)

Optional python modules (necessary for atomic databases):
- ASE
- Ovito

### Building the libraries

Clone the **cl-MDS** repository *recursively*:

    git clone --recursive http://github.com/mcaroba/cl-MDS.git

Execute the build script:

    cd cl-MDS/
    ./build_libraries.sh

Add the root directory to your Python path:

    echo "export PYTHONPATH=$(pwd):\$PYTHONPATH" >> ~/.bashrc
    source ~/.bashrc

### Basic Example  
Check examples/ folder (we will add more here and there soon).
For atomic databases, check examples/tutorial\_atoms\_file/ folder.


### Advanced features
__Customized sparse set__

__Using an existing clustering__

__Hierarchical clustering__

__Adding data to an existing cl-MDS map__


### Attribution

If you use **cl-MDS**, please provide a link to this GitHub repository and cite:

>**Patricia Hernández-León, Miguel A. Caro**. *Cluster-based multidimensional
>scaling embedding tool for data visualization*. [Phys. Scr. 99 066004
>(2024)](https://iopscience.iop.org/article/10.1088/1402-4896/ad432e).
