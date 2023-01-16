# Piquasso Boost

The Piquasso Boost library intends to improve the performance and the scaleability of the computationally most demanding components of the Piquasso bosonic quantum simulation package. 
The Piquasso Boost library is written in C/C++ providing a Python interface via [C++ extensions](https://docs.python.org/3/library/ctypes.html).
The library is equipped with the Threading Building Block ([TBB](https://github.com/oneapi-src/oneTBB)) library providing an efficient task oriented parallel programming model to achieve an optimal workload balance among the accessible execution units of the underlying hardware avoiding any over-subscription of the resources. 
Thus, the parallelized components of the Piquasso Boost library can be freely combined with each other without the cost of performance drop-down.

The Piquasso Boost library utilizes recently developed algorithms to ensure the most favorable scaling of the number of the floating point operations (FLOPS) with the problem size.
In order to reduce the computational time to the minimum we designed the structure of the code to also keep the number of memory operations (MEMOPS) as low as possible by the reuse of the data already loaded into the cache-line hierarchy of the CPU units whenever it is possible.
The register level parallelism via portable SIMD instruction are provided by the implementation of low level BLAS kernels in calculations involving double precision floating point representation.

When it comes to large scaled problems (for example boson sampling simulations involving 20-30 or more photons) the necessary computational precision is ensured by a mixture of
double and extended precision floating point operations while keeping the running time as low as possible.
The numerical stability of the library obtained by mixing different precision floating point representations is governed by heuristically determined internal parameters.
Also, the interplay of MPI and TBB parallel libraries implemented in the Piquasso Boost library (the compilation of the library with MPI support is optional) provides a high scaleability in HPC environments allowing to spawn heavy computational tasks across several cluster nodes.
 
The present package is supplied with Python building script and CMake tools to ease its deployment.
The Piquasso Boost library package can be built with both Intel and GNU compilers, and can be link against various CBLAS libraries installed on the system.
(So far OpenBLAS and the Intel MKL packages were tested.)
In the following we briefly summarize the steps to build, install and use the Piquasso Boost library. 


The project was supported by the Ministry of Innovation and Technology and the National Research, Development and Innovation
Office within the Quantum Information National Laboratory of Hungary.


### Contact Us

Have a question about the Piquasso Boost library? Don't hesitate to contact us by creating a new issue or directly at the following e-mails:

* Zoltán Zimborás (researcher): zimboras.zoltan@wigner.hu
* Zoltán Kolarovszki (developer): kolarovszki@inf.elte.hu
* Peter Rakyta (developer): peter.rakyta@ttk.elte.hu



### Dependencies

The dependencies necessary to compile and build the Piquasso Boost library from source are the followings:

* [CMake](https://cmake.org/) (>=3.10.2)
* C++/C [GNU](https://gcc.gnu.org/) (>=v4.8.1) or [Intel](https://software.intel.com/content/www/us/en/develop/tools/compilers/c-compilers.html) (>=14.0.1) compiler
* [TBB](https://github.com/oneapi-src/oneTBB) library
* [OpenBlas](https://www.openblas.net/) or [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html)
* [LAPACK](http://www.netlib.org/lapack/) 
* [LAPACKE](https://www.netlib.org/lapack/lapacke.html)
* [MPI](https://www.open-mpi.org/) library (optional)
* [Doxygen](https://www.doxygen.nl/index.html) (optional)

The Python interface of the Piquasso Boost library was developed and tested with Python 3.6 and 3.7.
The Python interface of the Piquasso Boost library needs the following packages to be installed on the system:

* [scikit-build](https://pypi.org/project/scikit-build/) (>=0.11.1)
* [Numpy](https://numpy.org/install/) (>=1.19.4)
* [scipy](https://www.scipy.org/install.html) (>=1.5.2)
* quantum-blackbird (==0.2.3)
* theboss (>=2.0.3)
* tbb-devel
* mpi4py
* pytest
* piquasso

**Note**: In some distributions, OpenBLAS might not come with CBLAS, it might
be needed to install CBLAS manually.

### Download the source of the Piquasso Boost library

The developer version of the Piquasso Boost library can be cloned from [GitHub repository](https://github.com/Budapest-Quantum-Computing-Group/piquassoboost).
After the Piquasso Boost repository is extracted into the directory **path/to/piquasso_boost/library** (which would be the path to the source code of the Piquasso Boost library), one can proceed to the compilation steps described in the next section.


### Initialize Piquasso submodule

In case the Piquasso Boost library was cloned from GitHub repositories, the first step is to activate the piquasso submodule by git commands (The piquasso submodule provides the high level Python API of the Piquasso project.):

$ git submodule init

$ git submodule update

The commands above initialize and pull down the piquasso submodule from GitHub sources. 

### Setting up environment variables

The Piquasso Boost library is equipped with a Python build script and CMake tools to ease the compilation and the deployment of the package. 
These scripts automatically finds all library dependencies needed to compile Piquasso Boost. 
The Piquasso Boost library is parallelized via Threading Building Block (TBB) libraries. 
The most straightforward way to get TBB development package installed on the system is to install the python package **tbb-devel** containing the most recent version of the TBB library (including the header files).
(The **tbb-devel** package can be installed in any python virtual environment, thus it is not needed to have administration privileges to have it.)  
If the TBB library is already present on the system (for example it was installed via the apt utility (sudo apt install libtbb-dev) or it was downloaded and built from source from 
[https://github.com/oneapi-src/oneTBB](https://github.com/oneapi-src/oneTBB)) and the user wants to use this version of the TBB library, it is possible by (optionally) setting the

$ export TBB_LIB_DIR=path/to/TBB/lib(64)

$ export TBB_INC_DIR=path/to/TBB/include

environment variables. The building script will look for TBB libraries and header files on the paths given by these environment variables.

CBLAS and LAPACK libaraies are another dependencies necessary to use the Piquasso Boost library. 
Since it is advised to have numpy linked against such a library (for example anaconda automatically brings numpy linked against Intel MKL or OpenBLAS) the building script will automatically find out the location of this library. 
(To check whether there is any CBLAS libarary behind numpy use commands **import numpy as np** and **np.show_config()** inside a python interpreter and check the given library locations.)
If there is no BLAS behind numpy, one can install system wide OpenBLAS by command

$ sudo apt-get install libopenblas-dev liblapack-dev liblapacke-dev

If one don't have administration privileges it is possible to build OpenBLAS (including LAPACK and LAPACKE interfaces) from source (for details see [OpenBLAS](https://github.com/xianyi/OpenBLAS)) and set the environment variable

$ export BLAS_LIB_DIR=path/to/OpenBLAS/lib(64)

to give a hint for the building scripts where to look for the OpenBLAS library.


The Piquasso Boost library can also deployed with MPI support to run large scaled calculations in HPC environments. 
The python package **mpi4py** provides the necessary dependencies for the MPI support (which also checks for system wide MPI libraries)
In order to enable the MPI support one should define the 

$ export PIQUASSOBOOST_MPI=1

environment variable in prior the build.
The Piquasso Boost library is supported with AVX/AVX2 and AVX512F kernels. 
The underlying architecture is determined automatically by building scripts, however the library provides a control switch to compile against AVX512F kernels when it is possible. 
The AVX152 kernels provide 10-15% speedup at the same CPU clock speed, however, since AVX512 mode usually locks down the CPU clock speed, in overall AVX512F kernels would perform slower than the
AVX/AVX2 kernel, if they are not limited by CPU clock speed lock. To check AVX512 capability during compilation and build the code against AVX512F kernels one need to define the

$ export USE_AVX512=1

environment variable. 
Finally, in order to  build Piquasso Boost library including the C test files define the

$ export PIQUASSOBOOST_CTEST=1

environment variable before compiling the library.


### Developer build


We recommend to install the Piquasso Boost package in the Anaconda environment. In order to install the necessary requirements, follow the steps below:

Creating new python environment: 

$ conda create -n pqboost python=3.10

Activate the new anaconda environment

$ conda activate pqboost

Install dependencies:

$ conda install numpy scipy pip pytest scikit-build tbb-devel tensorflow ninja

$ pip install quantum_blackbird theboss==2.0.3

For running pytest examples one should also install the Strawberry Fields package:

$ pip install strawberryfields

To initialize the correct piquasso package for interfacing with python issue the following commands:

$ git submodule init

$ git submodule update

After the basic environment variables are set and the dependencies are installed, the compilation of the package can be started by the Python command:

$ python3 setup.py build_ext

The command above starts the compilation of the SQUANDER C++ library and builds the necessary C++ Python interface extensions of the SQUANDER package in place.
After a successful build, one can register the SQUANDER package in the Python distribution in developer (i.e. editable) mode by command:

$ python -m pip install -e .


### Binary distribution

After the environment variables are set it is possible to build the Piquasso Boost binaries. 
In order to launch the compilation process from python, **[scikit-build](https://scikit-build.readthedocs.io/en/latest/)** package is necessary.
(It is optional to install the ninja package which speeds up the building process by parallel compilation.)
The binary wheel can be constructed by command

$ python3 setup.py bdist_wheel

in the root directory of the Piquasso Boost library.
(It is also possible to compile Piquasso Boost package without creating binary wheel with the command python **setup.py build_ext**)
The created Piquasso Boost wheel can be installed on the local machine by issuing the command from the directory **path/to/piquasso_boost/library/dist**

$ pip3 install piquassoboost-*.whl

We notice, that the created wheel is not portable, since it contains hard coded link to external libraries (TBB and CBLAS).


### Source distribution

A portable source distribution of the Piquasso Boost library can be created by a command launched from the root directory of the Piquasso Boost package:

$ python3 setup.py sdist

In order to create a source distribution it is not necessary to set the environment variables, since this script only collects the necessary files and pack them into a tar ball located in the directory **path/to/piquasso_boost/library/dist**. 
In order to install the Piquasso Boost package from source tar ball, see the previous section discussing the initialization of the environment variables.
The package can be compiled and installed by the command

$ pip3 install piquassoboost-*.tar.gz

issued from directory **path/to/piquasso_boost/library/dist**
(It is optional to install the ninja package which speeds up the building process by parallel compilation.)


### Test the Piquasso Boost library

After a succesfull intallation of the Piquasso Boost library one can test its functionalities by calling the tests scripts 

$ pytest tests

and

$ pytest -s performance_tests

issued in the root directory of the Piquasso Boost library.


