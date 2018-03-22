Pychan3d

Pychan3d is a python scripting library for modeling groundwater flow and solute transport in arbitrary unstructured
networks of 1-dimensional channels. The channel network approach has been developed by Moreno and Neretnieks (1993) and
implemented into the CHAN3D code by Gylling et al. (1999), after evidence showed strong channeling of the flow and
solute transport in deep crystalline rock systems. Pychan3d offers routines to rapidly generate networks of channels
based on fully populated rectangular lattices (original approach of Moreno and Neretnieks, and Gylling et al., see
examples 1, part1 and example 2), percolation lattices (see example 1, part2) and fully unstructured networks of
channels (example 3).


Getting Started

The source code can be downloaded from the github page https://github.com/bende937/pychan3d.


Requirements and installation (see also the installation page of the online wiki section)

To use the pychan3d routines you will need to have previously installed:
- Python 3.5.x
- Numpy 1.11.2
- Scipy 0.18.1

Some other dependencies are highly recommended for full functionality:

- vtk 6.3.0 (for visualization with the standalone software Paraview)
- Pyamg 3.2.1 (for Algebraic MultiGrid preconditioners)

Other useful packages to use in combination with pychan3d:

- matplotlib 1.5.3 (for 2d and simple 3d visualization)

All the previously mentioned packages are available within the Anaconda distribution, which is the recommended way
to install all packages. Once all these packages are installed, downloading, unzipping the pychan3d archive and adding
the directory containing the pychan3d folder to your PYTHONPATH allows you to call the routines from your scripts (see
examples).

For visualizing the results, we recommend to install the software Paraview which can open the .vtp, .pvd and other vtk
files that are created when calling the method .export2vtk() on a channel network object. Paraview is a very powerful
software that allows a wide range of operations to customize the visualization of 3D data.


Other packages can be useful but require more installation efforts:

- PETSc 3.6.4 (for high performance serial and parallel linear solvers and preconditioners, required some MPI
implementation to be installed in order to run in parallel)
- mpi4py 1.3.1 (MPI wrapper needed by petsc4py)
- petsc4py 3.7.0 (Python wrapper for PETSc)
- METIS 5
- PyMetis 2016.2


pychan3d can be installed from a terminal by moving to the pychan3d folder (containing the file setup.py) and issuing the command:
python setup.py install

Examples

Three examples are provided to illustrate some pychan3d workflows:
- example 1: generation of full lattice networks and percolation lattice networks, solving steady-state flow with
different solvers
- example 2: customizing the domain shape on the outer boundaries and including cavities, solving steady-state flow
- example 3: manually defining an arbitrary network of channels, solving steady-state flow, defining and running a
solute transport simulation

The examples can be run by opening a terminal and calling:
python path_to_pychan3d/examples/exampleX/exampleX.py

which will execute the script and generate the output files in your current working directory.


If you use pychan3d for your own work please cite:
Dessirier, Benoît, Chin-Fu Tsang, and Auli Niemi. "A new scripting library for modeling flow and transport in fractured rock with channel networks." Computers & Geosciences 111 (2018): 181-189.
