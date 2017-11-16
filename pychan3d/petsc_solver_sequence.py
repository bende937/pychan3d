import logging
logging.basicConfig(filename='log_petsc', format='%(asctime)s: %(message)s', level=logging.INFO)

import numpy as np

from mpi4py import MPI
comm = MPI.Comm.Get_parent()

import petsc4py
petsc4py.init()
from petsc4py import PETSc

int_parameters = np.empty((4,), dtype='i')
comm.Bcast([int_parameters, MPI.INT], root=0)
n = int_parameters[0]
nnz = int_parameters[1]
n_data = int_parameters[2]
maxiter = int_parameters[3]

indptr = np.empty((n+1,), dtype='i')
comm.Bcast([indptr, MPI.INT], root=0)
indices = np.empty((n_data,), dtype='i')
comm.Bcast([indices, MPI.INT], root=0)
data = np.empty((n_data,), dtype='d')
comm.Bcast([data, MPI.DOUBLE], root=0)
b = np.empty((n,), dtype='d')
comm.Bcast([b, MPI.DOUBLE], root=0)
tol = np.empty((1,), dtype='d')
comm.Bcast([tol, MPI.DOUBLE], root=0)
tol = tol[0]

logging.info('Starting PETSc process %i out of %i' % (PETSc.COMM_WORLD.getRank()+1, PETSc.COMM_WORLD.getSize()))
logging.info('...assembling A...')
petsc_matrix = PETSc.Mat().createAIJ(size=(n, n), comm=PETSc.COMM_WORLD)  # I didn't manage to pass directly the csr here
petsc_matrix.setPreallocationNNZ(nnz)
istart, iend = petsc_matrix.getOwnershipRange()
for r in xrange(istart, iend):
    for i in xrange(indptr[r], indptr[r+1]):
        petsc_matrix[r, indices[i]] = data[i]
petsc_matrix.assemblyBegin()
petsc_matrix.assemblyEnd()

logging.info('...configuring solver...')
ksp = PETSc.KSP()
ksp.create(PETSc.COMM_WORLD)
ksp.setType('cg')  # conjugate gradient solver
ksp.getPC().setType('bjacobi')  # incomplete LU factorization
ksp.setTolerances(rtol=tol, max_it=maxiter)
x, petsc_b = petsc_matrix.getVecs()
x.set(0)
petsc_b.setArray(b[istart:iend])
ksp.setOperators(petsc_matrix)
ksp.setFromOptions()
logging.info('...running solver...')
ksp.solve(petsc_b, x)
if ksp.converged:
    info = np.array(0, dtype='i')
    comm.Reduce([info, MPI.INT], None, op=MPI.SUM, root=0)
    logging.info('...solver done, successfully')
    res = np.array(x.getArray(), dtype='d')
    comm.Gather([res, MPI.DOUBLE], None, root=0)
else:
    info = np.array(1, dtype='i')
    comm.Reduce([info, MPI.INT], None, op=MPI.SUM, root=0)
    logging.warning('...solver done, no convergence with tolerance %e after %i iterations' % (tol, maxiter))
logging.info('...PETSc process %i out of %i done.' % (PETSc.COMM_WORLD.getRank()+1, PETSc.COMM_WORLD.getSize()))
comm.Disconnect()
