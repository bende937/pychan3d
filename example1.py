import pychan3d


########################################################################################################################
#  The first part shows the generation of a full Lattice network with lognormally distributed channel conductances
#  In this case we choose an even spacing of 1m between channels in all directions
dx, dy, dz = 1., 1., 1.
#  We decide to create a lattice of 50 nodes (49 channels) in each direction
nx, ny, nz = 49, 49, 49
#  The the total network will have a dimension of 49x49x49

#  We choose a mean log conductance of -6 (log10 [m**2/s]) and a standard deviation of 1.5 in each direction
meanlogc, sigma = -6., 1.5

#  Now we create the Lattice network
LN = pychan3d.LatticeNetwork(nx, ny, nz, dx, dy, dz, meanlogc, meanlogc, meanlogc, sigma, sigma, sigma, seed=987654321)
#  For lattice networks, one can assign boundary conditions on particular faces of the model,
#  for example here between X- and X+
LN.set_Hboundary_Xminus(0.)
LN.set_Hboundary_Xplus(10.)

#  Now we can solve the steady state flow system, for example using an AMG precondicitoner and a conjugate gradient
#  solver
LN.solve_steady_state_flow_pyamg()
# LN.solve_steady_state_flow_petsc(nproc=4)  # for this solver to work, you need PETSc and the corresponding python
# # modules to be installed

# One can do some post-processing
print("the steady state flow rate through the system is %e m^3/s" % (LN.get_total_inflow(), ))

#  And finally export the results for visualization in Paraview
LN.export2vtk('example1_part1_lattice_network')
########################################################################################################################

########################################################################################################################
#  The second part shows the generation of a percolation (sparse) lattice
#  In this case we also use an even spacing of 1m in all directions
d = 1.
#  And we also adopt a total lattice size of 49 channels (50 nodes) in each direction
nx, ny, nz = 49, 49, 49
#  The the total network will have a dimension of 49x49x49 as before.

#  Here we decide to use a constant log channel conductance of -10. (log10 [m**2/s])
meanlogc, sigma = -10., 0.
#  For percolation lattices as defined by Black et al. (2016), one needs two extra parameters to generate the lattice
#  PA is the probability for a channel to be open if the previous one in the same direction was already open
#  and PON is the probability for any channel to be open
PA, PON = 0.9, 0.04

#  Now we create the percolation network
SLN = pychan3d.SparseLatticeNetwork(nx, ny, nz, d, PA, PON, meanlogc, sigma, seed=981654127)

# One can fetch the indices of the nodes on some exterior faces of the domain to test percolation
ind_out = SLN.get_Xminus_indices()
ind_in = SLN.get_Xplus_indices()
perco = SLN.check_percolation(ind_in, ind_out)
print('checking percolation...')

if perco:  # if the network percolates (True with the provided seed value),
    SLN.remove_non_percolating_channels(ind_in, ind_out) #, we trim the non-percolating channel clusters
    ind_out = SLN.set_Hboundary_Xminus(0.)  #we assign boundary conditions and calculate the steady state flow solution
    ind_in = SLN.set_Hboundary_Xplus(10.)
    print('...starting solve...')
    SLN.solve_steady_state_flow_scipy_direct()
    SLN.export2vtk('example1_part2_sparse_lattice_network')

    print("the steady state flow rate through the system is %e m^3/s" % (SLN.get_total_inflow(), ))
else:
    print('Network is not percolating.')
########################################################################################################################