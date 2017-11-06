import pychan3d


#  In this example we create a full lattice network and carve different shape inside and on the outside of the domain
#  before calculating the steady state flow solution.

#  We will create a lattice network with a spacing of 10m in each direction
dx, dy, dz = 10., 10., 10.
# and 30 channels in the X-direction (length), 10 channels in the Y- and Z-directions (width and heigth)
nx, ny, nz = 30, 10, 10
#  So the resulting total dimensions are 300m x 100m x 100m

#  We choose the mean log conductance and a standard deviation for the log conductances which are assumed to be
#  lognormally distributed
logC, sigma = -6., 1.

#  We create the Lattice network, and use an offset to have the center of the box at coordinates (0., 0., 0.)
LN = pychan3d.LatticeNetwork(nx, ny, nz, dx, dy, dz, logC, logC, logC, sigma, sigma, sigma,
                             offset=[-150., -50., -50.], seed=987654321)
LN.export2vtk('example2_convex_full_network')

#  Now, we carve a cylindrical tunnel at the center of the network, the point on the axis of the cylinder and on the
#  'bottom' surface has coordinates (-51,0,0), the point on the axis and on the 'top' surface has coordinates (51,0,0),
#  the cylinder has radius of 2.5m, and we discard the points located inside the cylinder.
#  This cylinder can be visualized in Paraview by loading the companion file 'example2_cylindrical_cavity.vtp'.
#  We also collect the indices of the nodes located on the cylinder into an array (nodes1).
nodes1 = pychan3d.carve_cylinder(LN, [-51., 0., 0.], [51., 0., 0.], radius=2.5, carve_in=True, create_vtk_output=True)
#  We assign fixed head boundary conditions (0m) for all the nodes on the tunnel wall
for n in nodes1:
    LN.hbnds[n] = 0.
LN.export2vtk('example2_cylinder_carved_in')

#  Now, we trim the outer part of the domain to a cylindrical shape with a radius of 48m.
nodes2 = pychan3d.carve_cylinder(LN, [-151., 0., 0.], [151., 0., 0.], radius=48., carve_in=False, create_vtk_output=True)
#  We assign a fixed head boundary condition at the nodes located on the outer surface of the domain (10m)
for n in nodes2:
    LN.hbnds[n] = 10.
LN.export2vtk('example2_cylinder_carved_out')

#  Inside, the domain we will also carve a convex cavity defined by its corner points, here a cubic box (but it could be
# any convex shape defined a minimum of 4 non-coplanar points).
#  This box can be visualized in Paraview by loading the companion file 'example2_convex_cavity.vtp'.
points = [[34., -16., 16.], [34., -16., -16.], [34., 16., 16.],  [34., 16., -16.],
          [66., 16., 16.],  [66., 16., -16.],  [66., -16., 16.], [66., -16., -16.]]
nodes3 = pychan3d.carve_convex(LN, points, carve_in=True, create_vtk_output=True)
for n in nodes3:  # and we assign fixed head boundary conditions
    LN.hbnds[n] = 0.
LN.export2vtk('example2_convex_carved_in')

#  We also carve a spherical cavity centered at (-51,0,0) of radius 21m.
#  This sphere can be visualized in Paraview by loading the companion file 'example2_spherical_cavity.vtp'.
nodes4 = pychan3d.carve_sphere(LN, [-51., 0., 0.], 21., carve_in=True, create_vtk_output=True)
for n in nodes4:  # and we assign fixed head boundary conditions
    LN.hbnds[n] = 0.
LN.export2vtk('example2_sphere_carved_in')

#  finally we solve the steady state flow problem using the direct solver
LN.solve_steady_state_flow_scipy_direct()
#  and export the results to a vtk file for visualization in paraview
LN.export2vtk('example2_flow_data')
# you can compare your result with the provided file given that you did not change the seed value on line16
