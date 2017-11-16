import numpy as np
from numpy.linalg import norm
from scipy.spatial import ConvexHull
from pychan3d.core_network import Node, Channel


def normed(item, eps=1.e-3):
    n = norm(item)
    if n > eps:
        return item / n
    else:
        return item

def get_relative_abcissa_of_closest_point_to_center(p1, p2, c=np.array([0., 0., 0.])):
    return (np.inner(c, p2-p1) + np.inner(p1, p1-p2)) / np.inner(p2-p1, p2-p1)


def get_relative_abcissa_of_closest_points_between_two_lines(p1, p2, p3=np.array([0., 0., 0.]), p4=np.array([0., 0., 1.])):
    b, d, e = p2 - p1, p4 - p3, p1 - p3
    det = - np.inner(b, b) * np.inner(d, d) + np.inner(b, d) ** 2
    if det == 0.:
        return get_relative_abcissa_of_closest_point_to_center(p1, p2, p3), 0.
    else:
        return (np.inner(d, d) * np.inner(b, e) - np.inner(d, e) * np.inner(b, d)) / det, \
               (- np.inner(b, b) * np.inner(d, e) + np.inner(d, b) * np.inner(b, e)) / det


def carve_sphere(nwk, center, radius, carve_in=True, eps=1.e-3, create_vtk_output=False):
    center = np.array(center)
    if create_vtk_output:
        import vtk
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(center[0], center[1], center[2]); sphere.SetRadius(radius)
        sphere.SetPhiResolution(20); sphere.SetThetaResolution(20); sphere.Update()
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName('sphere_center=%f,%f,%f_radius=%f.vtp' % (tuple(center) + (radius,)))
        try:    writer.SetInputData(sphere.GetOutput())
        except AttributeError:  writer.SetInput(sphere.GetOutput())
        writer.Write()

    inside, outside, n0 = {}, {}, len(nwk.nodes)
    for k in nwk.channels.keys():
        p1, p2 = nwk.nodes[k[0]](), nwk.nodes[k[1]]()
        t = get_relative_abcissa_of_closest_point_to_center(p1, p2, center)
        p3 = p1 + t * (p2 - p1)
        c = nwk.channels[k]
        if norm(p3 - center) > radius:  # closest point is outside
            outside[k] = c
        elif norm(p1 - center) < radius - eps and norm(p2 - center) < radius - eps:  # both end points are inside
            inside[k] = c
        elif norm(p3 - center) > radius - eps and t > 0. and t < 1.:  # one hit, split channel and assign both parts
            outside[(k[0],len(nwk.nodes))] = Channel(c.conductance / t, c.length * t, c.width, c.aperture, c.flow)
            outside[(k[1],len(nwk.nodes))] = Channel(c.conductance / (1.-t),c.length * (1.-t),c.width,c.aperture,c.flow)
            nwk.add_node(p3)
        else:  # two potential hits
            dt = np.sqrt(radius**2 - np.inner(p3 - center, p3 - center)) / norm(p2 - p1)
            t1, t2 = t - dt, t + dt
            p4, p5 = p1 + t1 * (p2 - p1), p1 + t2 * (p2 - p1)
            if t1 < 0. and norm(p1 - center) < radius - eps:
                inside[(k[0], len(nwk.nodes))] = Channel(c.conductance / t2, c.length * t2, c.width, c.aperture, c.flow)
                outside[(k[1],len(nwk.nodes))]=Channel(c.conductance/(1.-t2),c.length*(1.-t2),c.width,c.aperture,c.flow)
                nwk.add_node(Node().fromarray(p5))
            elif t2 > 1. and norm(p2 - center) < radius - eps:
                outside[(k[0], len(nwk.nodes))] = Channel(c.conductance / t1, c.length *t1, c.width, c.aperture, c.flow)
                inside[(k[1], len(nwk.nodes))]=Channel(c.conductance/(1.-t1),c.length*(1.-t1),c.width,c.aperture,c.flow)
                nwk.add_node(Node().fromarray(p4))
            elif t1 > 0. and t2 < 1.:  # t1 and t2 in [0, 1]
                outside[(k[0], len(nwk.nodes))] = Channel(c.conductance / t1, c.length * t1, c.width, c.aperture,c.flow)
                inside[(len(nwk.nodes), len(nwk.nodes) + 1)] = \
                                   Channel(c.conductance / (t2 - t1), c.length * (t2 - t1), c.width, c.aperture, c.flow)
                nwk.add_node(Node().fromarray(p4))
                outside[(k[1],len(nwk.nodes))]=Channel(c.conductance/(1.-t2),c.length*(1.-t2),c.width,c.aperture,c.flow)
                nwk.add_node(Node().fromarray(p5))

    boundary_nodes = range(n0, len(nwk.nodes))
    if carve_in:    nwk.channels = outside
    else:           nwk.channels = inside
    dic = nwk.clean_network(return_dic=True)
    return [dic[item] for item in boundary_nodes]


def carve_cylinder(nwk, p1, p2, radius, carve_in=True, eps=1.e-3, create_vtk_output=False):
    p1, p2 = np.array(p1), np.array(p2)
    if create_vtk_output:
        import vtk
        source = vtk.vtkLineSource()
        source.SetPoint1(p1[0], p1[1], p1[2]); source.SetPoint2(p2[0], p2[1], p2[2])
        tube = vtk.vtkTubeFilter()
        tube.SetInputConnection(source.GetOutputPort())
        tube.SetRadius(radius); tube.SetNumberOfSides(20); tube.Update()
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName('cylinder_bottom=%3.1f,%3.1f,%3.1f_top=%3.1f,%3.1f,%3.1f_radius=%3.1f.vtp' %
                           (tuple(p1) + tuple(p2) + (radius,)))
        try:    writer.SetInputData(tube.GetOutput())
        except AttributeError:  writer.SetInput(tube.GetOutput())
        writer.Write()

    inside, outside, n0 = {}, {}, len(nwk.nodes)
    for k in nwk.channels.keys():
        p3, p4 = nwk.nodes[k[0]](), nwk.nodes[k[1]]()
        t, s = get_relative_abcissa_of_closest_points_between_two_lines(p3, p4, p1, p2)
        p5, p6 = p3 + t * (p4 - p3), p1 + s * (p2 - p1)
        dmin = norm(p6 - p5)
        c = nwk.channels[k]
        if dmin > radius:  # no possible hit ###########################
            outside[k] = c
        elif dmin > radius - eps:  # one hit with tolerance ############
            if s < 0. or s > 1. or t < 0. or t > 1.:  # hit is outside the bounds of the considered cylinder section
                outside[k] = c
            else:
                outside[(k[0], len(nwk.nodes))] = Channel(c.conductance / t, c.length * t, c.width, c.aperture, c.flow)
                outside[(k[1],len(nwk.nodes))] = Channel(c.conductance/(1.-t),c.length*(1.-t),c.width,c.aperture,c.flow)
                nwk.add_node(p5)
        else:  # two potential hits
            # determining intersections with finite cylinder ###########################################################
            a = np.arccos(np.abs(np.inner(normed(p2 - p1), normed(p4 - p3))))
            if a < eps:  # the channel is parallel to the axis of the cylinder
                t1, t2 = t, t + np.sign(np.inner(p2 - p1, p4 - p3)) * norm(p2 - p1) / norm (p4 - p3)
                t1, t2 = min(t1, t2), max(t1, t2)
            elif a > np.pi / 2. - eps:  # the channel is perpendicular to the axis of the cylinder
                if s < 0. or s > 1.:  # hit is outside of the finite section of the cylinder
                    outside[k] = c; continue
                else:
                    dt = np.sqrt(radius**2 - dmin**2) / norm(p4 - p3)
                    t1, t2 = t - dt, t + dt
            else:  # alpha strictly between 0 and pi/2
                ds = np.sqrt(radius**2 - dmin**2) / np.sin(a) / norm(p2 - p1)
                s1, s2 = s - ds, s + ds
                if s2 < 0. or s1 > 1.:
                    outside[k] = c; continue  # the hits are completely outside the finite section of the cylinder
                if s1 < 0.: dt1 = - s / np.cos(a) * norm(p4-p3) / norm(p2-p1)  # hit on the bottom face
                else:       dt1 = ds / np.cos(a) * norm(p4-p3) / norm(p2-p1)  # hit on the radial face
                if s2 > 1.: dt2 = (1. - s) / np.cos(a) * norm(p4-p3) / norm(p2-p1)  # hit on the top face
                else:       dt2 = ds / np.cos(a) * norm(p4-p3) / norm(p2-p1)  # hit on the radial face
                t1, t2 = t + dt1, t + dt2
            # re-assigning channel sections to inside and outside depending on the intersections #######################
            if (t1 < 0. and t2 < 0.) or (t1 > 1. and t2 > 1.):
                outside[k] = c
            elif (t1 < 0. and t2 > 1.):
                inside[k] = c
            elif (t1 < 1. and t2 > 1.):
                outside[(k[0], len(nwk.nodes))] = Channel(c.conductance / t1, c.length * t1, c.width, c.aperture,c.flow)
                inside[(k[1],len(nwk.nodes))] =Channel(c.conductance/(1.-t1),c.length*(1.-t1),c.width,c.aperture,c.flow)
                nwk.add_node(Node().fromarray(p3 + t1 * (p4 - p3)))
            elif (t1 < 0. and t2 > 0.):
                inside[(k[0], len(nwk.nodes))] = Channel(c.conductance / t2, c.length * t2, c.width, c.aperture, c.flow)
                outside[(k[1],len(nwk.nodes))]=Channel(c.conductance/(1.-t2),c.length*(1.-t2),c.width,c.aperture,c.flow)
                nwk.add_node(Node().fromarray(p3 + t2 * (p4 - p3)))
            elif t1 > 0. and t2 < 1.:
                outside[(k[0], len(nwk.nodes))] = Channel(c.conductance / t1, c.length * t1, c.width, c.aperture,c.flow)
                inside[(len(nwk.nodes), len(nwk.nodes) + 1)] = \
                                   Channel(c.conductance / (t2 - t1), c.length * (t2 - t1), c.width, c.aperture, c.flow)
                nwk.add_node(Node().fromarray(p3 + t1 * (p4 - p3)))
                outside[(k[1],len(nwk.nodes))]=Channel(c.conductance/(1.-t2),c.length*(1.-t2),c.width,c.aperture,c.flow)
                nwk.add_node(Node().fromarray(p3 + t2 * (p4 - p3)))

    boundary_nodes = range(n0, len(nwk.nodes))
    if carve_in:    nwk.channels = outside
    else:           nwk.channels = inside
    dic = nwk.clean_network(return_dic=True)
    return [dic[item] for item in boundary_nodes]


def carve_convex(nwk, vertices, carve_in=True, eps=1.e-3, create_vtk_output=False):
    vertices = np.array(vertices)
    hull = ConvexHull(vertices)

    import vtk
    points = vtk.vtkPoints()
    for point in hull.points:
        points.InsertNextPoint(point[0], point[1], point[2])
    polyhedronFacesIdList = vtk.vtkIdList()
    polyhedronFacesIdList.InsertNextId(hull.simplices.shape[0])
    for face in hull.simplices:
        polyhedronFacesIdList.InsertNextId(len(face))
        [polyhedronFacesIdList.InsertNextId(i) for i in face]
    uGrid = vtk.vtkUnstructuredGrid()
    uGrid.SetPoints(points)
    uGrid.InsertNextCell(vtk.VTK_POLYHEDRON, polyhedronFacesIdList)
    polyhedron = uGrid.GetCell(0)
    if create_vtk_output:
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName("convex.vtu")
        try:    writer.SetInputData(uGrid)
        except AttributeError:  writer.SetInput(uGrid)
        writer.Write()

    inside, outside, n0 = {}, {}, len(nwk.nodes)
    for k in nwk.channels.keys():
        p1, p2, c = nwk.nodes[k[0]](), nwk.nodes[k[1]](), nwk.channels[k]
        t1, x1, pcoords1, subId1 = vtk.mutable(0), [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], vtk.mutable(0)
        iD1 = polyhedron.IntersectWithLine(p1, p2, eps, t1, x1, pcoords1, subId1)
        p1_in, p2_in = polyhedron.IsInside(p1, eps), polyhedron.IsInside(p2, eps)
        if iD1 > 0:
            if p1_in:
                inside[(k[0], len(nwk.nodes))] = Channel(c.conductance / t1, c.length * t1, c.width, c.aperture, c.flow)
                outside[(k[1],len(nwk.nodes))]=Channel(c.conductance/(1.-t1),c.length*(1.-t1),c.width,c.aperture,c.flow)
                nwk.add_node(Node().fromarray(x1))
            elif p2_in:
                outside[(k[0], len(nwk.nodes))] = Channel(c.conductance / t1, c.length *t1, c.width, c.aperture, c.flow)
                inside[(k[1], len(nwk.nodes))]=Channel(c.conductance/(1.-t1),c.length*(1.-t1),c.width,c.aperture,c.flow)
                nwk.add_node(Node().fromarray(x1))
            else:
                t2, x2, pcoords2, subId2 = vtk.mutable(0), [0., 0., 0.], [0., 0., 0.], vtk.mutable(0)
                iD2 = polyhedron.IntersectWithLine(p2, p1, eps, t2, x2, pcoords2, subId2)
                outside[(k[0], len(nwk.nodes))] = Channel(c.conductance / t1, c.length * t1, c.width, c.aperture,c.flow)
                inside[(len(nwk.nodes), len(nwk.nodes) + 1)] = \
                                   Channel(c.conductance / (t2 - t1), c.length * (t2 - t1), c.width, c.aperture, c.flow)
                nwk.add_node(Node().fromarray(x1))
                outside[(k[1],len(nwk.nodes))]=Channel(c.conductance/(1.-t2),c.length*(1.-t2),c.width,c.aperture,c.flow)
                nwk.add_node(Node().fromarray(x2))
        else:
            if p1_in and p2_in: inside[k] = c
            else:               outside[k] = c

    boundary_nodes = range(n0, len(nwk.nodes))
    if carve_in:    nwk.channels = outside
    else:           nwk.channels = inside
    dic = nwk.clean_network(return_dic=True)
    return [dic[item] for item in boundary_nodes]


# if __name__ == "__main__":
#     from  pychan3d import LatticeNetwork
#     LN = LatticeNetwork(1, 0, 0, 2., 1., 1., offset=[0., 0., 0.])
#
#     print(LN.nodes)
#     print(LN.channels)
#     # carve_convex(LN, [[1., 0. , 0.],
#     #                   [0.5, -1, -1],[0.5, 1, -1],[2.25, 1, -1],[2.25, -1, -1],
#     #                   [0.5, -1, 1], [0.5, 1, 1], [2.25, 1, 1], [2.25, -1, 1]], carve_in=True)
#     carve_cylinder(LN, [1.5, 0.25, 0.25], [1., 0.25, 0.25], radius=0.5)
#     print(LN.nodes)
#     print(LN.channels)