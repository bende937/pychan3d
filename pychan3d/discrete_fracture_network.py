import numpy as np
from scipy.sparse import coo_matrix
from scipy.interpolate import interp1d
from scipy.spatial import Delaunay, delaunay_plot_2d
import matplotlib.pyplot as plt
from pychan3d.core_network import Node, Channel, Network, BND_COND
from collections import Counter

class DiscreteFractureNetwork(object):
    """
    Description
    """
    def __init__(self, nodes=None, fractures=None, fparams=None, intersections=None):
        """
        :param nodes: a list of node objects (defined in core_network) to define both fracture perimeters and intersections
        :param fractures: a list of lists of node indices to define the fracture perimeters
        :param fparameters: a list of lists of fracture parameters, the working version has 3 paramereters, T0,
        sigma_lnK and lambda
        :param intersections: a dictionary of lists of node indices to define the fracture intersections, the keys are
        tuples of pairs of fracture numbers
        """
        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes
        if fractures is None:
            self.fractures = []
        else:
            self.fractures = fractures
        if fparams is None:
            self.fparams = []
        else:
            self.fparams = fparams
        if intersections is None:
            self.intersections = {}
        else:
            self.intersections = {(min(k[0], k[1]), max(k[0], k[1])): intersections[k] for k in intersections.keys()}

    def __str__(self):
        return "DFN with %i fractures and %i intersections." % (len(self.fractures), len(self.intersections))

    def get_fracture_nodes(self):
        return np.array(self.nodes)[list(set([inner for outer in self.fractures for inner in outer]))]

    def get_intersection_nodes(self):
        return np.array(self.nodes)[list(set([inner for outer in self.intersections.values() for inner in outer]))]

    def from_file(self, filename):
        pass

    def clean_network(self):
        pass  # remove the fnodes that are not used in fractures and re-index fractures
        # remove the inodes that are not used in intersections and re-index intersections

    def export2vtk(self, filename):
        import vtk
        points = vtk.vtkPoints()
        for point in self.nodes:
            points.InsertNextPoint(point.x, point.y, point.z)

        # create file for fractures
        fractures = vtk.vtkCellArray()
        for fracture in self.fractures:
            polygon = vtk.vtkPolygon()
            polygon.GetPointIds().SetNumberOfIds(len(fracture))
            for n, i in enumerate(fracture):
                polygon.GetPointIds().SetId(n, i)
            fractures.InsertNextCell(polygon)
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(fractures)
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filename + "_fractures.vtp")
        try:
            writer.SetInputData(polydata)
        except AttributeError:
            writer.SetInput(polydata)
        writer.Write()

        # create file for intersections
        intersections = vtk.vtkCellArray()
        for intersection in self.intersections.values():
            polyline = vtk.vtkPolyLine()
            polyline.GetPointIds().SetNumberOfIds(len(intersection))
            for n, i in enumerate(intersection):
                polyline.GetPointIds().SetId(n, i)
            intersections.InsertNextCell(polyline)
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(intersections)
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filename + "_intersections.vtp")
        try:
            writer.SetInputData(polydata)
        except AttributeError:
            writer.SetInput(polydata)
        writer.Write()

    def get_fracture_center_normalvec_etc(self, frac):
        fcenter = np.sum([(1. / float(len(frac))) * node for node in np.array(self.nodes)[frac]], axis=0)
        fnormal = np.cross(self.nodes[frac[0]]() - fcenter(), self.nodes[frac[1]]() - fcenter())
        fnormal /= np.linalg.norm(fnormal)

        v = np.cross(fnormal, np.array([0., 0., 1.]))
        if np.linalg.norm(v) < 1.e-5:   R = np.eye(3)
        else:
            angle = np.arccos(np.dot(fnormal, np.array([0., 0., 1.])))
            axis = v / np.linalg.norm(v)
            nn = np.outer(axis, axis.T)
            nx = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
            R = np.cos(angle) * np.eye(3) + np.sin(angle) * nx + (1. - np.cos(angle)) * nn
        fpoints = np.array([(self.nodes[fi] - fcenter).rotate(R)()[:-1] for fi in frac])
        perimeter = zip(range(len(frac)), range(1, len(frac)) + [0])
        return fcenter, fnormal, R, fpoints, perimeter

    def export2channelnetwork(self, scheme='cacas', w=None, fparams={'transmissivity': 1.e-8}):
        if scheme is 'cacas':
            self.nodes = np.array(self.nodes)
            fnodes = [(1. / len(f)) * np.sum(self.nodes[f]) for f in self.fractures]
            ikeys = self.intersections.keys()
            inodes = [0.5*(self.nodes[self.intersections[k][0]] + self.nodes[self.intersections[k][-1]]) for k in ikeys]
            nodes = fnodes + inodes

            channels = np.array([[k[0], i + len(fnodes), i + len(fnodes), k[1]]
                                 for i, k in enumerate(ikeys)]).reshape(2*len(ikeys), 2)
            vectors = np.array([nodes[t[0]]() - nodes[t[1]]() for t in channels])
            length = np.array([np.linalg.norm(v) for v in vectors])
            length[length < 1.e-5] = 1.e-5  # if points are identical, we cheat by adding a small distance
            if w is None:
                ivectors = np.array([[self.nodes[self.intersections[k][0]]() -self.nodes[self.intersections[k][-1]]()]*2
                                     for k in ikeys]).reshape(2 * len(ikeys), 3)
                width = np.array([np.linalg.norm(ivectors[n]) - np.abs(np.dot(ivectors[n], vectors[n] / length[n]))
                                  for n, t in enumerate(channels)])
            else:
                width = np.array(w) * np.ones(length.shape)
            transm = np.array(fparams['transmissivity'])
            if transm.shape == (1,):
                transm = transm * np.ones(length.shape)
            elif transm.shape == (len(self.fractures),):
                transm = np.array([[transm[k[0]], transm[k[1]]] for k in ikeys]).flatten()
            # we could add a case where the transmissivity is based on the sqrt of the fracture area
            rows, cols = channels[:, 0], channels[:, 1]
            conductances = transm * width / length

            channels = {(rows[n], cols[n]): Channel(c, length=length[n], width=width[n]) for n, c in enumerate(conductances)}
            nwk = Network(nodes=nodes, channels=channels)
            nwk.set_channel_apertures()
            return nwk

            # import matplotlib.pyplot as plt
            # from mpl_toolkits.mplot3d import Axes3D
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # points = np.array([item() for item in self.nodes])
            #
            # for n1, n2 in ilinks:  # plotting intersection channels
            #     plt.plot(points[[n1, n2], 0], points[[n1, n2], 1], points[[n1, n2], 2],'r-')
            # for n1, n2 in flinks:  # plotting tri_links
            #     plt.plot(points[[n1, n2], 0], points[[n1, n2], 1], points[[n1, n2], 2],'g--')
            # for f in self.fractures:
            #     plt.plot(points[f, 0], points[f, 1], points[f, 2], 'k-')
            # plt.plot(points[:, 0], points[:, 1], points[:, 2], 'ko')  # plotting fpoints
            # plt.show()





# generate_small_random_network()

# exit()
# # test area
# nodes = [Node(0., 0., 0.),
#           Node(1., 0., 0.),
#           Node(1., 1., 0.),
#           Node(0., 1., 0.),
#           Node(1., 1., 1.),
#           Node(1., 1., 0.4),
#           Node(0.75, 0.75, 0.)]
# fractures = [[0, 1, 6, 3], [0, 1, 4], [0, 2, 4], [0, 5, 3]]
# fparams = {'transmissivity': [1.e-8, 1.e-10, 1.e-9, 1.e-9]}
# intersections = {(0, 1): [0, 1], (0, 2): [0, 6], (1, 2): [0, 4], (0, 3): [0, 3], (2, 3): [0, 5]}
#
# DFN = DiscreteFractureNetwork(nodes=nodes, fractures=fractures, fparams=fparams, intersections=intersections)
# print(DFN)
# DFN.export2vtk('test_dfn')
# CN1 = DFN.export2channelnetwork(scheme='cacas', w=0.1)
# CN1.export2vtk('test_cacas')
#
#
