import pickle
from .core_network import Node, Channel, Network, BND_COND
from .lattice_network import LatticeNetwork
from .sparse_lattice_network import SparseLatticeNetwork
from .core_transport import TransportSimulation
from .carving_utilities import carve_sphere, carve_cylinder, carve_convex
from .discrete_fracture_network import DiscreteFractureNetwork

__version__ = '0.1.0'


def load_network(filename, nwk=Network()):
    with open(filename, 'rb') as f:
        nwk.nodes = pickle.load(f)
        nwk.channels = pickle.load(f)
        nwk.hbnds = pickle.load(f)
        nwk.qbnds = pickle.load(f)
        nwk.heads = pickle.load(f)
        nwk.node_throughflows = pickle.load(f)
        nwk.boundary_flows = pickle.load(f)
        nwk.components = pickle.load(f)
        nwk.n_components = pickle.load(f)
        nwk.percolating_components = pickle.load(f)
    return nwk


def load_lattice_network(filename):
    with open(filename, 'rb') as f:
        lnwk = LatticeNetwork(nx=pickle.load(f), ny=pickle.load(f), nz=pickle.load(f),
                              dx=pickle.load(f), dy=pickle.load(f), dz=pickle.load(f), offset=pickle.load(f),
                              mean_logcx=pickle.load(f), mean_logcy=pickle.load(f), mean_logcz=pickle.load(f),
                              sigma_logcx=pickle.load(f), sigma_logcy=pickle.load(f), sigma_logcz=pickle.load(f),
                              initial_state=pickle.load(f))
        lnwk.hbnds = pickle.load(f)
        lnwk.qbnds = pickle.load(f)
        lnwk.set_heads(pickle.load(f))
    return lnwk


def load_sparse_lattice_network(filename):
    with open(filename, 'rb') as f:
        lnwk = SparseLatticeNetwork(nx=pickle.load(f), ny=pickle.load(f), nz=pickle.load(f),
                                    d=pickle.load(f), offset=pickle.load(f),
                                    PAx=pickle.load(f), PONx=pickle.load(f), PAy=pickle.load(f),
                                    PONy=pickle.load(f), PAz=pickle.load(f), PONz=pickle.load(f),
                                    mean_logcx=pickle.load(f), mean_logcy=pickle.load(f), mean_logcz=pickle.load(f),
                                    sigma_logcx=pickle.load(f), sigma_logcy=pickle.load(f), sigma_logcz=pickle.load(f),
                                    initial_state=pickle.load(f))
        lnwk.hbnds = pickle.load(f)
        lnwk.qbnds = pickle.load(f)
        lnwk.set_heads(pickle.load(f))
    return lnwk
