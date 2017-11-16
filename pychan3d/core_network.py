import numpy as np
from scipy.sparse import coo_matrix, dok_matrix, csgraph, linalg
from scipy.spatial.distance import cdist
from collections import Counter
import pickle
import logging
logging.basicConfig(filename='log_network', format='%(asctime)s: %(message)s', level=logging.INFO)


BND_COND = 1.e6


class Node(object):
    """This is a small container class for points in 3D.
    Its only attributes are three spaces coordinates x, y and z."""
    def __init__(self, x=0., y=0., z=0.):
        self.x, self.y, self.z = x, y, z

    def __str__(self):
        return "Point at %f,%f,%f" % (self.x, self.y, self.z)

    def fromarray(self, arr):
        self.x, self.y, self.z = arr[0], arr[1], arr[2]  # will ignore other elements if array too long
        return self

    def __repr__(self):
        return "(%f, %f, %f)" % (self.x, self.y, self.z)

    def __add__(self, other):
        return Node(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Node(self.x - other.x, self.y - other.y, self.z - other.z)

    def __rmul__(self, scalar):
        return Node(self.x * scalar, self.y * scalar, self.z * scalar)

    def __call__(self):
        return np.array([self.x, self.y, self.z])

    def rotate(self, rot_matrix):
        x, y, z = tuple(np.dot(rot_matrix, self()))
        return Node(x, y, z)


class Channel(object):
    """This is a class to contain all attributes necessary for a channel."""
    def __init__(self, conductance=np.nan, length=np.nan, width=np.nan, aperture=np.nan, flow=np.nan):
        self.conductance, self.length, self.width, self.aperture, self.flow = conductance, length, width, aperture, flow

    def __str__(self):
        return "Channel: C=%e m2/s, L=%e m, W=%e m, b=%e m" % (self.conductance, self.length, self.width, self.aperture)

    def __repr__(self):
        return "Channel: C=%e m2/s, L=%e m, W=%e m, b=%e m" % (self.conductance, self.length, self.width, self.aperture)

    def set_aperture_from_extended_cubic_law(self, flexp=3., flcoef=9.81e6/12.):
        self.aperture = (self.conductance * self.length / self.width / flcoef) ** (1. / flexp)

    def set_width_from_fws(self, fws):
        self.width = 0.5 * fws / self.length

    def fws(self):
        return 2. * self.length * self.width

    def vol(self):
        return self.length * self.width * self.aperture

    def tadv(self):
        return self.vol() / self.flow


class Network(object):
    """This is the main class to represent unstructured channel networks for flow in fractured rock."""

    def __init__(self, nodes=(), channels={}, hbnds={}, qbnds={}):
        logging.info("Creating Network with %i Nodes, %i Channels, %i fixed head bnd conditions, %i fixed flow bnd conditions..." % \
                     (len(nodes), len(channels), len(hbnds), len(qbnds)))
        self.nodes = list(nodes)
        self.channels = {(min(k[0], k[1]), max(k[0], k[1])): v for k, v in channels.items()}
        self.hbnds, self.qbnds = hbnds, qbnds
        self.heads, self.boundary_flows, self.node_throughflows = None, None, None
        self.components, self.n_components, self.percolating_components = None, 0., None
        logging.info('...done.')

    def __str__(self):
        return "Network with %i Nodes, %i Channels, %i fixed head bnd conditions, %i fixed flow bnd conditions." % \
               (len(self.nodes), len(self.channels), len(self.hbnds), len(self.qbnds))
    # ##################################################################################################################

    # network building and management methods ##########################################################################
    def __add__(self, nwk, eps=1.e-5, consolidate_nodes=True, check_channel_intersections=False):
        """Consolidate a network with another one.
        Note1: when consolidating nodes, the routine will identify the nodes in nwk with the closest one in self if they
        are within a distance eps. The result might become unpredictable if self has nodes with locations within 2*eps
        of each other. The user would have to define his own heuristics in that case.
        Note2: If two channels are redundant after consolidating, the channels properties from nwk take precedence over
        the channel properties from self in the returned object.
        Note3: Same behavior with boundary conditions, nwk takes precedence over self in the returned object.
        Note4: check_channel_intersections is not implemented yet."""
        n = len(self.nodes)
        dic = {i: i + n for i in range(len(nwk.nodes))}  # indices of nwk nodes in the consolidated network
        # check for tying nodes.
        if consolidate_nodes:
            dist = cdist(np.array([item() for item in self.nodes]), np.array([item() for item in nwk.nodes]))
            ind = np.arange(len(nwk.nodes))[np.sum(np.where(dist < eps, True, False), axis=0, dtype=bool)]
            for i in ind:
                dic[i] = np.argmin(dist[:, i])
        # defining the network to be returned
        nodes = self.nodes + nwk.nodes
        channels = self.channels.copy()
        channels.update({(dic[k[0]], dic[k[1]]): v for k, v in nwk.channels.items()})
        hbnds = self.hbnds.copy()
        hbnds.update({dic[k]: v for k, v in nwk.hbnds.items()})
        qbnds = self.qbnds.copy()
        qbnds.update({dic[k]: v for k, v in nwk.qbnds.items()})
        new_nwk = Network(nodes=nodes, channels=channels, hbnds=hbnds, qbnds=qbnds)
        new_nwk.clean_network()
        return new_nwk

    def add_node(self, node, h=np.nan, q=np.nan):
        if ~np.isnan(h): self.add_hboundary(len(self.nodes), h)
        if ~np.isnan(q): self.add_qboundary(len(self.nodes), q)
        self.nodes.append(node)

    def add_nodes(self, nodes, h=[np.nan], q=[np.nan]):
        self.hbnds.update({len(self.nodes) + r: hr for r, hr in enumerate(h) if ~np.isnan(hr)})
        self.qbnds.update({len(self.nodes) + r: qr for r, qr in enumerate(q) if ~np.isnan(qr)})
        self.nodes += list(nodes)

    def add_channel(self, n1, n2, chan):
        n1, n2 = min(n1, n2), max(n1, n2)
        if (n1, n2) in self.channels:
            self.add_node(0.5 * (self.nodes[n1]+self.nodes[n2]))
            chan1 = Channel(conductance=2.*chan.conductance, length=0.5*chan.length, width=chan.width, aperture=chan.aperture)
            chan2 = Channel(conductance=2.*chan.conductance, length=0.5*chan.length, width=chan.width, aperture=chan.aperture)
            self.channels.update({(n1, len(self.nodes)): chan1, (n2, len(self.nodes)): chan2})
        else:
            self.channels[(n1, n2)] = chan

    def add_channels(self, channels={}):
        for k, v in channels.items():
            self.add_channel(k[0], k[1], v)

    def add_hboundary(self, n, h):
        """Adds a fixed head boundary condition at node n (first argument). The head value is the second argument."""
        self.hbnds[n] = h

    def add_qboundary(self, n, q):
        """Adds a fixed flow boundary condition at node n (first argument). The flow value is the second argument.
        Positive flow values represent water entering the network at the node."""
        self.qbnds[n] = q
                
    def disconnect_node(self, ind):
        """Removes all channels connecting to one node. To remove it completely, call the "clean_network" method."""
        to_delete = set((i, ind) for i in range(ind))
        to_delete.update(set((ind, i) for i in range(ind+1, len(self.nodes))))
        self.channels = {k: self.channels[k] for k in (self.channels.keys() - to_delete)}

    def disconnect_nodes(self, inds):
        inds = set(inds)
        to_delete = set(k for k in self.channels.keys() if (k[0] in inds or k[1] in inds))
        self.channels = {k: self.channels[k] for k in (self.channels.keys() - to_delete)}

    def remove_node(self, ind):
        self.disconnect_node(ind)
        self.clean_network()

    def remove_nodes(self, inds):
        self.disconnect_nodes(inds)
        self.clean_network()

    def remove_channel(self, n1, n2):
        n1, n2 = min(n1, n2), max(n1, n2)
        del self.channels[(n1, n2)]

    def remove_channels(self, pairs):
        self.channels = {k: self.channels[k] for k in (self.channels.keys() - set(pairs))}

    def remove_hboundary(self, ind):
        del self.hbnds[ind]

    def remove_qboundary(self, ind):
        del self.qbnds[ind]

    def clean_network(self, return_dic=False):
        """This function detects nodes with no connecting channel, delete them and re-indexes channels accordingly."""
        logging.info("cleaning network...")
        connected_nodes = sorted(set(i for t in self.channels.keys() for i in t))
        if len(self.nodes) > len(connected_nodes):
            self.nodes = [self.nodes[i] for i in connected_nodes]
            if self.heads is not None:
                self.heads = self.heads[connected_nodes]
            dic = {old: new for new, old in enumerate(connected_nodes)}
            connected_nodes = set(connected_nodes)
            self.hbnds = {dic[k]: self.hbnds[k] for k in connected_nodes.intersection(set(self.hbnds.keys()))}
            self.new_qbnds = {dic[k]: self.qbnds[k] for k in connected_nodes.intersection(set(self.qbnds.keys()))}
            self.channels = {(dic[k[0]], dic[k[1]]): v for k, v in self.channels.items()}
            if self.heads is not None:
                self.calculate_extra_output()
        else:
            if return_dic:  dic = {i: i for i in range(len(self.nodes))}
        logging.info("...done.")
        if return_dic:
            return dic

    def set_heads(self, h):
        """Assigns the head field to h. And calculates the corresponding channel flows. A mass balance is calculated at
         each node and the residual is defined as boundary_flows."""
        h = np.array(h).flatten()
        if h.size == len(self.nodes):
            self.heads = h
        else:
            logging.error("Could not assign heads, wrong array size.")
            return
        self.calculate_extra_output()
    # ##################################################################################################################

    # percolation-related methods ######################################################################################
    def calculate_components(self):
        """This method populates the attribute self.components that tags each node with a integer representing a set of
        nodes it is connected to by channels."""
        logging.info('calculating connected components...')
        self.clean_network()
        rows, cols = zip(*self.channels.keys())
        self.n_components, self.components = csgraph.connected_components(
                                  coo_matrix((np.ones_like(rows, dtype=bool), (rows, cols)), shape=[len(self.nodes)]*2))
        logging.info('...identified %i (dis)connected component(s), done' % (self.n_components, ))
    
    def check_percolation(self, indices_in, indices_out):
        """This method checks if there are any components of the graph linking the two lists of indices. It populates
        the attribute self.percolating_components with a tuple of the input arguments and the set of percolating
        component indices."""
        self.calculate_components()
        comp_in, comp_out = set(self.components[i] for i in indices_in), set(self.components[i] for i in indices_out)
        self.percolating_components = (indices_in, indices_out, comp_in.intersection(comp_out))
        if len(self.percolating_components[2]) > 0: return True
        else:   return False

    def remove_non_percolating_channels(self, indices_in, indices_out):
        """This method removes nodes and channels on non-percolating components of the graph and then recalculates the
        graph components and percolating_components."""
        self.check_percolation(indices_in, indices_out)
        if len(self.percolating_components[2]) < 1:
            logging.info("no percolating component")
            return [], []
        logging.info("removing non-percolating components...")
        non_perco = set(i for i in range(len(self.nodes)) if self.components[i] not in self.percolating_components[2])
        self.disconnect_nodes(non_perco)
        dic = self.clean_network(return_dic=True)
        indices_in = set(dic.get(item, None) for item in indices_in)
        indices_out = set(dic.get(item, None) for item in indices_out)
        indices_in.discard(None); indices_out.discard(None)
        self.calculate_components()
        self.check_percolation(indices_in, indices_out)  # have to recalculate the indices_in and _out and return them
        logging.info("...done.")
        return indices_in, indices_out
    # ##################################################################################################################

    # solve and compute methods ########################################################################################
    def get_system_matrix(self):
        logging.info('assembling A...')
        nc, nh = len(self.channels), len(self.hbnds)
        n = 4 * nc + nh
        row, col = np.empty((n,), dtype=np.int), np.zeros((n,), dtype=np.int)
        data = np.empty((n,), dtype=np.double)
        hbnd_ind = np.fromiter(self.hbnds.keys(), dtype=np.int, count=nh)
        data[-nh:] = BND_COND * np.ones((nh,), dtype=np.double)
        row[-nh:] = hbnd_ind
        col[-nh:] = hbnd_ind
        r, c = zip(*self.channels.keys())
        for i, d in enumerate(self.channels.values()):
            cond = d.conductance
            row[i] = r[i]; row[i+nc] = c[i]; row[i+2*nc] = r[i]; row[i+3*nc] = c[i]
            col[i] = r[i]; col[i+nc] = c[i]; col[i+2*nc] = c[i]; col[i+3*nc] = r[i]
            data[i] = cond; data[i+nc] = cond; data[i+2*nc] = -cond; data[i+3*nc] = -cond
        matrix = coo_matrix((data, (row, col)), shape=[len(self.nodes)]*2)
        logging.info('...done')
        return matrix

    def get_system_rhs(self):
        logging.info('assembling b...')
        b = np.zeros_like(self.nodes, dtype=np.double)
        b[np.fromiter(self.hbnds.keys(), dtype=np.int, count=len(self.hbnds))] += BND_COND * \
                                                np.fromiter(self.hbnds.values(), dtype=np.double, count=len(self.hbnds))
        b[np.fromiter(self.qbnds.keys(), dtype=np.int, count=len(self.qbnds))] += np.fromiter(self.qbnds.values(),
                                                                                 dtype=np.double, count=len(self.qbnds))
        logging.info('...done')
        return b

    def solve_steady_state_flow_pyamg(self, tol=1.e-12, CF='RS', full_output=True):
        from pyamg import ruge_stuben_solver
        self.clean_network()
        logging.info('solving sparse system A.x=b ...')
        A = self.get_system_matrix().tocsr()
        M = ruge_stuben_solver(A, CF=CF).aspreconditioner(cycle='V')
        self.heads, info = linalg.cg(A, self.get_system_rhs(), tol=tol, M=M)
        logging.info('...done')
        if full_output:
            self.calculate_extra_output()
        return info

    def solve_steady_state_flow_scipy_direct(self, full_output=True):
        self.clean_network()
        logging.info('solving sparse system A.x=b ...')
        self.heads = linalg.spsolve(self.get_system_matrix().tocsr(), self.get_system_rhs())
        logging.info('...done')
        if full_output:
            self.calculate_extra_output()
        return 0

    def solve_steady_state_flow_petsc(self, nproc=1, tol=1.e-5, maxiter=5000, full_output=True, PETSc=None):
        """
        This method solves the steady state flow problem over the flow network.
        :param tol: float, relative tolerance of the iterative sparse matrix solver
        :param maxiter: int, maximum number of iterations for the sparse matrix solver
        :param full_output: bool, calculates channel_flows and boundary_flows corresponding to the head solution.
        :return: 0 upon successful completion, positive integer values in case of failure.
        """
        self.clean_network()
        matrix, b = self.get_system_matrix().tocsr(), self.get_system_rhs()
        nnz = np.max(matrix.indptr[1:]-matrix.indptr[:-1])

        import sys
        from mpi4py import MPI
        comm = MPI.COMM_SELF.Spawn(sys.executable, args=['../petsc_solver_sequence.py'], maxprocs=nproc)

        int_parameters = np.array([matrix.shape[0], nnz, matrix.nnz, maxiter], 'i')
        comm.Bcast([int_parameters, MPI.INT], root=MPI.ROOT)
        comm.Bcast([matrix.indptr, MPI.INT], root=MPI.ROOT)
        comm.Bcast([matrix.indices, MPI.INT], root=MPI.ROOT)
        comm.Bcast([matrix.data, MPI.DOUBLE], root=MPI.ROOT)
        comm.Bcast([b, MPI.DOUBLE], root=MPI.ROOT)
        comm.Bcast([np.array(tol, dtype='d'), MPI.DOUBLE], root=MPI.ROOT)

        info = np.array(0, dtype='i')
        comm.Reduce(None, [info, MPI.INT], op=MPI.SUM, root=MPI.ROOT)
        if info ==0:
            logging.info('...solver done, successfully')
            result = np.empty((matrix.shape[0],), dtype='d')
            comm.Gather(None, [result, MPI.DOUBLE], root=MPI.ROOT)
            self.heads = result
        else:
            logging.warning('...solver done, no convergence with tolerance %e after %i iterations' % (tol, maxiter))
        comm.Disconnect()

        if info == 0 and full_output:  # calculating extra output
            self.calculate_extra_output()
        return info

    def calculate_extra_output(self):
        logging.info('calculating extra output...')
        if self.heads is None:
            logging.error('run solve_steady_state_flow_scipy or solve_steady_state_flow_petsc or set_heads before calculating' \
                  'channel flows. _calculate_channel_flows call aborted.')
            return
        oldrows, oldcols = np.array(list(zip(*self.channels.keys())), dtype=np.int)
        cond = np.fromiter((item.conductance for item in self.channels.values()), dtype=np.double, count=len(self.channels))
        channel_flows = np.abs(self.heads[oldrows] - self.heads[oldcols]) * cond
        for i, cf in enumerate(channel_flows):
            self.channels[(oldrows[i], oldcols[i])].flow = cf
        rows = np.where(self.heads[oldrows] > self.heads[oldcols], oldrows, oldcols)  # reorder so flow>=0 is row -> col
        cols = np.where(self.heads[oldrows] > self.heads[oldcols], oldcols, oldrows)
        node_inflows = np.bincount(cols, weights=channel_flows, minlength=len(self.nodes))
        node_outflows = np.bincount(rows, weights=channel_flows, minlength=len(self.nodes))
        self.node_throughflows = np.vstack((node_inflows, node_outflows)).max(axis=0)
        self.boundary_flows = node_outflows - node_inflows
        logging.info('...done')

    def get_total_inflow(self):
        if self.boundary_flows is not None:
            return np.sum(self.boundary_flows[self.boundary_flows > 0.])
        else:
            logging.warning("flow simulation was not run or did not converge, get_total_inflow aborted.")
    
    def get_total_outflow(self):
        if self.boundary_flows is not None:
            return np.sum(self.boundary_flows[self.boundary_flows < 0.])
        else:
            logging.warning("flow simulation was not run or did not converge, get_total_outflow aborted.")

    def get_inflow_nodes_indices(self, eps=1.e-8):
        return np.arange(len(self.nodes))[self.boundary_flows > eps]

    def get_outflow_nodes_indices(self, eps=1.e-8):
        return np.arange(len(self.nodes))[self.boundary_flows < - eps]
    # ##################################################################################################################

    # reordering, topological ordering etc... ##########################################################################
    def set_channel_lengths(self, length=None, tort=1.):
        if length is None:
                for chan in self.channels.keys():
                    self.channels[chan].length = tort * np.linalg.norm(self.nodes[chan[0]]() - self.nodes[chan[1]]())
        else:
            try:
                for chan in self.channels.keys():
                    self.channels[chan].length = length[chan]
            except TypeError:
                for chan in self.channels.keys():
                    self.channels[chan].length = length

    def set_channel_widths(self, width):
        try:
            for chan in self.channels.keys():
                self.channels[chan].width = width[chan]
        except TypeError:
            for chan in self.channels.keys():
                self.channels[chan].width = width

    def set_channel_apertures(self, aperture=None, from_extended_cubic_law=True):
        if aperture is None and from_extended_cubic_law:
            for chan in self.channels.values():
                chan.set_aperture_from_extended_cubic_law()
        else:
            try:
                for chan in self.channels.keys():
                    self.channels[chan].aperture = aperture[chan]
            except TypeError:
                for chan in self.channels.keys():
                    self.channels[chan].aperture = aperture

    def reduce_simple_links(self, nodes_to_keep=None, nodes_to_convert=None):  # not tested yet
        """This method contracts all vertices of order 2 if they are not tagged in the nodes_to_keep argument.
        This method will re-index the nodes. As a commodity you can pass a list of indices to convert during the
        re-indexing."""
        nodes_to_keep = set(nodes_to_keep)
        nodes_to_keep.update(self.hbnds.keys())
        nodes_to_keep.update(self.qbnds.keys())
        mat = dok_matrix((len(self.nodes), len(self.nodes)), dtype=np.int)
        for k in self.channels.keys():
            mat[k[0], k[1]] = 1
            mat[k[1], k[0]] = 1
        for n in nodes_to_keep:
            mat[n, n] = 4
        mat = mat.tocsr()
        ind, ptr = mat.indices, mat.indptr
        c = mat.sum(axis=1).A1
        not_visited = set(n for n, i in enumerate(c) if i == 2)
        while len(not_visited) > 0:
            i0 = not_visited.pop()
            r1, r2 = ind[ptr[i0]:ptr[i0+1]]
            l1, l2 = [i0, r1], [i0, r2]
            while r1 in not_visited:
                not_visited.remove(r1)
                next_row = ind[ptr[r1]:ptr[r1+1]]
                r1 = next_row[next_row != l1[-2]][0]
                l1.append(r1)
            while r2 in not_visited:
                not_visited.remove(r2)
                next_row = ind[ptr[r2]:ptr[r2 + 1]]
                r2 = next_row[next_row != l2[-2]][0]
                l2.append(r2)
            l = l1[-1:0:-1] + l2
            links = [(min(j, l[i]), max(j, l[i])) for i, j in enumerate(l[1:])]
            cond = 1. / np.sum([1. / self.channels[l].conductance for l in links])
            leng = np.sum([self.channels[l].length for l in links])
            widt = np.average([self.channels[l].width for l in links], weights=[self.channels[l].length for l in links])
            aper = np.sum([self.channels[l].vol() for l in links]) / widt / leng
            self.add_channel(l[0], l[-1], Channel(cond, leng, widt, aper))
            self.remove_channels(links)
        dic = self.clean_network(return_dic=True)
        return [dic.get(item, None) for item in nodes_to_convert]

        # c = Counter([i for t in self.channels.keys() for i in t])
        # for n in nodes_to_keep:     c[n] = 0
        # for n in self.hbnds.keys(): c[n] = 0
        # for n in self.qbnds.keys(): c[n] = 0
        # endlinks = set(k for k in self.channels.keys() if ((c[k[0]]!=2 and c[k[1]]==2) or (c[k[0]]==2 and c[k[1]]!=2)))
        # midlinks = set(k for k in self.channels.keys() if (c[k[0]]==2) or c[k[1]]==2)
        # to_disconnect = set(i for t in midlinks for i in t)
        # print(len(midlinks))
        # while len(midlinks) > 0:
        #     links = [endlinks.pop()]
        #     midlinks.remove(links[0])
        #     if c[links[0][0]] == 2:    l0, l1 = links[0][1], links[0][0]
        #     else:   l0, l1 = links[0][0], links[0][1]
        #     while c[l1] == 2:
        #         next = midlinks.intersection(set((min(l1, i), max(l1, i)) for i in to_disconnect)).pop()
        #         midlinks.remove(next)
        #         links.append(next)
        #         if links[-1][0] in links[-2]: l1 = links[-1][1]
        #         else:   l1 = links[-1][0]
        #         if c[l1] == 2:  to_disconnect.remove(l1)  # trying to accelerate the operation as we go
        #     endlinks.remove(links[-1])
        #     # insert new channel and delete old ones here
        #     cond = 1. / np.sum([1. / self.channels[l].conductance for l in links])
        #     leng = np.sum([self.channels[l].length for l in links])
        #     widt = np.average([self.channels[l].width for l in links], weights=[self.channels[l].length for l in links])
        #     aper = np.sum([self.channels[l].vol() for l in links]) / widt / leng
        #     self.add_channel(l0, l1, Channel(cond, leng, widt, aper))
        #     self.remove_channels(links)
        #     ########################
        # assert len(endlinks) == 0
        # dic = self.clean_network(return_dic=True)
        # return [dic.get(item, None) for item in nodes_to_convert]

    def reorder2topological_ordering(self):
        """
        This method sorts the nodes (and node-based property) in decreasing order of heads following the flow solution.
        This gives a topological partial order over the flow graph
        :return: 2 utility functions to help transform indices and arrays backwards and forwards, in order:
            - original2topo(ind): useful to transform old node indices into new ones
            - topo2original(ind): inverse effect compared to the previous function
        """
        logging.info("Reordering self to flow topological order...")
        ifwd = np.argsort(self.heads)[-1::-1]  # obtaining a topological order on the DAG of the flowing self
        ibkd = np.argsort(ifwd)  # and the conversion back to original ordering
        self.channels = {(min(ibkd[k[0]],ibkd[k[1]]), max(ibkd[k[0]],ibkd[k[1]])): v for k, v in self.channels.items()}

        self.nodes = [self.nodes[i] for i in ifwd]
        self.heads = self.heads[ifwd]
        self.hbnds = {ifwd[k]: self.hbnds[k] for k in self.hbnds.keys()}
        self.qbnds = {ifwd[k]: self.qbnds[k] for k in self.qbnds.keys()}
        if self.boundary_flows is not None:
            self.boundary_flows = self.boundary_flows[ifwd]
        if self.node_throughflows is not None:
            self.node_throughflows = self.node_throughflows[ifwd]
        if self.components is not None:
            self.components = self.components[ifwd]
        if self.percolating_components is not None:
            self.percolating_components = (ibkd[self.percolating_components[0]], ibkd[self.percolating_components[1]],
                                           self.percolating_components[2])

        def orig2topo(ind, ibkd=ibkd):  # ibkd is passed as parameter to save its value at function creation
            return ibkd[ind]

        def topo2orig(ind, ifwd=ifwd):  # see comment above
            return ifwd[ind]

        logging.info("...done.")
        return orig2topo, topo2orig
    ####################################################################################################################

    # Save and export methods ##########################################################################################
    def save_network(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.nodes, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.channels, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.hbnds, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.qbnds, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.heads, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.node_throughflows, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.boundary_flows, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.components, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.n_components, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.percolating_components, f, protocol=pickle.HIGHEST_PROTOCOL)

    def export2vtk(self, filename):
        """This method call saves existing fields into a vtk file for visualization in e.g Paraview.
        This function has a dependency on VTK and the Python bindings vtk."""
        logging.info('exporting network to VTK format...')
        try:
            import vtk
            from vtk.util import numpy_support
        except ImportError:
            logging.warning('Package VTK cannot be found, abort export to VTK format.')
            return
        self.clean_network()

        points = vtk.vtkPoints()  # node-based quantities
        for n, node in enumerate(self.nodes):
            points.InsertPoint(n, node.x, node.y, node.z)
        if self.heads is not None:
            head = numpy_support.numpy_to_vtk(num_array=self.heads.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
            head.SetName("h (m)")
        if self.node_throughflows is not None:
            throughflows = numpy_support.numpy_to_vtk(self.node_throughflows.ravel(), True, array_type=vtk.VTK_FLOAT)
            throughflows.SetName("Node throughflows (m**3/s)")
        if self.boundary_flows is not None:
            boundary_flows = numpy_support.numpy_to_vtk(self.boundary_flows.ravel(), True, array_type=vtk.VTK_FLOAT)
            boundary_flows.SetName("Local boundary flows (m**3/s)")
        if self.components is not None:
            components = numpy_support.numpy_to_vtk(self.components.ravel(), True, array_type=vtk.VTK_INT)
            components.SetName("Connected components (id)")

        lines = vtk.vtkCellArray()  # channel-based quantities
        conductances, lengths, widths = vtk.vtkDoubleArray(), vtk.vtkDoubleArray(), vtk.vtkDoubleArray()
        apertures, flows = vtk.vtkDoubleArray(), vtk.vtkDoubleArray()
        conductances.SetName("C (m**2/s)")
        lengths.SetName("L (m)")
        widths.SetName("W (m)")
        apertures.SetName("b (m)")
        flows.SetName("Q (m**3/s)")
        for k, v in self.channels.items():
            lines.InsertNextCell(2)
            lines.InsertCellPoint(k[0])
            lines.InsertCellPoint(k[1])
            conductances.InsertNextValue(v.conductance)
            lengths.InsertNextValue(v.length)
            widths.InsertNextValue(v.width)
            apertures.InsertNextValue(v.aperture)
            flows.InsertNextValue(v.flow)

        profile = vtk.vtkPolyData()
        profile.SetPoints(points)  # node-based data
        if self.heads is not None:
            profile.GetPointData().AddArray(head)
        if self.node_throughflows is not None:
            profile.GetPointData().AddArray(throughflows)
        if self.boundary_flows is not None:
            profile.GetPointData().AddArray(boundary_flows)
        if self.components is not None:
            profile.GetPointData().SetScalars(components)
        profile.SetLines(lines)  # channel-based data
        profile.GetCellData().SetScalars(conductances)
        profile.GetCellData().AddArray(lengths)
        profile.GetCellData().AddArray(widths)
        profile.GetCellData().AddArray(apertures)
        profile.GetCellData().AddArray(flows)
            
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filename + ".vtp")
        try:
            writer.SetInputData(profile)
        except AttributeError:
            writer.SetInput(profile)
        writer.Write()
        logging.info('...done')
        # ##############################################################################################################

if __name__ == "__main__":
    nodes = [Node(0., 0., 0.), Node(1., 0., 0.), Node(1., 1., 0.), Node(0., 1., 0.)]
    channels = {(0, 1): Channel(1.e-8), (1, 2): Channel(1.e-9), (2, 3): Channel(1.e-10), (3, 0): Channel(1.e-7)}
    hbnds = {0: 1.}
    qbnds = {1: -1.e-9}
    N = Network(nodes=nodes, channels=channels, hbnds=hbnds, qbnds=qbnds)
    assert len(N.nodes) == 4
    assert len(N.channels) == 4
    assert len(N.hbnds) == 1
    assert len(N.qbnds) == 1

    N.add_channel(0, 2, Channel(1.e-8))
    assert len(N.channels) == 5

    N.remove_node(3)
    assert len(N.nodes) == 3
    assert len(N.channels) == 3

    N.add_node(Node(1., 0., 0.))
    assert len(N.nodes) == 4

    N.add_channels({(3, 0): Channel(1.e-7), (2, 3): Channel(1.e-10)})
    assert len(N.channels) == 5

    N.add_nodes([Node(1., 1., 1.), Node(1., 1., -1.)])
    assert len(N.nodes) == 6

    N.disconnect_nodes([4, 5])
    assert len(N.nodes) == 6

    N.clean_network()
    assert len(N.nodes) == 4

    N.remove_channels({(0, 3), (2, 3)})
    assert len(N.nodes) == 4
    assert len(N.channels) == 3

    N.calculate_components()
    print(N.n_components, N.components)
    N.remove_non_percolating_channels([0], [2])

    from pychan3d import LatticeNetwork
    LN1 = LatticeNetwork(nx=4, ny=0, nz=4, dx=1., dy=1, dz=1., mean_logcx=-7.)
    print(LN1)
    LN1.export2vtk('LN1')
    LN2 = LatticeNetwork(nx=2, ny=0, nz=2, dx=2., dy=1., dz=2., mean_logcx=-8.) #, offset=np.array([0., 1., 0.]))
    print(LN2)
    LN2.export2vtk('LN2')
    print(LN1 + LN2)

    print('passed.')
