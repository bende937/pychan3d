import numpy as np
import pychan3d.core_network as cn
import pickle
import logging


class SparseLatticeNetwork(cn.Network):
    def __init__(self, nx=100, ny=100, nz=100, d=1., PAx=0.9, PONx=0.08, mean_logcx=-10., sigma_logcx=0.,
            PAy=None, PONy=None, mean_logcy=None, sigma_logcy=0., PAz=None, PONz=None, mean_logcz=None, sigma_logcz=0.,
                                                                     offset=np.zeros((3,)), seed=987654321, initial_state=None):
        super(SparseLatticeNetwork, self).__init__()  # all attributes default to empty first, then get populated
        logging.info('Generating conductance field ...')
        if seed is None: prng = np.random.RandomState()
        else: prng = np.random.RandomState(seed)
        if initial_state is not None: prng.set_state(initial_state)
        self.initial_state = prng.get_state()

        self.nx, self.ny, self.nz, self.d, self.offset = nx, ny, nz, d, offset
        self.PAx, self.PONx, self.mean_logcx, self.sigma_logcx = PAx, PONx, mean_logcx, sigma_logcx
        if PAx != 1. and PONx != 0.:
            PNx = (PAx - 1.) / (1. - 1./PONx)
        elif PAx == 1.:
            PNx = 0.
        elif PONx == 0. and PAx == 0.:
            PNx = 0.
        else:
            PNx = 0.
        self.PNx = PNx
        
        if None in (PAy, PONy, mean_logcy, PAz, PONz, mean_logcz): 
            logging.info('incomplete 3D parametrization of conductances: assuming PAx, PONx, mean_logcx and sigma_logcx'
                         ' apply in all three directions')
            self.PAy, self.PONy, self.mean_logcy, self.sigma_logcy = PAx, PONx, mean_logcx, sigma_logcx
            self.PAz, self.PONz, self.mean_logcz, self.sigma_logcz = PAx, PONx, mean_logcx, sigma_logcx
            self.PNy, self.PNz = self.PNx, self.PNx
        else:
            logging.info('complete 3D parametrization of conductances: PA, PON, mean_logc and sigma_logc are decoupled'
                         ' in all three directions')
            self.PAy, self.PONy, self.mean_logcy, self.sigma_logcy = PAy, PONy, mean_logcy, sigma_logcy
            if PAy != 1.:
                PNy = (PAy-1.)/(1.-1./PONy)
            elif PAy == 1.:
                PNy = 0.
            elif PONy == 0. and PAy == 0.:
                PNy = 0.
            else:
                PNy = 0.
            self.PNy = PNy
            self.PAz, self.PONz, self.mean_logcz, self.sigma_logcz = PAz, PONz, mean_logcz, sigma_logcz
            if PAz != 1.:
                PNz = (PAz-1.)/(1.-1./PONz)
            elif PAz == 1.:
                PNz = 0.
            elif PONz == 0. and PAz == 0.:
                PNz = 0.
            else:
                PNz = 0.
            self.PNz = PNz

        self.nodes = [cn.Node(i * self.d + offset[0], j * self.d + offset[1], k * self.d + offset[2]) \
                      for k in range(self.nz + 1) for j in range(self.ny + 1) for i in range(self.nx + 1)]

        nz1 = nz + 1
        nz1ny1 = nz1 * (ny+1)
        def pos(i, j, k):
            return k + nz1 * j + nz1ny1 * i

        dicx, sample_x = set(), prng.rand(self.nx, self.ny + 1, self.nz + 1)
        for j in range(self.ny + 1):
            for k in range(self.nz + 1):
                previous = False
                if sample_x[0, j, k] < self.PONx:
                    dicx.add((pos(0,j,k), pos(1,j,k)))
                    previous = True
                for i in range(1, self.nx):
                    if (previous is True and sample_x[i, j, k] < self.PAx) or \
                            (previous is False and sample_x[i, j, k] < self.PNx):
                        dicx.add((pos(i,j,k), pos(i+1,j,k)))
                        previous = True
                    else:
                        previous = False
        if self.sigma_logcx == 0.:
            xvalues = [cn.Channel(10 ** self.mean_logcx) for i in range(len(dicx))]
        else:
            xvalues = [cn.Channel(10**c) for c in prng.normal(self.mean_logcx, self.sigma_logcx, size=len(dicx))]
        self.channels = {k: xvalues[n] for n, k in enumerate(dicx)}
        
        dicy, sample_y = set(), prng.rand(self.nx + 1, self.ny, self.nz + 1)
        for i in range(self.nx + 1):
            for k in range(self.nz + 1):
                previous = False
                if sample_y[i, 0, k] < self.PONy:
                    dicy.add((pos(i, 0, k), pos(i, 1, k)))
                    previous = True
                for j in range(1, self.ny):
                    if (previous is True and sample_y[i, j, k] < self.PAy) or \
                            (previous is False and sample_y[i, j, k] < self.PNy):
                        dicy.add((pos(i, j, k), pos(i, j + 1, k)))
                        previous = True
                    else:
                        previous = False
        if self.sigma_logcy == 0.:
            yvalues = [cn.Channel(10 ** self.mean_logcy) for i in range(len(dicy))]
        else:
            yvalues = [cn.Channel(10**c) for c in prng.normal(self.mean_logcy, self.sigma_logcy, size=len(dicy))]
        self.channels.update({k: yvalues[n] for n, k in enumerate(dicy)})
        
        dicz, sample_z = set(), prng.rand(self.nx + 1, self.ny + 1, self.nz)
        for i in range(self.nx + 1):
            for j in range(self.ny + 1):
                previous = False
                if sample_z[i, j, 0] < self.PONz:
                    dicz.add((pos(i, j, 0), pos(i, j, 1)))
                    previous = True
                for k in range(1, self.nz):
                    if (previous is True and sample_z[i, j, k] < self.PAz) or \
                            (previous is False and sample_z[i, j, k] < self.PNz):
                        dicz.add((pos(i, j, k), pos(i, j, k + 1)))
                        previous = True
                    else:
                        previous = False
        if self.sigma_logcz == 0.:
            zvalues = [cn.Channel(10 ** self.mean_logcz) for i in range(len(dicz))]
        else:
            zvalues = [cn.Channel(10**c) for c in prng.normal(self.mean_logcz, self.sigma_logcz, size=len(dicz))]
        self.channels.update({k: zvalues[n] for n, k in enumerate(dicz)})
        
        self.clean_network()

    def get_Xminus_indices(self, eps=1.e-3):
        return np.arange(len(self.nodes))[np.array([p.x < self.offset[0] + eps for p in self.nodes])]

    def get_Xplus_indices(self, eps=1.e-3):
        return np.arange(len(self.nodes))[np.array([p.x > self.offset[0]+self.nx*self.d - eps for p in self.nodes])]

    def get_Yminus_indices(self, eps=1.e-3):
        return np.arange(len(self.nodes))[np.array([p.y < self.offset[1] + eps for p in self.nodes])]

    def get_Yplus_indices(self, eps=1.e-3):
        return np.arange(len(self.nodes))[np.array([p.y > self.offset[1]+self.ny*self.d - eps for p in self.nodes])]

    def get_Zminus_indices(self, eps=1.e-3):
        return np.arange(len(self.nodes))[np.array([p.z < self.offset[2] + eps for p in self.nodes])]

    def get_Zplus_indices(self, eps=1.e-3):
        return np.arange(len(self.nodes))[np.array([p.z > self.offset[2]+self.nz*self.d - eps for p in self.nodes])]

    def set_Hboundary_Xminus(self, h, eps=1.e-3):
        for i in self.get_Xminus_indices(eps=eps):
            self.hbnds[i] = h
        
    def set_Hboundary_Xplus(self, h, eps=1.e-3):
        for i in self.get_Xplus_indices(eps=eps):
            self.hbnds[i] = h
    
    def set_Hboundary_Yminus(self, h, eps=1.e-3):
        for i in self.get_Yminus_indices(eps=eps):
            self.hbnds[i] = h
        
    def set_Hboundary_Yplus(self, h, eps=1.e-3):
        for i in self.get_Yplus_indices(eps=eps):
            self.hbnds[i] = h
        
    def set_Hboundary_Zminus(self, h, eps=1.e-3):
        for i in self.get_Zminus_indices(eps=eps):
            self.hbnds[i] = h
        
    def set_Hboundary_Zplus(self, h, eps=1.e-3):
        for i in self.get_Zplus_indices(eps=eps):
            self.hbnds[i] = h

    def save_network(self, filename, as_unstructured=False):
        if as_unstructured:
            super(SparseLatticeNetwork, self).save_network(filename)
        else:
            protocol = 0  # pickle.HIGHEST_PROTOCOL
            with open(filename, 'wb') as f:
                pickle.dump(self.nx, f, protocol=protocol)
                pickle.dump(self.ny, f, protocol=protocol)
                pickle.dump(self.nz, f, protocol=protocol)
                pickle.dump(self.d, f, protocol=protocol)
                pickle.dump(self.offset, f, protocol=protocol)
                pickle.dump(self.PAx, f, protocol=protocol)
                pickle.dump(self.PONx, f, protocol=protocol)
                pickle.dump(self.PAy, f, protocol=protocol)
                pickle.dump(self.PONy, f, protocol=protocol)
                pickle.dump(self.PAz, f, protocol=protocol)
                pickle.dump(self.PONz, f, protocol=protocol)
                pickle.dump(self.mean_logcx, f, protocol=protocol)
                pickle.dump(self.mean_logcy, f, protocol=protocol)
                pickle.dump(self.mean_logcz, f, protocol=protocol)
                pickle.dump(self.sigma_logcx, f, protocol=protocol)
                pickle.dump(self.sigma_logcy, f, protocol=protocol)
                pickle.dump(self.sigma_logcz, f, protocol=protocol)
                pickle.dump(self.initial_state, f, protocol=protocol)
                pickle.dump(self.hbnds, f, protocol=protocol)
                pickle.dump(self.qbnds, f, protocol=protocol)
                pickle.dump(self.heads, f, protocol=protocol)
