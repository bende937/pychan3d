import numpy as np
import pychan3d.core_network as cn
import pickle
import logging

class LatticeNetwork(cn.Network):
    def __init__(self, nx=20, ny=10, nz=10, dx=1., dy=1., dz=1., mean_logcx=-10., mean_logcy=None, mean_logcz=None,
                 sigma_logcx=0., sigma_logcy=None, sigma_logcz=None, offset=np.zeros((3,)), initial_state=None, seed=None):
        super(LatticeNetwork, self).__init__()

        if seed is None:
            self.prng = np.random.RandomState()  # for independent prngs
        else:
            self.prng = np.random.RandomState(seed)  # if you need to repeat the same prng
        if initial_state is not None:
            self.prng.set_state(initial_state)
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.mean_logcx, self.sigma_logcx = mean_logcx, sigma_logcx,
        if mean_logcy is None: self.mean_logcy = mean_logcx
        else: self.mean_logcy = mean_logcy
        if mean_logcz is None: self.mean_logcz = mean_logcx
        else: self.mean_logcz = mean_logcz
        if sigma_logcy is None: self.sigma_logcy = sigma_logcx
        else: self.sigma_logcy = sigma_logcy
        if sigma_logcz is None: self.sigma_logcz = sigma_logcx
        else: self.sigma_logcz = sigma_logcz
        self.offset, self.initial_state = offset, self.prng.get_state()

        self.nodes = [cn.Node(i*self.dx+offset[0], j*self.dy+offset[1], k*self.dz+offset[2]) \
                                        for i in range(self.nx + 1) for j in range(self.ny+1) for k in range(self.nz+1)]

        i, j, k = np.ogrid[0:self.nx+1, 0:self.ny+1, 0:self.nz+1]
        nz1 = self.nz + 1
        nz1ny1 = nz1 * (self.ny + 1)

        xkeys = [(pos, pos + nz1ny1) for pos in (k + nz1 * j + nz1ny1 * i[:-1, :, :]).flatten()]
        if self.sigma_logcx == 0.:
            xvalues = [cn.Channel(10**self.mean_logcx) for i in range(len(xkeys))]
        else:
            xvalues = [cn.Channel(10**c) for c in self.prng.normal(loc=self.mean_logcx, scale=self.sigma_logcx, size=len(xkeys))]
        self.channels = {xkeys[k]: xvalues[k] for k in range(len(xkeys))}

        ykeys = [(pos, pos + nz1) for pos in (k + nz1 * j[:, :-1, :] + nz1ny1 * i).flatten()]
        if self.sigma_logcy == 0.:
            yvalues = [cn.Channel(10 ** self.mean_logcy) for i in range(len(ykeys))]
        else:
            yvalues = [cn.Channel(10**c) for c in self.prng.normal(loc=self.mean_logcy, scale=self.sigma_logcy, size=len(ykeys))]
        self.channels.update({ykeys[k]: yvalues[k] for k in range(len(ykeys))})

        zkeys = [(pos, pos + 1) for pos in (k[:, :, :-1] + nz1 * j + nz1ny1 * i).flatten()]
        if self.sigma_logcz == 0.:
            zvalues = [cn.Channel(10 ** self.mean_logcz) for i in range(len(zkeys))]
        else:
            zvalues = [cn.Channel(10**c) for c in self.prng.normal(loc=self.mean_logcz, scale=self.sigma_logcz, size=len(zkeys))]
        self.channels.update({zkeys[k]: zvalues[k] for k in range(len(zkeys))})

        self.hbnds = {}
        self.qbnds = {}

    def get_Xminus_indices(self, eps=1.e-3):
        return np.arange(len(self.nodes))[np.array([p.x < self.offset[0] + eps for p in self.nodes])]

    def get_Xplus_indices(self, eps=1.e-3):
        return np.arange(len(self.nodes))[
            np.array([p.x > self.offset[0] + self.nx * self.dx - eps for p in self.nodes])]

    def get_Yminus_indices(self, eps=1.e-3):
        return np.arange(len(self.nodes))[np.array([p.y < self.offset[1] + eps for p in self.nodes])]

    def get_Yplus_indices(self, eps=1.e-3):
        return np.arange(len(self.nodes))[np.array([p.y > self.offset[1] + self.ny * self.dy - eps for p in self.nodes])]

    def get_Zminus_indices(self, eps=1.e-3):
        return np.arange(len(self.nodes))[np.array([p.z < self.offset[2] + eps for p in self.nodes])]

    def get_Zplus_indices(self, eps=1.e-3):
        return np.arange(len(self.nodes))[np.array([p.z > self.offset[2] + self.nz * self.dz - eps for p in self.nodes])]

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
            super(LatticeNetwork, self).save_network(filename)
        else:
            protocol = pickle.HIGHEST_PROTOCOL
            with open(filename, 'wb') as f:
                pickle.dump(self.nx, f, protocol=protocol)
                pickle.dump(self.ny, f, protocol=protocol)
                pickle.dump(self.nz, f, protocol=protocol)
                pickle.dump(self.dx, f, protocol=protocol)
                pickle.dump(self.dy, f, protocol=protocol)
                pickle.dump(self.dz, f, protocol=protocol)
                pickle.dump(self.offset, f, protocol=protocol)
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


if __name__ == "__main__":
    dx, dy, dz = 10., 10., 10.
    nx, ny, nz = 30, 10, 10
    logC, sigma = -6., 1.
    LN = LatticeNetwork(nx, ny, nz, dx, dy, dz, logC, logC, logC, sigma, sigma, sigma, offset=[-150., -50., -50.], seed=987654321)
    LN.export2vtk('test_LN')