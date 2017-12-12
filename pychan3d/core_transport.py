from .core_network import Network
from scipy.sparse import coo_matrix
from scipy.stats import rv_discrete, rv_continuous
from scipy.interpolate import interp1d
from scipy.special import erfcinv
import numpy as np
import logging
logging.basicConfig(filename='log_network', format='%(asctime)s: %(message)s', level=logging.INFO)


class CustomTimeInjection(rv_continuous):
    """This class provides a continuous random variable to describe the injection time of a particle given an injection
    following a custom (interpolated) concentration profile, i.e obtained by field measurements."""
    def __init__(self, t, c, kind='linear'):
        """
        :param t: a list or array of times corresponding to the concentrations in param "c" (no duplicates are allowed
        in param t or the call will fail).
        :param c: a list or array of concentrations applied at the times contained in param "t"
        :param kind: the kind of interpolation between given points (default is 'linear'). !Warning! if you over-ride
        the default some update will be required to ensure an exact normalization of the curve in self.set_interpolator
        and in self._cdf
        """
        super(CustomTimeInjection, self).__init__()
        if len(set(t)) != len(list(t)):  # test that there is no duplicate in t
            logging.error('Error! There are duplicates in the time array.')
            raise ValueError
        if len(list(t)) != len(list(c)):
            logging.error('Error! Arguments t and c are of different lengths.')
            raise ValueError
        order = np.argsort(t)  # sorting the time and concentration arrays
        self.t, self.c = np.array(t)[order], np.array(c)[order]
        self.norm = 0.5 * np.sum((self.c[:-1] + self.c[1:]) * (self.t[1:] - self.t[:-1]))  # calculating the total mass
        # assuming linear interpolation between points, update self.norm and the _cdf method if you pass other arguments
        # to the 'kind' argument.
        self.interpolator = interp1d(self.t, self.c / self.norm, fill_value=0., bounds_error=False, kind=kind)

    def _pdf(self, x):
        """Custom definition of the probability density function, using the interpolator.
        :param x: time
        :return: probability"""
        return self.interpolator(x)

    def _cdf(self, x):
        """Fast custom definition of cdf (not using self._pdf repeatedly but assuming linear interpolation).
        :param x: time
        :return: accumulated probability"""
        n = np.sum(x > self.t)  # fetching the index of the current value in the time array, to sum up to it.
        newt, newc = np.hstack([self.t[:n], x]), np.hstack([self.c[:n] / self.norm, self.interpolator(x)])
        return 0.5 * np.sum((newc[:-1] + newc[1:]) * (newt[1:] - newt[:-1]))

    #def _parse_args_rvs(self,*args, **kwds):
    #    return args, 0, 1, 1

class ArrivalTimeDistribution(rv_continuous):
    """This class provides a continuous random variable to describe the injection time of a particle given an injection
    following a custom (interpolated) concentration profile, i.e obtained by field measurements."""
    def __init__(self, t, kind='linear'):
        """
        :param t: a list or array of times corresponding to the concentrations in param "c" (no duplicates are allowed
        in param t or the call will fail).
        :param kind: the kind of interpolation between given points (default is 'linear'). !Warning! if you over-ride
        the default some update will be required to ensure an exact normalization of the curve in self.set_interpolator
        and in self._cdf
        """
        super(ArrivalTimeDistribution, self).__init__()
        if len(set(t)) != len(list(t)):  # test that there is no duplicate in t
            logging.error('Error! There are duplicates in the time array.')
            raise ValueError
        self.t = np.sort(np.array(t))
        self.cumul = (np.arange(self.t.shape[0])) / float(self.t.shape[0] - 1)
        self.interpolator = interp1d(self.t, self.cumul, fill_value=0., bounds_error=False, kind=kind)

    def _cdf(self, x):
        """Fast custom definition of cdf (not using self._pdf repeatedly but assuming linear interpolation).
        :param x: time
        :return: accumulated probability"""
        return self.interpolator(x)


class TransportSimulation(Network):
    """This is the main class to run a transport simulation by particle tracking."""
    def __init__(self, nwk, injection_nodes='all_in', sampling_nodes='all_out', n_particles=10000,
                 spatial_injection_mode='flux', time_injection_mode='instantaneous', c=None, t=None):
        """
        To create a transport simulation, one needs to pass the network in which transport is to be computed. For this,
        the network needs to have been run and have an existing flow solution. The two additional optional arguments are
        :param nwk:
        :param injection_nodes:
        :param sampling_nodes:
        :param n_particles: int, number of particles to track
        :param time_injection_mode: string to define the type of injection: 'instantaneous' or 'interpolated'
        :param c: (optional) if interpolated injection, c is an array of concentration values at times t
        :param t: (optional) the times corresponding to the injection concentration values c
        The interpolation is done using a spline fitting without smoothing.
        :param spatial_injection_mode: string to choose between 'flux' and 'resident'
        """
        super(TransportSimulation, self).__init__(nwk.nodes, nwk.channels, nwk.hbnds, nwk.qbnds)
        if nwk.heads is None:
            logging.error('Run the flow simulation before defining the transport problem. Aborted transport setup.')
            raise ValueError
        else:
            self.heads, self.boundary_flows, self.node_throughflows = nwk.heads,nwk.boundary_flows,nwk.node_throughflows
        self.orig2topo, self.topo2orig = self.reorder2topological_ordering()

        if injection_nodes is 'all_in':
            self.injection_nodes = self.get_inflow_nodes_indices()
        else:
            self.injection_nodes = self.orig2topo(injection_nodes)
        if sampling_nodes is 'all_out':
            self.sampling_nodes = self.get_outflow_nodes_indices()
        else:
            self.sampling_nodes = self.orig2topo(sampling_nodes)
        self.RMM = PerfectMixingModel(self)  # OBS! define after the injection and sampling nodes

        if time_injection_mode == 'instantaneous':
            self.particle_injection_times = np.zeros((n_particles,))
        elif time_injection_mode == 'interpolated':
            p = CustomTimeInjection(t, c)
            self.particle_injection_times = p.rvs(size=n_particles)
        else:
            logging.error("Unknown time_injection_mode, choose between 'instantaneous' and 'interpolated'")

        if spatial_injection_mode == 'resident':
            p = rv_discrete(values=(self.injection_nodes,
                                    [1. / self.injection_nodes.shape[0]] * self.injection_nodes.shape[0]))
            self.particle_injection_nodes = p.rvs(size=n_particles)
        elif spatial_injection_mode == 'flux':
            norm = np.sum(self.node_throughflows[self.injection_nodes])
            p = rv_discrete(values=(self.injection_nodes, self.node_throughflows[self.injection_nodes] / norm))
            self.particle_injection_nodes = p.rvs(size=n_particles)
        else:
            logging.error("Unknown spatial_injection_mode, choose between 'resident' and 'flux'.")

        self.particles = []

    def __str__(self):
        return "Transport simulation in network:\n" + super(TransportSimulation, self).__str__()

    def set_channel_matrix_models(self, channels='all', R=1., MPG=None):
        """
        :param channels: the 2-tuple coordinates of the channels to define
        :param R: the surface retardation coefficient, R=1 means pure advective time, R=0 gigves instant transport, R>1
        gives some retardation by sorption in the channel surface
        :param MPG: the material property group for diffusion and retardation in the rock matrix
        """
        if channels is 'all':
            channels = self.channels.keys()
        for k in channels:
            if MPG is not None:
                self.channels[k].transportModel = InfiniteSingleLayerMatrix(self.channels[k], R, MPG)
            else:
                self.channels[k].transportModel = NoMatrix(self.channels[k], R)

    def run_particle_tracking(self, tmax=1.e12):
        if hasattr(self, 'particle_injection_times'):
            for n, p in enumerate(self.particle_injection_nodes):
                self.particles.append(Particle(self, p, self.particle_injection_times[n]))
        else:
            logging.error('You have to call set_particle_number and injection_times() before you start the particle tracking.')

    def compute_time_distribution(self):
        # sort particles by outlet node
        tracked_times = [p.times[-1] for p in self.particles if p.nodes[-1] in self.sampling_nodes]
        recovery = float(len(tracked_times)) / len(self.particles)
        return ArrivalTimeDistribution(tracked_times), recovery

    def export_particles2vtk(self, filename, times, particles='all'):
        """
        :param filename:
        :param times:
        :param particles:
        """
        try:
            import vtk
            from vtk.util import numpy_support
        except ImportError:
            self.network.add2log(': Package VTK cannot be found, abort export to VTK format.\n')
            return
        if particles == 'all':
            particles = self.particles

        pvd = open(filename+'.pvd', 'w')
        pvd.write("""<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1"
         byte_order="LittleEndian"
         compressor="vtkZLibDataCompressor">
  <Collection>
  """)
        for t, time in enumerate(times):
            logging.info('time=%e'%(time,))
            pvd.write("""<DataSet timestep="%f" group="" part="0" file="%s_%010i.vtp"/>
""" % (time, filename, t))
            points = vtk.vtkPoints()
            for p, particle in enumerate(particles):
                position = particle(time)
                points.InsertPoint(p, position[0], position[1], position[2])
            profile = vtk.vtkPolyData()
            profile.SetPoints(points)
            ids = numpy_support.numpy_to_vtk(num_array=np.arange(len(particles)), deep=True, array_type=vtk.VTK_INT)
            profile.GetPointData().SetScalars(ids)
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(filename + "_%010i.vtp" % (t,))
            try:
                writer.SetInputData(profile)
            except:
                writer.SetInput(profile)
            writer.Write()
        pvd.write("""</Collection>
</VTKFile>
""")
        pvd.close()


class Particle(object):
    def __init__(self, trsim, inode, itime):
        self.trsim = trsim
        self.nodes = trsim.RMM(inode)  # this call determines the nodes through which the particle will pass
        self.times = [itime]  # the next line samples the time spent in each visited channel
        self.times += [self.trsim.channels[n, self.nodes[i+1]].transportModel() for i, n in enumerate(self.nodes[:-1])]
        self.times = np.cumsum(self.times)  # this call consolidates everything as times of passage after t=0

    def __str__(self):
        return 'Particle going through %i nodes in %f seconds.' % (len(self.nodes), self.times[-1])

    def __call__(self, time):
        """This method gives the interpolated particle position at the requested time."""
        pos = np.searchsorted(self.times, time)  # if pos=0, particle is at node 0, else pos-1 is last passed node, and pos is the next node to pass
        if pos == 0:  # particle has not been injected yet
            return self.trsim.nodes[self.nodes[0]]()
        elif pos >= len(self.times):  # particle has already left the system
            return self.trsim.nodes[self.nodes[-1]]()
        else:
            n1 = self.trsim.nodes[self.nodes[pos-1]]()
            n2 = self.trsim.nodes[self.nodes[pos]]()
            w1 = time - self.times[pos-1]
            w2 = self.times[pos] - time
            return (w1 * n2 + w2 * n1) / (w1 + w2)  # position of particle at time "time"


class RoutingMixingModel(object):  # generic mother class
    def __init__(self, transport_simulation):
        self.trsim = transport_simulation

    def __str__(self):
        return 'RoutingMixingModel of class %s over network: %s.' % (str(self.__class__), str(self.trsim.network))


class PerfectMixingModel(RoutingMixingModel):
    def __init__(self, transport_simulation):
        super(PerfectMixingModel, self).__init__(transport_simulation)
        n = len(self.trsim.nodes)
        rows, cols = zip(*self.trsim.channels.keys())
        self.P = coo_matrix((np.array([item.flow for item in self.trsim.channels.values()]) / self.trsim.node_throughflows[list(rows)],
                        (rows, cols)), shape=(n, n)).tocsr()  # P[n, :] are the routing probabilities at node n
        self.routers = {i: rv_discrete(values=(list(self.P[i, :].indices)+[-1],
                                               list(self.P[i, :].data)+[1.-self.P[i, :].data.sum()])) for i in range(n)}
        # for i in range(n):
        #     line = self.P[i, :]
        #     if line.indices.shape[0] > 0:
        #         self.routers[i] = rv_discrete(values=(list(line.indices)+[-1], list(line.data)+[1.-line.data.sum()]))

    def __call__(self, inode):
        node_list = []
        while inode is not -1:
            node_list.append(inode)
            # inode = self.routers.get(inode, rv_discrete(values=([-1],[1.]))).rvs()  #samples the discrete random variate
            inode = self.routers.get(inode, rv_discrete(values=([-1],[1.]))).rvs()  #samples the discrete random variate
        return node_list

    def estimate_tracer_recovery(self, spatial_injection_mode='flux'):
        tracer_recovery = np.zeros_like(self.trsim.nodes)
        if spatial_injection_mode == 'resident':
            tracer_recovery[self.trsim.injection_nodes] = 1. / float(self.trsim.injection_nodes.shape[0])
        elif spatial_injection_mode == 'flux':
            norm = np.sum(self.trsim.node_throughflows[self.trsim.injection_nodes])
            tracer_recovery[self.trsim.injection_nodes] = self.trsim.node_throughflows[self.trsim.injection_nodes] /norm
        else:
            logging.error("Unknown spatial_injection_mode, currently implemented types are 'resident' and 'flux'")

        for n in range(self.trsim.heads.shape[0]):  # Checking routing from injection nodes to sampling nodes
            line = self.P[n, :]
            tracer_recovery[line.indices] += tracer_recovery[n] * line.data
        recovery_fraction = np.sum(tracer_recovery[self.trsim.sampling_nodes])
        logging.info("The estimated mass recovery is %f percent." % (recovery_fraction * 100., ))
        return recovery_fraction


class MatrixModel(object):  # generic mother class
    def __init__(self, channel):
        self.channel = channel

    def __str__(self):
        return 'MatrixModel of class %s over network: %s.' % (str(self.__class__), str(self.channel))


class NoMatrix(MatrixModel):
    def __init__(self, channel, R):
        super(NoMatrix, self).__init__(channel)
        self. R = R

    def __call__(self):
        return self.R * self.channel.tadv()


class InfiniteSingleLayerMatrix(MatrixModel):
    def __init__(self, channel, R, MPG):
        super(InfiniteSingleLayerMatrix, self).__init__(channel)
        self.MPG, self.R = MPG, R

    def __call__(self):
        return self.R * self.channel.tadv() + \
               self.MPG ** 2 * (self.channel.fws() / self.channel.flow) ** 2 / 4. / (erfcinv(np.random.random())) ** 2

class FiniteSingleLayerMatrix(MatrixModel):
    def __init__(self, channel, R, MPG, tauD):
        super(FiniteSingleLayerMatrix, self).__init__(channel)
        self.MPG, self.R, self.tauD = MPG, R, tauD

    def __call__(self):
        return self.R * self.channel.tadv() + 0.
