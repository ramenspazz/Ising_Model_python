'''
Author: Ramenspazz

This file defines the simulation lattice to be used and sets up basic methods
to operate on the lattice
'''
# Typing imports
import queue
from typing import Callable, Optional, Union
from numbers import Number
from numpy import float64, int64, integer as Int, floating as Float, ndarray, number  # noqa
from numpy.typing import NDArray
# import numpy.typing as npt

import sys # noqa
import numpy as np
from numpy import array
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn import cluster
import LinkedLattice as lc
import DataAnalysis as DA
import random
from Node import Node
import PrintException as PE
import time
import InputFuncs as inF
import math
import multiprocessing as mltp
import MLTPQueue as MLTPqueue

from WaitListLock import WaitListLock
from pyQueue import LLQueue, QueueEmpty
# from pyQueueWithDict import DictQueue

# from scipy.constants import Boltzmann as k_boltzmann

GNum = Union[number, Number]
k_boltzmann = float64(8.617333262E-05)  # eV / K


def T_to_Beta(beta: float64) -> float64:
    return(float64(1 / (k_boltzmann * beta)))


def Beta_to_T(T: float64) -> float64:
    return(float64(1 / (k_boltzmann * T)))


class LatticeDriver:

    """
        TODO : write docstring
    """

    def __init__(self,
                 scale: number | Number,
                 Lshape: list[int] | NDArray[Int],
                 J: float64,
                 basis: Optional[NDArray] = None,
                 time: Optional[int | Int] = 1000) -> None:
        '''
            Parameters
            ----------

            scale : `int` | `float`
                - The scale of distances in centimeters

            Lshape : `list`[`int256`] | `NDArray`[`Int`]
                - Span of the x and y bounds for the lattice.

            basis : Optional[`NDArray`]
                - A numpy ndarray of a shape 2x2 that specifies the x basis
                on the first row and the y basis on the second row.
        '''
        try:
            self.voids: bool = None
            self.probs: list[int] = None
            self.rand_seed: int = None
            self.lat_spins = {
                'x': [],
                'y': [],
                'spin': []}
            self.saved_config: list = []
            self.FigPlt = go.FigureWidget()
            self.first_run = True
            self.Lshape = np.array([
                np.int64(Lshape[0]),
                np.int64(Lshape[1])
                ])
            self.lattice_spacing = scale
            self.threaded = True
            self.J = J
            self.time = time
            if self.Lshape[0] == 0 or self.Lshape[1] == 0:
                raise ValueError()
            elif self.Lshape[0] > 0 and self.Lshape[1] > 0:
                if basis is not None:
                    if isinstance(basis, ndarray):
                        input_basis = basis
                    else:
                        input_basis = array(basis)
                    self.LinkedLat: lc.LinkedLattice = lc.LinkedLattice(
                        scale, Lshape, J, basis_arr=(input_basis))
                elif basis is None:
                    self.LinkedLat: lc.LinkedLattice = lc.LinkedLattice(
                        scale, Lshape, J, basis_arr=None)
            self.ZERO_mtx: ndarray = np.zeros([self.time, 2])
            # Multithreading sum declarations
            self.qsize = self.LinkedLat.tc
            self.result_queue_sum = MLTPqueue.ThQueue(self.qsize)
            self.ready_sum = WaitListLock(self.qsize)
            self.finished_sum = mltp.Event()
            self.thread_pool_sum: list[mltp.Process] = []
            self.SumThreadsAlive = False
            # multithreading energy declarations
            self.result_queue_energy = MLTPqueue.ThQueue(self.qsize)
            self.ready_energy = WaitListLock(self.qsize)
            self.finished_energy = mltp.Event()
            self.thread_pool_energy: list[mltp.Process] = []
            self.EnergyThreadsAlive = False
            # multithreading path declarations
            # TODO must fix C3V, but testing goes on
            self.result_queue_path = MLTPqueue.ThQueue(self.qsize)
            self.was_seen = dict()
            self.cluster_queue = LLQueue()
            self.work_queue_path = mltp.Queue(0)
            self.ready_path = WaitListLock[float](self.qsize)
            self.finished_path = mltp.Event()
            self.thread_pool_path: list[mltp.Process] = []
            self.PathThreadsAlive = False
            self.cluster = MLTPqueue.ThQueue(0)
        except Exception:
            PE.PrintException()

    def __reformMTX__(self):
        self.ZERO_mtx: ndarray = np.zeros([self.time, 2])

    def __len__(self):
        """
            Returns
            -------
            Total number of nodes in the lattice
        """
        try:
            return(len(self.LinkedLat))
        except Exception:
            PE.PrintException()

    def __getitem__(self, *args) -> Node:
        """
            Returns
            -------
            `Node` at the requested location if it exists, else return None.
        """
        try:
            if args is None:
                sys.stderr.write("Args are None")
            if np.abs(self.Lshape[0]) > 0 and np.abs(self.Lshape[1]) > 0:
                return(self.LinkedLat.__getitem__(*args))
            else:
                raise IndexError(f"Index tup is out of range! tup={args[0]} is"
                                 " greater than the max index of"
                                 f"{self.Lshape[0]*self.Lshape[0]-1}!\n")
        except Exception:
            PE.PrintException()

    def __iter__(self) -> Node | None:
        for node in self.LinkedLat:
            yield node

    def __setitem__(self, index: list[int] | NDArray[Int], val: int | Int):
        """
            Purpose
            -------
            assigns value [val] to the object at the location of [index] from
            the `internal_arr` object.

            Parameters
            ----------

            index : `list`
           16     - index of the basis combination refrencing a node.

            val : `int` | `Int`
                - Value to set the spin of the node refrenced by index.

        """
        try:
            self.LinkedLat[index] = val
        except Exception:
            PE.PrintException()

    def __LaunchSumThreads__(self):
        for th_num in range(self.qsize):
            self.thread_pool_sum.append(mltp.Process(
                target=self.LinkedLat.Sum_Worker,
                args=(self.LinkedLat.bounds[th_num],
                      th_num,
                      self.result_queue_sum,
                      self.ready_sum,
                      self.finished_sum)))
            self.thread_pool_sum[th_num].start()
        self.SumThreadsAlive = True

    def __LaunchEnergyThreads__(self):
        for th_num in range(self.qsize):
            self.thread_pool_energy.append(mltp.Process(
                target=self.LinkedLat.Energy_Worker,
                args=(self.LinkedLat.bounds[th_num],
                      th_num,
                      self.result_queue_energy,
                      self.ready_energy,
                      self.finished_energy)))
            self.thread_pool_energy[th_num].start()
        self.EnergyThreadsAlive = True

    def __LaunchPathThreads__(self):
        for th_num in range(self.qsize):
            self.thread_pool_path.append(mltp.Process(
                target=self.LinkedLat.Path_Worker,
                args=(th_num,
                      self.cluster,
                      self.work_queue_path,
                      self.result_queue_path,
                      self.was_seen,
                      self.ready_path,
                      self.finished_energy)))
            self.thread_pool_path[th_num].start()
        self.PathThreadsAlive = True

    def __StopSumThreads__(self):
        if self.SumThreadsAlive is True:
            self.finished_sum.set()
            self.ready_sum.Start_Threads()
            for t in self.thread_pool_sum:
                t.join()
        self.thread_pool_sum.clear()
        # reset events
        self.finished_sum.clear()
        self.SumThreadsAlive = False

    def __StopEnergyThreads__(self):
        if self.EnergyThreadsAlive is True:
            self.finished_energy.set()
            self.ready_energy.Start_Threads()
            for t in self.thread_pool_energy:
                t.join()
        self.thread_pool_energy.clear()
        # reset events
        self.finished_energy.clear()
        self.EnergyThreadsAlive = False

    def __StopPathThreads__(self):
        if self.PathThreadsAlive is True:
            self.finished_path.set()
            self.ready_path.Start_Threads(0)
            for t in self.thread_pool_path:
                # for some reason I cant join the threads, so terminate will
                # have to do until I figure out why. Probably some dumb
                # mistake I made elsewhere.
                # t.join()
                t.terminate()
        self.thread_pool_path.clear()
        # reset events
        self.finished_path.clear()
        self.PathThreadsAlive = False

    def GetMagnitization(self) -> int64:
        cur_itt_sum = int64(0)
        self.ready_sum.Check()
        self.ready_sum.Start_Threads()
        for j in range(self.qsize):
            cur_itt_sum += self.result_queue_sum.get(block=True)
        return(cur_itt_sum)

    def GetEnergy(self) -> int64:
        cur_itt_energy = int64(0)
        self.ready_energy.Check()
        self.ready_energy.Start_Threads()
        for j in range(self.qsize):
            cur_itt_energy += self.result_queue_energy.get(block=True)
        return(cur_itt_energy)

    def GetCluster(self, balance_condition) -> None:
        self.ready_path.Check()
        self.ready_path.Start_Threads(balance_condition)
        for i in range(self.qsize):
            self.result_queue_path.get(block=True)
        return

    def reset(self) -> None:
        for i, node in enumerate(
                self.LinkedLat.range(0,
                                     self.LinkedLat.Shape[0] *
                                     self.LinkedLat.Shape[1])):
            node.set_spin(self.saved_config[i])
        return

    def update(self, set_state=False) -> None:
        """
            Purpose
            -------
            Updates the data representation of the lattice.
        """
        try:
            for i in range(self.Lshape[0]):
                for j in range(self.Lshape[1]):
                    cur = self[np.array([i, j])]
                    if self.first_run is True:
                        self.lat_spins.get('x').append(cur.get_coords()[0])
                        self.lat_spins.get('y').append(cur.get_coords()[1])
                        self.lat_spins.get('spin').append(cur.get_spin())
                    else:
                        self.lat_spins.get('spin')[i+j] = cur.get_spin()
                    if set_state is True and self.first_run is False:
                        self.saved_config.append(cur.get_spin())
            self.first_run = False
        except Exception:
            PE.PrintException()

    def plot(self):
        """
            Purpose
            -------
            Plot the current state of the spin lattice.
        """
        self.update()
        fig = px.scatter(self.lat_spins, x='x', y='y', color='spin')
        fig.show()

    def relax(self, time, beta):
        """
            Purpose
            -------
            Relax the lattice with a given `beta` value for `time` amount of
            itterations of the Wolff algorithm.
        """
        inF.print_stdout('Relaxing lattice...')
        self.WolffAlgorithm(time, beta, relax_call=True)
        inF.print_stdout('Relaxing done!', end='\n')

    # Reminder
    # Look into this http://mcwa.csi.cuny.edu/umass/izing/Ising_text.pdf
    def MetropolisAlgorithm(
            self,
            times,
            beta,
            initial_energy: Optional[float64] = None,
            spinenergy_call: Optional[bool] = False,
            relax_call: Optional[bool] = False,
            plot: Optional[bool] = False) -> ndarray | None:
        """
            Purpose
            -------
            Evolve the system and compute the metroplis approximation to the
            equlibrium state given a beta*J and number of monty carlo
            itterations to preform.
        """
        try:
            if self.SumThreadsAlive is False and relax_call is False:
                if np.array_equiv(self.ZERO_mtx.shape,
                                  np.array([times, 2])) is False:
                    self.set_time(times)
                self.__LaunchSumThreads__()
            if initial_energy is not None:
                cur_itt_energy = initial_energy
            else:
                cur_itt_energy = np.float64(0)
            if spinenergy_call is False:
                inF.print_stdout(
                    "MetropolisAlgorithm is 000.0% complete...")
            netSE_mtx = self.ZERO_mtx
            for itt_num in range(times):
                if spinenergy_call is False:
                    inF.print_stdout(
                        f"MetropolisAlgorithm is {100 * itt_num/times:.2f}%"
                        f" complete...")
                # select spins to flip
                while True:
                    # pick random point on array and flip spin
                    # randint is very slow so dont use it, ever please...
                    rand_xy = np.array([math.trunc((random.random()) *
                                        (self.Lshape[0] - 1)),
                                        math.trunc((random.random()) *
                                        (self.Lshape[1] - 1))])
                    node_i = self[rand_xy]
                    if node_i.get_spin() == 0:
                        continue
                    else:
                        break
                S_i = node_i.get_spin()
                # compute change in energy
                nbrs_Energy: np.int64 = 0
                for neighbor in node_i:
                    if neighbor.get_spin() == 0:
                        continue
                    nbrs_Energy += neighbor.get_spin()
                dE = 2 * S_i * nbrs_Energy * self.J
                if dE > 0 and random.random() < np.exp(-beta*dE):
                    cur_itt_energy += dE
                    node_i.flip_spin()
                if dE <= 0:
                    cur_itt_energy += dE
                    node_i.flip_spin()

                # begin calculating total spin of lattice
                if relax_call is False:
                    cur_itt_sum = self.GetMagnitization()
                    netSE_mtx[itt_num, 0] = cur_itt_sum
                    netSE_mtx[itt_num, 1] = cur_itt_energy
            # for itt_num in range(times): end
            if spinenergy_call is False:
                inF.print_stdout(
                    "MetropolisAlgorithm is 100% complete...")
            if spinenergy_call is False:
                self.__StopSumThreads__()
            if plot is True:
                self.PlotItterations(netSE_mtx, beta, save=False,
                                     auto_plot=True)
            if relax_call is False:
                return(netSE_mtx)
        except KeyboardInterrupt:
            print('\nKeyboard Inturrupt, exiting...\n')
            self.__StopEnergyThreads__()
            self.__StopSumThreads__()
            exit()
        except Exception:
            PE.PrintException()

    def WolffAlgorithm(self,
                       times,
                       beta,
                       initial_energy: Optional[float64] = None,
                       spinenergy_call: Optional[bool] = False,
                       relax_call: Optional[bool] = False,
                       plot: Optional[bool] = False) -> ndarray | None:
        """
            TODO : write docstring
        """
        try:
            if self.SumThreadsAlive is False and relax_call is False:
                if np.array_equiv(self.ZERO_mtx.shape,
                                  np.array([times, 2])) is False:
                    self.set_time(times)
                self.__LaunchSumThreads__()
                self.__LaunchPathThreads__()
            if initial_energy is not None:
                cur_itt_energy = initial_energy
            else:
                cur_itt_energy = np.float64(0)
            if spinenergy_call is False:
                inF.print_stdout(
                    "WolffAlgorithm is 000.0% complete...")
            netSE_mtx = self.ZERO_mtx
            balcond = 1-np.exp(-2*beta*self.J, dtype=np.float64)
            for itt_num in range(times):
                if spinenergy_call is False:
                    inF.print_stdout(
                        f"WolffAlgorithm is {100 * itt_num/times:.2f}%"
                        f" complete...")
                # pick random point
                while True:
                    # pick random point on array
                    # randint is very slow so dont use it when you need to
                    # call it successively, ever please...
                    rand_xy = np.array([math.trunc((random.random()) *
                                        (self.Lshape[0] - 1)),
                                        math.trunc((random.random()) *
                                        (self.Lshape[1] - 1))])
                    rand_node = self[rand_xy]
                    if rand_node.get_spin() == 0:
                        continue
                    else:
                        self.cluster.put(rand_node.get_index())
                        break
                # push the random node to the work queue
                for nbr in rand_node:
                    self.work_queue_path.put(nbr.get_index())
                self.was_seen[rand_node] = rand_node
                # wait for cluster to be generated
                self.GetCluster(balcond)
                self.was_seen.clear()
                try:
                    # evaluate energy change path integral
                    cur = self.LinkedLat[self.cluster.get(block=False)]
                    while True:
                        nbr_sum = float64(0)
                        for nbr in cur:
                            nbr_sum += nbr.get_spin()
                        cur_itt_energy += nbr_sum*cur.flip_spin()*self.J
                        cur = self.LinkedLat[self.cluster.get(block=False)]
                except queue.Empty:
                    # exit while loop when queue is empty
                    pass
                if relax_call is False:
                    cur_itt_sum = self.GetMagnitization()
                    netSE_mtx[itt_num, 0] = cur_itt_sum
                    netSE_mtx[itt_num, 1] = cur_itt_energy
            # for itt_num in range(times): end
            if spinenergy_call is False:
                inF.print_stdout(
                    "WolffAlgorithm is 100% complete...")
            if spinenergy_call is False:
                self.__StopSumThreads__()
                self.__StopPathThreads__()
            if plot is True:
                self.PlotItterations(netSE_mtx, beta, save=False,
                                     auto_plot=True)
            if relax_call is False:
                return(netSE_mtx)
        except Exception:
            PE.PrintException()

    def SpinEnergy(
            self,
            Beta: list | ndarray,
            times: int,
            method: Callable[[], ndarray],
            save: Optional[bool] = False,
            auto_plot: Optional[bool] = True) -> list:
        """
            Purpose
            -------
            Uses the Wolff algorithm to return the Magnitization -- VIA
            summing over the lattice, and Energy -- as derived from the
            partition function.

            Parameters
            ----------
            `Beta` : `list` | `ndarray`
                - Beta values to use representing the beta in the partition
                function in units of Kelvin per electronvolt.

            `times` : `int`
                - Number of itterations of the Wolff algorithm to preform.

            `method` : `Callable`[[], `ndarray`]
                - A function (internal to LatticeDriver as self is passed)
                that evolves the lattice.

            `save` : `Optional`[`bool`] = `False`
                - If set to `True`, saves plots after they are generated.

            `auto_plot` : `Optional`[`bool`] = `True`
                - If set to `True`, the function will automatically display
                plots as they become ready.
        """
        try:
            if np.array_equiv(self.ZERO_mtx.shape,
                              np.array([times, 2])) is False:
                self.set_time(times)

            magnitization = np.zeros(len(Beta), float64)
            E_mean = np.zeros(len(Beta), float64)
            E_std = np.zeros(len(Beta), float64)

            inF.print_stdout(
                "get_spin_energy is 000.0% complete...\r")

            self.__LaunchEnergyThreads__()
            start_time = time.time()
            initial_energy = self.GetEnergy()*self.J
            self.__StopEnergyThreads__()
            # print(f'\nInitial Energy is {initial_energy}')

            for i, beta in enumerate(Beta):
                inF.print_stdout(
                    f"get_spin_energy is {100 * i / len(Beta) :.1f}%"
                    f" complete...")
                if i != 0:
                    self.reset()

                netSE_mtx = method(
                    self,
                    times,
                    beta,
                    initial_energy=initial_energy,
                    spinenergy_call=True)

                magnitization[i] = DA.data_mean(netSE_mtx[-times:, 0])
                E_mean[i] = DA.data_mean(netSE_mtx[-times:, 1])
                E_std[i] = DA.std_dev(netSE_mtx[-times:, 1], E_mean[i])
            # for i, beta in enumerate(Beta): end

            self.__StopSumThreads__()
            self.__StopPathThreads__()
            end_time = time.time()
            inF.print_stdout(
                f'get_spin_energy is 100% complete in '
                f'{end_time-start_time:.8f} seconds!',
                end='\n')

            self.PlotSpinEnergy(Beta,
                                magnitization,
                                E_std,
                                save=save,
                                auto_plot=auto_plot,
                                times=times)
            return(netSE_mtx)
        except KeyboardInterrupt:
            print('\nKeyboard Inturrupt, exiting...\n')
            self.__StopEnergyThreads__()
            self.__StopSumThreads__()
            self.__StopPathThreads__()
            exit()
        except Exception:
            PE.PrintException()

    def PlotItterations(self, SE_mtx: ndarray,
                        Beta: list | ndarray,
                        times: Optional[int] = None,
                        quiet: Optional[bool] = True,
                        save: Optional[bool] = False,
                        auto_plot: Optional[bool] = True) -> None:
        """
        Parameters
        ----------
        SE_mtx : `ndarray`
            - spin energy matrix to plot.

        Beta : `list` | `ndarray`
            - list or ndarray of Beta range that generated the SE_mtx.

        times : Optional[`int`]
            - Number of times the system was evolved in time.

        quiet : Optional[`bool`] = `True`
            - If quiet is Flase, the method will print statistics about the
            data in SE_mtx.

        save : Optional[`bool`] = `False`
            - If True, the method will save the output plot.

        auto_plot : Optional[`bool`] = `True`
            - If True, the method will automatically plot the output plot.
        """
        num_nodes = len(self.LinkedLat)
        if quiet is False:
            spin_up = 0
            spin_dn = 0
            for val in SE_mtx[:, 0]:
                if val > 0:
                    spin_up += val
                elif val < 0:
                    spin_dn += val
            mean = DA.data_mean(SE_mtx[:, 0])
            stdev = DA.std_dev(SE_mtx[:, 0], mean, sample=False)
            inF.print_stdout(f"""
            Average up sum of spins density is {
                np.abs(spin_up) / num_nodes :.4f}%\n
            Average up sum of spins density is {
                np.abs(spin_dn) / num_nodes :.4f}%\n
            The mean is {mean}\n
            The Standard deviation of the mean is {
                stdev / np.sqrt(num_nodes) :.4f}\n
            """)
        fig = make_subplots(rows=1, cols=2)
        fig.layout.title = (f'Evolution of Average Spin n={num_nodes} and'
                            f' Energy for Beta*J={Beta*self.J}')
        fig.add_trace(go.Scatter(
            x=np.linspace(0, len(SE_mtx[:, 0])-1, len(SE_mtx[:, 0])),
            y=SE_mtx[:, 0] / num_nodes, mode='lines'), row=1, col=1)
        fig.update_xaxes(title_text='Time Steps', type='log', row=1, col=1)
        fig.update_yaxes(title_text='Average Spin', row=1, col=1)
        fig.add_trace(go.Scatter(
            x=np.linspace(0, len(SE_mtx[:, 1])-1, len(SE_mtx[:, 1])),
            y=SE_mtx[:, 1], mode='lines'), row=1, col=2)
        fig.update_xaxes(title_text='Time Steps', row=1, col=2)
        fig.update_yaxes(title_text='E / J', row=1, col=2)
        if save is True:
            if self.LinkedLat.rots == 3:
                rots = 6
            elif self.LinkedLat.rots == 6:
                rots = 3
            else:
                rots = self.LinkedLat.rots
            fname = ('Metropolis' + '_' + str(self.Lshape) + '_' +
                     str(rots) + '_' + str(Beta) +
                     str(times) + '.png')
            fig.write_image(fname)
        if auto_plot is True:
            fig.show()

    def PlotSpinEnergy(self,
                       Beta: ndarray,
                       magnitization: ndarray | list,
                       E_std: ndarray | list,
                       save: Optional[bool] = False,
                       auto_plot: Optional[bool] = True,
                       times: Optional[int] = None) -> None:
        """
            Plot the results of a Spin Energy call.
        """
        fig = make_subplots(rows=1, cols=2)
        if self.LinkedLat.rots == 3:
            rots = 6
        elif self.LinkedLat.rots == 6:
            rots = 3
        else:
            rots = self.LinkedLat.rots

        # BJ = np.multiply(Beta, self.J)
        temps = np.zeros(len(Beta))
        for i in range(len(Beta)):
            temps[i] = Beta_to_T(Beta[i])

        Cv = np.zeros(len(Beta))
        for i in range(len(Beta)):
            # print(f'E_std[[{i}] = {E_std[i]}')
            Cv[i] = E_std[i]**2 / (k_boltzmann * temps[i]**2)

        fig.layout.title = (f'Spin energy realtions plot,'
                            f' C{rots}V')
        xname = 'Kelvin'
        yname1 = 'Average Magnitization density'
        yname2 = 'Heat Capacity (eV/K)'
        fig.add_trace(go.Scatter(x=temps, y=magnitization/len(self),
                                 mode='lines+markers'), row=1, col=1)
        fig.update_xaxes(title_text=xname, row=1, col=1)
        fig.update_yaxes(title_text=yname1, row=1, col=1)
        fig.add_trace(
            go.Scatter(x=temps, y=Cv, mode='lines+markers'),
            row=1, col=2)
        fig.update_xaxes(title_text=xname, row=1, col=2)
        fig.update_yaxes(title_text=yname2, row=1, col=2)
        if save is True:
            fname = ('SpinEnergy' + str(self.Lshape) + '_' +
                     str(rots) + '_' + str(Beta[0]) + '-' +
                     str(Beta[len(Beta)-1]) + str(times) + '.png')
            fig.write_image(fname)
        if auto_plot is True:
            fig.show()

    def randomize(self,
                  voids: Optional[bool] = None,
                  probs: Optional[list] = None,
                  rand_seed: Optional[int] = None) -> None:
        """
            Parameters
            ----------
            voids : `bool`
                - Truns on random voids in the lattice

            probs : `list`[`int`]
                - list representing the probability of `1`, `0`, `-1`
                where the first two entries represent the the mean and
                standard-deviation of the normal distribution, and the
                percentage that a spin should be a void if enabled.

            rand_seed : Optional[`int`] = Node
                - Seed value to seed random with.
        """
        try:
            if (self.rand_seed is None and self.voids is None and
                    self.probs is None):
                self.voids = voids
                self.probs = probs
                self.rand_seed = rand_seed

            if sum(self.probs) != 100:
                raise ValueError('The sum of all probabilites must add to'
                                 ' 100%!')
            random.seed(self.rand_seed)
            x = self.LinkedLat.Shape[0]
            y = self.LinkedLat.Shape[1]
            if self.voids is True:
                my_list = ([-1]*self.probs[0] + [1]*self.probs[1] +
                           [0]*self.probs[2])
            elif self.voids is False:
                my_list = [-1]*self.probs[0] + [1]*self.probs[1]
            for i in range(x):
                for j in range(y):
                    node = self[np.array([i, j])]
                    if (self.LinkedLat.rots == 6 and
                            ((j + 1) % 3 == 0 and j != 0)):
                        node.set_spin(0)
                        continue
                    rand_num = random.choice(my_list)
                    if rand_num == 0:
                        self.LinkedLat.num_voids += 1
                    if node is not None:
                        node.set_spin(rand_num)
            if self.LinkedLat.num_voids == self.shape()[0]*self.shape()[1]:
                raise ValueError("All nodes are voids! Please choose different"
                                 " values for your gaussian distribution, or"
                                 " you\'re just reaaaaaally unlucky.")
            # setup internal display of lattice
            self.update()
            self.update(set_state=True)
            return(None)
        except Exception:
            PE.PrintException()

    def set_time(self, time) -> None:
        """
        Change the simulation time from the default of 1000 itterations.
        """
        self.time = time
        self.__reformMTX__()

    def shape(self) -> tuple:
        return(self.Lshape)
