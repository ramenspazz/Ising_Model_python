'''
Author: Ramenspazz

This file defines the simulation lattice to be used and sets up basic methods
to operate on the lattice
'''
# Typing imports
from typing import Optional, Union
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

GNum = Union[number, Number]
k_boltzmann = float64(8.617333262145E-05)  # eV / K


def T_to_Beta(beta: float64) -> float64:
    return np.round(float64(1 / (k_boltzmann * beta)), 6)


def Beta_to_T(T: float64) -> float64:
    return np.round(float64(1 / (k_boltzmann * T)), 6)


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
            self.initial_configuration: list = []
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

    def generate_lattice(self) -> None:
        """
            Purpose
            -------
            This function calls the `LinkedLattice` class member function
            `self.internal_arr.generate(dims)`.
        """
        self.LinkedLat.generate(self.Lshape)

    def get_energy(self):
        # $E/J = -\sum_{<i,j>} \sigma_i\sigma_j$
        return(self.LinkedLat.Nearest_Neighbor())

    def reset(self) -> None:
        if self.first_run is not True:
            for i, node in enumerate(
                    self.LinkedLat.range(0, self.LinkedLat.num_nodes)):
                node.set_spin(self.initial_configuration[i])
        return

    def update(self) -> None:
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
                        self.initial_configuration.append(cur.get_spin())
                    else:
                        self.lat_spins.get('spin')[i+j] = cur.get_spin()
            self.first_run = False
        except Exception:
            PE.PrintException()

    def plot(self):
        self.update()
        fig = px.scatter(self.lat_spins, x='x', y='y', color='spin')
        fig.show()

    def WolffSpinEnergy(self,
                        Beta: list | ndarray,
                        times: int,
                        save: Optional[bool] = False,
                        auto_plot: Optional[bool] = True) -> list:
        """
            TODO : write docstring
        """
        try:
            if np.array_equiv(self.ZERO_mtx.shape,
                              np.array([times, 2])) is False:
                self.set_time(times)

            netSE_mtx = self.ZERO_mtx
            magnitization = np.zeros(len(Beta), float64)
            E_mean = np.zeros(len(Beta), float64)
            E_std = np.zeros(len(Beta), float64)

            inF.print_stdout(
                "get_spin_energy is 000.0% complete...\r")
            start_time = time.time()

            qsize = self.LinkedLat.tc

            # multithreading object declarations for sum
            result_queue_SE = MLTPqueue.ThQueue(qsize)
            ready_SE = WaitListLock(qsize)
            finished_SE = mltp.Event()

            thread_pool_SE: list[mltp.Process] = []
            for th_num in range(self.LinkedLat.tc):
                thread_pool_SE.append(mltp.Process(
                    target=self.LinkedLat.SpinEnergy_Worker,
                    args=(self.LinkedLat.bounds[th_num],
                          th_num,
                          result_queue_SE,
                          ready_SE,
                          finished_SE)))
                thread_pool_SE[th_num].start()

            # # multithreading object declarations for sum
            # result_queue_sum = MLTPqueue.ThQueue(qsize)
            # ready_sum = WaitListLock(qsize)
            # finished_sum = mltp.Event()

            # thread_pool_sum: list[mltp.Process] = []
            # for th_num in range(self.LinkedLat.tc):
            #     thread_pool_sum.append(mltp.Process(
            #         target=self.LinkedLat.Sum_Worker,
            #         args=(self.LinkedLat.bounds[th_num],
            #               th_num,
            #               result_queue_sum,
            #               ready_sum,
            #               finished_sum)))
            #     thread_pool_sum[th_num].start()

            # # multithreading object declarations for energy
            # result_queue_energy = MLTPqueue.ThQueue(qsize)
            # ready_energy = WaitListLock(qsize)
            # finished_energy = mltp.Event()

            # thread_pool_energy: list[mltp.Process] = []
            # for th_num in range(self.LinkedLat.tc):
            #     thread_pool_energy.append(mltp.Process(
            #         target=self.LinkedLat.Energy_Worker,
            #         args=(self.LinkedLat.bounds[th_num],
            #               th_num,
            #               result_queue_energy,
            #               ready_energy,
            #               finished_energy)))
            #     thread_pool_energy[th_num].start()

            # ready_energy.Check()
            # ready_energy.Start_Threads()
            # for j in range(qsize):
            #     energy += result_queue_energy.get(block=True)

            # finished_energy.set()
            # ready_energy.Start_Threads()
            # for t in thread_pool_energy:
            #     t.join()

            # SE_vec = np.zeros(2, int64)
            # ready_SE.Check()
            # ready_SE.Start_Threads()
            # for j in range(qsize):
            #     SE_vec += result_queue_SE.get(block=True)

            node_q = LLQueue[Node]()
            for i, beta in enumerate(Beta):
                inF.print_stdout(
                    f"get_spin_energy is {100 * i / len(Beta) :.1f}%"
                    f" complete...")
                if i != 0:
                    self.reset()
                for itt_num in range(times):
                    # pick random point
                    while True:
                        # pick random point on array and flip spin
                        # randint is very slow so dont use it, ever please...
                        rand_xy = np.array([math.trunc((random.random()) *
                                            (self.Lshape[0] - 1)),
                                            math.trunc((random.random()) *
                                            (self.Lshape[1] - 1))])
                        select_node = self[rand_xy]
                        if select_node.get_spin() == 0:
                            continue
                        else:
                            node_q.push(select_node)
                            break
                    try:
                        while True:
                            cur_node = node_q.pop()
                            # check neighbors
                            for nbr in cur_node:
                                if nbr.get_spin == 0:
                                    continue
                                nbrs_Energy = 0
                                for Enbr in nbr:
                                    nbrs_Energy += Enbr.get_spin()
                                balcond = np.exp(
                                    -2*nbr.flip_test()*nbrs_Energy*beta*self.J,
                                    dtype=np.float64)
                                # input(f'enter to continue, balcond = {balcond}')  # noqa
                                # print(balcond)
                                if (nbr.get_spin() == -cur_node.get_spin()
                                        and random.random() < balcond):
                                    # print('flipped')
                                    nbr.flip_spin()
                                    node_q.push(nbr)
                    except QueueEmpty:
                        # exit while loop when queue is empty
                        pass
                    # get magnitization
                    SE_vec = np.zeros(2, int64)
                    ready_SE.Check()
                    ready_SE.Start_Threads()
                    for j in range(qsize):
                        SE_vec += result_queue_SE.get(block=True)
                    netSE_mtx[itt_num, :] = SE_vec
                # for itt_num in range(times): end

                magnitization[i] = DA.data_mean(netSE_mtx[-times:, 0])
                E_mean[i] = DA.data_mean(netSE_mtx[-times:, 1] / self.J)
                E_std[i] = DA.std_dev(netSE_mtx[-times:, 1]/self.J, E_mean[i])
            # for i, beta in enumerate(Beta): end
            finished_SE.set()
            ready_SE.Start_Threads()
            for t in thread_pool_SE:
                t.join()

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
            finished_SE.set()
            ready_SE.Start_Threads()
            for t in thread_pool_SE:
                t.terminate()
            exit()
        except Exception:
            PE.PrintException()

    def plot_metrop(self, SE_mtx: ndarray,
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
        fig = make_subplots(rows=1, cols=2)
        if self.LinkedLat.rots == 3:
            rots = 6
        elif self.LinkedLat.rots == 6:
            rots = 3
        else:
            rots = self.LinkedLat.rots

        BJ = np.multiply(Beta, self.J)  # in eV / K
        Cv = np.zeros(len(BJ))
        for i in range(len(BJ)):
            Cv[i] = E_std[i]**2 * BJ[i] * k_boltzmann

        temps = np.zeros(len(BJ))
        for i in range(len(BJ)):
            temps[i] = Beta_to_T(Beta[i])

        fig.layout.title = (f'Spin energy realtions plot,'
                            f' C{rots}V')
        xname = 'Kelvin'
        yname1 = 'Average Magnitization density'
        yname2 = 'Heat Capacity (eV/K)'
        fig.add_trace(go.Scatter(x=temps, y=magnitization,
                                 mode='lines+markers'), row=1, col=1)
        fig.update_xaxes(title_text=xname, type='log', row=1, col=1)
        fig.update_yaxes(title_text=yname1, row=1, col=1)
        fig.add_trace(
            go.Scatter(x=temps, y=Cv, mode='lines+markers'),
            row=1, col=2)
        fig.update_xaxes(title_text=xname, type='log', row=1, col=2)
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
