'''
Author: Ramenspazz

This file defines the simulation lattice to be used and sets up basic methods
to operate on the lattice
'''
# Typing imports
from queue import Empty
from typing import Optional, Union
from numbers import Number
from numpy import integer as Int, floating as Float, ndarray, number
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
import MLTPQueue as queue
from time import sleep

GNum = Union[number, Number]
k_boltzmann = 1.380649E-23


class LatticeDriver:

    """
        TODO : write docstring
    """

    def __init__(self,
                 scale: number | Number,
                 Lshape: list[int] | NDArray[Int],
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
            self.lat_spins = {
                'x': [],
                'y': [],
                'spin': []}
            self.FigPlt = go.FigureWidget()
            self.first_run = True
            self.Lshape = np.array([
                np.int64(Lshape[0]),
                np.int64(Lshape[1])
                ])
            self.lattice_spacing = scale
            self.threaded = True
            self.time = time
            if self.Lshape[0] == 0 or self.Lshape[1] == 0:
                raise ValueError()
            elif self.Lshape[0] > 0 and self.Lshape[1] > 0:
                if basis is not None:
                    if isinstance(basis, ndarray):
                        input_basis = basis
                    else:
                        input_basis = array(basis)
                    self.internal_arr: lc.LinkedLattice = lc.LinkedLattice(
                        scale, Lshape, basis_arr=(input_basis))
                elif basis is None:
                    self.internal_arr: lc.LinkedLattice = lc.LinkedLattice(
                        scale, Lshape, basis_arr=None)
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
            return(len(self.internal_arr))
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
                return(self.internal_arr.__getitem__(*args))
            else:
                raise IndexError(f"Index tup is out of range! tup={args[0]} is"
                                 " greater than the max index of"
                                 f"{self.Lshape[0]*self.Lshape[0]-1}!\n")
        except Exception:
            PE.PrintException()

    def __iter__(self) -> Node | None:
        for node in self.internal_arr:
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
            self.internal_arr[index] = val
        except Exception:
            PE.PrintException()

    def generate_lattice(self) -> None:
        """
            Purpose
            -------
            This function calls the `LinkedLattice` class member function
            `self.internal_arr.generate(dims)`.
        """
        self.internal_arr.generate(self.Lshape)

    def get_energy(self):
        # $E/J = -\sum_{<i,j>} \sigma_i\sigma_j$
        return(self.internal_arr.Nearest_Neighbor())

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
                    else:
                        self.lat_spins.get('spin')[i+j] = cur.get_spin()
            self.first_run = False
        except Exception:
            PE.PrintException()

    def plot(self):
        self.update()
        fig = px.scatter(self.lat_spins, x='x', y='y', color='spin')
        fig.show()

    def get_spin_energy(self,
                        BJs: list | ndarray,
                        __times: Optional[int] = None,
                        save: Optional[bool] = False,
                        auto_plot: Optional[bool] = True) -> list:
        """
            Purpose
            -------
            Computes the energy of the lattice at thermal equlibrium and
            relates this to T=J/(Beta*J*k) = 1/(Beta*k) ~ 1/(Beta*J)
        """
        try:
            inF.print_stdout(
                "get_spin_energy is 000.0% complete...\r")
            if np.array_equiv(self.ZERO_mtx.shape,
                              np.array([__times, 2])) is False:
                self.set_time(__times)
            netSE_mtx = self.ZERO_mtx
            mean_spin = np.zeros(len(BJs))
            E_means = np.zeros(len(BJs))
            E_stds = np.zeros(len(BJs))

            # multithreading object declarations
            qsize = self.internal_arr.tc
            res_sum = queue.MyQueue(qsize)
            st_queue_sum = queue.MyQueue(qsize)
            start_sum = mltp.Event()
            wait_sum = mltp.Event()
            finished_sum = mltp.Event()

            thread_pool_sum: list[mltp.Process] = []
            for itt_num in range(self.internal_arr.tc):
                thread_pool_sum.append(mltp.Process(
                    target=self.internal_arr.__Sum_Worker__,
                    args=(self.internal_arr.bounds[itt_num],
                          res_sum,
                          st_queue_sum,
                          start_sum,
                          wait_sum,
                          finished_sum)))
                thread_pool_sum[itt_num].start()

            prev_sum = 0
            rand_xy: list[Node] = []
            flip_num = math.ceil(4 * np.log2(len(self.internal_arr)))

            if __times is None:
                times = 1
            else:
                times = __times
            if self.time != times:
                self.set_time(times)

            start_time = time.time()
            energy = self.internal_arr.__threadlauncher__(
                    self.internal_arr.__NN_Worker__, True)

            for i, bj in enumerate(BJs):
                inF.print_stdout(
                    f"get_spin_energy is {100 * i / len(BJs) :.1f}%"
                    f" complete...")
                prev_energy = energy

                netSE_mtx = self.ZERO_mtx
                for itt_num in range(times):
                    rand_xy.clear()

                    # select spins to flip
                    for nth_flip in range(flip_num):
                        # pick random point on array and flip spin
                        # randint is very slow so dont use it, ever please...
                        test = np.array([math.trunc((random.random()) *
                                        (self.Lshape[0] - 1)),
                                        math.trunc((random.random()) *
                                                   (self.Lshape[1] - 1))])
                        if self[test].get_spin() == 0 or self[test] in rand_xy:
                            continue
                        else:
                            rand_xy.append(self[test])
                    dE_tot = 0
                    for node_i in rand_xy:
                        if node_i.get_spin() == 0:
                            netSE_mtx[itt_num, 0] = prev_sum
                            netSE_mtx[itt_num, 1] = prev_energy
                            continue

                        # compute change in energy
                        E_i: np.int64 = 0
                        E_f: np.int64 = 0
                        for neighbor in node_i:
                            if neighbor.get_spin() == 0:
                                continue
                            E_i += node_i.get_spin() * neighbor.get_spin()
                            E_f += -1 * node_i.get_spin() * neighbor.get_spin()

                        # change state
                        dE = E_f-E_i

                        if (dE > 0) and ((random.uniform(0, 1)) <
                                         np.exp(-bj * dE)):
                            node_i.flip_spin()
                        elif dE <= 0:
                            node_i.flip_spin()
                        dE_tot += dE

                    # begin calculating total spin of lattice
                    wait_sum.clear()
                    wait_sum.set()
                    wait_sum.clear()
                    while st_queue_sum.qsize() != qsize:
                        # print(start_queue.qsize())
                        pass
                    try:
                        for j in range(qsize):
                            # empty the start queue, yeah I wish there was a
                            # better way to do this...
                            # https://xkcd.com/292/
                            # mutter mutter velociraptors...
                            st_queue_sum.get()
                    except Empty:
                        pass
                    start_sum.set()
                    while res_sum.qsize() != qsize:
                        # wait for results to populate the results queue
                        pass
                    try:
                        for j in range(qsize):
                            netSE_mtx[itt_num, 0] += res_sum.get()
                    except Empty:
                        pass
                    # reset event variables
                    start_sum.clear()
                    start_sum.set()
                    start_sum.clear()
                    wait_sum.set()
                    sleep(0.00001)
                    wait_sum.clear()
                    wait_sum.set()

                    prev_sum = netSE_mtx[itt_num, 0]
                    energy += dE_tot
                    prev_energy = energy
                    netSE_mtx[itt_num, 1] = energy

                # for itt_num in range(times):

                spins = netSE_mtx[:, 0]
                energies = netSE_mtx[:, 1]

                mean_spin[i] = (DA.data_mean(spins[-times:]) /
                                len(self.internal_arr))
                E_means[i] = DA.data_mean(energies[-times:])
                E_stds[i] = DA.std_dev(energies[-times:], E_means[i])

            finished_sum.set()
            start_sum.set()
            wait_sum.set()

            for t in thread_pool_sum:
                t.join()

            end_time = time.time()
            inF.print_stdout(
                f'get_spin_energy is 100% complete in '
                f'{end_time-start_time:.8f} seconds!',
                end='\n')
            self.plot_spin_energy(BJs, mean_spin, E_stds, save=save,
                                  auto_plot=auto_plot, times=times)
            return([mean_spin, E_means, E_stds])
        except KeyboardInterrupt:
            print('Keyboard Inturrupt, exiting...\n')
            finished_sum.set()
            wait_sum.set()
            for t in thread_pool_sum:
                t.terminate()
            exit()
        except Exception:
            PE.PrintException()

    def plot_spin_energy(self, bjs: ndarray,
                         ms: ndarray | list,
                         E_stds: ndarray | list,
                         save: Optional[bool] = False,
                         auto_plot: Optional[bool] = True,
                         times: Optional[int] = None) -> None:
        fig = make_subplots(rows=1, cols=2)
        fig.layout.title = (f'Spin energy realtions plot,'
                            f' C{self.internal_arr.rots}V')
        xname = 'Scaled Temperature'
        yname1 = 'Average Spin'
        yname2 = 'Heat Capacity per total spins'
        fig.add_trace(go.Scatter(x=1/bjs, y=ms, mode='lines'), row=1, col=1)
        fig.update_xaxes(title_text=xname, row=1, col=1)
        fig.update_yaxes(title_text=yname1, row=1, col=1)
        fig.add_trace(go.Scatter(x=1/bjs,
                                 y=E_stds*bjs/self.__len__(),
                                 mode='lines'), row=1,
                      col=2)
        fig.update_xaxes(title_text=xname, row=1, col=2)
        fig.update_yaxes(title_text=yname2, row=1, col=2)
        if save is True:
            fname = ('SpinEnergy' + str(self.Lshape) + '_' +
                     str(self.internal_arr.rots) + '_' + str(bjs[0]) + '-' +
                     str(bjs[len(bjs)-1]) + str(times) + '.png')
            fig.write_image(fname)
        if auto_plot is True:
            fig.show()

# TODO: Look into this http://mcwa.csi.cuny.edu/umass/izing/Ising_text.pdf
# TODO: the worm algorithm.
    def metropolis(self, times: int | Int, BJ: float | Float,
                   progress: Optional[bool] = None,
                   quiet: Optional[bool] = False,
                   save: Optional[bool] = False,
                   auto_plot: Optional[bool] = True) -> ndarray:
        """
            Purpose
            -------
            Evolve the system and compute the metroplis approximation to the
            equlibrium state given a beta*J and number of monty carlo
            itterations to preform.
        """
        try:
            energy = self.internal_arr.__threadlauncher__(
                self.internal_arr.__NN_Worker__, True)

            # multithreading object declarations
            qsize = self.internal_arr.tc
            res = queue.MyQueue(qsize)
            start_queue = queue.MyQueue(qsize)
            start_itt = mltp.Event()
            wait_until_set = mltp.Event()
            finished = mltp.Event()

            thread_pool: list[mltp.Process] = []
            for itt_num in range(self.internal_arr.tc):
                thread_pool.append(mltp.Process(
                    target=self.internal_arr.__Sum_Worker__,
                    args=(self.internal_arr.bounds[itt_num],
                          res,
                          start_queue,
                          start_itt,
                          wait_until_set,
                          finished)))
                thread_pool[itt_num].start()

            prev_sum = 0
            prev_energy = energy
            rand_xy = []
            flip_num = math.trunc(np.log10(len(self.internal_arr)))

            if self.time != times:
                self.set_time(times)
            netSE_mtx = self.ZERO_mtx
            if progress is True:
                inF.print_stdout(
                    'Computing Metropolis Algorithm with iterations'
                    f'={times}...')

                for itt_num in range(times):
                    rand_xy.clear()

                    # select spins to flip
                    for nth_flip in range(flip_num):
                        # pick random point on array and flip spin
                        # randint is evil and slow dont use it. this does the
                        # same thing about two orders of magnitude faster.
                        test = np.array([math.trunc((random.random()) *
                                        (self.Lshape[0] - 1)),
                                math.trunc((random.random()) *
                                           (self.Lshape[1] - 1))])
                        if self[test].get_spin() == 0 or self[test] in rand_xy:
                            continue
                        else:
                            rand_xy.append(self[test])
                    dE_tot = 0
                    for node_i in rand_xy:
                        if node_i.get_spin() == 0:
                            netSE_mtx[itt_num, 0] = prev_sum
                            netSE_mtx[itt_num, 1] = prev_energy
                            continue

                        # compute change in energy
                        E_i: np.int64 = 0
                        E_f: np.int64 = 0
                        for neighbor in node_i:
                            if neighbor.get_spin() == 0:
                                continue
                            E_i += node_i.get_spin() * neighbor.get_spin()
                            E_f += -1 * node_i.get_spin() * neighbor.get_spin()

                        # change state with designated probabilities
                        dE = E_f-E_i

                        if (dE > 0) and ((random.uniform(0, 1)) <
                                         np.exp(-BJ * dE)):
                            node_i.flip_spin()
                        elif dE <= 0:
                            node_i.flip_spin()
                        dE_tot += dE

                    # begin calculating total spin of lattice
                    wait_until_set.clear()
                    wait_until_set.set()
                    wait_until_set.clear()
                    while start_queue.qsize() != qsize:
                        # print(start_queue.qsize())
                        pass
                    try:
                        for j in range(qsize):
                            # empty the start queue, yeah I wish there was a
                            # better way to do this...
                            # https://xkcd.com/292/
                            # mutter mutter velociraptors...
                            start_queue.get()
                    except Empty:
                        pass
                    start_itt.set()
                    while res.qsize() != qsize:
                        # wait for results to populate the results queue
                        pass
                    try:
                        for j in range(qsize):
                            netSE_mtx[itt_num, 0] += res.get()
                    except Empty:
                        pass
                    # reset event variables
                    start_itt.clear()
                    start_itt.set()
                    start_itt.clear()
                    wait_until_set.set()
                    sleep(0.00001)
                    wait_until_set.clear()
                    wait_until_set.set()

                    prev_sum = netSE_mtx[itt_num, 0]
                    energy += dE_tot
                    prev_energy = energy
                    netSE_mtx[itt_num, 1] = energy

                # for itt_num in range(times):
                yield(netSE_mtx)
            # yield loop?

            finished.set()
            start_itt.set()
            wait_until_set.set()

            for t in thread_pool:
                t.join()

            if progress is True:
                inF.print_stdout('Metropolis Algorithm complete!')
            if quiet is False:
                self.plot_metrop(netSE_mtx, BJ, save=save, auto_plot=auto_plot)
            finished.set()
        except KeyboardInterrupt:
            print('\nKeyboard Inturrupt, exiting...\n')
            finished.set()
            wait_until_set.set()
            for t in thread_pool:
                t.terminate()
            exit()
        except Exception:
            PE.PrintException()

    def plot_metrop(self, SE_mtx: ndarray, BJ: list | ndarray,
                    times: Optional[int] = None,
                    quiet: Optional[bool] = True,
                    save: Optional[bool] = False,
                    auto_plot: Optional[bool] = True) -> None:
        """
        Parameters
        ----------
        SE_mtx : `ndarray`
            - spin energy matrix to plot.

        BJ : `list` | `ndarray`
            - list or ndarray of BJ range that generated the SE_mtx.

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
        num_nodes = len(self.internal_arr)
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
                            f' Energy for BJ={BJ}')
        fig.add_trace(go.Scatter(
            x=np.linspace(0, len(SE_mtx[:, 0])-1, len(SE_mtx[:, 0])),
            y=SE_mtx[:, 0] / num_nodes, mode='lines'), row=1, col=1)
        fig.update_xaxes(title_text='Time Steps', row=1, col=1)
        fig.update_yaxes(title_text='Average Spin', row=1, col=1)
        fig.add_trace(go.Scatter(
            x=np.linspace(0, len(SE_mtx[:, 1])-1, len(SE_mtx[:, 1])),
            y=SE_mtx[:, 1], mode='lines'), row=1, col=2)
        fig.update_xaxes(title_text='Time Steps', row=1, col=2)
        fig.update_yaxes(title_text='E / J', row=1, col=2)
        if save is True:
            fname = ('Metropolis' + '_' + str(self.Lshape) + '_' +
                     str(self.internal_arr.rots) + '_' + str(BJ) +
                     str(times) + '.png')
            fig.write_image(fname)
        if auto_plot is True:
            fig.show()

    def randomize(self,
                  voids: bool,
                  probs: list,
                  rand_seed: Optional[int] = None) -> None:
        """
            Parameters
            ----------
            voids : `bool`
                - Truns on random voids in the lattice

            probs : `list`[`float`]
                - list representing the probability of `1`, `0`, `-1`
                where the first two entries represent the the mean and
                standard-deviation of the normal distribution, and the
                percentage that a spin should be a void if enabled.

            rand_seed : Optional[`int`] = Node
                - Seed value to seed random with.
        """
        try:
            if rand_seed is not None:
                if sum(probs) != 100:
                    raise ValueError('The sum of all probabilites must add to'
                                     '100%!')
                random.seed(rand_seed)
                if voids is True:
                    my_list = [-1]*probs[0] + [1]*probs[1] + [0]*probs[2]
                elif voids is False:
                    my_list = [-1]*probs[0] + [1]*probs[1]
                for node in self.internal_arr:
                    rand_num = random.choice(my_list)
                    if rand_num == 0:
                        self.internal_arr.num_voids += 1
                    if node is not None:
                        node.set_spin(rand_num)
            if self.internal_arr.num_voids == self.shape()[0]*self.shape()[1]:
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
