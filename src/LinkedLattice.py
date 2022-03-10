"""
Author: Ramenspazz

This file defines the Node class and the LinkedLattice class.
"""
from __future__ import annotations
from queue import Empty
from random import random
from time import sleep
# from random import random
# Typing imports
from typing import Optional, TypedDict, Union, Callable
from numbers import Number

# Math related
import numpy.typing as npt  # noqa
from numpy.typing import NDArray
from numpy import float64, int64, int8, integer as Int, floating as Float, ndarray, number  # noqa E501
from Node import Node

# Functions and Libraries
from SupportingFunctions import GetIndex, Array2Bytes, DividendRemainder
from SupportingFunctions import RoundNum
import numpy as np
import PrintException as PE
import InputFuncs as inF
import concurrent.futures as CF
import multiprocessing as mltp
from threading import RLock
import MLTPQueue as MLTPqueue

# ignore warning on line 564
import warnings

from WaitListLock import WaitListLock
from pyQueue import LLQueue, QueueEmpty  # noqa
warnings.filterwarnings("ignore", category=RuntimeWarning)


GNum = Union[number, Number]
MIN_INT = pow(-2, 31)
MAX_INT = pow(2, 31) - 1


class CoordinateDict(TypedDict):
    name: bytes
    value: Node


class NDArrayDict(TypedDict):
    name: int
    value: ndarray


class LinkedLattice:
    """
        Overview
        --------
        This class defines a linkedlist of spins that can be `-1` or `1` for
        the spin state and `0` to represent a empty lattice position.
    """
    def __init__(self,
                 scale_factor: Float,
                 __shape: list,
                 J: float64,
                 basis_arr: Optional[NDArray[Float]] = None,
                 threads: Optional[int] = None):
        """
            Parameters
            ----------
            Scaling factor : `float`
                Scaling factor from unit num_nodes basis vectors.
            basis_arr : Optional[`NDArray[Float]`]
                A `2x2` numpy 'array' of `floats`. Defaults to standard R2
                Euclidean basis if no basis is specified.
        """
        try:
            self.lock = RLock()
            self.J = J
            self.scale_factor: Float = scale_factor
            self.node_dict: CoordinateDict = {}
            self.cord_dict: CoordinateDict = {}
            self.num_nodes: int = int(0)
            self.num_voids: int = int(0)
            self.origin_node: Node = Node([0, 0], [0, 0])
            self.Shape = __shape
            self.b1_plus_b2_len: float = None
            self.nbrs_list: list[list[ndarray]] = list()
            self.visited = list()
            self._sum: int | Int = int(0)
            self.rots = 0
            self.basis_arr = basis_arr
            self.bounds: list = []
            if threads is None:
                self.tc = mltp.cpu_count()  # threadcount
            else:
                self.tc = threads
            self.setup_basis(basis_arr)
            self.setup_multithreading()
            self.__calcneighbor__()
            inF.print_stdout('Generating structure...')
            self.__threadlauncher__(self.__generation_worker__, False,
                                    generate_call=True)
            inF.print_stdout('Generation complete!', end='\n')
        except Exception:
            PE.PrintException()

    def __calcneighbor__(self) -> None:
        """
            Purpose
            -------
            Construct and returns a dictonary of a possible lattice points that
            are nearest neighbors to an origin point that can be arbitrarily
            translated.
        """
        b1 = self.basis_arr[0]
        b2 = self.basis_arr[1]  # TODO: maybe include this too?

        self.nbrs_list.append([b1, np.array([1, 0], int8)])
        self.nbrs_list.append([-1*b1, np.array([-1, 0], int8)])
        self.nbrs_list.append([b2, np.array([0, 1], int8)])
        self.nbrs_list.append([-1*b2, np.array([0, -1], int8)])
        if self.rots == 3 or self.rots == 6:
            self.nbrs_list.append([b1 - b2, np.array([1, -1], int8)])
            self.nbrs_list.append([b2 - b1, np.array([-1, 1], int8)])

    def __iter__(self):
        return(self.__next__())

    def __next__(self) -> Node:
        try:
            if self.origin_node is None:
                raise ValueError("Iterator : origin node is None!")
            for i in range(self.num_nodes):
                yield(self[GetIndex(i, self.Shape[0])])
        except Exception:
            PE.PrintException()

    def __setitem__(self, __NodeIndex: list | ndarray,
                    __value: int) -> None:
        """
            Sets the `Node` spin value at `__NodeIndex` equal to the __value.

            Parameters
            ----------
            __NodeIndex : `list` of coordinates
                The coordinates are a 1D `list` with two entries, where the
                first entry represents the x and the second y the coordinate
                respectivly.
            __value : `int`
                Value to assign at the specified __NodeIndex
        """
        try:
            if type(__NodeIndex) == tuple or type(__NodeIndex) == list:
                cor = np.array(__NodeIndex)
            else:
                __NodeIndex
            lookup = self.node_dict.get(Array2Bytes(cor))
            if lookup is not None:
                lookup.set_spin = __value
            elif lookup is None:
                raise ValueError(f"Node {__NodeIndex} does not exist!")
        except Exception:
            PE.PrintException()

    def __getitem__(self,
                    __NodeIndex: ndarray | int | Int) -> Node:
        """
            Parameters
            ----------
            __NodeIndex : `ndarray`[`Int`] | `int`
                The coordinates are a 1D `ndarray` with two entries, where the
                first entry represents the x and the second y the coordinate
                respectivly.

            Returns
            -------
                node : `Node`
                    `Node` refrence at location `__NodeIndex`.
                `None`
                    No existing `Node` was found, return `None`
        """
        try:
            if type(__NodeIndex) == int:
                return(self[GetIndex(__NodeIndex)])
            else:
                return(self.node_dict.get(Array2Bytes(__NodeIndex)))
        except Exception:
            PE.PrintException()

    def __len__(self):
        """
            Returns
            -------
            num_nodes : `int`
                Number of total nodes.
        """
        return(self.num_nodes - self.num_voids)

    def __generation_worker__(self, bounds: list | ndarray) -> None:
        """
            Parameters
            ---------
            bounds : `list` | `ndarray` -> ArrayLike
                - Bounds for worker summation.

            Returns
            -------
            spin_sum : `int` | `numpy.integer`
                - This threads final parital sum to return.
        """
        try:
            lower_B = bounds[0]
            upper_B = bounds[1]
            for i in range(lower_B, upper_B):
                index = GetIndex(i, self.Shape[0])
                coord = RoundNum(index.dot(self.basis_arr), 10)
                node = self[index]
                if node is None:
                    node = Node(coord, index)
                    self.append(node)
                # generate the neighbors for the node
                for nbr in self.nbrs_list:
                    coord = RoundNum(node.get_coords() + nbr[0], 10)
                    index = node.get_index() + nbr[1]
                    x_in_bounds = ((index[0] >= 0) and
                                   (index[0] < self.Shape[0]))
                    y_in_bounds = ((index[1] >= 0) and
                                   (index[1] < self.Shape[1]))
                    if x_in_bounds and y_in_bounds:
                        with self.lock:
                            possible_nb = self[index]
                        if possible_nb is not None:
                            node.add_link(possible_nb)
                        else:
                            new_node = Node(coord, index)
                            self.append(new_node, node)
        except Exception:
            PE.PrintException()

    def __threadlauncher__(self,
                           run_function: Callable[[], list],
                           has_retval: bool,
                           generate_call: Optional[bool] = False,
                           threads: Optional[int] = None) -> int | None:
        """
            TODO : write docstring
        """
        try:
            thread_count = threads if threads is not None else self.tc
            with CF.ThreadPoolExecutor(max_workers=thread_count) as exe:
                if thread_count == 1:
                    futures = {exe.submit(
                        run_function,
                        [0, self.Shape[0]*self.Shape[1]])}
                else:
                    futures = {exe.submit(
                        run_function,
                        bound): bound for bound in self.bounds}
                if has_retval:
                    res = 0
                    for future in CF.as_completed(futures):
                        try:
                            data = future.result()
                        except Exception:
                            return(0)
                        else:
                            res += data
            if has_retval:
                return(res)
            else:
                return
        except Exception:
            PE.PrintException()

    def setup_basis(self, basis_arr):
        if basis_arr is not None:
            if isinstance(basis_arr, ndarray) is not True:
                basis_arr = np.array(basis_arr)
            self.basis_arr: NDArray[Float] = basis_arr
            # Begin by calculating the inner-products of each possible
            # combination to use in the next step.
            a = np.float64(self.basis_arr[0].dot(self.basis_arr[1]))
            b = np.float64(self.basis_arr[0].dot(self.basis_arr[0]))
            c = np.float64(self.basis_arr[1].dot(self.basis_arr[1]))
            arg = np.float64(a / (np.sqrt(b) * np.sqrt(c)))
            # This is A*B/|AB|=cos(C), using the a, b and c above
            __rots = np.float64(2*np.pi/np.arccos(arg))
            __ciel = np.ceil(__rots)
            __floor = np.floor(__rots)
            # d_up and d_dn are the percent change between the
            # unmutated number and the resulting floor and
            # cieling operation respectivly.
            d_up = np.abs(1-__rots/__ciel)
            d_dn = np.abs(1-__rots/__floor)
            if d_dn < 10**(-12):
                self.rots: int | np.float64 = int(np.floor(__rots))
            elif d_up < 10**(-12):
                self.rots: int | np.float64 = int(np.ceil(__rots))
            else:
                self.rots: int | np.float64 = np.float64(__rots)
        elif basis_arr is None:
            self.basis_arr: NDArray[Float] = np.array(
                [[1, 0], [0, 1]])
            self.rots: int = 4

    def setup_multithreading(self):
        x = self.Shape[0]
        y = self.Shape[1]
        if self.tc != 1:
            DivRem = DividendRemainder(x*y, self.tc)
            # print(f'\nDivisor and Remainder = {DivRem}')
        if self.tc == 1:
            self.bounds.append([0, x*y-1])
        elif not ((DivRem[0] == 0) or (DivRem[0] == 1)):
            # create tc instances of DivRem[0] size to sum
            # tc instances of DivRem[0] Nodes total.
            for i in range(self.tc):
                lower = i*DivRem[0]
                upper = (i+1)*DivRem[0] - 1
                self.bounds.append([lower, upper])
            if DivRem[1] > 0:
                # append the extra
                self.bounds.append([(self.tc+1)*DivRem[0] - 1, x*y-1 - 1])
        elif DivRem[0] == 0:
            self.tc = 1
            self.bounds.append([0, x*y-1 - 1])
        elif DivRem[0] == 1 and DivRem[1] != 0:
            self.bounds.append([0, self.tc - 1])
            self.bounds.append([self.tc, x*y-1 - 1])
            self.tc = 2

    def range(self, start: int, stop: int, step: Optional[int] = None) -> Node:
        try:
            for i in range(start, stop):
                yield self[GetIndex(i, self.Shape[0])]
        except Exception:
            PE.PrintException()

    def Sum_Worker(self,
                   bounds: list | ndarray,
                   thread_num: int,
                   result_queue: MLTPqueue.ThQueue,
                   ready_sum: WaitListLock,
                   finished_sum: mltp._EventType) -> int | Int:
        """
            Parameters
            ----------
            bounds : `list` | `ndarray` -> ArrayLike
                - Bounds for worker summation.

            results_queue : `multithreading`.Queue
                -
            start_queue : `queue`.`MyQueue`
                - User defined class from stack exchange that `super`'s the
                multiprocessing queue class.

            start_itt : `multithreading`.`_EventType`
                -

            wait_until_set : `multithreading`.`_EventType`
                -

            finished : `multithreading`.`_EventType`

            Returns
            -------
            spin_sum : `int` | `numpy.integer`
                - This threads final parital sum to return.
        """
        lower_B = bounds[0]
        upper_B = bounds[1]
        while True:
            psum: int64 = 0
            ready_sum.Wait(thread_num)
            if finished_sum.is_set() is True:
                break
            for node in self.range(lower_B, upper_B):
                node.unmark_node()
                psum += node.get_spin()
            result_queue.put_nowait(psum)
        return

    def Energy_Worker(self,
                      bounds: list | ndarray,
                      thread_num: int,
                      result_queue: MLTPqueue.ThQueue,
                      ready_energy: WaitListLock,
                      finished_energy: mltp._EventType) -> int | Int:
        lower_B = bounds[0]
        upper_B = bounds[1]
        while True:
            if finished_energy.is_set() is True:
                break
            ready_energy.Wait(thread_num)
            energy: int64 = 0
            for node in self.range(lower_B, upper_B):
                nbr_psum: int64 = 0
                if node.get_spin() == 0:
                    continue
                for nbr in node:
                    nbr_psum += nbr.get_spin()
                energy += nbr_psum * node.get_spin()
            result_queue.put_nowait(energy)
        return

    def SpinEnergy_Worker(self,
                          bounds: list | ndarray,
                          thread_num: int,
                          result_queue: MLTPqueue.ThQueue,
                          ready_SE: WaitListLock,
                          finished_SE: mltp._EventType) -> int | Int:
        lower_B = bounds[0]
        upper_B = bounds[1]
        while True:
            SE_vec = np.zeros(2, int64)
            psum: int64 = 0
            if finished_SE.is_set() is True:
                break
            ready_SE.Wait(thread_num)
            for node in self.range(lower_B, upper_B):
                node.unmark_node()
                if node.get_spin() == 0:
                    continue
                psum = 0
                SE_vec[0] += node.get_spin()
                for nbr in node:
                    psum += nbr.get_spin()
                SE_vec[1] += psum * node.get_spin()
            SE_vec[1] *= -1
            result_queue.put_nowait(SE_vec)
        return

    def Path_Worker(self,
                    thread_num: int,
                    cluster: MLTPqueue.ThQueue,
                    work_queue_path: MLTPqueue.ThQueue,
                    result_queue: MLTPqueue.ThQueue,
                    ready_path: WaitListLock[float],
                    finished_path: mltp._EventType):
        """
            Purpose
            -------
            This thread worker selects a cluster of spins to be flipped.
        """
        while True:
            balcond = ready_path.Wait(thread_num)
            if finished_path.is_set() is True:
                break
            while True:
                try:
                    cur = self[work_queue_path.get(timeout=0.001)]
                    cur_Si = cur.get_spin()
                except Empty:
                    # break loop when stack is empty or timeout
                    break
                for nbr in cur:
                    nbr_Si = nbr.get_spin()
                    if (nbr_Si == 0 or nbr_Si != cur_Si or
                            nbr.marked is True):
                        continue
                    nbr.mark_node()
                    if random() < balcond:
                        nbr_index = nbr.get_index()
                        cluster.put_nowait(nbr_index)
                        work_queue_path.put_nowait(nbr_index)
            result_queue.put_nowait(1)
        return

    def append(self,
               child: list[Node] | Node,
               parent: Optional[Node] = None) -> None:
        """
            Add a Node to the lattice. Adds Node as the origin Node if no other
            Nodes exist. If passing a list Nodes with a specified parent, this
            function will add each element of the list to the parent Node.

            Parameters
            ----------
            child : `list[Node]` | `Node`
                Node to append.
            parent : Optional[`Node`]
                Node that child gets linked to.
        """
        try:
            # This works by taking the new nodes basis combination and first
            # computes the angle between the two vectors and finds how many
            # neighbors a given node can have maximum.
            if child == parent:
                raise ValueError("""
                Can not append, Child and Parrent are the same!
                """)
            elif isinstance(child, list) and parent is not None:
                name = Array2Bytes(parent.get_index())
                self.node_dict[name] = parent
                self.num_nodes += 1
                for neighbor in child:
                    parent.add_link(neighbor)
                    name = Array2Bytes(neighbor.get_index())
                    self.node_dict[name] = neighbor
                    self.num_nodes += 1
            elif isinstance(child, list) is True:
                for neighbor in child:
                    parent.add_link(neighbor)
                    name = Array2Bytes(neighbor.get_combination())
                    self.node_dict[name] = neighbor
                    self.num_nodes += 1
            elif parent is None:
                if self.origin_node is None:
                    self.origin_node = child
                name = Array2Bytes(child.get_index())
                self.node_dict[name] = child
                self.num_nodes += 1
            elif parent is not None:
                parent.add_link(child)
                name = Array2Bytes(child.get_index())
                self.node_dict[name] = child
                self.num_nodes += 1
        except Exception:
            PE.PrintException()

    def print_dict(self) -> None:
        """
            Purpose
            -------
            Prints the objects node dictionary, `node_dict` to `stdout`.
        """
        inF.print_stdout('')
        for key, val in self.node_dict.items():
            inF.print_stdout(f"key={key}, value={val}")
        inF.print_stdout('')

    def NodesNeighbors(self, node: Node) -> None:
        """
            Purpose
            -------
            Checks the generated dictionary from __calcneighbor__ and returns
            `True` or `False`.
        """
        try:
            for nbr in self.nbrs_list:
                coord = RoundNum(node.get_coords() + nbr[0], 10)
                index = node.get_index() + nbr[1]
                x_in_bounds = ((index[0] >= 0) and
                               (index[0] < self.Shape[0]))
                y_in_bounds = ((index[1] >= 0) and
                               (index[1] < self.Shape[1]))
                if (x_in_bounds and y_in_bounds) is True:
                    possible_nb = self[index]
                    if possible_nb is not None:
                        node.add_link(possible_nb)
                    else:
                        new_node = Node(coord, index)
                        self.append(new_node, node)
        except Exception:
            PE.PrintException()
