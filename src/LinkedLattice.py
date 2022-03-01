"""
Author: Ramenspazz

This file defines the Node class and the LinkedLattice class.
"""
from __future__ import annotations
from time import sleep
# Typing imports
from typing import Optional, TypedDict, Union, Callable
from numbers import Number

# Math related
import numpy.typing as npt  # noqa
from numpy.typing import NDArray
from numpy import int8, integer as Int, floating as Float, ndarray, number  # noqa E501
from Node import Node

# Functions and Libraries
from SupportingFunctions import GetIndex, ArrayHash, DividendRemainder
from SupportingFunctions import RoundNum
import numpy as np
import PrintException as PE
import InputFuncs as inF
import concurrent.futures as CF
import multiprocessing as mltp
from threading import RLock
import MLTPQueue as queue

# ignore warning on line 564
import warnings
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
        self.lock = RLock()
        self.scale_factor: Float = scale_factor
        self.node_dict: CoordinateDict = {}
        self.cord_dict: CoordinateDict = {}
        self.num_nodes: int = int(0)
        self.num_voids: int = int(0)
        self.origin_node: Node = Node([0, 0], [0, 0])
        self.__shape = __shape
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
        self.__threadlauncher__(self.__generation_worker__, False, threads=1)

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
        x = self.__shape[0]
        y = self.__shape[1]
        if self.tc != 1:
            DivRem = DividendRemainder(x, y, self.tc)
        if self.tc == 1:
            self.bounds.append([0, x*y-1])
        elif not ((DivRem[0] == 0) or (DivRem[0] == 1)):
            if DivRem[1] == 0:
                # tc evenly divides the area so split the alttice into
                # tc instances of DivRem[0] Nodes total.
                for i in range(self.tc):
                    lower = i*DivRem[0]
                    upper = (i+1)*DivRem[0] - 1
                    self.bounds.append([lower, upper])
            else:
                # create tc instances of DivRem[0] size to sum
                # tc instances of DivRem[0] Nodes total.
                for i in range(self.tc):
                    lower = i*DivRem[0]
                    upper = (i+1)*DivRem[0] - 1
                    self.bounds.append([lower, upper])
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
                yield self[GetIndex(i, self.__shape[0])]
        except Exception:
            PE.PrintException()

    def __threadlauncher__(self,
                           run_function: Callable[[], list],
                           has_retval: bool,
                           threads: Optional[int] = None) -> int | None:
        """
            TODO : write docstring
        """
        thread_count = threads if threads is not None else self.tc
        with CF.ThreadPoolExecutor(max_workers=thread_count) as exe:
            if thread_count == 1:
                futures = {exe.submit(
                    run_function,
                    [0, self.__shape[0]*self.__shape[1]])}
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

    def __Sum_Worker__(self,
                       bounds: list | ndarray,
                       results_queue: mltp.Queue,
                       start_queue: queue.MyQueue,
                       finish_queue: queue.MyQueue,
                       start_itt,
                       wait_until_set,
                       finished: mltp) -> int | Int:
        """
            Parameters
            ----------
            bounds : `list` | `ndarray` -> ArrayLike
                - Bounds for worker summation.

            Returns
            -------
            spin_sum : `int` | `numpy.integer`
                - This threads final parital sum to return.
        """
        lower_B = bounds[0]
        upper_B = bounds[1]
        added_to_queue = False
        while not finished.is_set():
            if added_to_queue is False:
                start_queue.put_nowait(1)
                added_to_queue = True
            start_itt.wait()
            psum: np.int64 = 0
            for node in self.range(lower_B, upper_B):
                psum += node.spin_state
            results_queue.put_nowait(psum)
            wait_until_set.wait(timeout=1)
            added_to_queue = False
        return

    def __NN_Worker__(self, bounds: list) -> int | Int:
        psum: np.int64 = 0
        lower_B = bounds[0]
        upper_B = bounds[1]
        for node in self.range(lower_B, upper_B):
            for nbr in node:
                psum += nbr.get_spin()
        return(psum)

    def __iter__(self):
        try:
            if self.origin_node is None:
                raise ValueError("Iterator : origin node is None!")
            for i in range(self.num_nodes):
                yield(self[GetIndex(i, self.__shape[0])])
        except Exception:
            PE.PrintException()

    def __next__(self):
        try:
            if self.origin_node is None:
                raise ValueError("Iterator : origin node is None!")
            for i in range(self.num_nodes):
                yield(self[GetIndex(i, self.__shape[0])])
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
            lookup = self.node_dict.get(ArrayHash(cor))
            if lookup is not None:
                lookup.set_spin = __value
            elif lookup is None:
                raise ValueError(f"Node {__NodeIndex} does not exist!")
        except Exception:
            PE.PrintException()

    def __getitem__(self,
                    __NodeIndex: list | ndarray | int | Int) -> Node | None:
        """
            Parameters
            ----------
            __NodeIndex : `list` of coordinates
                The coordinates are a 1D `list` with two entries, where the
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
                if __NodeIndex > self.num_nodes - 1:
                    raise IndexError(f'Index {__NodeIndex} is out of bounds'
                                     f' for max index of {self.num_nodes-1:}!')
                return(self[GetIndex(__NodeIndex)])
            else:
                lookup = np.array(__NodeIndex) if (type(__NodeIndex) == tuple or type(__NodeIndex) == list) else __NodeIndex  # noqa E501 lazy
                return(self.node_dict.get(ArrayHash(lookup)))
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
                name = ArrayHash(parent.get_index())
                self.node_dict[name] = parent
                self.num_nodes += 1
                for neighbor in child:
                    parent.add_link(neighbor)
                    name = ArrayHash(neighbor.get_index())
                    self.node_dict[name] = neighbor
                    self.num_nodes += 1
            elif isinstance(child, list) is True:
                for neighbor in child:
                    parent.add_link(neighbor)
                    name = ArrayHash(neighbor.get_combination())
                    self.node_dict[name] = neighbor
                    self.num_nodes += 1
            elif parent is None:
                if self.origin_node is None:
                    self.origin_node = child
                name = ArrayHash(child.get_index())
                self.node_dict[name] = child
                self.num_nodes += 1
            elif parent is not None:
                parent.add_link(child)
                name = ArrayHash(child.get_index())
                self.node_dict[name] = child
                self.num_nodes += 1
        except Exception:
            PE.PrintException()

    def get_root(self) -> Node:
        """Returns the origin (A.K.A. root) node."""
        return(self.origin_node)

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

    def set_basis(self, input_basis: NDArray[Float]) -> None:
        """
            Parameters
            ----------
            input_basis : `NDArray[Float]`
                Specifies a basis for the crystal.
        """
        for basis_vector in input_basis:
            self.basis_arr.append(basis_vector)

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
                               (index[0] < self.__shape[0]))
                y_in_bounds = ((index[1] >= 0) and
                               (index[1] < self.__shape[1]))
                if (x_in_bounds and y_in_bounds) is True:
                    possible_nb = self[index]
                    if possible_nb is not None:
                        node.add_link(possible_nb)
                    else:
                        new_node = Node(coord, index)
                        self.append(new_node, node)
        except Exception:
            PE.PrintException()

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
                index = GetIndex(i, self.__shape[0])
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
                                   (index[0] < self.__shape[0]))
                    y_in_bounds = ((index[1] >= 0) and
                                   (index[1] < self.__shape[1]))
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
