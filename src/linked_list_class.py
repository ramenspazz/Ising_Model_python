"""
Author: Ramenspazz

This file defines the Node class and the LinkedLattice class.
"""
from __future__ import annotations
# Typing imports
from typing import Optional, TypedDict, Union, Callable
from numbers import Number
# import matplotlib.pyplot as plt
import numpy.typing as npt  # noqa
from numpy.typing import NDArray
from numpy import int64, int8, integer as Int, floating as Float, ndarray, number  # noqa E501
# Functions and Libraries
import tkinter  # noqa : TODO use it later
import numpy as np
import PrintException as PE
from threading import RLock
import multiprocessing
import concurrent.futures as CF
import input_funcs as inF
# ignore warning on line 564
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


GNum = Union[number, Number]
MIN_INT = pow(-2, 31)
MAX_INT = pow(2, 31) - 1


def get_index(i: int, x_range: int) -> ndarray:
    """
        Returns
        -------
        index : `ndarray`[`int`]
            - A 1D numpy ndarray with 2 entries listing a (x,y) pair given an
            input integer `i` that maps to the 2D plane.
    """
    try:
        return(np.array(
            [i % x_range,
             int(i / x_range)]))
    except Exception:
        PE.PrintException()


def ArrayHash(input_arr: ndarray) -> bytes:
    """
        Pray IEEE 754 is not in the array.

        Returns
        -------
        hash : `bytes`
            - byte hash of array using the numpy function `tobytes`.
    """
    try:
        return(hash(input_arr.tobytes()))
    except Exception:
        PE.PrintException()


def Dividend_Remainder(n: int | Int,
                       m: int | Int,
                       t: int | Int) -> list[int | Int]:
    """
        Purpose
        -------
        Returns a list with the dividend and remainder of (nm)/t.
        Runs in Theta(log(n)), Omega(n), O(nlog(n))

        Returns
        -------
        Remainder : `int`
            - Represents the remainder after division.

        Dividend : `int`
            - Represents a whole number p for n*m>=pt, where n, m, p, and t are
            all integers.

    """
    try:
        t = int(t)
        area = int(n * m)

        if not (MAX_INT > area > MIN_INT):
            raise ValueError('Input n and m are too large!')
        if t > area:
            return(0, area)
        elif t == area:
            return(1, 0)

        test = int(0)
        div = int(0)
        div_overflow = int(0)
        OF_on = int(0)
        prev = int(0)
        while True:
            test = int(t * div_overflow) + int(t << div)

            if prev > area and test > area:
                return([int((t << div) / t + div_overflow),
                        area-t*np.floor(area / t)])

            elif prev < area and test > area:
                return([div,
                        area-t*np.floor(area / t)])

            if test == area:
                return([int((t << div) / t + div_overflow),
                        area-t*np.floor(area / t)])

            elif test < area:
                prev = test
                div += 1
                continue

            elif prev < area and OF_on == 0:
                div_overflow += int((t << (div - 1)) / t)
                prev = test
                OF_on = 1
                div = 1
                continue
    except Exception:
        pass


def round_num(input: Float,
              figures: Int,
              round_fig: Optional[Int] = None) -> Float:
    """
        Purpose
        -------
        Rounds a number to a given number of figures, and optionally equates a
        number to zero if it is within the number of significant figures
        passed.

        Parameters
        ----------
        input : `float`
            The input number to round.
        figures : `int`
            The number of significant figures to round the input to.
        round_fig : Optional[`int`]
            The number of sigificant figures representing how close absolute
            value of the input must be to zero to round the number to
            `int`(`0`).
    """
    try:
        if isinstance(input, ndarray) is True:
            ret_val = np.zeros(input.shape)
            for i in range(len(input)):
                if round_fig is not None:
                    ret_val[i] = np.round(input[i], round_fig)
                else:
                    ret_val[i] = np.round(input[i], figures)
                    if ret_val[i] is -0.0:  # intentional check for -0.0
                        ret_val[i] = 0
            return(ret_val)
        else:
            if np.abs(input) < 10**(-figures):
                return(0)
            else:
                if round_fig is not None:
                    return(np.round(input, round_fig))
                else:
                    return(np.round(input, figures))
    except Exception:
        PE.PrintException()


def array_isclose(A: ndarray | list,
                  B: ndarray | list,
                  round_fig: Optional[float] = 1E-09) -> bool:
    try:
        if np.array_equiv(A.shape, B.shape) is False:
            raise ValueError('Shape of the two inputs must be equal!'
                             f' Shapes are {A.shape} and {B.shape}.')
        for A_val, B_val in zip(A, B):
            if np.abs(A_val - B_val) <= round_fig:
                continue
            else:
                return(False)
        return(True)
    except Exception:
        PE.PrintException()


class Node:
    """
        Overview
        --------
        This class defines an object that stores a x-y position as a list and a
        refrence to its neighbors stored in a `list`. This is used by the
        LinkedLattice class to store a linkedlist style lattice of Nodes. Spin
        states are `-1` or `+1` with respect to the z axis spin 0 denotes an
        empty node, start all nodes as empty.
    """
    def __init__(self,
                 coords: NDArray[Float],
                 combination: Optional[list | NDArray[Int]] = None,
                 spin_state: Optional[int] = None) -> None:
        """
            Parameters
            ----------
            coords : `list[Float]` | `NDArray[Float]`
                Array of x and y coordinate pair for the `Node`.
            combination : Optional[`list` | `NDArray[Int]`]
                An index of the basis combination of the `Node`.
            spin_state : Optional[`int`]
                An `int` of `-1`, `0`, or `1` representing the spin
                state or if the `Node` is a void
        """
        try:
            if spin_state is None or spin_state == 0:
                self.spin_state = 0
            elif spin_state == 1 or -1:
                self.spin_state = spin_state
            else:
                raise ValueError(
                    f"""Optional parameter spin_state must be 1, 0, -1 or None!
                        Got ({spin_state}, type={type(spin_state)})
                        """)
            if combination is not None:
                self.combination: NDArray[Int] = np.array(combination, int64)
            elif combination is None:
                self.combination: NDArray[Int] = None
            self.coords: NDArray[Float] = np.array(coords)
            self.links = list()
        except Exception:
            PE.PrintException()

    def __len__(self):
        return(len(self.links))

    def __iter__(self) -> Node:
        try:
            for link in self.links:
                yield(link)
        except Exception:
            PE.PrintException()

    def __setitem__(self, __value: int) -> None:
        self.spin_state = __value

    def __rmul__(self, LHS: int | Node) -> int:
        """
            Overview
            --------
            Multiplication operator overload for nodes.

            Parameters
            ----------
            RHS : `int` | `Node`
                Right hand side for multiplication.
        """
        if isinstance(LHS, Node) is True:
            return(self.spin_state * LHS.spin_state)
        else:
            return(self.spin_state * LHS)

    def __mul__(self, RHS: int | Node) -> int:
        """
            Overview
            --------
            Multiplication operator overload for nodes.

            Parameters
            ----------
            RHS : `int` | `Node`
                Right hand side for multiplication.
        """
        if isinstance(RHS, Node) is True:
            return(self.spin_state * RHS.spin_state)
        else:
            return(self.spin_state * RHS)

    def add_link(self, new_links: Node) -> None:
        """
            Adds a link to a Node. The links can either be a `list[Node]` or a
            single `Node`.
        """
        try:
            if isinstance(new_links, Node) is True:
                self.links.append(new_links)
            elif isinstance(new_links, list) is True:
                for item in new_links:
                    self.links.append(item)
            else:
                raise ValueError("Incorrent type for new_links!"
                                 " Expected type Node or list[Node] but got "
                                 f"{type(new_links)}!")
        except Exception:
            PE.PrintException()

    def get_connected(self) -> Node:
        """
            Returns
            -------
            generator to next neighbor
        """
        return(self.links)

    def num_connected(self) -> int:
        """
            Returns
            -------
            nNumLinks : `int`
                The integer number of nodes connected to `self`.
        """
        return(len(self.links))

    def get_coords(self) -> ndarray | str:
        return(self.coords)

    def get_coords_and_spin(self) -> ndarray:
        return(np.array([*self.coords, self.get_spin()]))

    def get_index(self) -> NDArray | str:
        """
            Parameters
            ----------
            ReturnString : Optional[`bool`]
                Flag to trigger returning a `str` instead of a `NDArray[int]`
                representing the basis combination of this nodes coordinates.
            Returns
            -------
                combination : `NDArray[int]` | `str`
                    The basis combination of this nodes coordinates.
        """
        return(self.combination)

    def get_spin(self) -> int:
        """
            Returns
            -------
            An integer representation of the spinstate as `-1` or `1`. Can
            return `0` to represent a lattice void.
        """
        return(self.spin_state)

    def set_spin(self, spin) -> None:
        self.spin_state = spin

    def flip_spin(self):
        """
            Purpose
            -------
            Flips the spin of the node if it is `1` or `-1`. If the spin is `0`
            then do nothing.
        """
        try:
            if self.get_spin() == 0:
                return
            elif self.get_spin() == 1:
                self.spin_state = -1
                return
            elif self.get_spin() == -1:
                self.spin_state = 1
                return
        except Exception:
            PE.PrintException()

    def flip_test(self) -> int:
        try:
            if self.get_spin() != 0:
                return(-1 if self.get_spin() == 1 else -1)
            else:
                return(0)
        except Exception:
            PE.PrintException()


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
            self.tc = multiprocessing.cpu_count()  # threadcount
        else:
            self.tc = threads
        self.setup_basis(basis_arr)
        self.setup_multithreading()
        self.__calcneighbor__()
        self.__threadlauncher__(self.__generation_worker__, False)

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
            Div, Rem = Dividend_Remainder(x, y, self.tc)
        if self.tc == 1:
            self.bounds.append([0, x*y-1])
        elif not ((Div == 0) or (Div == 1)):
            if Rem == 0:
                # tc evenly divides the area so split the alttice into
                # tc instances of Div Nodes total.
                for i in range(self.tc):
                    lower = i*Div
                    upper = (i+1)*Div - 1
                    self.bounds.append([lower, upper])
            else:
                # create tc instances of Div size to sum
                # tc instances of Div Nodes total.
                for i in range(self.tc):
                    lower = i*Div
                    upper = (i+1)*Div - 1
                    self.bounds.append([lower, upper])
                # append the extra
                self.bounds.append([(self.tc+1)*Div - 1, x*y-1 - 1])
        elif Div == 0:
            self.tc = 1
            self.bounds.append([0, x*y-1 - 1])
        elif Div == 1 and Rem != 0:
            self.bounds.append([0, self.tc - 1])
            self.bounds.append([self.tc, x*y-1 - 1])
            self.tc = 2

    def range(self, start: int, stop: int, step: Optional[int] = None) -> Node:
        try:
            for i in range(start, stop):
                yield self[get_index(i, self.__shape[0])]
        except Exception:
            PE.PrintException()

    # TODO analyse this function and include the generation function
    # TODO was working on generation then got tired
    def __threadlauncher__(self,
                           run_function: Callable[[], list],
                           has_retval: bool,
                           args_list: Optional[list] = None,
                           threads: Optional[int] = None) -> int | None:
        """
            TODO : write docstring
        """
        thread_count = threads if threads is not None else self.tc
        if thread_count == 1:
            bounds = [0, self.__shape[0]*self.__shape[1]]
        else:
            bounds = self.bounds
        with CF.ThreadPoolExecutor(max_workers=thread_count) as exe:
            if args_list is not None:
                futures = {exe.submit(
                    run_function,
                    bound, args_list): bound for bound in bounds}
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

    def __Sum_Worker__(self, bounds: list | ndarray) -> int | Int:
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
        # Begin by creating a visit list for keeping track of nodes we still
        # need to visit and a visited list that will keep track of what has
        # already been visited.
        try:
            sum: np.int64 = 0
            lower_B = bounds[0]
            upper_B = bounds[1]
            for node in self.range(lower_B, upper_B):
                sum += node.get_spin()
            return(sum)
        except Exception:
            PE.PrintException()

    def __NN_Worker__(self, bounds: list) -> int | Int:
        try:
            sum: np.int64 = 0
            lower_B = bounds[0]
            upper_B = bounds[1]
            for node in self.range(lower_B, upper_B):
                for nbr in node:
                    sum += nbr.get_spin()
            return(sum)
        except Exception:
            PE.PrintException()

    def __iter__(self):
        try:
            if self.origin_node is None:
                raise ValueError("Iterator : origin node is None!")
            for i in range(self.num_nodes):
                yield(self[get_index(i, self.__shape[0])])
        except Exception:
            PE.PrintException()

    def __next__(self):
        try:
            if self.origin_node is None:
                raise ValueError("Iterator : origin node is None!")
            for i in range(self.num_nodes):
                yield(self[get_index(i, self.__shape[0])])
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
                return(self[get_index(__NodeIndex)])
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
                coord = round_num(node.get_coords() + nbr[0], 10)
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
            upper_B = bounds[1] + 1
            for i in range(lower_B, upper_B):
                index = get_index(i, self.__shape[0])
                coord = round_num(index.dot(self.basis_arr), 10)
                node = self[index]
                if node is None:
                    node = Node(coord, index)
                    self.append(node)
                # generate the neighbors for the node
                for nbr in self.nbrs_list:
                    coord = round_num(node.get_coords() + nbr[0], 10)
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
