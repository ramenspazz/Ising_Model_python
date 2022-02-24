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
from numpy import int64, int8, integer as Int, floating as Float, ndarray, number
# Functions and Libraries
import tkinter  # noqa : TODO use it later
import sys
import numpy as np
import PrintException as PE
import threading as td  # noqa TODO : use it later
import multiprocessing
import concurrent.futures as CF
import input_funcs as inF
# ignore warning on line 564
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


GNum = Union[number, Number]
MIN_INT = pow(-2, 31)
MAX_INT = pow(2, 31) - 1


def ArrayHash(input_arr: ndarray) -> bytes:
    """
        Mutating function that returns a hash of an array. Will replace any
        instance of -0.0 with 0.
    """
    try:
        if len(input_arr.shape) != 1:
            raise ValueError('Input array must be 1 dimensional!')
        for i in range(len(input_arr)):
            if input_arr[i] is -0.0:  # explicitly look for -0.0, not an error
                input_arr[i] = 0
        return(input_arr.tobytes())
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


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = np.linalg.norm(v1)
    v2_u = np.linalg.norm(v2)
    return(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def distance_between(A: ndarray, B: ndarray) -> float | int:
    try:
        if np.array_equiv(A.shape, B.shape) is False:
            raise ValueError('Shape of the two inputs must be equal!'
                             f' Shapes are {A.shape} and {B.shape}.')
        return(round_num(np.sqrt((A[0]-B[0])**2 + (A[1] - B[1])**2), 5))
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
            self.foward_link: Node = None
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

    def set_forward_link(self, next: Node) -> None:
        self.foward_link = next

    def get_foward_link(self) -> Node:
        return(self.foward_link)

    def get_coords(self) -> ndarray | str:
        return(self.coords)

    def get_coords_and_spin(self) -> ndarray:
        return(np.array([*self.coords, self.spin_state]))

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

    def flip_spin(self):
        """
            Purpose
            -------
            Flips the spin of the node if it is `1` or `-1`. If the spin is `0`
            then do nothing.
        """
        try:
            if self.spin_state == 0:
                sys.stderr.write("empty node, can not flip spin!")
            elif self.spin_state == 1:
                self.spin_state = -1
            elif self.spin_state == -1:
                self.spin_state = 1
            else:
                raise ValueError(
                    f"""
                    Expected spin_state to  be -1, 0 or 1. Got
                    ({self.spin_state}, type: {type(self.spin_state)}).
                    """)
        except Exception:
            PE.PrintException()

    def flip_test(self) -> int:
        try:
            if self.spin_state != 0:
                return(-1 if self.spin_state == 1 else -1)
            else:
                return(0)
        except Exception:
            PE.PrintException()


def rotation_mtx(arg: float | Float) -> NDArray:
    """
        Returns
        -------
        This function returns a 2D rotation matrix of a specified argument.

        Parameters
        ----------
        arg : `float`
            The angle the rotation matrix will rotate counter-clockwise from.
    """
    temp = np.array([
        [np.cos(arg), -np.sin(arg)],
        [np.sin(arg), np.cos(arg)]
    ])
    return(temp)


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
                 basis_arr: Optional[NDArray[Float]] = None):
        """
            Parameters
            ----------
            Scaling factor : `float`
                Scaling factor from unit num_nodes basis vectors.
            basis_arr : Optional[`NDArray[Float]`]
                A `2x2` numpy 'array' of `floats`. Defaults to standard R2
                Euclidean basis if no basis is specified.
        """
        self.scale_factor: Float = scale_factor
        # This is a list of 2 basis vectors in R2
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
        self.b1_plus_b2_len: float = None
        self.nbrs_list: list[list[ndarray]] = list()
        self.__calcneighbor__()
        self.node_dict: CoordinateDict = {}
        self.cord_dict: CoordinateDict = {}
        self.num_nodes: int = int(0)
        self.num_voids: int = int(0)
        self.origin_node: Node = None
        self.__shape = __shape
        self.__threadlauncher__(self.__generation_worker__, False, threads=1)
        self.visited = list()
        self._sum: int | Int = int(0)
        self.fll_generated = False

    def make_linear_linkedlist(self) -> None:
        try:
            cur = self[0, 0]
            for i in range(self.num_nodes):
                coords_p1 = [
                        (i+1) % self.__shape[0],
                        int((i+1) / self.__shape[0])]
                next_node = self[coords_p1]
                if next_node is not None:
                    cur.set_forward_link(next_node)
                    cur = next_node
                else:
                    raise IndexError(
                        f"Index {i} is out of bounds in linked list!")
            self.fll_generated = True
            inF.print_stdout("Sucessfully generated fowardly linked list!")
        except IndexError:
            self.fll_generated = True
            return
        except Exception:
            PE.PrintException()

    def range(self, start: int, stop: int, step: Optional[int] = None):
        curr: Node = self.origin_node
        if step is None:
            step = 1
        curr: Node = self.origin_node
        index_check = stop - start
        index = 0
        curr = self[start]
        while curr is not None and index_check >= index:
            yield(curr)
            for i in range(step):
                curr = curr.get_foward_link()
                if curr is None:
                    return
            index += step
        return

    # TODO analyse this function and include the generation function
    # TODO was working on generation then got tired
    def __threadlauncher__(self,
                           run_function: Callable[[], list],
                           has_retval: bool,
                           args_list: Optional[list] = None,
                           threads: Optional[int] = None) -> int | None:
        res = 0
        if threads is None:
            tc = multiprocessing.cpu_count()  # threadcount
        else:
            tc = threads
        bounds: list = list()
        x = self.__shape[0]
        y = self.__shape[1]
        if tc != 1:
            Div, Rem = Dividend_Remainder(x, y, tc)
        if tc == 1:
            bounds.append([0, x*y-1])
        elif not ((Div == 0) or (Div == 1)):
            if Rem == 0:
                # tc evenly divides the area so split the alttice into
                # tc instances of Div Nodes total.
                for i in range(tc):
                    lower = i*Div
                    upper = (i+1)*Div - 1
                    bounds.append([lower, upper])
            else:
                # create tc instances of Div size to sum
                # tc instances of Div Nodes total.
                for i in range(tc):
                    lower = i*Div
                    upper = (i+1)*Div - 1
                    bounds.append([lower, upper])
                # append the extra
                bounds.append([(tc+1)*Div - 1, x*y-1 - 1])
        elif Div == 0:
            tc = 1
            bounds.append([0, x*y-1 - 1])
        elif Div == 1 and Rem != 0:
            bounds.append([0, tc - 1])
            bounds.append([tc, x*y-1 - 1])
            tc = 2
        with CF.ThreadPoolExecutor(max_workers=tc) as exe:
            if args_list is not None:
                futures = {exe.submit(
                    run_function,
                    bound, args_list): bound for bound in bounds}
            else:
                futures = {exe.submit(
                    run_function,
                    bound): bound for bound in bounds}
            if has_retval:
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
            return()

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
                    sum += nbr.spin_state
            return(sum)
        except Exception:
            PE.PrintException()

    def __iter__(self):
        try:
            if self.origin_node is None:
                raise ValueError("Iterator : origin node is None!")
            curr: Node = self.origin_node
            while curr is not None:
                yield(curr)
                curr = curr.get_foward_link()
            return
        except Exception:
            PE.PrintException()

    def __next__(self):
        try:
            if self.origin_node is None:
                raise ValueError("Iterator : origin node is None!")
            curr: Node = self.origin_node
            while curr is not None:
                yield(curr)
                curr = curr.get_foward_link()
            return
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
                lookup.spin_state = __value
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
                cur = self.origin_node
                for i in range(__NodeIndex):
                    cur = self.origin_node.get_foward_link()
                return(cur)

            else:
                lookup = np.array(__NodeIndex) if type(__NodeIndex) == tuple else __NodeIndex  # noqa E501 lazy
                retval = self.node_dict.get(ArrayHash(lookup))
                return(retval)
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
        if self.rots != 4:
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
                coord = round_num(node.get_coords() + nbr[0], 5)
                index = node.get_index() + nbr[1]
                x_in_bounds = ((index[0] >= 0) and
                               (index[0] < self.__shape[0]))
                y_in_bounds = ((index[1] >= 0) and
                               (index[1] < self.__shape[1]))
                if (x_in_bounds and y_in_bounds) is True:
                    cord_hash = ArrayHash(coord)
                    possible_nb = self.cord_dict.get(cord_hash)
                    if possible_nb is not None:
                        node.add_link(possible_nb)
                    else:
                        new_node = Node(coord, index)
                        self.append(new_node)
                        node.add_link(new_node)
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
                index = np.array([
                        i % self.__shape[0],
                        int(i / self.__shape[0])])
                coord = round_num(index.dot(self.basis_arr), 5)
                node = self[index]
                if node is None:
                    node = Node(coord, index)
                    self.append(node)
                # generate the neighbors for the node
                for nbr in self.nbrs_list:
                    nbr_coord = round_num(coord + nbr[0], 5)
                    index = index + nbr[1]
                    x_in_bounds = ((index[0] >= 0) and
                                   (index[0] < self.__shape[0]))
                    y_in_bounds = ((index[1] >= 0) and
                                   (index[1] < self.__shape[1]))
                    if (x_in_bounds and y_in_bounds) is True:
                        cord_hash = ArrayHash(nbr_coord)
                        possible_nb = self.cord_dict.get(cord_hash)
                        if possible_nb is not None:
                            node.add_link(possible_nb)
                        else:
                            new_node = Node(nbr_coord, index)
                            self.append(new_node)
                            node.add_link(new_node)
        except Exception:
            PE.PrintException()

    # def generate(self) -> None:
    #     try:
    #         if self.basis_arr is None:
    #             raise ValueError("Error, Basis must first be defined!")
    #         inF.print_stdout(" Generating, Please wait...")
    #         # Begin creating the origin node
    #         self.append(Node([0, 0], [0, 0]))
    #         cur = self.origin_node
    #         for i in range(self.__shape[0]):
    #             for j in range(self.__shape[1]):
    #                 if i == j == 0:
    #                     continue
    #                 combo = [i, j]
    #                 next_node = Node(
    #                     np.array(combo).dot(self.basis_arr),
    #                     combo)
    #                 self.append(next_node)
    #                 cur.set_forward_link(next_node)
    #                 cur = next_node
    #         self.fll_generated = True
    #         for value in self.node_dict.values():
    #             self.cord_dict[ArrayHash(value.get_coords())] = value
    #         self.__threadlauncher__(self.__generation_worker__, False, threads=1)
    #         inF.print_stdout(' Generation complete!', end='\n')
    #     except Exception:
    #         PE.PrintException()
