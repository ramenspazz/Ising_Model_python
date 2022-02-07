"""
Author: Ramenspazz

This file defines the Node class and the LinkedLattice class.
"""
from __future__ import annotations
# Typing imports
from typing import Optional, Union, TYPE_CHECKING  # noqa F401
from numbers import Number
import numpy.typing as npt  # noqa
from numpy.typing import NDArray
from numpy import integer as Int, floating as Float, ndarray, number
# Functions and Libraries
import tkinter  # noqa : TODO use it later
import sys
import numpy as np
from numpy.linalg import norm
import PrintException as PE
import re

GNum = Union[number, Number]
r_str: str = ['-0.0', '-0.', '-0']


def StripString(input: NDArray[number] | GNum | str,
replace: list[str]) -> str:  # noqa E128
    """
        Purpose
        -------
        This function is used to sanitize input to remove all forms of
        parenthaticals, and to remove the IEEE `-0`, `-0.`, and `-0.0`.
        This function only removes complete patterns and not a pattern embeded
        within another complete number; IE, -0.01 contains -0.0. but is a
        seperate number entirely even though the pattern -0.0 is indeed
        contained within. I honestly dont know why they declined to remove
        these negative zero abominations but whatever. Sthap messing with my
        floating points IEEE :(

        Parameters
        ----------
            input : `NDArray` | `Number` | `str`
                The input to sanatize.
            replace : `list[str]`
                A list of strings containing the patterns to be removed.
    """
    if isinstance(input, str) is not True:
        name = str(input)
    else:
        name = input
    # check if the first character is a space
    checks = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    name = name.translate({ord(c): None for c in '[](),'})
    for repl in replace:
        __iter = re.finditer(repl, name)
        hits = list()
        for i, val in enumerate(__iter):
            # get the begining and ending index of the search pattern of
            # a checks list item.
            temp = val.span()
            try:
                next_char = name[temp[1]]
                if next_char not in checks:
                    hits.append(temp[0])
            except IndexError:
                # If at the end of the string and we attempt to overstep the
                # boundary, we can throw an error and catch it to signal that
                # the search pattern happens at the end of the string. After
                # this we add the index to the list to be shift out splice
                # index later.
                hits.append(temp[0])
                continue
        for i, val in enumerate(hits):
            name = name[:val - i] + name[val + 1 - i:]
    # remove whitespace at the begining of the string
    if name[0] == ' ':
        name = name[1:len(name)]
    # remove whitespace after first complete number
    temp_str = name.split(' ')
    name = ''
    for i, item in enumerate(temp_str):
        if item == '':
            continue
        else:
            if not (i == len(temp_str) - 1):
                name += item + ' '
            else:
                name += item
    return(name)


def round_num(input: Float, figures: Int,
round_fig: Optional[Int] = None) -> Float:  # noqa E128
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
        if np.abs(input) < 10**(-figures):
            return(0)
        else:
            if round_fig is not None:
                return(np.round(input, round_fig))
            else:
                return(np.round(input, figures))
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
        coords: NDArray[Float],  # noqa E128
        combination: Optional[list | NDArray[Int]] = None,  # noqa E128
        spin_state: Optional[int] = None) -> None:  # noqa E125
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
                    f"""\nOptional parameter spin_state must be 1, 0, -1 or None!
                        \nGot ({spin_state}, type={type(spin_state)})
                        \n""")
            if combination is not None:
                self.combination: NDArray[Int] = np.array(combination)
            elif combination is None:
                self.combination: NDArray[Int] = None
            self.coords: NDArray[Float] = np.array(coords)
            self.links = list()
        except Exception:
            PE.PrintException()

    def __len__(self):
        return(len(self.links))

    def __iter__(self):
        try:
            # Check if the node has any neighbors
            if len(self.links) > 0:
                for link in self.links:
                    yield link
            else:
                yield self
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
            else:
                raise ValueError(f"""
                Incorrent type for new_links!
                Expected type Node or list[Node] but got {type(new_links)}!\n
                """)
        except Exception:
            PE.PrintException()

    def get_connected(self) -> 'list[Node]':
        """
            Returns
            -------
            links : `list[Node]`
                A list of nodes connected to `self`.
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

    def get_coords(self,
    ReturnString: Optional[bool] = None) -> list[GNum] | str:  # noqa E128
        if ReturnString is None:
            return(self.coords.tolist())
        elif ReturnString is True:
            return(StripString(self.coords, r_str))

    def get_combination(self,
    ReturnString: Optional[bool] = None) -> NDArray[Int] | str:  # noqa E128
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
        if ReturnString is None:
            return(self.combination)
        elif ReturnString is True:
            return(StripString(self.combination, r_str))

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
            if not (self.spin_state == 0):
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


class LinkedLattice:
    """
        Overview
        --------
        This class defines a linkedlist of spins that can be `-1` or `1` for
        the spin state and `0` to represent a empty lattice position.
    """
    def __init__(self,
        scale_factor: Float,  # noqa E128
        __shape: list,  # noqa E128
        basis_arr: Optional[NDArray[Float]] = None):  # noqa E128
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
            a = np.float128(self.basis_arr[0].dot(self.basis_arr[1]))
            b = np.float128(self.basis_arr[0].dot(self.basis_arr[0]))
            c = np.float128(self.basis_arr[1].dot(self.basis_arr[1]))
            arg = np.float128(a / (np.sqrt(b) * np.sqrt(c)))
            # This is A*B/|AB|=cos(C), using the a, b and c above
            __rots = np.float128(2*np.pi/np.arccos(arg))
            __ciel = np.ceil(__rots)
            __floor = np.floor(__rots)
            # d_up and d_dn are the percent change between the
            # unmutated number and the resulting floor and
            # cieling operation respectivly.
            d_up = np.abs(1-__rots/__ciel)
            d_dn = np.abs(1-__rots/__floor)
            if d_dn < 10**(-12):
                self.rots: int | np.float128 = int(np.floor(__rots))
            elif d_up < 10**(-12):
                self.rots: int | np.float128 = int(np.ceil(__rots))
            else:
                self.rots: int | np.float128 = np.float128(__rots)
        elif basis_arr is None:
            self.basis_arr: NDArray[Float] = np.array(
                [[1, 0], [0, 1]])
            self.rots: int = 4
        self.y_length: float = None
        self.neighbors_ref: dict = self.__calcneighbor__()
        self.node_dict: dict[str, Node] = dict()
        self.num_nodes: int = int(0)
        self.num_voids: int = int(0)
        self.origin_node: Node = None
        self.__shape = __shape
        self.generate(__shape)
        self.visited = list()

    def __calcneighbor__(self) -> dict:
        """
            Purpose
            -------
            Construct and returns a dictonary of a possible lattice points that
            are nearest neighbors to an origin point that can be arbitrarily
            translated.
        """
        coords = dict()
        b1 = self.basis_arr[0]
        b2 = self.basis_arr[1]
        __rots = 6 if self.rots == 3 else self.rots

        if isinstance(self.rots, Float) is True:
            rot_range = np.linspace(0, np.int64(100 * __rots),
                np.int64(100 * __rots))  # noqa E128
        else:
            rot_range = range(__rots)

        for n in rot_range:
            c_rot = rotation_mtx(2 * np.pi * n / self.rots)
            temp1 = b1.dot(c_rot)
            temp2 = b2.dot(c_rot)
            for i in range(2):
                temp1[i] = round_num(temp1[i], 10)
                temp2[i] = round_num(temp2[i], 10)
            coords[StripString(str(temp1), r_str)] = temp1
            coords[StripString(str(temp2), r_str)] = temp2
        self.y_length = (np.array([1, -1]).dot(self.basis_arr)).dot(
            np.array([0, 1]))
        theta = np.pi / __rots
        p_vec = self.y_length * b1

        if isinstance(self.rots, Float) is True:
            rot_range = np.linspace(0, np.int64(100 * __rots),
                np.int64(100 * 2 * __rots))  # noqa E128
        else:
            rot_range = range(2 * __rots)

        for n in rot_range:
            temp3 = p_vec.dot(rotation_mtx(n*theta))
            for i in range(2):
                temp3[i] = round_num(temp3[i], 10)
            coords[StripString(str(temp3), r_str)] = temp3

        return(coords)

    def is_neighbor(self, origin, possible_neighbor) -> bool:
        """
            Purpose
            -------
            TODO
        """
        for item in self.neighbors_ref.values():
            temp = np.arccos((possible_neighbor-origin).dot(origin) /
            (norm(origin) * norm(possible_neighbor-origin)))  # noqa E128
            temp = round_num(temp, 7, round_fig=10)
            # print(temp)
            if temp == 0 or (1-temp/np.pi) < 1E-07:
                return(False)
            if (not (self.y_length == norm(possible_neighbor - origin))
            or not (1 == norm(possible_neighbor - origin))):  # noqa E128
                a = origin.dot(self.basis_arr)+item
                b = possible_neighbor.dot(self.basis_arr)
                for i in range(len(a)):
                    a[i] = round_num(a[i], 10)
                    b[i] = round_num(b[i], 10)
                if np.array_equiv(a, b) is True:
                    return(True)
            else:
                continue
        return(False)

    def Sum(self):
        sum = 0
        for i in range(self.__shape[0]):  # TODO
            for j in range(self.__shape[1]):
                sum += self[i, j].get_spin()
        return(sum)

    def Nearest_Neighbor(self) -> np.int256:
        sum: np.int256 = 0
        visited: list[Node] = list()
        sum = self.__NNGenerator__(self.origin_node, sum, visited)
        return(sum)

    def __NNGenerator__(self, _node: Node, spin_sum: np.int256,
    visited: list[Node]) -> np.int256:  # noqa E128
        """
            Parameters
            ---------
            _node : `Node`
                cur node
            spin_sum : `numpy`.`int256`
                cur sum
            visited : `list`[`Node`]
                list of already used nodes
        """
        visited.append(_node)
        neighbors = _node.get_connected()
        if not _node.get_spin() == 0:
            for neighbor in neighbors:
                temp = neighbor.get_spin()
                if temp == 0:
                    continue
                spin_sum += neighbor.get_spin()
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.append(neighbor)
                return(self.__NNGenerator__(neighbor, spin_sum, visited))
            else:
                continue
        return(spin_sum)

    def __setitem__(self, __NodeIndex: list | NDArray,
    __value: int) -> None:  # noqa E128
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
            lookup = self.node_dict.get(StripString(__NodeIndex, r_str))
            if lookup is not None:
                lookup.spin_state = __value
            elif lookup is None:
                raise ValueError(f"Node {__NodeIndex} does not exist!")
                pass
        except Exception:
            PE.PrintException()

    def __getitem__(self, __NodeIndex: list | NDArray) -> Node | None:
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
            return(self.node_dict.get(StripString(__NodeIndex, r_str)))
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

    def append(self, child: list[Node] | Node,
    parent: Optional[Node] = None) -> None:  # noqa E128
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
            # TODO : check the immediate surrounds of the basis combination
            # to see if another node exists that is not linked to the node
            # to be appeneded.
            # This works by taking the new nodes basis combination and first
            # computes the angle between the two vectors and finds how many
            # neighbors a given node can have maximum.
            if child == parent:
                raise ValueError("""
                Can not append, Child and Parrent are the same!\n
                """)
            elif isinstance(child, list) and parent is not None:
                name = StripString(parent.get_combination(), r_str)
                self.node_dict[name] = parent
                self.num_nodes += 1
                for neighbor in child:
                    parent.add_link(neighbor)
                    name = StripString(
                        neighbor.get_combination(ReturnString=True), r_str)
                    self.node_dict[name] = neighbor
                    self.num_nodes += 1
            elif isinstance(child, list) is True:
                for neighbor in child:
                    parent.add_link(neighbor)
                    name = StripString(
                        neighbor.get_combination(ReturnString=True), r_str)
                    self.node_dict[name] = neighbor
                    self.num_nodes += 1
            elif parent is None and self.origin_node is None:
                self.origin_node = child
                name = StripString(child.get_combination(), r_str)
                self.node_dict[name] = child
                self.num_nodes += 1
            elif parent is not None:
                parent.add_link(child)
                name = StripString(child.get_combination(), r_str)
                self.node_dict[name] = child
                self.num_nodes += 1
        except Exception:
            PE.PrintException()

    def get_root(self) -> Node:
        """Returns the origin (A.K.A. root) node."""
        return self.origin_node

    def print_dict(self) -> None:
        """
            Purpose
            -------
            Prints the objects node dictionary, `node_dict` to `stdout`.
        """
        sys.stdout.write('\n')
        for key, val in self.node_dict.items():
            sys.stdout.write(f"key={key}, value={val}\n")
        sys.stdout.write('\n')

    def set_basis(self, input_basis: NDArray[Float]) -> None:
        """
            Parameters
            ----------
            input_basis : `NDArray[Float]`
                Specifies a basis for the crystal.
        """
        for basis_vector in input_basis:
            self.basis_arr.append(basis_vector)

    def generate(self, dims: list) -> None:
        try:
            if self.basis_arr is None:
                raise ValueError("Error, Basis must first be defined!")
            # elif isinstance(self.rots, Union[int, Int]) is not True:
            #     # if the "symmetry" is irrational, use linear_generation
            #     self.linear_generation(dims)
            #     return
            cur_cor = np.array([0, 0])
            cur_ind = np.array([np.int64(0), np.int64(0)])
            CurNode = None
            xp1 = False
            yp1 = False
            xm1 = False
            ym1 = False
            xp = np.array([1, 0])
            yp = np.array([0, 1])
            used = dict()
            neighbors: list[Node] = list()
            first_run = True
            check = None
            sys.stdout.write("\n Generating, Please wait...\r")
            while cur_ind[0] <= dims[0]-1:
                while cur_ind[1] <= dims[1]-1:
                    if first_run:
                        # create the origin node
                        CurNode = Node(cur_cor, cur_ind)
                        self.origin_node = CurNode
                        first_run = False
                    elif self[cur_ind] is None:
                        # create a node indexed by its basis combination
                        cur_cor = cur_ind.dot(self.basis_arr)
                        CurNode = Node(cur_cor, cur_ind) # noqa
                    else:
                        # set the current node to a previously generated Node
                        CurNode = used.get(StripString(cur_ind, r_str))

                    # Begin checking and creating neighbors
                    if cur_ind[0]+1 < dims[0]: # noqa
                        xp1 = True
                        cor = cur_ind+xp
                        check = self[cor]
                        if check is None:
                            neighbors.append(
                                Node(cor.dot(self.basis_arr), cor))
                        elif check is not None:
                            neighbors.append(check)
                        else:
                            raise Exception("IDKWtF")

                    if cur_ind[1]+1 < dims[1]:
                        yp1 = True
                        cor = cur_ind+yp
                        check = self[cor]
                        if check is None:
                            neighbors.append(
                                Node(cor.dot(self.basis_arr), cor))
                        elif check is not None:
                            neighbors.append(check)
                        else:
                            raise Exception("IDKWtF")

                    if cur_ind[0]-1 >= 0:
                        xm1 = True
                        cor = cur_ind-xp
                        check = self[cor]
                        if check is None:
                            neighbors.append(
                                Node(cor.dot(self.basis_arr), cor))
                        elif check is not None:
                            neighbors.append(check)
                        else:
                            raise Exception("IDKWtF")

                    if cur_ind[1]-1 >= 0:
                        ym1 = True
                        cor = cur_ind-yp
                        check = self[cor]
                        if check is None:
                            neighbors.append(
                                Node(cor.dot(self.basis_arr), cor))
                        elif check is not None:
                            neighbors.append(check)
                        else:
                            raise Exception("IDKWtF")
                    if (xp1 and yp1) is True:
                        cor = cur_ind+xp+yp
                        check = self[cor]
                        testcheck = self.is_neighbor(cur_ind, cor)
                        if check is None and testcheck:
                            neighbors.append(
                                Node(cor.dot(self.basis_arr), cor))
                        elif check is not None and testcheck:
                            neighbors.append(used.get(StripString(cor, r_str)))

                    if (xp1 and ym1) is True:
                        cor = cur_ind+xp-yp
                        check = self[cor]
                        testcheck = self.is_neighbor(cur_ind, cor)
                        if check is None and testcheck:
                            neighbors.append(
                                Node(cor.dot(self.basis_arr), cor))
                        elif check is not None and testcheck:
                            neighbors.append(used.get(StripString(cor, r_str)))

                    if (xm1 and yp1) is True:
                        cor = cur_ind-xp+yp
                        check = self[cor]
                        testcheck = self.is_neighbor(cur_ind, cor)
                        if check is None and testcheck:
                            neighbors.append(
                                Node(cor.dot(self.basis_arr), cor))
                        elif check is not None and testcheck:
                            neighbors.append(used.get(StripString(cor, r_str)))

                    if (xm1 and ym1) is True:
                        cor = cur_ind-xp-yp
                        check = self[cor]
                        testcheck = self.is_neighbor(cur_ind, cor)
                        if check is None and testcheck:
                            neighbors.append(
                                Node(cor.dot(self.basis_arr), cor))
                        elif check is not None and testcheck:
                            neighbors.append(used.get(StripString(cor, r_str)))

                    # add neighbors to the Node
                    self.append(neighbors, CurNode)
                    for item in neighbors:
                        temp = item.get_combination(ReturnString=True)
                        used[temp] = item

                    neighbors.clear()
                    yp1, ym1 = False, False

                    cur_ind = cur_ind + np.array([0, 1])  # increment y
                    cur_cor = cur_ind.dot(self.basis_arr)

                xp1, xm1 = False, False
                cur_ind[1] = 0

                cur_ind = cur_ind + np.array([1, 0])  # increment x
            sys.stdout.write(' Generation complete!            \n')
        except Exception:
            PE.PrintException()

    def linear_generation(self, dims: list) -> None:
        for i in range(dims[0]):
            for j in range(dims[1]):
                if i == 0 and j == 0:
                    cur = Node([0, 0], [0, 0])
                    self.append(cur)
                    last = cur
                    continue
                # construct a bravis lattice
                cur_cor = i * self.basis_arr[0] + j * self.basis_arr[1]
                cur = Node(cur_cor, [i, j])
                self.append(cur, last)
                last = cur

###########
# Testing #
###########
# TODO
# new = Node(0,0)
# test.append(new)
# basis_list = []
# basis_list.append([0,1])
# basis_list.append([1,0])
# print(basis_list[0]+basis_list[1])
# for i in range(4):
#     for j in range(4):
#         pass

# test.append(node_list, new)
# print(new.get_connected())
# TODO

# test = LinkedLattice(scale_factor=1,basis_arr=[[1, 0], [0.5, np.sqrt(3)/2]]) # noqa
# test.generate([10, 10])
# for item in test:
#     print(item.get_coords())

# new = Node(0, 0)
# test.append(new)
# old = new
# for i in range(10):
#     for j in range(10):
#         if i == 0 and j == 0:
#             continue
#         else:
#             new = Node([i, j], [i, j])
#             test.append(new, parent=old)
#             old = new

# test[0, 1] = 1
# print(test_node)
# print(test_node.spin_state)
# test.append(test_node, test_node)
# testing append error for child and parent being the same

# TODO

    # print(item.get_connected())
    # print('\n')

# print(test[0, 0])
# test.printdict()
# TODO
