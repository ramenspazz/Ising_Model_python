"""
Author: Ramenspazz

This file defines the Node class.
"""
from __future__ import annotations
# Typing imports
from typing import Optional
# import matplotlib.pyplot as plt
from numpy.typing import NDArray
from numpy import int64, int8, integer as Int, floating as Float, ndarray  # noqa E501
# Functions and Libraries
import numpy as np
import PrintException as PE


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
                 spin_state: Optional[int8] = None) -> None:
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
            self.marked: bool = False
            self.links = list()
        except Exception:
            PE.PrintException()

    def __len__(self):
        return(len(self.links))

    def __iter__(self) -> Node:
        return(self.__next__())

    def __next__(self) -> Node:
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

    def get_coords(self) -> ndarray | str:
        return(self.coords)

    def get_index(self) -> ndarray | str:
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

    def flip_spin(self) -> int:
        """
            Purpose
            -------
            Flips the spin of the node if it is `1` or `-1`. If the spin is `0`
            then do nothing.
        """
        try:
            if self.get_spin() == 0:
                return(0)
            elif self.get_spin() == 1:
                self.spin_state = -1
                return(-1)
            elif self.get_spin() == -1:
                self.spin_state = 1
                return(1)
        except Exception:
            PE.PrintException()

    def mark_node(self):
        self.marked = True

    def unmark_node(self):
        self.marked = False

    def flip_test(self) -> int:
        try:
            if self.get_spin() != 0:
                return(-1 if self.get_spin() == 1 else -1)
            else:
                return(0)
        except Exception:
            PE.PrintException()
