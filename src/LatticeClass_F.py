'''
This file defines the simulation lattice to be used and sets up basic methods
to operate on the lattice
'''
# Typing imports
from typing import Optional, Union
from numbers import Number
# import numpy.typing as npt
from numpy.typing import NDArray
from numpy import integer as Int, floating as Float, ndarray, number

import sys # noqa
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, generate_binary_structure # noqa
from scipy.interpolate import griddata # noqa
import linked_list_class as lc
import random
from random import randint
import PrintException as PE

GNum = Union[number, Number]


class lattice_class:
    """
        TODO : write docstring
    """

    def __init__(
        self, scale: number | Number,
        __shape: list[int] | NDArray[Int | Float],
        basis: Optional[NDArray] = None) -> None: # noqa E125
        '''
            Parameters
            ----------

            scale : `int` | `float`
                - The scale of distances in centimeters

            __shape : `list`[`int256`] | `NDArray`[`Int`]
                - Span of the x and y bounds for the lattice.

            basis : Optional[`NDArray`]
                - A numpy ndarray of a shape 2x2 that specifies the x basis
                on the first row and the y basis on the second row.
        '''
        try:
            self.__shape = __shape
            self.lattice_spacing = scale
            if self.__shape[0] == 0 or self.__shape[1] == 0:
                raise ValueError()
            elif self.__shape[0] > 0 and self.__shape[1] > 0:
                if basis is not None:
                    if isinstance(basis, ndarray):
                        input_basis = basis
                    else:
                        input_basis = array(basis)
                    self.internal_arr: lc.LinkedLattice = lc.LinkedLattice(
                        scale, __shape, basis_arr=(input_basis))
                elif basis is None:
                    self.internal_arr: lc.LinkedLattice = lc.LinkedLattice(
                        scale, __shape, basis_arr=None)
        except Exception:
            PE.PrintException()

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

    def __getitem__(self, *args) -> lc.Node:
        """
            Returns
            -------
            `Node` at the requested location if it exists, else return None.
            (TODO : CURRENTLY ERRORS [IndexError] ON OUT OF RANGE)
        """
        try:
            if args is None:
                sys.stderr.write("IS NONE")
            if np.abs(self.__shape[0]) > 0 and np.abs(self.__shape[1]) > 0:
                return(self.internal_arr.__getitem__(*args))
            else:
                raise IndexError(f"""Index tup is out of range! tup={args[0]}
                is greater than the max index of {self.cols*self.rows-1}!\n""")
        except Exception:
            PE.PrintException()

    def __iter__(self) -> lc.Node | None:
        for item in self.internal_arr:
            yield item

    def __setitem__(self, index: list[int] | NDArray[Int], val: int | Int):
        """
            Purpose
            -------
            assigns value [val] to the object at the location of [index] from
            the `internal_arr` object.

            Parameters
            ----------

            index : `list`
                - index of the basis combination refrencing a node.

            val : `int` | `Int`
                - Value to set the spin of the node refrenced by index.

        """
        try:
            self.internal_arr[index] = val
        except Exception:
            PE.PrintException()

    def display(self) -> None:
        """
            Purpose
            -------
            Uses the matplotlib function matplotlib.pyplot.scatter to plot the
            spin states of the lattice nodes.
        """
        try:
            U = list()
            V = list()
            W = list()
            for i in range(self.__shape[0]):
                for j in range(self.__shape[1]):
                    coord = self[i, j].get_coords()
                    U.append(coord[0])
                    V.append(coord[1])
                    W.append(self[i, j].get_spin())
            plt.scatter(U, V, c=W)
            plt.show()
        except Exception:
            PE.PrintException()

    def generate_lattice(self) -> None:
        """
            Purpose
            -------
            This function calls the `LinkedLattice` class member function
            `self.internal_arr.generate(dims)`.
        """
        self.internal_arr.generate(self.__shape)

    def get_energy(self):
        # $E/J = -\sum_{<i,j>} \sigma_i\sigma_j$
        return(self.internal_arr.Nearest_Neighbor())

    def get_spin_energy(self, BJs=None, __times=None) -> list:
        """
            Purpose
            -------
            Computes the energy of the lattice using nearest neighbors.
        """
        ms = np.zeros(len(BJs))
        E_means = np.zeros(len(BJs))
        E_stds = np.zeros(len(BJs))
        if __times is None:
            times = 1
        else:
            times = __times
        for i, bj in enumerate(BJs):
            spins, energies = self.metropolis(times, bj)
            ms[i] = spins[-times:].mean()/(self.rows*self.cols)
            E_means[i] = energies[-times:].mean()
            E_stds[i] = energies[-times:].std()
        return([ms, E_means, E_stds])

# TODO: Look into this http://mcwa.csi.cuny.edu/umass/izing/Ising_text.pdf
# TODO: the worm algorithm.
    def metropolis(self, times: int | Int, BJ: float | Float) -> ndarray:
        """
            Purpose
            -------
            Evolve the system and compute the metroplis approximation to the
            equlibrium state given a beta*J and number of monty carlo
            itterations to preform.
        """
        try:
            energy = self.get_energy()
            SE_mtx: ndarray = np.zeros([times, 2])
            for i in range(0, times):
                # 2. pick random point on array and flip spin
                while True:
                    rand_x = randint(0, self.__shape[0] - 1)
                    rand_y = randint(0, self.__shape[1] - 1)
                    node_i = self.internal_arr[rand_x, rand_y]  # initial spin
                    if node_i is not None and node_i.get_spin() == 0:
                        # keep looking for a viable random node
                        continue
                    else:
                        # exit random node selection loop
                        break
                # compute change in energy with nearest neighbors
                E_i: np.int256 = 0
                E_f: np.int256 = 0
                for neighbor in node_i:
                    if neighbor is None:
                        continue
                    E_i += -node_i.get_spin() * neighbor.get_spin()
                    E_f += -node_i.flip_test() * neighbor.get_spin()

                # change state with designated probabilities
                dE = E_f-E_i
                if (dE > 0) and ((randint(0, 100) / 100) < np.exp(-BJ * dE)):
                    self.internal_arr[rand_x, rand_y].flip_spin()
                    energy += dE

                elif dE <= 0:
                    self.internal_arr[rand_x, rand_y].flip_spin()
                    energy += dE

                SE_mtx[i, 0] = self.internal_arr.Sum()
                SE_mtx[i, 1] = energy
            del energy
            return(SE_mtx)
        except Exception:
            PE.PrintException()

    def randomize(
        self, rand_condition: float,
        rand_seed: Optional[int] = None,
        quiet: Optional[bool] = True) -> None: # noqa E125
        """
            Purpose
            -------
            TODO

            Parameters
            ----------
            rand_condition : `float`
                - Number representing the probability of `1` or `-1`.

            rand_seed : Optional[`int`]
                - Seed value to seed random with.
        """
        try:
            if rand_seed is not None:
                if quiet is not True:
                    sys.stdout.write(f"Generating Seed = {rand_seed}\n")
                random.seed(rand_seed)
                for i in range(self.__shape[0]):
                    for j in range(self.__shape[1]):
                        rand_num = 1 if random.gauss(
                            0.5, 0.5) >= rand_condition else -1
                        self.internal_arr[i, j] = rand_num
            return(None)
        except Exception:
            PE.PrintException()

    def shape(self) -> tuple:
        return(self.__shape)

# lc_test = lattice_class(1, [10, 10], [[1, 0], [0.5, np.sqrt(3)/2]])
# lc_test = lattice_class(1, [10, 10], [[0.5, np.sqrt(3)/2], [0.5, -np.sqrt(3)/2]]) # noqa

# lc_test.randomize(0.64)
# # lc_test.display()
# for item in lc_test:
#     if item is None:
#         continue
#     else:
#         print(item.coords)
