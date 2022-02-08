'''
Author: Ramenspazz

This file defines the simulation lattice to be used and sets up basic methods
to operate on the lattice
'''
# Typing imports
from typing import Optional, Union
from numbers import Number
from numpy import integer as Int, floating as Float, ndarray, number
from numpy.typing import NDArray
# import numpy.typing as npt

import sys # noqa
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
from matplotlib.colors import BoundaryNorm
from matplotlib import cm # noqa E402
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy import rand # noqa E402
from scipy.ndimage import convolve, generate_binary_structure # noqa
from scipy.interpolate import griddata
import linked_list_class as lc
import Data_Analysis as DA
import random
from random import randint
import PrintException as PE

GNum = Union[number, Number]

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__  # noqa 
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


class lattice_class:
    """
        TODO : write docstring
    """

    def __init__(
        self, scale: number | Number,
        Lshape: list[int] | NDArray[Int],
        basis: Optional[NDArray] = None) -> None: # noqa E125
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
            self.Lshape = Lshape
            self.lattice_spacing = scale
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
            if np.abs(self.Lshape[0]) > 0 and np.abs(self.Lshape[1]) > 0:
                return(self.internal_arr.__getitem__(*args))
            else:
                raise IndexError(f"""Index tup is out of range! tup={args[0]}
                is greater than the max index of {self.Lshape[0]*self.Lshape[0]-1}!\n""")
        except Exception:
            PE.PrintException()

    def __iter__(self) -> lc.Node | None:
        pass

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
            for i in range(self.Lshape[0]):
                for j in range(self.Lshape[1]):
                    cur = self[i, j]
                    U.append(cur.get_coords()[0])
                    V.append(cur.get_coords()[1])
                    W.append(cur.get_spin())
            plt.scatter(U, V, c=W, cmap='viridis')
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
        self.internal_arr.generate(self.Lshape)

    def get_energy(self):
        # $E/J = -\sum_{<i,j>} \sigma_i\sigma_j$
        return(self.internal_arr.Nearest_Neighbor())

    def get_spin_energy(self, BJs: list | ndarray, __times: Optional[int] = None) -> list:
        """
            Purpose
            -------
            Computes the energy of the lattice at thermal equlibrium and
            relates this to T=J/(Beta*J*k) = 1/(Beta*k) ~ 1/(Beta*J)
        """
        ms = np.zeros(len(BJs))
        E_means = np.zeros(len(BJs))
        E_stds = np.zeros(len(BJs))
        if __times is None:
            times = 1
        else:
            times = __times
        for i, bj in enumerate(BJs):
            SE_mtx = self.metropolis(times, bj)
            spins = SE_mtx[:, 0]
            energies = SE_mtx[:, 1]
            ms[i] = spins[-times:].mean()/len(self.internal_arr)
            E_means[i] = energies[-times:].mean()
            E_stds[i] = energies[-times:].std()
            sys.stdout.write(f"get_spin_energy is {100 * i / len(BJs) :.2f}% complete...       \r")
            if True or isnotebook() is True:
                self.plot_metrop(SE_mtx, bj)
        sys.stdout.write(f"get_spin_energy is {100 :.2f}% complete!       \n")
        self.plot_spin_energy(BJs, ms, E_stds)
        return(ms, E_means, E_stds)
    
    def plot_spin_energy(self, bjs: ndarray, a: ndarray | list, c: ndarray | list) -> None:

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(1/bjs, a, 'o--', label=r"<m> vs $\left(\frac{k}{J}\right)T$")
        ax1.legend()
        ax2.plot(1/bjs, c*bjs, 'x--', label=r'$C_V / k^2$ vs $\left(\frac{k}{J}\right)T$')
        ax2.legend()
        plt.show()

    def plot_metrop(self, SE_mtx, BJ, quiet=True):
        """
        Parameters
        ---------- 
        SE_mtx 
        BJ
        """
        num_nodes = len(self.internal_arr)
        if quiet is False:
            spin_up = 0
            spin_dn = 0
            spin0 = 0
            for val in SE_mtx[:,0]:
                if val > 0:
                    spin_up += val
                elif val < 0:
                    spin_dn += val
            total = np.abs(spin_up) + np.abs(spin_dn)
            mean = DA.data_mean(SE_mtx[:, 0])
            stdev = DA.std_dev(SE_mtx[:, 0], mean, sample=False)
            print(f"""
            Average up sum of spins density is {np.abs(spin_up) / num_nodes :.4f}%\n
            Average up sum of spins density is {np.abs(spin_dn) / num_nodes :.4f}%\n
            The mean is {mean}\n
            The Standard deviation of the mean is {stdev / np.sqrt(num_nodes) :.4f}\n
            """)
        fig, axes = plt.subplots(
            1, 2, figsize=(12, 4),
            num=f'Evolution of Average Spin n={num_nodes**2} and Energy for BJ={BJ}')
        ax = axes[0]
        ax.plot(SE_mtx[:, 0] / num_nodes)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel(r'Average Spin $\bar{m}$')
        ax.grid()
        ax = axes[1]
        ax.plot(SE_mtx[:, 1])
        ax.set_xlabel('Time Steps')
        ax.set_ylabel(r'Energy $E/J$')
        ax.grid()
        fig.tight_layout()
        plt.show()

# TODO: Look into this http://mcwa.csi.cuny.edu/umass/izing/Ising_text.pdf
# TODO: the worm algorithm.
    def metropolis(self, times: int | Int, BJ: float | Float, quiet: Optional[bool] = True) -> ndarray:
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
                    rand_x = randint(0, self.Lshape[0] - 1)
                    rand_y = randint(0, self.Lshape[1] - 1)
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
                # if quiet is False and i % 100 == 0:
                #     self.display()
                SE_mtx[i, 0] = self.internal_arr.Sum()
                SE_mtx[i, 1] = energy
            del energy
            if quiet is False:
                self.plot_metrop(SE_mtx, BJ, self.Lshape)
            return(SE_mtx)
        except Exception:
            PE.PrintException()

    def randomize(self,
        voids: bool,
        probs: list,
        rand_seed: Optional[int] = None,
        quiet: Optional[bool] = True) -> None: # noqa E125
        """
            Parameters
            ----------
            voids : `bool`
                - Truns on random voids in the lattice

            probs : `list`[`float`]
                - list representing the probability of `1`, `0`, `-1`
                where the two entries represent the two bounds such that
                if 0 < rand < a then spin = -1, if a <= rand <= b then spin = 0,
                else spin = 1.

            rand_seed : Optional[`int`]
                - Seed value to seed random with.
        """
        try:
            if rand_seed is not None:
                if quiet is not True:
                    sys.stdout.write(f"Generating Seed = {rand_seed}\n")
                random.seed(rand_seed)
                for i in range(self.Lshape[0]):
                    if voids is True:
                        for j in range(self.Lshape[1]):
                            rand_num = (randint(-50,50) * probs[0] + probs[1]) % 3 - 1
                            self[i, j] = rand_num
                            if rand_num == 0:
                                self.internal_arr.num_voids += 1
                            
                    elif voids is False:
                        for j in range(self.Lshape[1]):
                            self[i, j] = -1 if randint(1,100) % 2 == 0 else 1

            return(None)
        except Exception:
            PE.PrintException()

    def shape(self) -> tuple:
        return(self.Lshape)

