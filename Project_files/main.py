"""
Author: Ramenspazz

Purpose: main driver file for ISING model simulation
"""
# Begin by importing external libraries using path_setup
# NOTE : Must be ran first, thus PEP8-E402
import path_setup
path_setup.path_setup()
import sys  # noqa E402
import numpy as np # noqa E402
import datetime as dt # noqa E402
from LatticeDriver import LatticeDriver as lt  # noqa E402
from LatticeDriver import T_to_Beta, Beta_to_T  # noqa E402
import PrintException as PE  # noqa E402
import InputFuncs as inF  # noqa E402
from getpass import getpass  # noqa E402
import random as rnd  # noqa E402
import DataAnalysis as DA  # noqa E402
import warnings  # noqa E402
from SupportingFunctions import generate_random, rand_time  # noqa E402
warnings.filterwarnings("ignore", category=SyntaxWarning)
# sys.setrecursionlimit(100000000)  # increase recursion limit


zeroC = 273.15


def main(*args, **kwargs) -> int:
    try:
        N = 32
        M = 32
        size = [N, M]
        total_time = 1000
        # for some reason I feel this value is smaller than it should be but
        # it is giving me the expected graphs 0_o
        J = 0.01  # eV, interation energy
        T1 = zeroC
        T2 = 100+zeroC
        a = T_to_Beta(T1)
        b = T_to_Beta(T2)
        print(f'a={a}/eV, b={b}/eV')
        num_points = 10
        step = (b-a)/num_points
        Beta = np.arange(a, b, step)

        output = ''

        lt_c4v_up = lt(1, size, J)
        lt_c3v_up = lt(1, size, J, basis=[[1, 0], [0.5, np.sqrt(3)/2]])
        lt_c6v_up = lt(1, size, J, [[0.5, np.sqrt(3)/2], [0.5, -np.sqrt(3)/2]])

        lt_c4v_dn = lt(1, size, J)
        lt_c3v_dn = lt(1, size, J, [[1, 0], [0.5, np.sqrt(3)/2]])
        lt_c6v_dn = lt(1, size, J, [[0.5, np.sqrt(3)/2], [0.5, -np.sqrt(3)/2]])

        inF.print_stdout(
            "would you like to save plots automatically? (y/n): ")
        output = inF.key_input(['y', 'n'])
        auto_save = True if output == 'y' else False

        inF.print_stdout(
            "wouild you like to automatically display plots when they are"
            " ready? (y/n): ")
        output = inF.key_input(['y', 'n'])
        auto_plot = True if output == 'y' else False

        while True:
            inF.print_stdout(
                'Enter 0 for seeded random or 1 to input probabilities'
                ' or q to quit: ')

            output: str = inF.key_input(['0', '1', 'q'])
            if output == '0':
                inF.print_stdout("option 0 chosen.", end='\n')
                # DOCtest seed = 1644121893
                # good seed 1644144314
                seed = 1644121893
                lt_c4v_up.randomize(voids=True, probs=[15, 80, 5],
                                    rand_seed=seed)
                lt_c3v_up.randomize(voids=False, probs=[20, 80],
                                    rand_seed=seed)
                lt_c6v_up.randomize(voids=True, probs=[15, 80, 5],
                                    rand_seed=seed)
                lt_c4v_dn.randomize(voids=True, probs=[80, 15, 5],
                                    rand_seed=seed)
                lt_c3v_dn.randomize(voids=False, probs=[80, 20],
                                    rand_seed=seed)
                lt_c6v_dn.randomize(voids=True, probs=[80, 15, 5],
                                    rand_seed=seed)

            elif output == '1':
                inF.print_stdout("option 1 chosen.", end='\n')
                inF.print_stdout('Enable voids (y/n)?')
                output = inF.key_input(['y', 'n'])
                voids_enable = True if output == 'y' else False
                rand_n = 2 if voids_enable is False else 3
                seed = rand_time()

                lt_c4v_up.randomize(voids=voids_enable,
                                    probs=generate_random(rand_n),
                                    rand_seed=seed)
                lt_c3v_up.randomize(voids=voids_enable,
                                    probs=generate_random(rand_n),
                                    rand_seed=seed)
                lt_c6v_up.randomize(voids=voids_enable,
                                    probs=generate_random(rand_n),
                                    rand_seed=seed)
                lt_c4v_dn.randomize(voids=voids_enable,
                                    probs=generate_random(rand_n),
                                    rand_seed=seed)
                lt_c3v_dn.randomize(voids=voids_enable,
                                    probs=generate_random(rand_n),
                                    rand_seed=seed)
                lt_c6v_dn.randomize(voids=voids_enable,
                                    probs=generate_random(rand_n),
                                    rand_seed=seed)

            elif output == 'q':
                inF.cls()
                inF.print_stdout('Goodbye', end='\n')
                exit()

            # relax_itt_num = 100
            # relax_beta = 0
            # lt_c4v_up.relax(relax_itt_num, relax_beta)
            # lt_c4v_up.update(set_state=True)
            # lt_c3v_up.relax(relax_itt_num, relax_beta)
            # lt_c3v_up.update(set_state=True)
            # lt_c6v_up.relax(relax_itt_num, relax_beta)
            # lt_c6v_up.update(set_state=True)
            # lt_c4v_dn.relax(relax_itt_num, relax_beta)
            # lt_c4v_dn.update(set_state=True)
            # lt_c3v_dn.relax(relax_itt_num, relax_beta)
            # lt_c3v_dn.update(set_state=True)
            # lt_c6v_dn.relax(relax_itt_num, relax_beta)
            # lt_c6v_dn.update(set_state=True)

            if auto_plot is True:
                lt_c4v_up.plot()
                lt_c3v_up.plot()
                lt_c6v_up.plot()
                lt_c4v_dn.plot()
                lt_c3v_dn.plot()
                lt_c6v_dn.plot()

            inF.print_stdout(
                f"Temperature range= [{T1},{T2}]Kelvin."
                f" Number of sample points in the range = {num_points}."
                " Change (y/n)? ")

            output = inF.key_input(['y', 'n'])

            if output == 'y':
                print('y\n')
                T1 = float(input('T1 = '))
                T2 = float(input('T2 = '))
                num_points = float(input('number of sample points = '))
                a = T_to_Beta(T1)
                b = T_to_Beta(T2)
                step = (b-a)/num_points
                Beta = np.arange(a, b, step)

            lt_c4v_up.SpinEnergy(Beta, total_time, lt.WolffAlgorithm,
                                 save=auto_save, auto_plot=auto_plot)

            lt_c3v_up.SpinEnergy(Beta, total_time, lt.WolffAlgorithm,
                                 save=auto_save, auto_plot=auto_plot)

            lt_c6v_up.SpinEnergy(Beta, total_time, lt.WolffAlgorithm,
                                 save=auto_save, auto_plot=auto_plot)

            lt_c4v_dn.SpinEnergy(Beta, total_time, lt.WolffAlgorithm,
                                 save=auto_save, auto_plot=auto_plot)

            lt_c3v_dn.SpinEnergy(Beta, total_time, lt.WolffAlgorithm,
                                 save=auto_save, auto_plot=auto_plot)

            lt_c6v_dn.SpinEnergy(Beta, total_time, lt.WolffAlgorithm,
                                 save=auto_save, auto_plot=auto_plot)

    except KeyboardInterrupt:
        inF.cls()
        inF.print_stdout("Keyboard Interrupt, closing can take a moment"
                         " But usually succeedes...",
                         end='\n')
        exit()
    except Exception:
        PE.PrintException()


if __name__ == '__main__':
    main()
