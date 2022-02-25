"""
Author: Ramenspazz

Purpose: main driver file for ISING model simulation
"""
# Begin by importing external libraries using path_setup
# NOTE : Must be ran first, thus PEP8-E402
import math
import path_setup
path_setup.path_setup()
import sys  # noqa E402
import time  # noqa E402
import matplotlib.pyplot as plt # noqa E402
import numpy as np # noqa E402
import datetime as dt # noqa E402
from LatticeClass_F import lattice_class as lt # noqa E402
import PrintException as PE  # noqa E402
import input_funcs as inF  # noqa E402
from getpass import getpass  # noqa E402
from random import random  # noqa E402
import Data_Analysis as DA  # noqa E402
import warnings  # noqa E402
import cProfile  # noqa E402
warnings.filterwarnings("ignore", category=SyntaxWarning)
sys.setrecursionlimit(1000000)  # increase recursion limit


def rand_time() -> int:
    out = int(dt.datetime.now().strftime('%s'))
    sys.stdout.write(f"Time Seed = {out}\n")
    return(int(dt.datetime.now().strftime('%s')))


def main(*args, **kwargs) -> int:
    try:
        N = 128
        M = 128
        size = [N, M]
        total_time = math.trunc(np.sqrt(N*M))
        a = 0.1
        b = 10
        num_points = 100
        step = (b-a)/num_points
        BJs = np.arange(a, b, step)
        output = ''

        lt_c4v = lt(1, size)
        lt_c3v = lt(1, size, [[1, 0], [0.5, np.sqrt(3)/2]])
        lt_c6v = lt(1, size, [[0.5, np.sqrt(3)/2], [0.5, -np.sqrt(3)/2]])

        # print('lt_a connected')
        # for item in lt_a[1, 1]:
        #     print(item)
        # print('lt_b connected')
        # for item in lt_b[1, 1]:
        #     print(item)
        # print('lt_c connected')
        # for item in lt_c[1, 1]:
        #     print(item)

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
                lt_c4v.randomize(voids=True, probs=[15, 80, 5],
                                 rand_seed=seed)
                lt_c3v.randomize(voids=True, probs=[49, 49, 2],
                                 rand_seed=seed)
                lt_c6v.randomize(voids=True, probs=[80, 15, 5],
                                 rand_seed=seed)

            elif output == '1':
                inF.print_stdout("option 1 chosen.", end='\n')
                inF.print_stdout('Enable voids (y/n)?')
                output = inF.key_input(['y', 'n'])
                voids_enable = True if output == 'y' else False

                lt_c4v.randomize(voids=voids_enable, probs=[
                    random(), random()],
                    rand_seed=rand_time(), quiet=False)

                lt_c3v.randomize(voids=voids_enable, probs=[
                     random(), random()],
                     rand_seed=rand_time(), quiet=False)

                lt_c6v.randomize(voids=voids_enable, probs=[
                     random(), random()],
                     rand_seed=rand_time(), quiet=False)

            elif output == 'q':
                inF.cls()
                inF.print_stdout('Goodbye', end='\n')
                exit()

            if auto_plot is True:
                lt_c4v.display()
                lt_c3v.display()
                lt_c6v.display()

            inF.print_stdout(
                f"BJ range= [{a},{b}]. Steps= {step}. Change (y/n)? ")

            output = inF.key_input(['y', 'n'])

            if output == 'y':
                # TODO probably use a different function other than getpass
                inF.print_stdout('a = ')
                a = float(getpass(''))
                inF.print_stdout('b = ')
                b = float(getpass(''))
                inF.print_stdout('step = ')
                step = float(getpass(''))

            BJs = np.arange(a, b, step)  # noqa

            # # Uncomment the next 4 lines below if you want, but not
            # # really a reason to as the metropolis algorithm gets
            # # called anyways from the get_spin_energy function.
            # lt_a.metropolis(total_time, BJ, progress=True,
            #                 save=auto_save, auto_plot=auto_plot)
            # lt_b.metropolis(total_time, BJ, progress=True,
            #                 save=auto_save, auto_plot=auto_plot)
            # lt_c.metropolis(total_time, BJ, progress=True,
            #                 save=auto_save, auto_plot=auto_plot)

            lt_c4v.get_spin_energy(BJs, total_time, save=auto_save,
                                   auto_plot=auto_plot)

            lt_c3v.get_spin_energy(BJs, total_time, save=auto_save,
                                   auto_plot=auto_plot)

            lt_c6v.get_spin_energy(BJs, total_time, save=auto_save,
                                   auto_plot=auto_plot)
    except KeyboardInterrupt:
        inF.cls()
        inF.print_stdout("Keyboard Interrupt, closing...", end='\n')
        exit()
    except Exception:
        PE.PrintException()


if __name__ == '__main__':
    main()
