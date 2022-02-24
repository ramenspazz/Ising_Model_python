"""
Author: Ramenspazz

Purpose: main driver file for ISING model simulation
"""
# Begin by importing external libraries using path_setup
# NOTE : Must be ran first, thus PEP8-E402
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
warnings.filterwarnings("ignore", category=SyntaxWarning)
sys.setrecursionlimit(1000000)  # increase recursion limit


def rand_time() -> int:
    out = int(dt.datetime.now().strftime('%s'))
    sys.stdout.write(f"Time Seed = {out}\n")
    return(int(dt.datetime.now().strftime('%s')))


def main(*args, **kwargs) -> int:
    try:
        # all listed times are on a Ryzen 7 3700x
        # 32x32     => ~44.75876808 seconds runtime
        # 64x64     => ~81.04408002 seconds runtime
        # 128x128   => ~245.7723527 seconds runtime
        N = 20
        M = 20
        size = [N, M]
        total_time = 1000
        a = 0.1
        b = 2
        step = 0.05
        BJs = np.arange(a, b, step)
        BJ = 0.1  # noqa
        output = ''

        lt_a = lt(1, size)
        lt_b = lt(1, size, [[1, 0], [0.5, np.sqrt(3)/2]])
        # print(lt_b.internal_arr.cord_dict)
        lt_c = lt(1, size, [[0.5, np.sqrt(3)/2], [0.5, -np.sqrt(3)/2]])

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
                lt_a.randomize(voids=True, probs=[45, 45, 10],
                               rand_seed=seed, quiet=False)
                lt_b.randomize(voids=True, probs=[55, 40, 5],
                               rand_seed=seed, quiet=False)
                lt_c.randomize(voids=True, probs=[30, 65, 5],
                               rand_seed=seed, quiet=False)

            elif output == '1':
                inF.print_stdout("option 1 chosen.", end='\n')
                inF.print_stdout('Enable voids (y/n)?')
                output = inF.key_input(['y', 'n'])
                voids_enable = True if output == 'y' else False

                lt_a.randomize(voids=voids_enable, probs=[
                    random(), random()],
                    rand_seed=rand_time(), quiet=False)

                lt_b.randomize(voids=voids_enable, probs=[
                     random(), random()],
                     rand_seed=rand_time(), quiet=False)

                lt_c.randomize(voids=voids_enable, probs=[
                     random(), random()],
                     rand_seed=rand_time(), quiet=False)

            elif output == 'q':
                inF.cls()
                inF.print_stdout('Goodbye', end='\n')
                exit()

            if auto_plot is True:
                lt_a.display()
                lt_b.display()
                lt_c.display()

            print('lt_a connected')
            for item in lt_b[1, 1].get_connected():
                print(item)
            print('lt_b connected')
            for item in lt_b[1, 1].get_connected():
                print(item)
            print('lt_c connected')
            for item in lt_c[1, 1].get_connected():
                print(item)

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
            lt_a.metropolis(total_time, BJ, progress=True,
                            save=auto_save, auto_plot=auto_plot)
            lt_b.metropolis(total_time, BJ, progress=True,
                            save=auto_save, auto_plot=auto_plot)
            lt_c.metropolis(total_time, BJ, progress=True,
                            save=auto_save, auto_plot=auto_plot)
            # # lt_d.metropolis(total_time, BJ, quiet=False)

            # get_spin_energy is 100% complete in 34.30839276s on my home
            # desktop with n=36, m=42, threads=16 on a Ryzen 7 3700X @
            # 2.6-4.5Ghz using the seed=1644121893 with gaussian prob settings
            # of [0.25, 0.4].
            lt_a.get_spin_energy(BJs, total_time, save=auto_save,
                                 auto_plot=auto_plot)
            lt_b.get_spin_energy(BJs, total_time, save=auto_save,
                                 auto_plot=auto_plot)
            lt_c.get_spin_energy(BJs, total_time, save=auto_save,
                                 auto_plot=auto_plot)
    except KeyboardInterrupt:
        inF.cls()
        inF.print_stdout("Keyboard Interrupt, closing...", end='\n')
        exit()
    except Exception:
        PE.PrintException()


if __name__ == '__main__':
    main()
