"""
Author: Ramenspazz

Purpose: main driver file for ISING model simulation
"""
# Begin by importing external libraries using path_setup
# NOTE : Must be ran first, thus PEP8-E402
import path_setup
path_setup.path_setup()
import math  # noqa E402
import sys  # noqa E402
import matplotlib.pyplot as plt # noqa E402
import numpy as np # noqa E402
import datetime as dt # noqa E402
from LatticeDriver import LatticeDriver as lt # noqa E402
import PrintException as PE  # noqa E402
import InputFuncs as inF  # noqa E402
from getpass import getpass  # noqa E402
from random import random as rng  # noqa E402
import random as rnd  # noqa E402
import DataAnalysis as DA  # noqa E402
import warnings  # noqa E402
warnings.filterwarnings("ignore", category=SyntaxWarning)
# sys.setrecursionlimit(1000000)  # increase recursion limit


def rand_time() -> int:
    out = int(dt.datetime.now().strftime('%s'))
    sys.stdout.write(f"Time Seed = {out}\n")
    return(int(dt.datetime.now().strftime('%s')))


def generate_random(gen_num: int) -> list:
    """
        Generates 2 or 3 random numbers whos sum is 100
    """
    if gen_num == 2:
        rand_a = rnd.randint(0, 100)
        rand_b = 100 - rand_a
        return([rand_a, rand_b])
    elif gen_num == 3:
        rand_a = rnd.randint(0, 98)
        if rand_a == 0:
            rand_b = rnd.randint(0, 99)
        else:
            rand_b = rnd.randint(0, 100-rand_a-1)
        rand_c = 100 - rand_a - rand_b
        return([rand_a, rand_b, rand_c])


def main(*args, **kwargs) -> int:
    try:
        N = 12
        M = 12
        size = [N, M]
        total_time = 100  # 4*math.trunc(np.sqrt(N*M))
        print(total_time)
        a = 0.1
        b = 10+a
        num_points = 100
        step = (b-a)/num_points
        BJs = np.arange(a, b, step)
        output = ''

        lt_c4v_up = lt(1, size)
        lt_c3v_up = lt(1, size, [[1, 0], [0.5, np.sqrt(3)/2]])
        lt_c6v_up = lt(1, size, [[0.5, np.sqrt(3)/2], [0.5, -np.sqrt(3)/2]])

        lt_c4v_dn = lt(1, size)
        lt_c3v_dn = lt(1, size, [[1, 0], [0.5, np.sqrt(3)/2]])
        lt_c6v_dn = lt(1, size, [[0.5, np.sqrt(3)/2], [0.5, -np.sqrt(3)/2]])

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
                lt_c4v_up.randomize(voids=True, probs=[42, 53, 5],
                                    rand_seed=seed)
                lt_c3v_up.randomize(voids=True, probs=[15, 80, 5],
                                    rand_seed=seed)
                lt_c6v_up.randomize(voids=True, probs=[15, 80, 5],
                                    rand_seed=seed)
                lt_c4v_dn.randomize(voids=True, probs=[80, 15, 5],
                                    rand_seed=seed)
                lt_c3v_dn.randomize(voids=True, probs=[80, 15, 5],
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

            if auto_plot is True:
                lt_c4v_up.plot()
                lt_c3v_up.plot()
                lt_c6v_up.plot()
                lt_c4v_dn.plot()
                lt_c3v_dn.plot()
                lt_c6v_dn.plot()

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

            lt_c4v_up.get_spin_energy(BJs, total_time, save=auto_save,
                                      auto_plot=auto_plot)

            lt_c3v_up.get_spin_energy(BJs, total_time, save=auto_save,
                                      auto_plot=auto_plot)

            lt_c6v_up.get_spin_energy(BJs, total_time, save=auto_save,
                                      auto_plot=auto_plot)

            lt_c4v_dn.get_spin_energy(BJs, total_time, save=auto_save,
                                      auto_plot=auto_plot)

            lt_c3v_dn.get_spin_energy(BJs, total_time, save=auto_save,
                                      auto_plot=auto_plot)

            lt_c6v_dn.get_spin_energy(BJs, total_time, save=auto_save,
                                      auto_plot=auto_plot)
    except KeyboardInterrupt:
        inF.cls()
        inF.print_stdout("Keyboard Interrupt, closing...", end='\n')
        exit()
    except Exception:
        PE.PrintException()


if __name__ == '__main__':
    main()
