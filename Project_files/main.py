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
from random import random  # noqa E402
import Data_Analysis as DA  # noqa E402
sys.setrecursionlimit(1000000)  # increase recursion limit


def rand_time() -> int:
    out = int(dt.datetime.now().strftime('%s'))
    sys.stdout.write(f"Time Seed = {out}\n")
    return(int(dt.datetime.now().strftime('%s')))


def main(*args, **kwargs) -> int:
    try:
        N = 16
        M = 16
        size = [N, M]
        total_time = 1000
        a = 0.1
        b = 2
        step = 0.05
        BJs = np.arange(20.1, 50.1, step)
        BJ = 0.1  # noqa

        lt_a = lt(1, size)
        lt_b = lt(1, size, [[1, 0], [0.5, np.sqrt(3)/2]])
        lt_c = lt(1, size, [[0.5, np.sqrt(3)/2], [0.5, -np.sqrt(3)/2]])
        # lt_d = lt(1, size, [[0.128, np.e], [3.02398, -np.e]])
        while True:
            
            
            inF.print_stdout(
                'Enter 0 for seeded random or 1 for time based or q to quit: ')
    
            output: str = inF.key_input(['0', '1', 'q'])
            
            # output = "0"
            if output == '0':
                inF.print_stdout("option 0 chosen..\n")
                # DOCtest seed = 1644121893
                # good seed 1644144314
                seed = 1644121893
                lt_a.randomize(voids=True, probs=[0.25, 0.4], rand_seed=seed,
                            quiet=False)
                lt_b.randomize(voids=True, probs=[0.25, 0.4], rand_seed=seed,
                               quiet=True)
                lt_c.randomize(voids=True, probs=[0.25, 0.4], rand_seed=seed,
                               quiet=True)
                # lt_d.randomize(voids=True, probs=[0.25, 0.4], rand_seed=seed,
                #                quiet=True)

            elif output == '1':
                inF.print_stdout("option 1 chosen.\n")
                lt_a.randomize(voids=True, probs=[
                    random(), random()],
                    rand_seed=rand_time(), quiet=False)

                lt_b.randomize(voids=False, probs=[
                     random(), random()],
                     rand_seed=rand_time(), quiet=False)

                lt_c.randomize(voids=False, probs=[
                     random(), random()],
                     rand_seed=rand_time(), quiet=False)

                # lt_d.randomize(voids=False, probs=[
                #      random(), random()],
                #      rand_seed=rand_time(), quiet=False)
            elif output == 'q':
                inF.cls()
                print('\n')
                exit()


            lt_a.display()
            lt_b.display()
            lt_c.display()
            # lt_d.display()
            
            
            inF.print_stdout(f"Step and range for BJ is {a} to {b} with steps {step}. Change (y/n)?")

            output = inF.key_input(['y', 'n'])
            
            if output == 'y':
                inF.print_stdout('a = ')
                a = input('\r')
                inF.print_stdout('b = ')
                b = input('\r')
                inF.print_stdout('step = ')
                step = input('\r')

            BJs = np.arange(0.1, 50.1, step)  # noqa

            # Uncomment the next 4 lines below if you want, but not
            # really a reason to as the metropolis algorithm gets
            # called anyways from the get_spin_energy function.
            # lt_a.metropolis(total_time, BJ, quiet=False)
            # lt_b.metropolis(total_time, BJ, quiet=False)
            # lt_c.metropolis(total_time, BJ, quiet=False)
            # lt_d.metropolis(total_time, BJ, quiet=False)

            # get_spin_energy is 100% complete in 34.30839276s on my home desktop
            # with n=36, m=42, threads=16 on a Ryzen 7 3700X @ 2.6-4.5Ghz using the
            # seed=1644121893 with gaussian prob settings of [0.25, 0.4]
            lt_a.get_spin_energy(BJs, total_time, quiet=False)
            # lt_b.get_spin_energy(BJs, total_time, quiet=False)
            # lt_c.get_spin_energy(BJs, total_time, quiet=False)
            # lt_d.get_spin_energy(BJs, total_time, quiet=False)

        return(0)
    except KeyboardInterrupt:
        inF.cls()
        inF.print_stdout("Keyboard Interrupt...", end='\n')
        exit()
    except Exception:
        PE.PrintException()


if __name__ == '__main__':
    main()
