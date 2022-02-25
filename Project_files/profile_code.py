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
import cProfile  # noqa E402
import pstats  # noqa E402
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
        N = 32
        M = 32
        size = [N, M]
        total_time = 200
        a = 0.1
        b = 2
        step = 0.05
        BJs = np.arange(a, b, step)

        lt_a = lt(1, size)
        lt_b = lt(1, size, [[1, 0], [0.5, np.sqrt(3)/2]])
        lt_c = lt(1, size, [[0.5, np.sqrt(3)/2], [0.5, -np.sqrt(3)/2]])

        auto_save = False
        auto_plot = False

        seed = 1644121893
        lt_a.randomize(voids=True, probs=[45, 45, 10],
                       rand_seed=seed)
        lt_b.randomize(voids=True, probs=[55, 40, 5],
                       rand_seed=seed)
        lt_c.randomize(voids=True, probs=[30, 65, 5],
                       rand_seed=seed)

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
    with cProfile.Profile() as pr:
        main()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats()
    stats.dump_stats(filename='profile.prof')
