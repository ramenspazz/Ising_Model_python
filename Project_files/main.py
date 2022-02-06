"""
Author: Ramenspazz

Purpose: main driver file for ISING model simulation
"""
# Begin by importing external libraries using path_setup
# NOTE : Must be ran first, thus PEP8-E402
import path_setup
path_setup.path_setup()
import sys  # noqa E402
import matplotlib.pyplot as plt # noqa E402
import numpy as np # noqa E402
import datetime as dt # noqa E402
from LatticeClass_F import lattice_class as lt # noqa E402
import PrintException as PE  # noqa E402
import input_funcs as inF  # noqa E402
from random import random  # noqa E402
import Data_Analysis as DA


def plot_metrop(SE_mtx, BJ, size):
    spin_up = 0
    spin_dn = 0
    spin0 = 0
    for val in SE_mtx[:,0]:
        if val > 0:
            spin_up += val
        elif val == 0:
            spin0 += 1
        elif val < 0:
            spin_dn += val
    total = np.abs(spin_up)+np.abs(spin_dn)
    sq_tot = np.abs(spin_up)**2+np.abs(spin_dn)**2
    mean = DA.data_mean(SE_mtx[:, 0])
    stdev = DA.std_dev(SE_mtx[:, 0], mean, sample=False)
    print(f"""
    The percent of time spent spin up on average is {np.abs(spin_up) / (size[0]*size[1])}\%\n
    The percent of time spent spin dn on average is {np.abs(spin_dn) / (size[0]*size[1])}\%\n
    The mean is {mean}\n
    The Standard deviation of the mean is {stdev / np.sqrt(mean)}\n
    """)
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4),
        num=f'Evolution of Average Spin n={(size[0]*size[1])**2} and Energy for BJ={BJ}')
    ax = axes[0]
    ax.plot(SE_mtx[:, 0] / (size[0]*size[1]))
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


def rand_time() -> int:
    out = int(dt.datetime.now().strftime('%s'))
    sys.stdout.write(f"Time Seed = {out}\n")
    return(int(dt.datetime.now().strftime('%s')))


def main(*args, **kwargs) -> int:
    try:
        N = 8
        M = 8
        size = [N, M]
        lt_a = lt(1, size)
        lt_b = lt(1, size, [[1, 0], [0.5, np.sqrt(3)/2]])
        lt_c = lt(1, size, [[0.5, np.sqrt(3)/2], [0.5, -np.sqrt(3)/2]])
        lt_d = lt(1, size, [[0.128, np.e], [3.02398, -np.e]])
        
        # good seed 1644144314
        sys.stdout.write(
            '\nEnter 0 for seeded random or 1 for time based:\n')

        output: str = inF.key_input(['0', '1'])

        if output == '0':
            # DOCtest seed = 1644121893
            seed = 1644121893
            lt_a.randomize(voids=True , probs=[0.49, 0.56], rand_seed=seed, quiet=False)
            lt_b.randomize(voids=False, probs=[0.75], rand_seed=seed)
            lt_c.randomize(voids=False, probs=[1-0.75], rand_seed=seed)
            lt_d.randomize(voids=False, probs=[1-0.75], rand_seed=seed)

        else:
            lt_a.randomize(voids=True , probs=[0.49, 0.56], rand_seed=rand_time(), quiet=False)
            lt_b.randomize(voids=False, probs=[0.49], rand_seed=rand_time(), quiet=False)
            lt_c.randomize(voids=False, probs=[0.49], rand_seed=rand_time(), quiet=False)
            lt_d.randomize(voids=False, probs=[0.49], rand_seed=rand_time(), quiet=False)

        # lt_a.display()
        # lt_b.display()
        # lt_c.display()
        # lt_d.display()

        # BJs = np.arange(0.1, 2, 0.05)
        total_time = (N*M)**2
        BJ = -10

        SE_mtx = lt_a.metropolis(total_time, BJ)
        plot_metrop(SE_mtx, BJ, size)
        SE_mtx = lt_b.metropolis(total_time, BJ)
        plot_metrop(SE_mtx, BJ, size)
        SE_mtx = lt_c.metropolis(total_time, BJ)
        plot_metrop(SE_mtx, BJ, size)
        SE_mtx = lt_d.metropolis(total_time, BJ)
        plot_metrop(SE_mtx, BJ, size)

        return(0)

    except Exception:
        PE.PrintException()


if __name__ == '__main__':
    main()
