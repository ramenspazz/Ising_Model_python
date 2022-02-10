# Instructions
Ran from command-line as python -O Project_files/main.py or from the included jupyter notebook.

Uses the env_export file provided to install the enviroment VIA anaconda using the command:

```BASH
conda env create -f env_export.yml
```

I don't use windows so this yml appears to only work on Unix.

If manually install a python 3.10 venv, install with:

```BASH
conda env create -n <name> python=3.10 matplotlib astropy sympy scipy
```

Then use the enviroment named <name> to run the files on Windows (but better yet use ubuntu or something... it's free...).

## Note
- All multithreading appears to be working properly. If this is not the case for you, let me know.
- I reccomend commenting out the lines with lt_d in it, they run slower and technically the same as lt_a rn.
