# Python version used : 3.10.0
NOTE: will run on Python 3.9.5 (NOT 3.9.0) in windows but some things seem to act a little dodgy in edge cases. Unix is reccomended for all the nice and shiny python features.

If at anytime you want to know what python version something will run with, on any operating system, run the command ```python --version``` in your powershell/cmd/terminal to print out the version of python that will be used for all calls to ```python``` on your system path.

# Instructions (UNIX and UNIX like systems)
Install Anaconda for UNIX one of two ways: (WARNING, might need to be ran as sudo)
```apt install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6```, or download and install from https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh.

Then use the env_export file provided in the root directory to install the enviroment VIA anaconda using the command:

```BASH
conda env create -f env_export.yml
```

Then run ```conda activate PY10``` to activate the installed enviroment, then finally run ```python -O Project_files/main.py``` and the program should start... Or... start it from the included jupyter notebook (do not reccomend).

# A quick note about using windows and existing bugs that stem from Microsoft depending on your runtime setup (network-boot vs usb launched vs native OS vs VM)
TL;DR

> Windows sucks for program development, bugs currently exist that Microsoft isnt willing to fix.

In my test VM of Windows on my computer and on my USB bootable Windows installation for testing, the multithreading library in python correctly launches the required number of threads, but in Microsoft Windows some threads are locked by the global interpreter lock (GIL) and thus do no work. This causes 16 threads to launch, but only 6 of them to do work and can cause race conditions and hanging. This issue does not exist in OSX or UNIX. If you can get it running in native Windows then awesome, please let me know, but it is not my experience with my testing VM and USB bootable that this works properly. My friend could get it working on his laptop, but I am unsure if the other threads were doing work or not, I was not able to test that, only verify that the program ran and the expected output was generated for the program test case.

You have been warned.

# Instructions if you're using Windows and are not UNIX shy
The truly easiet way to do this on windows is to run the WSL2.0 tools for Windows. LINK: https://docs.microsoft.com/en-us/windows/wsl/install. Then use the WSL terminal to run the UNIX and UNIX like systems commands. If you are a glutton for punishment and enjoy exercises in futility, read on.

# Instructions if you're using Windows and are UNIX shy

Download miniconda from the https://docs.conda.io/en/latest/miniconda.html.

Then run from the installed anaconda powershell, run the command:
```conda create -n <name here> python=3.10 matplotlib numpy scipy sympy astropy```

Then activate the newly created python enviroment with ```conda activate <name here>``` and finally run the file main.py like so:
```
<your path to the conda env> -O <the path to> main.py\main.py
```

If you can get the jupyter notebook working, I do not recommend using it at all. It runs on average ~2x slower (I will show you my data if you dont believe me) than the main.py file. Save yourself the headache and just run the .py file. Jupyter notebooks suck and IDK why people want to use them what with all the overhead in them.

IF YOU INSIST on using them with python 3.10.0, you need to change a registry key in Computer\HKEY_CURRENT_USER\SOFTWARE\Python\ContinuumAnalytics\InstallPath named ExecutablePath to the location of the conda env you created. This might not work and honestly isnt worth the time to get it to work.

## Note
- All multithreading appears to be working properly. If this is not the case for you, let me know.
- I reccomend commenting out the lines with lt_d in it, they run slower and technically the same as lt_a rn.
