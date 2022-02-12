# Instructions (UNIX and UNIX like systems)
Install Anaconda for UNIX one of two ways: (WARNING, might need to be ran as sudo)
```apt install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6```
Or download and install from https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh.

Then use the env_export file provided in the root directory to install the enviroment VIA anaconda using the command:

```BASH
conda env create -f env_export.yml
```

Then run:
```BASH
conda activate PY10
```

Then finally run:
```Bash
python -O Project_files/main.py
```

Or... from the included jupyter notebook (do not reccomend).

# Instructions if you are unfortunate enough to use Windows
The truly easiet way to do this on windows is to run the WSL2.0 tools for Windows. LINK: https://docs.microsoft.com/en-us/windows/wsl/install. Then use the WSL terminal to run the UNIX and UNIX like systems commands. If you are a glutton for punishment and enjoy exercises in futility, read on.

# Instructions if you use Windows and are UNIX shy

Download miniconda from the https://docs.conda.io/en/latest/miniconda.html.

Then run from the installed anaconda powershell:
```
conda create -n <name here> python=3.10 matplotlib numpy scipy sympy astropy
```

Then run the file main.py like so:
```
<your path to the conda env> -O <the path to> main.py\main.py
```      
      
If you can get the jupyter notebook working, I do not recommend using it at all. It runs on average ~2x slower (I will show you my data if you dont believe me) than the main.py file. Save yourself the headache and just run the .py file. Jupyter notebooks suck and IDK why people want to use them what with all the overhead in them.

IF YOU INSIST on using them with python 3.1, you need to change a registry key in Computer\HKEY_CURRENT_USER\SOFTWARE\Python\ContinuumAnalytics\InstallPath named ExecutablePath to the location of the conda env you created. This might not work and honestly isnt worth the time to get it to work.

# OSX...
You might be able to run all this in the terminal, but my fiend using OSX 12.2.1 (21D62) could not get it to run due to weird permissions issues with read and write to sys.stdout or sys.stderr which is really dumb...

Getta real operating system yoooo. Y U spend so much on a whole lot of proprietary nonsense that costs on average 3.2x an equivlant system?

## Note
- All multithreading appears to be working properly. If this is not the case for you, let me know.
- I reccomend commenting out the lines with lt_d in it, they run slower and technically the same as lt_a rn.
