# Python version used : 3.10.0
If at anytime you want to know what python version something will run with, on any operating system, run the command ```python --version``` in your powershell/cmd/terminal to print out the version of python that will be used for all calls to ```python``` on your system path.

> NOTE: will run on Python 3.9.5 (NOT 3.9.0) in Windows but some things seem to act a little dodgy in edge cases (programmer speak for not recommended). Unix is reccomended for all the nice and shiny python features.

# Instructions (UNIX and UNIX like systems)
Install Anaconda or miniconda for UNIX. I recommend using miniconda. (WARNING, some commands might need to be ran as sudo):
- Choose the correct version of miniconda for your distribution from https://docs.conda.io/en/latest/miniconda.html#linux-installers.
- chmod +x the file and install

Then use the env_export file provided in the root directory to install the enviroment VIA anaconda using the command ```conda env create -f env_export.yml```.

Next, run ```conda activate PY10``` to activate the installed enviroment and then finally run ```python -O Project_files/main.py``` and the program should start... Or... start it from the included jupyter notebook (do not recommend).

# Instructions if you're using Windows and are not UNIX shy
Install WSL2.0:
- Open powershell elevated as admin and run the command ```wsl --install```, then restart when prompted to.
- Upon reboot and logging back into Windows, a command prompt will pop up eventually. Wait for the install to finish.
- Then in this prompt window, setup your username and password.

Setup miniconda:
- Download in Windows: https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
- Then we transfer to this file to WSL by first opening a Windows explorer instance at the WSL installation directory by running the command (in the WSL window) ```explorer.exe .```.
- Next, drag and drop the previously downloaded file to the directory of your choice (I recomend somewhere in your home folder located at ```/home/<your user name>```.

Install miniconda on ubuntu:
- Begin by locating the directory and setting the current directory (```cd <directory>```) to the location of the file ```Miniconda3-latest-Linux-x86_64.sh```.
- Run the command ```sudo chmod +x Miniconda3-latest-Linux-x86_64.sh``` to enable the file to be ran.
- Run the file with the command ```./Miniconda3-latest-Linux-x86_64.sh```.
- Follow the prompts (default is fine for this usage) and select yes at the end of the installation to instantize the new installation.

Follow the Unix instructions from here, but make sure you are in your home folder before starting by running the command : cd ```/home/<your user name>```
WSL for some reason doesnt set the default directory to your home folder in some test cases I have looked into.

# Instructions if you're using Windows and are UNIX shy
Note: On my Windows test setup, Microsoft is deleting python as donwloaded from anaconda and miniconda right now in order to push their own version of python 3.10 on the windows store. This is BS and why I recomend using WSL instead.

Download miniconda from the https://docs.conda.io/en/latest/miniconda.html.

Then run from the installed anaconda powershell, run the command:
```conda create -n <name here> python=3.10 matplotlib numpy scipy sympy astropy```

Then activate the newly created python enviroment with ```conda activate <name here>``` and finally run the file main.py like so:
```
<your path to the conda env> -O <the path to> main.py\main.py
```
Next set your launcher with your program of choice to be the anaconda powershell launcher and you are done!

If you can get the jupyter notebook working, I do not recommend using it at all. It runs on average ~2x slower (I will show you my data if you don't believe me) than the main.py file. Save yourself the headache and just run the .py file. Jupyter notebooks suck and IDK why people want to use them what with all the overhead in them.

IF YOU INSIST on using them with python 3.10.0, you need to change a registry key in ```Computer\HKEY_CURRENT_USER\SOFTWARE\Python\ContinuumAnalytics\InstallPath``` named ```ExecutablePath``` to the location of the conda env you created. This might not work and honestly isn't worth the time to get it to work.



## Notes
- I reccomend commenting out the lines with lt_d in it, they run slower and technically the same as lt_a rn.
- In my test VM of Windows on my computer and on my USB bootable Windows installation for testing, the multithreading library in python correctly launches the required number of threads, but in Microsoft Windows some threads are locked by the global interpreter lock (GIL) and thus do no work. This causes 16 threads to launch, but only 6 of them to do work and can cause race conditions and hanging. This issue does not exist in OSX or UNIX. If you can get it running in native Windows then awesome, please let me know, but it is not my experience with my testing VM and USB bootable that this works properly. My friend could get it working on his laptop, but I am unsure if the other threads were doing work or not, I was not able to test that, only verify that the program ran and the expected output was generated for the program test case. WSL displays these bugs as well but still runs fine, and it is infuriating :<

You have been warned.
