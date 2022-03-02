Important notes at the bottom of the README

# Project Brief
This is a program written in pure python that simulates the Ising model of atomic spins. The Ising model of spins is a simple, but powerful model that displays the basics of phase transitions with temperature. My model also supports the natural crystal symmetries of C3V, C4V, C5V and C6V with voids in the lattice.

# Python version required: 3.10.0+
If at anytime you want to know what python version something will run with, on any operating system, run the command ```python --version``` in your powershell/cmd/terminal to print out the version of python that will be used for all calls to ```python``` on your system path.

> NOTE: will run on Python 3.9.5 (NOT 3.9.0) in Windows but some things seem to act a little dodgy in edge cases (programmer speak for not recommended). Unix is recommended for all the nice and shiny python features and somewhat required for some feautres to work properly.

# Instructions (UNIX and UNIX like systems, like OSX)
## Install Anaconda or miniconda for UNIX. I recommend using miniconda. (WARNING, some commands might need to be ran as sudo):
- Choose the correct version of miniconda for your distribution from https://docs.conda.io/en/latest/miniconda.html#linux-installers.
- chmod +x the file and install

## Setting up the enviroment:
- Use the ```env_export.yml``` file provided in the root directory of the project to install the enviroment VIA miniconda using the command ```conda env create -f env_export.yml```.
- Run ```conda activate PY10``` to activate the installed enviroment

## Option 1: Running the py program:
- Run ```python -O Project_files/main.py``` and the program should start or start it from the included jupyter notebook.
## Option 2: Running the jupyter notebook
> Note: if you use WSL for this, you need to start the jupyter notebook service as a computation server. Please read on jupyter's website how to do this.

It is easiest to use VS code to set the computation kernel to the miniconda python venv. To do this in VS code:
- Open the notebook in VS code
- Hit ```ctrl+p``` to open the command pallet at the top center of your screen.
- Type ```select notenook kernel``` and select the name of the python3.10 venv (PY10 if you used my yml file to generate your env).
- Run the notebook

# Instructions if you're using Windows and are not UNIX shy
## Install WSL2.0:
- Open powershell elevated as admin and run the command ```wsl --install```, then restart when prompted to.
- Upon reboot and logging back into Windows, a command prompt will pop up eventually. Wait for the install to finish.
- Then in this prompt window, setup your username and password.

## Setup and install miniconda:
- Download in Windows: https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
- Then we transfer to this file to WSL by first opening a Windows explorer instance at the WSL installation directory by running the command (in the WSL window) ```explorer.exe .```.
- Next, drag and drop the previously downloaded file to the directory of your choice (I recomend somewhere in your home folder located at ```/home/<your user name>```.
- Next locate the directory you placed the file in and set the current directory (```cd <directory>```) to the location of the file ```Miniconda3-latest-Linux-x86_64.sh```.
- Run the command ```sudo chmod +x Miniconda3-latest-Linux-x86_64.sh``` to enable the file to be ran.
- Run the file with the command ```./Miniconda3-latest-Linux-x86_64.sh```.
- Follow the prompts (default is fine for this usage) and select yes at the end of the installation to instantize the new installation.

Follow the Unix instructions from §([Setting up the enviroment](https://github.com/ramenspazz/Ising_Model_python#setting-up-the-enviroment)), but make sure you are in your home folder before starting by running the command : ```cd /home/<your user name here>```. WSL for some reason doesnt set the default directory to your home folder in some test cases I have looked into.

# Instructions if you're using Windows and are UNIX shy
> Note: On my Windows test setup, Microsoft is deleting python as downloaded from anaconda and miniconda (not WSL) right now in order to push their own version of python 3.10 on the windows store. This is BS and why I recomend using WSL instead.
> Note: You can pip install the python3.10 dependencies but this is bad practice.

Download and install miniconda:
- Navigate to https://docs.conda.io/en/latest/miniconda.html and select the correct version for your system.
- From the installed anaconda powershell, run the command ```conda create -n <name here> python=3.10 matplotlib numpy scipy sympy astropy```.
- Activate the newly created python enviroment with ```conda activate <name here>```.
- Run the file main.py like so: ```<your path to the conda env> -O <the path to>\main.py```
- Finally set your launcher with your program of choice to be the anaconda powershell launcher and you are done!

# Important Notes
- I recommend using the script file ```main.py``` instead of the jupyter notebook. Jupyter notebooks just needlessly complicate things. It YMMV when it comes to using Visual Studio Code jupyter enviroments (works fine for me), versus the jupyter provided enviroment (doesn't work fine for me).
- In my test VM of Windows on my computer and on my USB bootable Windows installation for testing, the multithreading library in python correctly launches the required number of threads, but in Microsoft Windows some threads are locked by the global interpreter lock (GIL) and thus do no work. This causes 16 threads to launch, but only 6 of them to do work and can cause race conditions and hanging. This issue does not exist in OSX or UNIX. If you can get it running in native Windows then awesome, please let me know, but it is not my experience with my testing VM and USB bootable that this works properly. My friend could get it working on his laptop, but I am unsure if the other threads were doing work or not, I was not able to test that, only verify that the program ran and the expected output was generated for the program test case. WSL displays these bugs as well but still runs fine, and it is infuriating :<

You have been warned.

