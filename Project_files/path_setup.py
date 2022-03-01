# This file imports a directory and exports it to the system path so
# that the directory is discoverable
import os
import sys

# Preinitialization of project
# This is responsible for running either compiled code or
# python bytecode depending on the state of __debug__
# Note : __debug__ is True unless python is ran with
# the -O (nato phonetic october) flag.


def path_setup():
    # import external modules
    nb_dir = os.path.split(os.getcwd())[0]  # noqa F841
    abs_nb_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir))
    module_dir = abs_nb_dir + '/src'
    if module_dir not in sys.path:
        sys.path.append(module_dir)
        # sys.stdout.write(f'Added to system path: {module_dir} \n')
    else:
        sys.stdout.write(f"""
        Module directory already in system path: {module_dir} \n
        """)
    return abs_nb_dir
