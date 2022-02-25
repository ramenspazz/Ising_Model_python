import sys
import linecache


def PrintException():
    """
    Prints a caught `Exception` to stderr and  calls `exit()`.
    """
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    sys.stderr.write(
        f'\nEXCEPTION IN ({filename}\n At LINE {lineno} "{line.strip()}"): {exc_obj}\n' # noqa E501
        )
    exit()
