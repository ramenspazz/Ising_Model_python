import platform
import sys
import select
from typing import Optional
from numbers import Number
plat = platform.system().lower()

if plat == 'windows':
    print('Windows System')

    def getch() -> str:
        out = input()
        out.replace('\n', '')
        return(out)
elif plat == 'linux':
    print('Linux System')

    import tty
    import termios

    def getch(timeout: Optional[Number] = None) -> str:
        """
            Parameters
            ----------
            timeout : Optional[`Number`] strictly positive
                Sets timeout for inactivity in seconds on Unix and Windows


            Returns
            -------
            output : `str`
                The key pressed

            NOTES
            -----
            Only supports sockets on Windows.
        """
        if timeout is None:
            timeout = None
        elif timeout < 0:
            raise ValueError('Timeout must be strictly positive!')
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            setup_term(fd)
            try:
                rw, wl, xl = select.select([fd], [], [], timeout)
            except select.error:
                return
            if rw:
                return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def setup_term(fd, when=termios.TCSAFLUSH):
        mode = termios.tcgetattr(fd)
        mode[tty.LFLAG] = mode[tty.LFLAG] & ~(termios.ECHO | termios.ICANON)
        termios.tcsetattr(fd, when, mode)

blnk_ln = '                                                                   '


def print_stdout(msg: str, end: Optional[str] = None) -> None:
    cls()
    if '\n' in msg:
        msg = msg.translate({ord(c): None for c in '\n'})
    if end is not None:
        sys.stdout.write('\r' + str(msg) + end)
    else:
        sys.stdout.write('\r' + str(msg))
    sys.stdout.flush()  # important


def cls():
    sys.stdout.write('\r' + blnk_ln)
    sys.stdout.flush()  # important


def poll_key(check: Optional[list[str]] = None) -> str | list[bool, str]:
    """
        Purpose
        -------
        Waits for input and returns either the key pressed or a True or False
        value.

        Parameters
        ----------
        check : Optional[`list`[`str`]]
            A list of characters to check the input against

        Returns
        -------
        `str` of the pressed character or `bool` if check was not None.
    """
    __in = ''
    __in = getch()
    if check is not None and __in in check:
        return([True, __in])
    elif check is not None:
        return(False, __in)
    else:
        return(__in)


def key_input(check: list[str]) -> str:
    out = [False, '']
    while out[0] is not True:
        out = poll_key(check)
    return(out[1])
