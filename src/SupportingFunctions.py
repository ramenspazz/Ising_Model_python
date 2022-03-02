"""
Author: Ramenspazz

This file defines the Node class and the LinkedLattice class.
"""
from __future__ import annotations
# Typing imports
from typing import Optional
# import matplotlib.pyplot as plt
from numpy import int64, integer as Int, floating as Float, ndarray  # noqa E501
import math

# Functions and Libraries
from numba import njit
import numpy as np
import PrintException as PE

# ignore warning on line 138
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


@njit
def GetIndex(i: int, x_range: int) -> ndarray:
    """
        Returns
        -------
        index : `ndarray`[`int`]
            - A 1D numpy ndarray with 2 entries listing a (x,y) pair given an
            input integer `i` that maps to the 2D plane.
    """
    return(np.array(
        [i % x_range,
            int(i / x_range)]))


def Array2Bytes(input_arr: ndarray) -> bytes:
    """
        Pray IEEE 754 is not in the array.

        Returns
        -------
        `bytes`(`input_arr`) : `bytes`
            - byte representation of ndarray.
    """
    return(hash(input_arr.tobytes()))


def DividendRemainder(n: int | Int,
                      m: int | Int,
                      t: int | Int) -> list[int | Int]:
    """
        Purpose
        -------
        Returns a list with the dividend and remainder of (nm)/t.
        Runs in Theta(log(n)), Omega(n), O(nlog(n))

        Returns
        -------
        Remainder : `int`
            - Represents the remainder after division.

        Dividend : `int`
            - Represents a whole number p for n*m>=pt, where n, m, p, and t are
            all integers.

    """
    t = int(t)
    area = int(n * m)

    # if not (MAX_INT > area > MIN_INT):
    #     raise ValueError('Input n and m are too large!')
    if t > area:
        return([0, area])
    elif t == area:
        return([1, 0])

    test = 0
    div = 0
    div_overflow = 0
    OF_on = 0
    prev = 0
    while True:
        test = math.trunc(t * div_overflow) + math.trunc(t << div)

        if prev > area and test > area:
            return([math.trunc((t << div) / t + div_overflow),
                    area-t*np.floor(area / t)])

        elif prev < area and test > area:
            return([div,
                    area-t*np.floor(area / t)])

        if test == area:
            return([math.trunc((t << div) / t + div_overflow),
                    area-t*np.floor(area / t)])

        elif test < area:
            prev = test
            div += 1
            continue

        elif prev < area and OF_on == 0:
            div_overflow += math.trunc((t << (div - 1)) / t)
            prev = test
            OF_on = 1
            div = 1
            continue


def RoundNum(input: Float,
             figures: Int,
             round_fig: Optional[Int] = None) -> Float:
    """
        Purpose
        -------
        Rounds a number to a given number of figures, and optionally equates a
        number to zero if it is within the number of significant figures
        passed.

        Parameters
        ----------
        input : `float`
            The input number to round.
        figures : `int`
            The number of significant figures to round the input to.
        round_fig : Optional[`int`]
            The number of sigificant figures representing how close absolute
            value of the input must be to zero to round the number to
            `int`(`0`).
    """
    if isinstance(input, ndarray) is True:
        # pythons round is faster than np.round
        ret_val = np.zeros(input.shape)
        for i in range(len(input)):
            if round_fig is not None:
                ret_val[i] = round(input[i], round_fig)
            else:
                ret_val[i] = round(input[i], figures)
                if ret_val[i] is -0.0:  # intentional check for -0.0
                    ret_val[i] = 0
        return(ret_val)
    else:
        if np.abs(input) < 10**(-figures):
            return(0)
        else:
            if round_fig is not None:
                return(round(input, round_fig))
            else:
                return(round(input, figures))


def Array_IsClose(A: ndarray | list,
                  B: ndarray | list,
                  round_fig: Optional[float] = 1E-09) -> bool:
    try:
        if np.array_equiv(A.shape, B.shape) is False:
            raise ValueError('Shape of the two inputs must be equal!'
                             f' Shapes are {A.shape} and {B.shape}.')
        for A_val, B_val in zip(A, B):
            if np.abs(A_val - B_val) <= round_fig:
                continue
            else:
                return(False)
        return(True)
    except Exception:
        PE.PrintException()
