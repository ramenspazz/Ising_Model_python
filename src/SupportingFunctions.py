"""
Author: Ramenspazz

This file defines the Node class and the LinkedLattice class.
"""
from __future__ import annotations
# Typing imports
from typing import Optional
# import matplotlib.pyplot as plt
from numpy import int64, integer as Int, floating as Float, ndarray  # noqa E501

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


def DividendRemainder(dividend: int | Int,
                      divisor: int | Int) -> list[int | Int]:
    """
        Purpose
        -------
        Returns a list with the quotient and remainder of `dividend` / `divisor`.
        I designed this to run in exponential jumps of the power of 2 using the
        left bitshift operator, so it functions faster than the standard
        implimentation of the remainder algorithm that I have seen.
        Runs in Omega(log(n)), Theta(n), O(nlog(n))

        Returns
        -------
        [Quotient, Remainder] : `list`
            - ``list` containing the integer quotient and remainder of division.

    """
    # Might be necessary but for now doesnt appear to be relavant for my use
    # case. Included just incase, just uncomment.
    # if not (MAX_INT > area > MIN_INT):
    #     raise ValueError('Input n and m are too large!')
    if divisor > dividend:
        return([0, dividend])
    elif divisor == dividend:
        return([1, 0])

    test = np.int64(0)
    div_power = np.int64(0)
    quotient = np.int64(0)
    prev = np.int64(0)
    while True:
        test = (divisor << div_power) + (divisor * quotient)

        if test == dividend:
            return([2 << div_power + quotient, 0])

        elif test > dividend and prev < dividend:
            if prev + divisor > dividend:
                return([quotient + (2 << div_power - 2), dividend - prev])
            quotient += 2 << (div_power - 2)
            div_power = np.int64(0)
            prev = quotient * divisor
            continue

        elif test < dividend:
            prev = test
            div_power += 1
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
