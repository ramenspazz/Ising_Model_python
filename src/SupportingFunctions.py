"""
Author: Ramenspazz

This file defines the Node class and the LinkedLattice class.
"""
from __future__ import annotations
import sys
# Typing imports
from typing import Optional
# import matplotlib.pyplot as plt
from numpy import int64, integer as Int, floating as Float, ndarray

# Functions and Libraries
import numpy as np
import PrintException as PE
import datetime as dt
import random as rnd

# # ignore warning for is comparison to -0.0
# import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning)


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
            - ``list` containing the integer quotient and remainder of
            division.

    """
    # Might be necessary but for now doesnt appear to be relavant for my use
    # case. Included just incase, just uncomment and define MAX_INT and
    # MIN_INT.
    # if not (MAX_INT > area > MIN_INT):
    #     raise ValueError('Input n and m are too large!')
    if divisor > dividend:
        return([0, dividend])
    elif divisor == dividend:
        return([1, 0])
    elif dividend % divisor == 0:
        return([int64(dividend/divisor), 0])

    test = int64(0)
    div_power = int64(0)
    quotient = int64(0)
    prev = int64(0)
    while True:
        test = (divisor << div_power) + (divisor * quotient)

        if test > dividend and prev < dividend:
            if prev + divisor > dividend:
                return([quotient + (2 << div_power - 2), dividend - prev])
            quotient += 2 << (div_power - 2)
            div_power = int64(0)
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


def rand_time() -> int:
    out = int(dt.datetime.now().strftime('%s'))
    sys.stdout.write(f"Time Seed = {out}\n")
    return(int(dt.datetime.now().strftime('%s')))


def generate_random(gen_num: int) -> list:
    """
        Generates 2 or 3 random numbers whos sum is 100
    """
    if gen_num == 2:
        rand_a = rnd.randint(0, 100)
        rand_b = 100 - rand_a
        return([rand_a, rand_b])
    elif gen_num == 3:
        rand_a = rnd.randint(0, 98)
        if rand_a == 0:
            rand_b = rnd.randint(0, 99)
        else:
            rand_b = rnd.randint(0, 100-rand_a-1)
        rand_c = 100 - rand_a - rand_b
        return([rand_a, rand_b, rand_c])
