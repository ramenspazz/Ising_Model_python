import numpy as np
import numpy.typing as npt
from typing import Union
from numbers import Number
import random

# test_str = "10  0.0"

# asdf = test_str.split(' ')
# test_str = ''
# for i, item in enumerate(asdf):
#     if item == '':
#         continue
#     else:
#         if not (i == len(asdf) - 1):
#             test_str += item + ' '
#         else:
#             test_str += item
# print(test_str)

test = 1.2
print(isinstance(test, Union[np.number, Number]))
test = np.float64(1.2)
print(isinstance(test, Union[np.number, Number]))
print([0, 1] is npt.ArrayLike)