#!/usr/bin/python3.5

import time
import array
import numpy as np
import numexpr as ne

# Creating large arrays
N = 1000000
list_a = list(range(N))
list_b = list(range(N))
array_a = array.array('i', list_a)
array_b = array.array('i', list_b)
numpy_a = np.array(list_a)
numpy_b = np.array(list_b)

# Python List
start_time = time.time()
sum([a * b for a, b in zip(list_a, list_b)])
print("Python List: {:.5f} seconds".format(time.time() - start_time))

# Array Module
start_time = time.time()
sum([a * b for a, b in zip(array_a, array_b)])
print("Array Module: {:.5f} seconds".format(time.time() - start_time))

# NumPy
start_time = time.time()
np.sum(numpy_a * numpy_b)
print("NumPy: {:.5f} seconds".format(time.time() - start_time))

# NumExpr
start_time = time.time()
ne.evaluate("sum(a*b)", {'a': numpy_a, 'b': numpy_b})
print("NumExpr: {:.5f} seconds".format(time.time() - start_time))