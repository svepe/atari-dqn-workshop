from time import process_time

import numpy as np
import cupy as cp

x = 2 * np.ones((10000, 10000))

# Run calculations on CPU
start = process_time()
for i in range(32):
    x *= 2
print(process_time() - start)

# Move to GPU
x = cp.asarray(x)

# Run calculations on GPU
start = process_time()
for i in range(32):
    x *= 2
print(process_time() - start)


# Move back to CPU
x = x.get()

# Run calculations on CPU
start = process_time()
for i in range(32):
    x *= 2
print(process_time() - start)
