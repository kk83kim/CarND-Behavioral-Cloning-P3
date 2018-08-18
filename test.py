import numpy as np

A = np.arange(20).reshape(5,4)
row_idx = [0, 1, 2, 3, 4]
col_idx = [1, 0, 2, 1, 3]
print(A)
print("")
# print(A[row_idx, col_idx])
print(A[range(A.shape[0]), col_idx])