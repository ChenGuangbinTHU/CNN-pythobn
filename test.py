import numpy as np
from scipy.signal import convolve
'''
A = np.arange(0,12).reshape(2,2,3)
AA = np.arange(0,12).reshape(3,4)

# print(A)

kernel = np.array([
    [[1,0],
    [0,2]],
    [[1,0],
    [0,1]]
])

# print kernel.shape

# kernel = kernel[None,:,:]
# print(kernel)

kernel = kernel.transpose((1,2,0))
A = A.transpose((1,2,0))

print(kernel)

B = convolve(A,kernel,mode='valid')

print(B)
print(B.shape)
'''

kernel = np.array([
    [[1,0],
    [0,2]],
    [[1,0],
    [0,2]],
    [[1,0],
    [0,2]],
    [[1,0],
    [0,2]],
])

print(kernel.shape)

print(kernel[:,:,1])

print(np.pad(kernel,((0,0),(1,1),(2,2)),'constant'))


def im2col_sliding_broadcasting(A, BSZ, stepsize=1):
    # Parameters
    M,N = A.shape
    col_extent = N - BSZ[1] + 1
    row_extent = M - BSZ[0] + 1

    # Get Starting block indices
    start_idx = np.arange(BSZ[0])[:,None]*N + np.arange(BSZ[1])

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:,None]*N + np.arange(col_extent)

    # Get all actual indices & index into input array for final output
    return np.take (A,start_idx.ravel()[:,None] + offset_idx.ravel()[::stepsize])

mat = np.arange(0,16).reshape(4,2,2)

print(convolve(mat[:],))
# print(mat[:,::-1,::-1])
# a = np.arange(-36,0).reshape(6,6)
# print(mat.sum(axis=(1,2)))
# print(mat[:,1:-1,1:-1])
# print(mat.repeat(2,axis=2).repeat(2,axis=3).shape)
# print(a[None,:])
# print(mat * a)


# print(B)