import numpy as np
from scipy import signal


def padding(matrix, pad):
    return np.pad(matrix, (pad, pad), 'constant', constant_values=((0, 0), (0, 0)))


def conv2d_forward1(input, W, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out0.97
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_out (#output channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after convolution
    '''
    N, c_in, h_in, w_in = input.shape
    c_out = W.shape[0]
    input = np.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant',
                   constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))
    ret = []
    for n in range(N):
        u = []
        for q in range(c_out):
            uq = np.zeros((h_in, w_in))
            for p in range(c_in):
                uq = uq + signal.convolve2d(input[n][p], np.rot90(W[q][p], 2), mode='valid')
            uq = uq + b[q]
            u.append(uq)
        ret.append(u)
    ret = np.array(ret)     # ret.shape: (100L, 4L, 28L, 28L)
    # print "y.shape after Conv:", ret.shape
    return ret


def conv2d_backward1(input, grad_output, W, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_out (#output channel) x h_out x w_out
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_W: gradient of W, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        grad_b: gradient of b, shape = c_out
    '''
    N, c_in, h_in, w_in = input.shape
    N, c_out, h_out, w_out = grad_output.shape
    grad_output = grad_output[np.ix_(np.arange(N), np.arange(c_out), np.arange(pad, h_out-pad), np.arange(pad, w_out-pad))]
    grad_input = np.zeros(input.shape)
    for n in range(N):
        for p in range(c_in):
            for q in range(c_out):
                grad_input[n][p] = grad_input[n][p] + signal.convolve2d(grad_output[n][q], W[q][p], mode='full')
    grad_W = np.zeros(W.shape)
    for q in range(c_out):
        for p in range(c_in):
            for n in range(N):
                grad_W[q][p] = grad_W[q][p] + signal.convolve2d(input[n][p], np.rot90(grad_output[n][q], 2), mode='valid')
    grad_b = np.sum(grad_output, axis=(0, 2, 3))
    # print "[Backward] grad.shape after Conv:", grad_input.shape
    return grad_input, grad_W, grad_b


def avgpool2d_forward1(input, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_in (#input channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after average pooling over input
    '''
    input = np.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant',
                   constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))
    N, c_in, new_h_in, new_w_in = input.shape
    h_out, w_out = new_h_in / kernel_size, new_w_in / kernel_size
    output = np.zeros((N, c_in, h_out, w_out))
    k2 = kernel_size * kernel_size
    for n in range(N):
        for p in range(c_in):
            for i in range(h_out):
                for j in range(w_out):
                    for di in range(kernel_size):
                        for dj in range(kernel_size):
                            output[n][p][i][j] += input[n][p][i*kernel_size+di][j*kernel_size+dj]
                    output[n][p][i][j] /= k2
    # print "y.shape after AvgPool:", output.shape
    return output


def upsample(matrix, pooling_size):
    return matrix.repeat(pooling_size, axis=0).repeat(pooling_size, axis=1)


def avgpool2d_backward1(input, grad_output, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_in (#input channel) x h_out x w_out
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
    '''
    input = np.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant',
                   constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))
    N, c_in, h_out, w_out = grad_output.shape
    grad_input = np.zeros(input.shape)
    k2 = kernel_size * kernel_size
    for n in range(N):
        for p in range(c_in):
            grad_input[n][p] = upsample(grad_output[n][p], kernel_size) / k2
    # print "[Backward] grad.shape after AvgPool:", grad_input.shape
    return grad_input
