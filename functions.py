import numpy as np
from scipy.signal import convolve
import time
from utils import log

def conv2d_forward(input, W, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_out (#output channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after convolution
    '''
    # f=open('test.txt','a+w')
    # print('conv forward',file=f)
    # print(input,file=f)
    # f.close()
    # log('conv forward',input)
    n, c_in, h_in, w_in = input.shape
    c_out, _, _, _ = W.shape
    # input = np.pad(input,pad,'constant')
    output = np.zeros([n,c_out,h_in+2*pad-kernel_size+1,w_in+2*pad-kernel_size+1])
    # print(output.shape)
    for i in range(c_out):
        for j in range(c_in):
            pad_input = np.pad(input[:,j,:,:],((0,0),(pad,pad),(pad,pad)),'constant')
            kernel = W[i,j,:,:]
            kernel = kernel[None,:,:]
            conv_result = convolve(pad_input,kernel,'valid')+b[i]
            # print(conv_result.shape)
            # print(conv_result.sum(axis=2).shape)
            output[:,i,:,:] = output[:,i,:,:]+conv_result
            # print(i)
            # print(output)
            # print(conv_result.shape)
            # print(kernel.shape)
            # print(pad_input.shape)
            # output[i] = np.sum(output[i],np.sum())
    # print(output.shape)
    return output
    pass


def conv2d_backward(input, grad_output, W, b, kernel_size, pad):
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
        grad_W: gradient of W, shape = n (#sample) x c_out (#output channel) x h_out x w_out
        grad_b: gradient of b, shape = c_out
    '''
    # print('conv back',file=f)
    # print(input,file = f)
    # time.sleep(1)
    # log('conv back',input)
    n, c_out, h_out, w_out = grad_output.shape
    _, c_in, h_in, w_in = input.shape
    #print(input.shape)
    #print(grad_output.shape)
    grad_input = np.zeros([n,c_in,h_in,h_out])
    grad_W = np.zeros(W.shape)
    grad_b = np.zeros([c_out])
    rot_grad = grad_output[:,:,::-1,::-1]
    for i in range(c_out):
        for j in range(c_in):
            kernel = W[i,j,:,:]
            kernel = kernel[None,:,:]
            grad = grad_output[:,i,:,:]
            conv = convolve(grad,kernel,'full')[:,1:-1,1:-1]
            grad_input[:,j,:,:] = grad_input[:,j,:,:] + conv
            for k in range(n):
                grad_W[i,j,:,:] = grad_W[i,j,:,:] + convolve(input[k,j,:,:],rot_grad[k,i,:,:],'valid')
            # grad_W[:,i,:,:] = grad_W[:,i,:,:]+
    grad_b = np.sum(grad_output,axis=(0,2,3))
    # print(grad_b.shape)
    # print('grad_input')
    # print(c_in)
    # print(grad_W.shape)
    # print(grad_input.shape)
    return grad_input,grad_W,grad_b
    pass


def pooling(mat, kernel_size):
    n, M, N = mat.shape
    K = 2
    L = 2

    MK = M // K
    NL = N // L
    return mat[:, :MK*K, :NL*L].reshape(n, MK, K, NL, L).mean(axis=(2, 4))

def avgpool2d_forward(input, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_in (#input channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after average pooling over input
    '''
    # print('pool forward',file=f)
    # print(input,file=f)
    # time.sleep(1)
    # log('pool forward',input)
    n, c_in, h_in, w_in = input.shape
    output = np.zeros([n, c_in, (h_in+2*pad)/kernel_size, (w_in+2*pad)/kernel_size])
    for i in range(c_in):
        pad_input = np.pad(input[:,i,:,:],((0,0),(pad,pad),(pad,pad)),'constant')
        output[:,i,:,:] = pooling(pad_input,kernel_size)
    # print(output.shape)
    # print('-------')
    # print(output)
    # exit(1)
    return output
    pass


def avgpool2d_backward(input, grad_output, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_in (#input channel) x h_out x w_out
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
    '''
    # log('pool back',input)
    # print('pool back',file=f)
    # print(input,file=f)
    output = grad_output.repeat(kernel_size,axis=2).repeat(kernel_size,axis=3)/(kernel_size**2)
    # f = open('poolinginfo.txt','a+w')
    # f.write('grad_output')
    # f.write(str(grad_output)+'\n')
    # f.write('output')
    # f.write(str(output)+'+\n')
    # f.close()
    # print(output.shape)
    return output
    pass
