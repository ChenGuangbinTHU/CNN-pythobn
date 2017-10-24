from __future__ import division
import numpy as np
from utils import log


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        return 0.5 * np.mean(np.sum(np.square(input - target), axis=1))

    def backward(self, input, target):
        return (input - target) / len(input)


# class SoftmaxCrossEntropyLoss(object):
#     def __init__(self, name):
#         self.name = name

#     def forward(self, input, target):
#         '''Your codes here'''
#         # print('loss forward')
#         # print(input)
#         # time.sleep(1)
#         # print(input.shape)
#         # print(input)
#         # log('loss forward',input)
#         exp_input = np.exp(input)
#         # print(exp_input.shape)
#         exp_sum = np.sum(exp_input,axis=1)[:,None]
#         if 0 in exp_sum or np.inf in exp_sum:
#             print(input)
#             print(exp_input)
#             print(exp_sum)
#             exit(1)
#         # print(exp_sum)
#         # exit(1)
#         softmax = exp_input / exp_sum
#         self.softmax = softmax
#         # print(np.sum(softmax,axis=1))
#         # print(softmax.shape)
#         # print(target.shape)
#         output = -1 * np.sum(softmax*np.log(softmax),axis=1)
#         # print(output.shape)
#         return output
#         pass

#     def backward(self, input, target):
#         '''Your codes here'''
#         # print(self.softmax)
#         # print('loss back')
#         # print(input)
#         # time.sleep(1)
#         # log('loss back',input)
#         return -1 * (target-self.softmax) * input / len(input)
#         pass
class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        ex = np.exp(input)
        H = np.log(ex / np.sum(ex, axis=1).reshape(-1, 1))   # 100*10, ln_hk^(n)
        return -np.mean(np.sum(target * H, axis=1))

    def backward(self, input, target):
        ex = np.exp(input)
        softmax_output = ex / np.sum(ex, axis=1).reshape(-1, 1)
        return (softmax_output - target) / len(input)
