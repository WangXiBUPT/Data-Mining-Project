from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):    #input size = target size = (100,10)
        '''Your codes here'''
        return np.sum(np.square(input-target))/(2.0*input.shape[0])


    def backward(self, input, target):
        '''Your codes here'''
        return (input - target) / input.shape[0]
