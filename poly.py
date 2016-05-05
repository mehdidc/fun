import sys
import os

import numpy as np
from collections import OrderedDict

from lasagne.generative.capsule import Capsule
from lasagne.easy  import BatchOptimizer, LightweightModel
from lasagne import init, layers

import theano.tensor as T
import theano

from lasagne.layers import Layer


class RBFLayer(Layer):
    def __init__(self, incoming,
                 num_units,
                 output_w, output_h,
                 W_coef=init.GlorotUniform(),
                 W_mean=init.GlorotUniform(),
                 W_std=init.GlorotUniform(),
                 b_coef=init.Constant(0.), 
                 b_mean=init.Constant(0.),
                 b_std=init.Constant(0.),
                 **kwargs):
        super(RBFLayer, self).__init__(incoming, **kwargs)
        num_inputs = int(np.prod(self.input_shape[1:]))

        self.num_units = num_units

        self.W_coef = self.add_param(W_coef, (num_inputs, num_units))
        self.b_coef = self.add_param(b_coef, (num_units,),
                                    regularizable=False)

        self.W_mean = self.add_param(W_mean, (num_inputs, num_units))
        self.b_mean = self.add_param(b_mean, (num_units,),
                                    regularizable=False)
        self.W_std = self.add_param(W_std, (num_inputs, num_units))
        self.b_std = self.add_param(b_std, (num_units,),
                                    regularizable=False)
        self.output_w = output_w
        self.output_h = output_h

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_w, self.output_h)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            input = input.flatten(2)
        
        coef = T.dot(input, self.W_coef) + self.b_coef
        mean = T.dot(input, self.W_mean) + self.b_mean
        std = T.dot(input, self.W_std) + self.b_std
        
        x, y = grid(self.output_w, self.output_h)
        x = x.flatten(1)
        y = y.flatten(1)
        a = T.exp(-(x.dimshuffle('x', 0, 'x') - 
                 mean.dimshuffle(0, 'x', 1)) ** 2 / std.dimshuffle(0, 'x', 1)**2)
        a = a * coef.dimshuffle(0, 'x', 1).sum(axis=2)
        print(a.ndim)
        return T.exp(-(a - y) ** 2)


class Model:
    def get_all_params(self, **t):
        return list(set(self.x_to_z.get_all_params(**t) + 
                        self.z_to_x.get_all_params(**t)))

import theano
import theano.tensor as T

def grid(w, h):
    w_seq = T.arange(1, w + 1)
    h_seq = T.arange(1, h + 1)
    g_x, g_y = T.alloc(0, (w, h)), T.alloc(0, (w, h))
    
    return (T.concatenate([w_seq.dimshuffle(0, 'x')] * h, 1),
            T.concatenate([h_seq.dimshuffle('x', 0)] * w, 0))

def loss_function(model, tensors):
    return ((model.z_to_x.get_output(tensors["X"]) - tensors["X"])**2).sum(axis=1).mean()

w, h = 28, 28

x_in = layers.InputLayer((None, w*h))
z = layers.DenseLayer(x_in, 200)
x_out = RBFLayer(z, 10, w, h)

model = Model()

model.x_to_z = LightweightModel([x_in], [z])
model.z_to_x = LightweightModel([z], [x_out])

input_variables = OrderedDict()
input_variables["X"] = dict(tensor_type=T.matrix)

batch_optimizer = BatchOptimizer()
functions = dict()

capsule = Capsule(input_variables, 
                  model, loss_function, 
                  batch_optimizer=batch_optimizer, 
                  functions=functions)
from lasagne.datasets.mnist import MNIST

data = MNIST()
data.load()
X = data.X[0:100].astype(np.float32)
capsule.fit(X=X)
