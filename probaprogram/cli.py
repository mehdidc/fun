import matplotlib
matplotlib.use('agg')  # NOQA

from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam, RMSprop # NOQA
from helpers import data_discretization, DocumentVectorizer, generate_text, dispims_color, categ, generate_text_deterministic

import time
import numpy as np

from skimage.io import imsave

from shape import Sampler, Point, to_img, to_img2, render

class Vectorize(object):

    def __init__(self, max_nbparts, max_nbpoints):
        self.max_nbparts = max_nbparts
        self.max_nbpoints = max_nbpoints

    def fit(self, data):
        return self

    def transform(self, data):
        x = np.zeros((len(data), self.max_nbparts * self.max_nbpoints + self.max_nbparts, 3))
        x[:, :, 2] = 1
        for i, example in enumerate(data):
            k = 0
            for part in example:
                for point in part:
                    x[i, k, 0] = point.x
                    x[i, k, 1] = point.y
                    x[i, k, 2] = 0
                    k += 1
                x[i, k, 0] = 0
                x[i, k, 1] = 0
                x[i, k, 2] = 1
                k += 1
        return x

    def inverse_transform(self, X):
        data = []
        for x in X:
            parts = []
            for cell in x:
                if cell[2] == 0:
                    x, y = cell[0], cell[1]
                    point = Point(x=x, y=y)
                    part.append(point)
                else:
                    parts.append(part)
                    part = []
        return data

class Discretize(object):

    def __init__(self, nbins=2, minval=0, maxval=1):
        self.bins = np.linspace(minval, maxval, nbins)

    def fit(self, data):
        return self

    def _transform(self, data, fn=lambda x:x):
        new_data = []
        for example in data:
            new_example = []
            for part in example:
                new_part = []
                for point in part:
                    x = fn(point.x)
                    y = fn(point.y)
                    new_point = Point(x=x, y=y)
                    new_part.append(new_point)
                new_example.append(new_part)
            new_data.append(new_example)
        return new_data

    def transform(self, data):
        def fn(x):
            return np.argmin(np.abs(self.bins - x))
        return self._transform(data, fn=fn)

    def inverse_transform(self, data):
        def fn(x):
            return self.bins[x]
        return self._transform(data, fn=fn)


def train():
    # Load data
    sampler = Sampler()
    data = [sampler.sample() for i in range(10)]

    discretize = Discretize(nbins=10)
    vectorize = Vectorize(max_nbparts=sampler.nbparts[1], max_nbpoints=sampler.nbpoints[1])

    data = discretize.transform(data)
    data = vectorize.transform(data)
    # Params

    nb_epochs = 1000
    outdir = 'out'
    nb_hidden = 128
    batch_size = 128
    width, height = 16, 16
    T = data.shape[1]
    D = data.shape[2]

    # Image model
    xim = Input(batch_shape=(batch_size, width, height), dtype='float32')
    x = xim
    x = Flatten()(x)
    x = Dense(128)(x)
    x = Activation('relu')(x)
    # Sequence Model
    x = RepeatVector(T)(x)
    h = x
    h = SimpleRNN(128, init='orthogonal',return_sequences=True)(h)
    pred_x = Activation('softmax', name='x')(TimeDistributed(Dense(D))(h))
    pred_y = Activation('softmax', name='y')(TimeDistributed(Dense(D))(h))
    pred_stop = Activation('sigmoid', name='stop')(TimeDistributed(Dense(1))(h))
    model = Model(input=xim, output=[pred_x, pred_y, pred_stop])

    optimizer = RMSprop(lr=0.001,  #(0.0001 for rnn_next_char, 0.001 for conv_aa)
                        #clipvalue=100,
                        rho=0.95,
                        epsilon=1e-8)
    model.compile(
        loss={
            'x': 'categorical_crossentropy',
            'y': 'categorical_crossentropy',
            'stop': 'binary_crossentropy'
        },
        optimizer=optimizer)
    avg_loss = -np.log(1./D)

    for epoch in range(nb_epochs):
        t = time.time()

        data = [sampler.sample() for i in range(1024)]
        images = np.array([to_img2(render(d), w=width, h=height) for d in data])
        images = np.float32(images)

        data = discretize.transform(data)
        data = vectorize.transform(data)

        print(images.shape)

        batch_losses = []
        for s in iterate_minibatches(len(outp), batchsize=batch_size, exact=True):
            x_mb = inp[s]
            y_mb = outp[s]
            y_mb = categ(y_mb, D=D)
            model.fit(x_mb, y_mb, nb_epoch=1, batch_size=batch_size, verbose=0)
            loss = model.evaluate(x_mb, y_mb,
                                  verbose=0, batch_size=batch_size)
            avg_loss = avg_loss * 0.999 + loss * 0.001
            batch_losses.append(loss)
            nb += 1
        print(np.mean(batch_losses))
        temp = 2
        cur = np.ones((batch_size, 1)) * vect._word2int[1]
        gen = generate_text(pred_func, vect,
                            cur=cur,
                            nb=batch_size,
                            max_length=T,
                            way='proba',
                            temperature=temp)
        #gen = generate_text_deterministic(pred_func, vect, cur=cur, nb=batch_size, max_length=T)
        #gen = gen.argmax(axis=-1)
        #gen = model.predict(inp[0:100]).argmax(axis=-1)
        gen = vect.inverse_transform(gen)
        #gen = inp[0:100]

        gen = undiscretize(gen)
        gen = np.array(gen, dtype='float32')

        gen = gen.reshape((gen.shape[0], width, height, 1))
        gen = gen * np.ones((1, 1, 1, 3))
        img = dispims_color(gen, border=1)
        imsave('{}/{:05d}.png'.format(outdir, epoch), img)


def iterate_minibatches(nb,  batchsize, exact=False):
    if exact:
        r = range(0, (nb/batchsize) * batchsize, batchsize)
    else:
        r = range(0, nb, batchsize)
    for start_idx in r:
        S = slice(start_idx, start_idx + batchsize)
        yield S

def floatX(x):
    return np.array(x).astype(np.float32)


if __name__ == '__main__':
    train()
