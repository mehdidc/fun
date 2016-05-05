import os
import matplotlib as mpl
mpl.use('Agg') # NOQA
from invoke import task
from caffezoo.vgg import VGG
from caffezoo.googlenet import GoogleNet
from lasagnekit.datasets.imagenet import ImageNet
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import LeakyReLU


def build_dense(input_dim=784, h_dim=1500, output_dim=1000):
    init = 'glorot_normal'
    model = Sequential()
    shape = (np.prod(input_dim),)
    model.add(Reshape(shape, input_shape=input_dim))
    model.add(Dense(h_dim, init=init))
    # model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dense(output_dim, init=init))
    return model


@task
def train():
    import pandas as pd
    from keras.optimizers import SGD, Adam, RMSprop, Adadelta  # NOQA
    import matplotlib.pyplot as plt
    imagenet_categories = open(os.getenv("DATA_PATH")+ "/imagenet/synset_words.txt").readlines()
    imagenet_categories = map(lambda cat: cat.split(" ")[0], imagenet_categories)
    int_to_cat = {i: cat for i, cat in enumerate(imagenet_categories)}
    size = (224, 224)
    batch_size = 128

    # dataset
    imagenet = ImageNet(size=size, nb=batch_size, crop=True,
                        categories=imagenet_categories)
    imagenet.rng = np.random.RandomState(2)
    imagenet.load()

    # teacher
    teacher = GoogleNet(layer_names=["loss3/classifier"],
                        input_size=size, resize=False)
    teacher._load()
    y = teacher.transform(imagenet.X[0:1])
    output_dim = y.shape[1]

    # student
    c = 3
    input_dim = (c,) + size
    output_dim
    h_dim = 1400

    student = build_dense(
            input_dim=input_dim,
            h_dim=h_dim,
            output_dim=output_dim)
    optim = Adam(lr=0.001, decay=0.0001, beta_1=0.9, beta_2=0.999)

    student.compile(loss='mean_squared_error', optimizer=optim)
    callbacks = []
    nb_epochs = 200000
    avg_mse = 0
    avg_acc = 0
    B = 0.9
    losses = []
    accs = []
    avg_accs = []
    avg_losses = []

    with open("out/arch.json", "w") as fd:
        fd.write(student.to_json())

    if os.path.exists("out/weights.h5"):
        student.load_weights("out/weights.h5")

    for t in range(nb_epochs):

        print("Epoch {}".format(t))
        imagenet.load()
        X = imagenet.X
        y = teacher.transform(X)
        print(y.shape)
        student.fit(X=X, y=y,
                    nb_epoch=1,
                    batch_size=batch_size,
                    callbacks=callbacks)

        y_pred = student.predict(X)
        mse = ((y_pred - y) ** 2).sum(axis=1).mean()
        avg_mse = avg_mse * B + (1 - B) * mse
        fix = 1 - B ** (1 + t)
        avg_mse_fix = avg_mse / fix
        print("MSE", mse, avg_mse_fix)

        avg_losses.append(avg_mse_fix)
        losses.append(mse)

        pred_teacher = y.argmax(axis=1)
        pred_student = y_pred.argmax(axis=1)
        acc = (pred_teacher == pred_student).mean()
        avg_acc = avg_acc * B + (1 - B) * acc
        avg_acc_fix = avg_acc / fix
        print("ACC w.r.t teacher", acc, avg_acc_fix)
        avg_accs.append(avg_acc_fix)
        accs.append(acc)

        if t % 100 == 0:
            fig = plt.figure()
            plt.plot(losses)
            plt.savefig("out/loss.png")
            plt.close(fig)
            plt.plot(accs)
            plt.savefig("out/acc.png")
            plt.close(fig)

            plt.plot(avg_losses)
            plt.savefig("out/avg_loss.png")
            plt.close(fig)
            plt.plot(avg_accs)
            plt.savefig("out/avg_acc.png")
            plt.close(fig)

            pd.Series(losses).to_csv("out/losses.csv")
            pd.Series(accs).to_csv("out/acc.csv")
            pd.Series(avg_losses).to_csv("out/avg_losses.csv")
            pd.Series(avg_accs).to_csv("out/avg_acc.csv")
            student.save_weights("out/weights.h5", overwrite=True)
