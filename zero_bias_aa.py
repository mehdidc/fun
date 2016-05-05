import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA
from skimage.io import imread
from skimage.transform import resize
if __name__ == "__main__":
    from lasagnekit import easy
    from lasagnekit.datasets.mnist import MNIST
    from lasagnekit.datasets.fonts import Fonts
    from sklearn.datasets import load_digits

    import glob

    images = list(glob.glob("/home/mehdi/Dropbox/photos-france/*.JPG"))
    images = images[0:10]
    #images = [
    #    "sample.png",
    #    "sample2.png"
    #]
    resize_h, resize_w = 200, 200
    X = []
    for filename in images:
        img = imread(filename)
        img = resize(img, (resize_h, resize_w), preserve_range=True)
        X.append(img)
    X = np.array(X)
    plt.ion()

    c, real_width, real_height = 3, resize_h, resize_w
    mean = X.mean(axis=(0, 2, 3))[np.newaxis, :, np.newaxis, np.newaxis]
    X = X - mean

    X_ = X
    X = X.reshape((X.shape[0], np.prod(X.shape[1:])))


    #X = StandardScaler().fit_transform(X)


    width, height = 10, 10 
    X_full = X
    nb_input = c * width * height
    nb_hidden = 10
    nb_hidden_sqrt = int(np.sqrt(nb_hidden))
    theta = 8 
    alpha = 0.01
    batch_size = 10

    W = np.random.uniform(-0.01, 0.01, size=(nb_input, nb_hidden))

    features = W.T.reshape((nb_hidden, c, width, height))
    space = 2
    
    F = np.zeros((c, nb_hidden_sqrt * (width+space), nb_hidden_sqrt * (height + space)))


    online_error = 0.
     
    i = 0
    b = 0
    full = np.arange(X_full.shape[0])

    def get_X():
        subset = np.random.randint(0, X_full.shape[0], size=batch_size)
        #subset = full[b:b+batch_size]
        #b = (b +  batch_size) % full.shape[0]
        X = X_full[subset]
        startx, starty = np.random.randint(0, real_width - width + 1), np.random.randint(0, real_height - height + 1)
        X__ = X_[:, startx:startx+width, starty:starty+height]
        X__ = X__.reshape((X_.shape[0], width*height*c))
        return X__

    X = get_X()
    h_tilde = np.dot(X, W)
    h = (h_tilde > theta) * h_tilde
    X_hat = np.dot(h, W.T)
    #plt.hist(h_tilde.flatten())
    #plt.show(block=True)
    
    mem = np.zeros(W.shape)
    index = 0
    while True:
        if index == 100:
            break
        index += 1
        X = get_X()
        # forward
        h_tilde = np.dot(X, W)
        h = (h_tilde > theta) * h_tilde
        X_hat = np.dot(h, W.T)

        #backward
        d_X_hat = (X_hat - X) / X.shape[0]
        d_h = np.dot(d_X_hat, W)
        d_h_tilde = d_h * (h_tilde > theta)

        d_W = np.dot(X.T, d_h_tilde) + np.dot(d_X_hat.T, h)
        
        #mem = mem * 0.9 + (d_W*d_W) * 0.1
        mem += d_W*d_W

        W -= alpha * (d_W / np.sqrt(mem + 1e-6))
        
        error = 0.5 * ((X - X_hat)**2).sum(axis=1).mean()
        online_error = (error + i * online_error) / (i + 1)
        assert not np.isnan(online_error)

        #if i % 10000 == 0:
        #    alpha /= 2
        
        if i % 100 == 0:
            k = 0
            for a in range(nb_hidden_sqrt):
                for b in range(nb_hidden_sqrt):
                    F[:, (a*width+a*space):(a+1)*width+a*space, (b*height+b*space):(b+1)*height+b*space] = features[k]
                    k += 1
            F_n = F.transpose((1, 2, 0)).copy()

            a = F_n.min(axis=(1, 2))[:, np.newaxis, np.newaxis]
            b = F_n.max(axis=(1, 2))[:, np.newaxis, np.newaxis]
            F_n = (F_n - a) / (b - a)
            plt.imshow(F_n)
            plt.draw()
            time.sleep(0.001)
            plt.pause(0.001)

        print(i*batch_size, online_error)
        i += 1
