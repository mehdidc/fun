import numpy as np

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1. - tanh(x) ** 2

def softmax(x):
    x_ = np.exp(x)
    return x_ / x_.sum(axis=1)[:, np.newaxis]

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return 1.*(x > 0)

def softmax(x):
    x_ = np.exp(x)
    return x_ / x_.sum(axis=1)[:, np.newaxis]

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1. - tanh(x) ** 2

from scipy.special import expit

def sigmoid(x):
    #assert np.all(x > -300)
    return 1./(1. + np.exp(-x))

def d_sigmoid(x):
    #assert np.all(x > -300)
    return sigmoid(x) * (1 - sigmoid(x))


def d_sigmoid_(x):
    return x * (1 - x)

def d_tanh_(x):
    return 1 - x*x



def forward(Wz, Wr, Uz, Ur, W, U, bz, br, b, X):
    
    nb_timesteps = X.shape[1]
    
    H = np.zeros((X.shape[0], X.shape[1], Wz.shape[0]))

    Z = np.zeros(H.shape)
    R = np.zeros(H.shape)
    H_tilde = np.zeros(H.shape)

    h_tm1 = np.zeros((H.shape[0], H.shape[2]))
    
    for t in range(nb_timesteps):
        x_t = X[:, t, :]
        z_t = sigmoid(np.dot(x_t, Wz.T) + np.dot(h_tm1, Uz.T) + bz)
        r_t = sigmoid(np.dot(x_t, Wr.T) + np.dot(h_tm1, Ur.T) + br)
        h_t_tilde = tanh(np.dot(x_t, W.T) + r_t * np.dot(h_tm1, U.T) + b)
        h_t = z_t * h_tm1 + (1 - z_t) * h_t_tilde
        Z[:, t, :] = z_t
        R[:, t, :] = r_t
        H_tilde[:, t, :] = h_t_tilde
        H[:, t, :] = h_t
        h_tm1 = h_t
    return Z, R, H_tilde, H

def backward(Wz, Wr, Uz, Ur, W, U, bz, br, b, X, Z, R, H_tilde, H, d_H):

    nb_timesteps = X.shape[1]
    d_h_tp1 = np.zeros((H.shape[0], H.shape[2]))
    
    d_Wz = np.zeros_like(Wz)
    d_Wr = np.zeros_like(Wr)
    d_Uz = np.zeros_like(Uz)
    d_Ur = np.zeros_like(Ur)
    d_W = np.zeros_like(W)
    d_U = np.zeros_like(U)
    d_bz = np.zeros_like(bz)
    d_br = np.zeros_like(br)
    d_b = np.zeros_like(b)
    d_h_tilde_tp1 = 0
    
    for t in reversed(range(1, nb_timesteps)):
        x_t = X[:, t, :]
        h_t = H[:, t, :]
        h_tm1 = H[:, t - 1, :]
        z_t = Z[:, t, :]
        r_t = R[:, t, :]
        h_t_tilde = H_tilde[:, t, :]
        if t < (nb_timesteps) - 1:
            h_tp1 = H[:, t + 1, :]
            h_tilde_tp1 = H_tilde[:, t + 1, :]
            z_tp1 = Z[:, t + 1, :]
            r_tp1 = R[:, t + 1, :]
        else:
            h_tp1 = 0
            h_tilde_tp1 = np.zeros((h_t_tilde.shape[0], h_t_tilde.shape[1]))
            d_h_tilde_tp1 = np.zeros_like(h_tilde_tp1)
            z_tp1 = 0
            r_tp1 = 0

        d_h_t = d_H[:, t, :] + d_h_tp1 * z_tp1 + np.dot(d_h_tilde_tp1 * d_tanh_(h_tilde_tp1), U) * r_tp1
        d_h_tilde_t = d_h_t * (1 - z_t)
        d_r_t = d_h_tilde_t *  d_tanh_(h_t_tilde) * np.dot(U, h_tm1.T).T
        d_z_t = d_h_t * (h_tm1 - h_t_tilde)

        d_Wz += np.dot((d_z_t * d_sigmoid_(z_t)).T, x_t)
        d_Wr += np.dot((d_r_t * d_sigmoid_(r_t)).T, x_t)
        d_Uz += np.dot((d_z_t * d_sigmoid_(z_t)).T, h_tm1)
        d_Ur += np.dot((d_r_t * d_sigmoid_(r_t)).T, h_tm1)
        d_W += np.dot((d_h_tilde_t * d_tanh_(h_t_tilde)).T, x_t)
        d_U += np.dot((d_h_tilde_t * d_tanh_(h_t_tilde) * r_t).T, h_tm1)

        d_bz += (d_z_t * d_sigmoid_(z_t)).sum(axis=0)
        d_br += (d_r_t * d_sigmoid_(r_t)).sum(axis=0)
        d_b += (d_h_tilde_t * d_tanh_(h_t_tilde) * r_t).sum(axis=0)
        
        d_h_tilde_tp1 = d_h_tilde_t
        
    return (Wz, Wr, Uz, Ur, U, bz, br, b), (d_Wz, d_Wr, d_Uz, d_Ur, d_U, d_bz, d_br, d_b)

def forward_softmax(W, b, X):
    nb_examples, nb_timesteps, nb_outputs = X.shape[0], X.shape[1], W.shape[1]
    nb_outputs = W.shape[0]
    Y = np.zeros((nb_examples, nb_timesteps, nb_outputs))
    P = np.zeros(Y.shape)
    for t in range(nb_timesteps):
        x_t = X[:, t, :]
        p_t = (np.dot(x_t, W.T) + b)
        P[:, t, :] = p_t
        y_t = softmax(p_t) 
        Y[:, t, :] = y_t
    return Y, P

def backward_softmax(W, b, X, P, Y, Y_pred):
    nb_examples, nb_timesteps = Y.shape[0], Y.shape[1]
    nb_inputs = W.shape[1]
    d_X = np.zeros((nb_examples, nb_timesteps, nb_inputs))
    d_W = np.zeros(W.shape)
    d_b = np.zeros(W.shape[0])
    for t in reversed(range(nb_timesteps)):
        y_t = Y[:, t]
        x_t = X[:, t, :]
        
        d_h_t = -Y_pred[:, t, :].copy()
        d_h_t[np.arange(d_h_t.shape[0]), y_t] += 1
        d_h_t /= -float(Y_pred.shape[0])

        d_W += np.dot(d_h_t.T, x_t)
        d_b += d_h_t.sum(axis=0)
        d_X[:, t, :] = np.dot(d_h_t, W)
    return d_X, d_W, d_b


def loss(Y_hat, y):
    L = 0.
    nb_timesteps = Y_hat.shape[1]
    for t in range(nb_timesteps):
        y_t = y[:, t]
        y_hat_t = Y_hat[:, t, :]
        L += -np.log(y_hat_t[np.arange(y_hat_t.shape[0]), y_t])
    return L.mean()


def build_data(filename):
    fd = open(filename, "r")
    data = fd.read()
    fd.close()
    
    chars = (list(set(c for c in data)))
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }

    return data, char_to_ix, ix_to_char

def sample(Wxh, Whh, bh, vocab_size, W, b, h, seed_ix, n, rng=np.random):
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(x[:, 0].T, Wxh.T) + np.dot(Whh, h) + bh)
    y = np.dot(W, h) + b
    p = np.exp(y) / np.sum(np.exp(y))
    ix = rng.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes


if __name__ == "__main__":
    import sys
    data, char_to_ix, ix_to_char = build_data("pomodoro.csv")

    def rnd_func(rng, shape):
        #if np.all(shape == shape[0]):
        #    return np.eye(shape)
        return rng.randn(*shape)*0.01
    rng = np.random
    rng.seed(1)
     
    nb_features = len(char_to_ix)
    nb_outputs = nb_features
    nb_hidden = 10
    # softmax layer
    
    Wz = rnd_func(rng, (nb_features, nb_hidden)).T
    Wr = rnd_func(rng, (nb_features, nb_hidden)).T
    W  = rnd_func(rng, (nb_features, nb_hidden)).T
    Wy  = rnd_func(rng, (nb_hidden, nb_outputs)).T


    Uz = rnd_func(rng, (nb_hidden, nb_hidden)).T
    Ur = rnd_func(rng, (nb_hidden, nb_hidden)).T
    U  = rnd_func(rng, (nb_hidden, nb_hidden)).T


    bz = np.zeros(nb_hidden)
    br = np.zeros(nb_hidden)
    b = np.zeros(nb_hidden)

    by =  np.zeros(nb_outputs)

    # optimization hyper-parameters
    learning_rate = 0.01
    def gradient_check(updates, Wz, Wr, Uz, Ur, W, U, bz, br, b, Wy, by, X, y, subsample=False, epsilon=1e-6):
        for param, d_param in updates:
            print(param.shape)
            param_ = param.ravel()
            d_param_ = d_param.ravel()
            for i in range(param_.shape[0]):
                initial = param_[i]
                param_[i] = initial - epsilon

                Z, R, H_tilde, H = forward(Wz, Wr, Uz, Ur, W, U, bz, br, b, X)
                Y_pred, P = forward_softmax(Wy, by, H)
         
                L_before = (loss(Y_pred, y))

                param_[i] = initial + epsilon

                Z, R, H_tilde, H = forward(Wz, Wr, Uz, Ur, W, U, bz, br, b, X)
                Y_pred, P = forward_softmax(Wy, by, H)
 
                L_after = (loss(Y_pred, y))
                grad = (L_after - L_before) / (2. * epsilon)
                print(grad, d_param_[i])
                if not np.allclose(grad, d_param_[i]):
                    print("Gradient check failed")
                #assert (grad * d_param_[i]) > 0
                param_[i] = initial
    p = 0
    seq_length = 10

    batch_size = 1
    epoch = 0
    smooth_loss = -np.log(1.0/nb_features)*seq_length # loss at iteration 0
    while True:
        X = np.zeros((batch_size, seq_length, nb_features), dtype=np.float32)
        y = np.zeros((batch_size, seq_length), dtype=np.int32)
        for i in range(batch_size):
            inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
            targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
            if p+seq_length+1 >= len(data) or len(targets) < len(inputs):
                p = 0
                inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
                targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
            for t, t_inp in enumerate(inputs):
                X[i, t, t_inp] = 1
            y[i] = targets
            p += seq_length
        
        nb_timesteps = len(inputs)
 
        # forward
        Z, R, H_tilde, H = forward(Wz, Wr, Uz, Ur, W, U, bz, br, b, X)
        Y_pred, P = forward_softmax(Wy, by, H)
        #backward
        updates = []
        d_H, d_Wy, d_by = backward_softmax(Wy, by, H, P, y, Y_pred)
        updates.append((Wy, d_Wy))
        updates.append((by, d_by))
        
        params, d_params = backward(Wz, Wr, Uz, Ur, W, U, bz, br, b, X, Z, R, H_tilde, H,d_H)
        # update
        for param, d_param in (zip(params, d_params)):
            updates.append((param, d_param))
        gradient_check(updates, Wz, Wr, Uz, Ur, W, U, bz, br, b, Wy, by, X, y)
        for i, (param, d_param) in enumerate(updates):
            d_param = np.clip(d_param, -1, 1) # clip to mitigate exploding gradients
            d = learning_rate * d_param
            param -= d 

        L = (loss(Y_pred, y))
        smooth_loss = smooth_loss * 0.999 + L * 0.001

        if epoch % 100 == 0:
            # loss
            print(epoch, smooth_loss, L)
        if epoch % 1000 == 0:
            """
            print("sampling...")
            print("------------")
            print("\n")
            X_initial = np.zeros((1, nb_features))
            c = X[0, 0].argmax()
            X_initial[np.arange(X_initial.shape[0]), c] = 1
            h = np.zeros(nb_hidden)
            s = sample(Wxh, Whh, bh, nb_features, W, b, h, c, 100, rng=rng)
            print("".join([ix_to_char[d] for d in s]))
            print("\n")
            """
        epoch += 1
        break
