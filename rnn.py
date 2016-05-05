import numpy as np

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1. - tanh(x) ** 2

def softmax(x):
    x_ = np.exp(x)
    return x_ / x_.sum(axis=1)[:, np.newaxis]

def forward(Wxh, Whh, bh, X):
    
    nb_timesteps = X.shape[1]
    
    H = np.zeros((X.shape[0], X.shape[1], Whh.shape[0]))
    h_tm1 = np.zeros((H.shape[0], H.shape[2]))
    
    for t in range(nb_timesteps):
        x_t = X[:, t, :]
        h_t = tanh(np.dot(x_t, Wxh.T) + np.dot(h_tm1, Whh.T) + bh)
        H[:, t, :] = h_t
        h_tm1 = h_t
    return H

def backward(Wxh, Whh, bh, X, H, d_H):

    nb_timesteps = X.shape[1]
    d_h_tp1 = np.zeros((H.shape[0], H.shape[2]))
    
    d_Wxh = np.zeros_like(Wxh)
    d_Whh = np.zeros_like(Whh)
    d_bh = np.zeros_like(bh)
    dhnext = np.zeros(bh.shape[0])
    for t in reversed(range(nb_timesteps)):
        h_t = H[:, t, :]
        if t > 0:
            h_tm1 = H[:, t - 1, :]
        else:
            h_tm1 = np.zeros(h_tm1.shape)
        x_t = X[:, t, :]
        
        d_h_t = d_H[:, t, :] * (1 - h_t * h_t) + dhnext.T * (1 - h_t * h_t)
        
        d_Wxh += np.dot(d_h_t.T, x_t)
        d_Whh += np.dot(d_h_t.T, h_tm1)
        d_bh += d_h_t.sum(axis=0)
        d_h_tp1 = d_h_t.copy()
        dhnext = np.dot(Whh.T, d_h_t.T)
    return (Wxh, Whh, bh), (d_Wxh, d_Whh, d_bh)

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
    Wxh = rnd_func(rng, (nb_hidden, nb_features))
    Whh = rnd_func(rng, (nb_hidden, nb_hidden))
    W = rnd_func(rng, (nb_outputs, nb_hidden))
    bh = np.zeros((nb_hidden,))
    b = np.zeros(nb_outputs)
    # optimization hyper-parameters
    learning_rate = 1e-2
    def gradient_check(updates, Wxh, Whh, bh, W, b, X, y, subsample=False, epsilon=1e-6):
        for param, d_param in updates:
            print(param.shape)
            param_ = param.ravel()
            d_param_ = d_param.ravel()
            for i in range(param_.shape[0]):
                initial = param_[i]
                param_[i] = initial - epsilon
                
                H = forward(Wxh, Whh, bh, X)
                Y_pred, P = forward_softmax(W, b, H)
                L_before = (loss(Y_pred, y))

                param_[i] = initial + epsilon

                H = forward(Wxh, Whh, bh, X)
                Y_pred, P = forward_softmax(W, b, H)
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
        H = forward(Wxh, Whh, bh, X)
        Y_pred, P = forward_softmax(W, b, H)
        #backward
        updates = []
        d_H, d_W, d_b = backward_softmax(W, b, H, P, y, Y_pred)
        updates.append((W, d_W))
        updates.append((b, d_b))
        
        params, d_params = backward(Wxh, Whh, bh,
                                    X, H, d_H)
        # update
        for param, d_param in reversed(zip(params, d_params)):
            updates.append((param, d_param))
        gradient_check(updates, Wxh, Whh, bh, W, b, X, y)
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
        epoch += 1
        break
