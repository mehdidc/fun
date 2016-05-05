import numpy as np

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
    assert np.all(x > -300)
    return 1./(1. + np.exp(-x))

def d_sigmoid(x):
    assert np.all(x > -300)
    return sigmoid(x) * (1 - sigmoid(x))

def d_sigmoid_(x):
    return x * (1 - x)

def d_tanh_(x):
    return 1 - x*x

def forward_softmax(W, b, X):
    nb_examples, nb_timesteps, nb_outputs = X.shape[0], X.shape[1], W.shape[1]
    nb_outputs = W.shape[1]
    Y = np.zeros((nb_examples, nb_timesteps, nb_outputs))
    P = np.zeros(Y.shape)
    for t in range(nb_timesteps):
        x_t = X[:, t, :]
        p_t = (np.dot(x_t, W) + b)
        P[:, t, :] = p_t
        y_t = softmax(p_t) 
        Y[:, t, :] = y_t
    return Y, P

def backward_softmax(W, b, X, P, Y, Y_pred):
    nb_examples, nb_timesteps = Y.shape[0], Y.shape[1]
    nb_inputs = W.shape[0]
    d_X = np.zeros((nb_examples, nb_timesteps, nb_inputs))
    d_W = np.zeros(W.shape)
    d_b = np.zeros(W.shape[1])
    for t in (range(nb_timesteps)):
        y_t = Y[:, t]
        x_t = X[:, t, :]
        d_h_t = -Y_pred[:, t, :].copy() 
        d_h_t[np.arange(d_h_t.shape[0]), y_t] += 1
        d_h_t /= -float(Y_pred.shape[0])
        d_W += np.dot(x_t.T, d_h_t)
        d_b += d_h_t.sum(axis=0)
        d_X[:, t, :] = np.dot(d_h_t, W.T)
    return d_X, d_W, d_b

def forward(layers_params, X):
    nb_examples, nb_timesteps, nb_features = X.shape
    result = []

    H_tm1 = X
    for l, layer_params in enumerate(layers_params):
        W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o, b_i, b_f, b_c, b_o = layer_params
        nb_hidden = U_f.shape[1]

        H = np.zeros((nb_examples, nb_timesteps, nb_hidden))
        C = np.zeros(H.shape)
        F = np.zeros(H.shape)
        O = np.zeros(H.shape)
        C_tilde = np.zeros(H.shape)
        I = np.zeros(H.shape)

        for t in range(nb_timesteps):
            x_t = H_tm1[:, t, :]
            if t > 0:
                h_tm1 = H[:, t - 1, :]
                C_tm1 = C[:, t - 1, :]
            else:
                h_tm1 = np.zeros((nb_examples, nb_hidden))
                C_tm1 = np.zeros(h_tm1.shape)

            i_t = sigmoid(np.dot(x_t, W_i) + np.dot(h_tm1, U_i) + b_i)
            C_t_tilde = tanh(np.dot(x_t, W_c) + np.dot(h_tm1, U_c) + b_c)
            f_t = sigmoid(np.dot(x_t, W_f) + np.dot(h_tm1, U_f) + b_f)
            C_t = i_t * C_t_tilde + f_t * C_tm1
            o_t = sigmoid(np.dot(x_t, W_o) + np.dot(h_tm1, U_o) + b_o)
            h_t = o_t * tanh(C_t)
            H[:, t, :] = h_t
            C[:, t, :] = C_t
            F[:, t, :] = f_t
            O[:, t, :] = o_t
            C_tilde[:, t, :] = C_t_tilde
            I[:, t, :] = i_t

        result.append((H_tm1, H, C, F, O, C_tilde, I))
        H_tm1 = H
    return H, result

def sample(layers_params, W, b, nb_timesteps, X_initial, rng=np.random):
    nb_examples = X_initial.shape[0]
    samples = np.zeros((nb_examples, nb_timesteps))
    x_t = X_initial

    C_tm1 = [None] * len(layers_params)
    H_tm1 = [None] * len(layers_params)
    for t in range(nb_timesteps):
        # go through each layer
        for l, layer_params in enumerate(layers_params):
            W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o, b_i, b_f, b_c, b_o = layer_params
            if t == 0:
                c_tm1 = np.zeros((nb_examples, W_c.shape[1]))
                h_tm1 = np.zeros((nb_examples, W_c.shape[1]))
            else:
                c_tm1 = C_tm1[l]
                h_tm1 = H_tm1[l]

            i_t = sigmoid(np.dot(x_t, W_i) + np.dot(h_tm1, U_i) + b_i)
            C_t_tilde = tanh(np.dot(x_t, W_c) + np.dot(h_tm1, U_c) + b_c)
            f_t = sigmoid(np.dot(x_t, W_f) + np.dot(h_tm1, U_f) + b_f)
            C_t = i_t * C_t_tilde + f_t * c_tm1
            C_tm1[l] = C_t.copy()

            o_t = sigmoid(np.dot(x_t, W_o) + np.dot(h_tm1, U_o) + b_o)
            h_t = o_t * tanh(C_t)
            H_tm1[l] = h_t.copy()

            x_t = h_t
        # go through softmax
        p_t = (np.dot(x_t, W) + b)
        y_t = softmax(p_t)
        pred = np.zeros(y_t.shape[0], dtype=np.int32)
        T = 1
        for i in range(y_t.shape[0]):
            pred[i] = rng.choice(range(y_t.shape[1]), p=y_t[i]**T/(y_t[i]**T).sum())
        x_t = np.zeros(y_t.shape)
        x_t[np.arange(x_t.shape[0]), pred]= 1
        samples[:, t] = pred
    return samples

def zeros_like(L):
    return [np.zeros(l.shape) for l in L]
def backward(layers_params,
             forward_result,
             d_X_next_layer):
    
    nb_examples, nb_timesteps, _ = d_X_next_layer.shape

    d_layer_params = []
    for layer_params, result in reversed(zip(layers_params, forward_result)):
        H_tm1, H, C, F, O, C_tilde, I = result 
        W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o, b_i, b_f, b_c, b_o = layer_params
        d_W_i, d_W_f, d_W_c, d_W_o, d_U_i, d_U_f, d_U_c, d_U_o, d_b_i, d_b_f, d_b_c, d_b_o = zeros_like(layer_params)

        nb_hidden = d_X_next_layer.shape[2]

        nb_units = H_tm1.shape[2]
        d_X = np.zeros((nb_examples, nb_timesteps, nb_units))
        d_h_tm1 = 0
        d_c_tp1 = 0
        for t in reversed(range(1, nb_timesteps)):
            x_t = H_tm1[:, t, :]
            h_t = H[:, t, :]
            c_t = C[:, t, :]
            
            d_h_t = d_h_tm1  + d_X_next_layer[:, t, :] 

            c_tm1 = C[:, t - 1, :]
            h_tm1 = H[:, t - 1, :]

            f_t = F[:, t, :]

            if t < (nb_timesteps) - 1:
                f_tp1 = F[:, t + 1, :]
            else:
                f_tp1 = np.zeros((nb_examples, H.shape[2]))

            o_t = O[:, t, :]
            c_t_tilde = C_tilde[:, t, :]
            i_t = I[:, t, :]
            f_t = F[:, t, :]
            d_o_t = d_h_t * tanh(c_t)
            d_c_t = d_h_t * o_t * d_tanh(c_t) +  d_c_tp1 * f_tp1

            d_i_t = d_c_t * c_t_tilde
            d_c_t_tilde = d_c_t * i_t
            d_f_t = d_c_t * c_tm1
            
            d_x_t = ( np.dot(d_o_t*d_sigmoid_(o_t), W_o.T) + 
                      np.dot(d_f_t*d_sigmoid_(f_t), W_f.T) +
                      np.dot(d_i_t*d_sigmoid_(i_t), W_i.T) +
                      np.dot(d_c_t_tilde*d_tanh_(c_t_tilde), W_c.T))
            d_X[:, t, :] = d_x_t

            d_h_tm1 = (np.dot(d_f_t*d_sigmoid_(f_t), U_f.T)+ 
                       np.dot(d_c_t_tilde*d_tanh_(c_t_tilde), U_c.T)+ 
                       np.dot(d_i_t*d_sigmoid_(i_t), U_i.T) + 
                       np.dot(d_o_t * d_sigmoid_(o_t), U_o.T)
                       )
            
            d_W_i += np.dot(x_t.T, d_i_t * d_sigmoid_(i_t))
            d_W_f += np.dot(x_t.T, d_f_t * d_sigmoid_(f_t))
            d_W_c += np.dot(x_t.T, d_c_t_tilde * d_tanh_(c_t_tilde))
            d_W_o += np.dot(x_t.T, d_o_t * d_sigmoid_(o_t))

            d_U_i += np.dot(h_tm1.T, d_i_t * d_sigmoid_(i_t))
            d_U_f += np.dot(h_tm1.T, d_f_t * d_sigmoid_(f_t))
            d_U_c += np.dot(h_tm1.T, d_c_t_tilde * d_tanh_(c_t_tilde))
            d_U_o += np.dot(c_t.T, d_o_t * d_sigmoid_(o_t))
            
            d_b_i += (d_i_t * d_sigmoid_(i_t)).sum(axis=0)
            d_b_f += (d_f_t * d_sigmoid_(f_t)).sum(axis=0)
            d_b_c += (d_c_t_tilde * d_tanh_(c_t_tilde)).sum(axis=0)
            d_b_o += (d_o_t * d_sigmoid_(o_t)).sum(axis=0)

            d_c_tp1 = d_c_t
            
        d = d_W_i, d_W_f, d_W_c, d_W_o, d_U_i, d_U_f, d_U_c, d_U_o, d_b_i, d_b_f, d_b_c, d_b_o 
        d_layer_params.append(d)
        d_X_next_layer = d_X
    return reversed(d_layer_params), d_X

def create_layer(nb_features, nb_hidden, rng=None, rnd_func=None):
    if rng is None:
        rng = np.random

    if rnd_func is None:
        def rnd_func(rng, shape):
            return rng.uniform(-0.01, 0.01, size=shape)
    
    W_i, W_f, W_c = [rnd_func(rng, (nb_features, nb_hidden)) for i in range(3)]
    U_i, U_f, U_c = [rnd_func(rng, (nb_hidden, nb_hidden)) for i in range(3)]

    W_o = rnd_func(rng, (nb_features, nb_hidden))
    U_o = rnd_func(rng, (nb_hidden, nb_hidden))
    
    b_i, b_f, b_c = [np.zeros(nb_hidden) for i in range(3)]
    b_o = np.zeros(nb_hidden)

    return W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o, b_i, b_f, b_c, b_o

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
    
    chars = sorted(list(set(c for c in data)))
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }

    return data, char_to_ix, ix_to_char
    
if __name__ == "__main__":

    from bokeh.plotting import cursession, figure, show, output_server
    use_bokeh = False

    if use_bokeh is True:
        output_server("lstm")
        print(cursession().__dict__)
        p = figure()
        p.line([], [], name="learning_curve_train", legend="learning curve train", color="blue")
        show(p)
        renderer = p.select(dict(name="learning_curve_train"))
        curve_train_ds = renderer[0].data_source


    import sys
    data, char_to_ix, ix_to_char = build_data("pomodoro.csv")

    def rnd_func(rng, shape):
        #if np.all(shape == shape[0]):
        #    return np.eye(shape[0])
        return rng.uniform(-0.001, 0.001, size=shape)
    rng = np.random
    rng.seed(100)
     
    nb_features = len(char_to_ix)
    nb_outputs = nb_features
    nb_hidden = 10
    #layers
    layer1 = create_layer(nb_features, nb_hidden, rng=rng, rnd_func=rnd_func)
 
    layers = [layer1]
    # softmax layer
    W = rnd_func(rng, (nb_hidden, nb_outputs))
    b = np.zeros(nb_outputs)
    # optimization hyper-parameters
    learning_rate = 0.1
    
    def gradient_check(updates, layers, W, b, X, y, subsample=False, epsilon=1e-5):
        for param, d_param in updates:
            print(param.shape)
            param_ = param.ravel()
            d_param_ = d_param.ravel()
            for i in range(param_.shape[0]):
                initial = param_[i]
                param_[i] = initial - epsilon
                
                H, forward_result = forward(layers, X)
                Y_pred, P = forward_softmax(W, b, H)
                L_before = (loss(Y_pred, y))

                param_[i] = initial + epsilon

                H, forward_result = forward(layers, X)
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
    nb_timesteps = seq_length

    batch_size = 1
    epoch = 0
    smooth_loss = -np.log(1.0/nb_features)*seq_length # loss at iteration 0
    mem = None
    while True:
        p = 0
        #if epoch % 10000 == 0:
        #    learning_rate /= 2
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
        # forward
        H_, forward_result = forward(layers, X)
        (H_tm1, H, C, F, O, C_tilde, I) = forward_result[0]

        Y_pred, P = forward_softmax(W, b, H)
               
        #backward
        updates = []


        d_S, d_W, d_b = backward_softmax(W, b, H, P, y, Y_pred)
        updates.append((W, d_W))
        updates.append((b, d_b))
        d_layer_params, d_X = backward(layers,
                                       forward_result,
                                       d_S)
        # update
        for params, d_params in reversed(zip(layers, d_layer_params)):
            updates.extend(zip(params, d_params))
        
        if mem is None:
            mem = [None] * len(updates)
        for i, (param, d_param) in enumerate(updates):
            #d_param = np.clip(d_param, -10, 10) # clip to mitigate exploding gradients
            d = learning_rate * d_param


            if mem[i] is None:
                mem[i] = d_param * d_param
            else:
                mem[i] += d_param * d_param
            d /= np.sqrt(mem[i] + 1e-8)
            param -= d 
        

        gradient_check(updates, layers, W, b, X, y)

        L = (loss(Y_pred, y))
        smooth_loss = smooth_loss * 0.999 + L * 0.001

        if epoch % 100 == 0:
            # loss
            print(epoch, smooth_loss, L)
            print("sampling...")
            print("------------")
            print("\n")
            X_initial = np.zeros((1, nb_features))
            c = X[0, 0].argmax()
            X_initial[np.arange(X_initial.shape[0]), c] = 1
            for s in sample(layers, W, b, 100, X_initial, rng=rng):
                print("".join([ix_to_char[d] for d in s]))
        epoch += 1
        if use_bokeh is True:
            curve_train_ds.data["x"].append(epoch * batch_size)
            curve_train_ds.data["y"].append(smooth_loss)
            cursession().store_objects(curve_train_ds)
        break
