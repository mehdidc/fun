import numpy as np

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def d_sigmoid(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - tanh(x)**2

def softmax(x):
    x_ = np.exp(x)
    return x_ / x_.sum(axis=1)[:, np.newaxis]

nb_examples = 1
nb_features = 5
nb_hidden = 10

def rnd_func(shape):
    return np.random.normal(0.01, size=shape)

def gradient_check(params, loss):
    for param in params:
        print(param.shape)
        param_ = param.ravel()
        for i in range(param_.shape[0]):
            initial = param_[i]
            param_[i] = initial - epsilon
            L_before = loss()
            
            param_[i] = initial + epsilon
            L_after = loss()
            grad = (L_after - L_before) / (2. * epsilon)
            print(grad, d_param_[i])
            if not np.allclose(grad, d_param_[i]):
                print("Gradient check failed")
            #assert (grad * d_param_[i]) > 0
            param_[i] = initial

def delta_check(compute_loss, names=None, epsilon=1e-5):
    L, V = compute_loss()

    V_names = V.keys()
    for name, value in V.items():
        if name not in names:
            continue
        param_ = value.ravel()

        for i in range(param_.shape[0]):

            val = param_[i]
            
            V_ = dict()
            V_[name] = V[name].copy()

            p = V_[name].ravel()
            p[i] = val - epsilon

            L_before, _ = compute_loss(V_)
            
            V_ = dict() 
            V_[name] = V[name].copy()

            p = V_[name].ravel()
            p[i] = val + epsilon

            L_after, _ = compute_loss(V_)

            param_[i] = val
            grad = (L_after - L_before) / (2 * epsilon)
            print(name, grad)


"""
def compute_loss(V=None):
    if V is None:
        V = dict()
    np.random.seed(5)
    o_t = V.get("o_t", rnd_func((nb_examples, nb_hidden)))
    V["o_t"] = o_t
    c_t = V.get("c_t", rnd_func((nb_examples, nb_hidden)))
    V["c_t"] = c_t
    h_t = V.get("h_t", o_t * tanh(c_t))
    V["h_t"] = h_t
    return h_t.sum(), V

L, V = compute_loss()
h_t, o_t, c_t = V["h_t"], V["o_t"], V["c_t"]
d_h_t = np.ones(h_t.shape)
d_o_t = d_h_t * tanh(c_t)
d_c_t = d_h_t * o_t * d_tanh(c_t)

delta_check(compute_loss)

print(d_h_t)
print(d_o_t)
print(d_c_t)
"""

"""
np.random.seed(5)
W_i = rnd_func((nb_features, nb_hidden))
U_i = rnd_func((nb_hidden, nb_hidden))

def compute_loss(V=None):
    if V is None:
        V = dict()
    np.random.seed(5)
    V["x_t"] = V.get("x_t", rnd_func((nb_examples, nb_features)))
    V["h_tm1"] = V.get("h_tm1", rnd_func((nb_examples, nb_hidden)))
    b_i = np.zeros(nb_hidden)
    V["W_i"] = V.get("W_i", W_i.copy())
    V["U_i"] = V.get("U_i", U_i.copy())
    V["i_t"] = V.get("i_t", sigmoid(np.dot(V["x_t"], V["W_i"]) + np.dot(V["h_tm1"], V["U_i"]) + b_i))
    return V["i_t"].sum(), V

L, V = compute_loss()
x_t, h_tm1, i_t = V["x_t"], V["h_tm1"], V["i_t"]

d_i_t = np.ones(h_tm1.shape)
d_x_t = np.dot(d_i_t * d_sigmoid(i_t), W_i.T)
d_h_tm1 = np.dot(d_i_t * d_sigmoid(i_t), U_i.T)

d_W_i = np.dot(x_t.T, d_i_t * d_sigmoid(i_t))
d_U_i = np.dot(h_tm1.T, d_i_t * d_sigmoid(i_t))

print(d_x_t)
print(d_h_tm1)
delta_check(compute_loss, names=["U_i"])
print(d_U_i)

"""

nb_timesteps = 200

np.random.seed(5)
nb_outputs = nb_features
W = rnd_func((nb_hidden, nb_outputs))
b = np.zeros(nb_outputs)

def compute_loss(V=None):
    if V is None:
        V = dict()
    np.random.seed(5)
    V["x"] = V.get("x", rnd_func((nb_examples, nb_timesteps, nb_hidden)))
    h = np.zeros((nb_examples, nb_timesteps, nb_outputs))
    y = np.zeros((nb_examples, nb_timesteps, nb_outputs))
 
    V["W"] = V.get("W", W)
    for t in range(nb_timesteps):
        x_t = V["x"][:, t, :]
        h_t = np.dot(x_t, V["W"]) + b
        h[:, t, :] = h_t
        y[:, t, :] = softmax(h_t)
    V["h"] = V.get("h", h)
    V["y"] = V.get("y", y)
    l = 0 
    for  t in range(nb_timesteps):
        y_t = V["y"][:, t, :]
        l += -np.log(y_t[:, 0])
    return l.mean(), V

L, V = compute_loss()
d_W = np.zeros(W.shape)
for t in range(nb_timesteps):
    x_t = V["x"][:, t, :]
    d_h_t = -V["y"][:, t, :].copy()
    d_h_t[np.arange(d_h_t.shape[0]), 0] += 1
    d_h_t /= -d_h_t.shape[0]
    d_W += np.dot(x_t.T, d_h_t)
print(d_W)
delta_check(compute_loss, names=["W"])
