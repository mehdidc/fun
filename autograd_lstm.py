import autograd.numpy as np
from autograd import grad

def forward(X, W_x, W_h):
    T = X.Shape[1]

    for t in range(T):
        H = np.dot(H, W_h) + np.dot(X, W_x)
    return np.dot(X, W)

def get_loss(X, W, y):
    return ((forward(X, W) - y) ** 2).sum()

get_grad = grad(get_loss, argnum=1)

X = np.random.uniform(-0.1, 0.1, size=(100, 10))
W = np.random.uniform(-0.1, 0.1, size=(10, 50))
y = np.random.uniform(-0.1, 0.1, size=(100, 50))

d_W = get_grad(X, W, y)
print(W - d_W)
