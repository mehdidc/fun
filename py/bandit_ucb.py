import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sample_from(i):
    if i == 0:
        return np.random.normal(5, 1)
    elif i == 1:
        return np.random.normal(4, 1)

narms = 2

m = [0.] * narms
s = [0] * narms

selected = []
for i in range(1000):
    thetas = [m[arm] + np.sqrt(2 * np.log(i + 1) / (s[arm] + 1))
              for arm in range(narms)]
    arm = np.argmax(thetas)
    q = sample_from(arm)
    s[arm] += 1
    m[arm] = (q + i * m[arm]) / (i + 1)
    selected.append(arm)
    for arm in range(narms):
        print( "Selected {} : {}".format(arm, (np.array(selected) == arm).mean() ))
