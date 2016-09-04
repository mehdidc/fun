import numpy as np


not_specified = None


class ParticleFilter(object):

    def __init__(self, q, q_sample, p_y_given_x, p_x_given_last_x,
                 nb_particles=10, random_state=2):
        self.q = q
        self.q_sample = q_sample
        self.p_y_given_x = p_y_given_x
        self.p_x_given_last_x = p_x_given_last_x
        self.nb_particles = nb_particles
        self.rng = np.random.RandomState(random_state)
        self.x_prev = []

    def update(self, y):
        # sample from q(x|y_i, x_1, x_2, ..., x_{i - 1})
        x = self.q_sample(y, self.x_prev, self.nb_particles,
                          rng=self.rng)
        # compute a (alpha)
        x_last = self.x_prev[-1] if len(self.x_prev) > 0 else not_specified
        a = self.p_y_given_x(y, x) * self.p_x_given_last_x(x, x_last)
        a /= self.q(x, y, self.x_prev)
        # resample
        W = a / a.sum()
        x = resample_(x, W, rng=self.rng)
        self.x_prev.append(x)

    @property
    def x(self):
        return self.x_prev[-1]


def resample_(x, w, nb=None, rng=np.random):
    if nb is None:
        nb = len(x)
    where = rng.multinomial(1, w, size=nb).argmax(axis=1)
    return x[where]

if __name__ == "__main__":
    from scipy.stats import norm

    def q(x, y, x_prev):
        if len(x_prev) == 0:
            return np.array([norm(3, 1).pdf(x_i) for x_i in x])
        else:
            x_last = x_prev[-1]
            return np.array([norm(x_last[i], 1).pdf(x_i)
                             for i, x_i in enumerate(x)])

    def q_sample(y, x_prev, nb_samples, rng):
        if len(x_prev) == 0:
            return np.array([rng.normal(3, 1) for i in range(nb_samples)])
        else:
            x_last = x_prev[-1]
            return np.array([rng.normal(x_last[i], 1)
                            for i in range(nb_samples)])

    def p_y_given_x(y, x):
        return np.array([norm(2 * x_i, 0.0001).pdf(y) for x_i in x])

    def p_x_given_last_x(x, x_last):
        if x_last is not_specified:
            return np.array([norm(5, 1).pdf(x_i) for x_i in x])
        else:
            return np.array([norm(x_last[i], 1).pdf(x_i)
                            for i, x_i in enumerate(x)])

    particle_filter = ParticleFilter(q, q_sample, p_y_given_x,
                                     p_x_given_last_x,
                                     nb_particles=100)
    import matplotlib.pyplot as plt
    import time
    plt.ion()
    plt.ylim([0, 10])
    plt.xlim([0, 300])
    plt.axhline(y=5, c='red')
    l, = plt.plot([], [])
    plt.show()
    X = []
    Y = []
    for y in range(300):
        particle_filter.update(y)
        x_est = particle_filter.x.mean()
        Y.append(y)
        X.append(x_est)
        l.set_xdata(Y)
        l.set_ydata(X)
        plt.draw()
        time.sleep(0.01)
        plt.pause(0.0001)
    plt.show()
