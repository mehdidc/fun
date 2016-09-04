if __name__ == '__main__':
    import numpy as np
    from hmmlearn.hmm import MultinomialHMM
    from hmm import MultinomialHMM as MultinomialHMM2
    hmm = MultinomialHMM(n_components=2)
    X = [0] * 200 + [1] * 200
    #X = map(lambda d: d, X)
    X = np.array(X)[:, np.newaxis]
    #X = open('zero_bias_aa.py').read()[0:100]
    hmm.fit(X)
    print(hmm.score(X))
    print(hmm.sample(400)[0][:, 0])

    hmm = MultinomialHMM2(n_states=2, verbose=0, n_repeats=10)
    hmm.fit(X)
    print(hmm.loglikelihood(X))
    print(hmm.generate(400)[:, 0])
