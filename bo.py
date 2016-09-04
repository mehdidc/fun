from __future__ import print_function
from __future__ import division

from sklearn.datasets import make_classification
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC

from bayes_opt import BayesianOptimization

# Load data set and target values
data, target = make_classification(n_samples=2500,
                                   n_features=45,
                                   n_informative=12,
                                   n_redundant=7)

def func(C, gamma):
    return cross_val_score(SVC(C=C, gamma=gamma, random_state=2),
                           data, target, 'f1', cv=5).mean()

if __name__ == "__main__":

    space = {'C': (0.001, 100), 'gamma': (0.0001, 0.1)}
    svcBO = BayesianOptimization(func, space, verbose=1)
    points = {
        1: {'C':1, 'gamma': 2},
        #2: {'C':1, 'gamma': 3}
    }
    svcBO.initialize(points)
    svcBO.maximize(init_points=0, n_iter=1)
