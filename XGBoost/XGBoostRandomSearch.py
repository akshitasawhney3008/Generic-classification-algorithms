import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import pickle

# Configuration section
iter = 5
cvCount = 7
seed = 42

# List of best parameters
best_params = []

# Random search over parameters
for i in range(iter):
    X_train = np.load('X_train_' + str(i) + '.npy')
    Y_train = np.load('Y_train_' + str(i) + '.npy')

    booster = ['gbtree', 'gblinear', 'dart']
    eta = [x for x in np.linspace(0.1, 1, num=25)]
    gamma = [x for x in np.linspace(0, 100, num=75)]
    max_depth = [int(x) for x in np.linspace(1, 10, num=5)]
    tree_method = ['auto']
    grow_policy = ['depthwise', 'lossguide']

    grid = {'booster': booster,
            'eta': eta,
            'gamma': gamma,
            'max_depth': max_depth,
            'tree_method': tree_method,
            'grow_policy': grow_policy
            }
    xgb = XGBClassifier(random_state=seed)
    print('Searching')
    xgb_random = RandomizedSearchCV(estimator=xgb, param_distributions=grid, n_iter=100, cv=cvCount, random_state=seed, verbose=2)
    xgb_random.fit(X_train, Y_train.ravel())
    best_params.append(xgb_random.best_params_)

with open('ListOfBestParamsRS.pkl', 'wb') as f:
    pickle.dump(best_params, f)

print('Done')
