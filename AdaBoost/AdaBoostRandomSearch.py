import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
import pickle

# Configuration section
iter = 1
cvCount = 5
seed = 42

# List of best parameters
best_params = []

# Random search over parameters
for i in range(iter):
    X_train = np.load('X_train_' + str(i) + '.npy')
    Y_train = np.load('Y_train_' + str(i) + '.npy')

    base_estimator = []
    for j in range(1, 20):
        base_estimator.append(DecisionTreeClassifier(max_depth=j))
    n_estimators = [int(x) for x in np.linspace(25, 250, num=100)]
    algorithm = ['SAMME', 'SAMME.R']

    grid = {'base_estimator': base_estimator,
            'n_estimators': n_estimators,
            'algorithm': algorithm
           }
    adb = AdaBoostClassifier(random_state=seed)
    print('Searching')
    adb_random = RandomizedSearchCV(estimator=adb, param_distributions=grid, n_iter=100, cv=cvCount, random_state=seed, verbose=2)
    adb_random.fit(X_train, Y_train.ravel())
    best_params.append(adb_random.best_params_)

with open('ListOfBestParamsRS.pkl', 'wb') as f:
    pickle.dump(best_params, f)

print('Done')
