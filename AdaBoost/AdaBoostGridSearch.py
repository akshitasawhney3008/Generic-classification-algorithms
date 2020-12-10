import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
import pickle

# Configuration section
iter = 1
cvCount = 5
seed = 42

# Load list of best parameters from Random Search
with open('ListOfBestParamsRS.pkl', 'rb') as f:
    best_params = pickle.load(f)

best_params_gs = []
# Grid search over parameters
for i in range(iter):
    X_train = np.load('X_train_' + str(i) + '.npy')
    Y_train = np.load('Y_train_' + str(i) + '.npy')
    rsbp = best_params[i]

    base_estimator = [rsbp['base_estimator']]
    n_estimators = [int(x) for x in np.linspace(max(10, rsbp['n_estimators'] - 25), min(300, rsbp['n_estimators'] + 25), num=50)]
    algorithm = [rsbp['algorithm']]

    grid = {'base_estimator': base_estimator,
            'n_estimators': n_estimators,
            'algorithm': algorithm
           }

    adb = AdaBoostClassifier(random_state=seed)
    # Instantiate the grid search object
    grid_search = GridSearchCV(estimator=adb, param_grid=grid, cv=cvCount, n_jobs=-1, verbose=2)
    # Fit the grid search to the data
    grid_search.fit(X_train, Y_train.ravel())
    best_params_gs.append(grid_search.best_params_)

with open('ListOfBestParamsGS.pkl', 'wb') as f:
    pickle.dump(best_params_gs, f)

print('Done')
