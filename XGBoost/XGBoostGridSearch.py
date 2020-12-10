import numpy as np
from xgboost import XGBClassifier
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

    booster = [rsbp['booster']]
    eta = [x for x in np.linspace(max(0.1, rsbp['eta'] - 0.2), min(1, rsbp['eta'] + 0.2), num=10)]
    gamma = [x for x in np.linspace(max(0, rsbp['gamma'] - 20), min(100, rsbp['gamma'] + 20), num=100)]
    max_depth = [int(x) for x in np.linspace(max(1, rsbp['max_depth'] - 3), min(10, rsbp['max_depth'] + 3), num=6)]
    tree_method = [rsbp['tree_method']]
    grow_policy = [rsbp['grow_policy']]

    param_grid = {
        'booster': booster,
        'eta': eta,
        'gamma': gamma,
        'max_depth': max_depth,
        'tree_method': tree_method,
        'grow_policy': grow_policy
    }

    xgb = XGBClassifier(random_state=seed)
    # Instantiate the grid search object
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=cvCount, n_jobs=-1, verbose=2)
    # Fit the grid search to the data
    grid_search.fit(X_train, Y_train.ravel())
    best_params_gs.append(grid_search.best_params_)

with open('ListOfBestParamsGS.pkl', 'wb') as f:
    pickle.dump(best_params_gs, f)

print('Done')
