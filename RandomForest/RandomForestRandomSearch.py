import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

iter = 4
cvCount = 10

best_params = []

for i in range(iter):
    X_train = np.load('X_train_' + str(i) + '.npy')
    Y_train = np.load('Y_train_' + str(i) + '.npy')

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=50, stop=300, num=150)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(1, 100, num=50)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestClassifier()
    print('Searching...')
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=grid, n_iter=100, cv=cvCount, verbose=2,
                                   random_state=42)

    rf_random.fit(X_train, Y_train.ravel())
    best_params.append(rf_random.best_params_)

for best_param in best_params:
    print(best_param)