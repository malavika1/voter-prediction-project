import utilities as ut
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform as sp_rand
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import numpy as np

def random_search(X_train, y_train, X_test):
    trees = range(10, 120, 3)
    param_grid = {'max_depth': [1, 2, 3, 4, 5], 'n_estimators': trees}
    model = RandomForestClassifier()

    rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, \
     n_iter=200, cv=5)
    rsearch.fit(X_train, y_train)
    print(rsearch)
    # summarize the results of the random parameter search
    print('Random Search best score: ', 100 * (1 - rsearch.best_score_))
    print('Random Search best depth: ', rsearch.best_estimator_.max_depth)
    print('Random Search best num_trees: ', rsearch.best_estimator_.n_estimators)

    #score = np.mean(cross_val_score(rsearch, X_train, y_train, cv=5, scoring='accuracy'))
    #print('Cross val score: ', score)

    ut.write_output_file(rsearch.predict(X_test), file_name='random_random_foreset_search.csv')

def main():
    X_train, y_train, X_test, X_test_2 = ut.import_data()
    print 'got data\n'
    random_search(X_train, y_train, X_test)
    
if __name__ == '__main__':
    print 'starting:\n'
    main()
