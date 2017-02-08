import utilities as ut
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform as sp_rand
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import numpy as np
import decimal

def drange(x, y, jump):
  while x < y:
    yield float(x)
    x += decimal.Decimal(jump)

def random_search(X_train, y_train, X_test):
    param_grid = {'alpha': sp_rand()}
    model = Ridge()

    rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=1000, scoring='accuracy')
    rsearch.fit(X_train, y_train)
    print(rsearch)
    # summarize the results of the random parameter search
    print('Random Search best score: ', rsearch.best_score_)
    print('Random Search best alpha: ', rsearch.best_estimator_.alpha)

    #score = np.mean(cross_val_score(rsearch, X_train, y_train, cv=5, scoring='accuracy'))
    #print('Cross val score: ', score)

    ut.write_output_file(rsearch.predict(X_test), file_name='ridge_random_search.csv')


def grid_search(X_train, y_train, X_test):
    alphas = np.array([0.00000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1])
    model = Ridge()

    grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas), scoring='accuracy')
    grid.fit(X_train, y_train)

    scores = np.array([x[1] for x in grid.grid_scores_])

    print('Grid Search Best score: ', grid.best_score_)
    #score = np.mean(cross_val_score(grid, X_train, y_train, cv=5, scoring='accuracy'))
    #print('Cross val score: ', score)
    #print(grid.best_estimator_.alpha)
    #print(scores)

    plt.scatter(alphas, scores, c='red', alpha=0.5)

    plt.title('Ridge Regression Accuracy vs. Alpha Value')
    plt.xlabel('alpha')
    plt.ylabel('Score')
    plt.savefig('graphs/ridge.png')
    plt.show()

    ut.write_output_file(grid.predict(X_test), file_name='ridge_grid_search.csv')


def main():
    X_train, y_train, X_test = ut.import_data()

    grid_search(X_train, y_train, X_test)
    random_search(X_train, y_train, X_test)

if __name__ == '__main__':
    main()
