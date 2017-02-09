import utilities as ut
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

import numpy as np

def run_one(X_train, y_train, X_test):
    #hidden_layer_sizes=(4) => 0.773686
    #hidden_layer_sizes=(5), adamsolver => 0.77314
    # hidden_layer_sizes=(4), adamsolver => 0.7739
    # hidden_layer_sizes=(3), adamsolver => 0.7724
    # hidden_layer_sizes=(6), adamsolver => 0.7717
    # hidden_layer_sizes=(3, 2), adamsolver => 0.77347
    # hidden_layer_sizes=(3, 2), adamsolver => 0.77265
    # hidden_layer_sizes=(2, 2), adamsolver =>0.765
    # hidden_layer_sizes=(5, 2), adamsolver =>0.7745
    # hidden_layer_sizes=(6, 2), adamsolver =>0.7735
    # hidden_layer_sizes=(4, 2), adamsolver =>0.7728
    # hidden_layer_sizes=(5, 3), adamsolver =>0.77401153
    # hidden_layer_sizes=(6, 3), adamsolver =>0.77401153
    # hidden_layer_sizes=(7, 3), adamsolver =>0.77288
    # hidden_layer_sizes=(4, 3), adamsolver =>0.7726
    mlp = MLPClassifier(hidden_layer_sizes=(4), solver='sgd',learning_rate_init=0.005,max_iter=500)

    mlp.fit(X_train, y_train)

    print(mlp.score(X_train,y_train))
    score = np.mean(cross_val_score(mlp, X_train, y_train, cv=5, scoring='accuracy'))
    print('Cross val score: ', score)

    ut.write_output_file(mlp.predict(X_test), file_name='mlp.csv')

def grid_search(X_train, y_train, X_test):
    parameters = {
     'hidden_layer_sizes': [(3), (4), (5), (6), (5, 2), (4,2), (3,2), (6, 3), (5, 3), (4, 3)],
     'solver': ['adam', 'sgd']
     }
    model = MLPClassifier(verbose=True, learning_rate_init=0.01,max_iter=500, solver='adam')
    classifier = GridSearchCV(estimator=model,  param_grid=parameters)
    classifier.fit(X_train, y_train)
    print(classifier.best_estimator_)
    print("Best parameters found:")
    print(classifier.best_params_)
    print("With a training score of:")
    print(classifier.best_score_)

    score = np.mean(cross_val_score(classifier, X_train, y_train, cv=5, scoring='accuracy'))
    print('Cross val score: ', score)

    ut.write_output_file(classifier.predict(X_test), file_name='mlp_gridsearch.csv')


def main():

    X_train, y_train, X_test = ut.import_data()

    run_one(X_train, y_train, X_test)
    grid_search(X_train, y_train, X_test)


if __name__ == '__main__':
    main()
