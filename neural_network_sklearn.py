import utilities as ut
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

import numpy as np

def run_one(X_train, y_train, X_test):
    #hidden_layer_sizes=(4) => 0.773686
    #hidden_layer_sizes=(), adamsolver => 0.774135
    # hidden_layer_sizes=(4), adamsolver => 0.7735
    # hidden_layer_sizes=(3), adamsolver => 0.7724
    # hidden_layer_sizes=(6), adamsolver => 0.7717
    mlp = MLPClassifier(hidden_layer_sizes=(), solver='adam',learning_rate_init=0.01,max_iter=500)

    mlp.fit(X_train, y_train)

    print(mlp.score(X_train,y_train))
    score = np.mean(cross_val_score(mlp, X_train, y_train, cv=5, scoring='accuracy'))
    print('Cross val score: ', score)

    ut.write_output_file(mlp.predict(X_test), file_name='mlp.csv')

def grid_search(X_train, y_train, X_test):
    parameters = {
     'hidden_layer_sizes': [(3), (4), (5), (6), (10, 2), (5, 2), (4,2), (3,2)],
     'activation': ['relu'],
     'learning_rate_init': [1, 0.1, 0.01, 0.001, 1e-05],
     'max_iter': [500],
     'alpha': [0.1, 0.01, 10],
     'solver': ['adam', 'sgd'],
     'tol': [0.01],
     'learning_rate': ['invscaling', 'constant']
     }
    model = MLPClassifier(verbose=True)
    classifier = GridSearchCV(estimator=model, param_grid=parameters)
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
    #grid_search(X_train, y_train, X_test)


if __name__ == '__main__':
    main()
