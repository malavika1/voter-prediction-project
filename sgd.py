import utilities as ut
from sklearn.linear_model import SGDClassifier



# Stochastic gradient descent
# and k fold cross validation?


def main():
    X_train, y_train, X_test = ut.import_data()

    # Possible loss parameters: 'hinge', 'modified_huber', 'log'
    # Penalty parameter: 'l2', 'l1', 'elasticnet'
    clf = SGDClassifier(loss='hinge', penalty='l2')
    clf.fit(X_train, y_train)

    score = clf.score(X_train, y_train)
    print('Score:')
    print(score)

    ut.write_output_file(clf.predict(X_test), file_name='sgd.csv')

if __name__ == '__main__':
    main()
