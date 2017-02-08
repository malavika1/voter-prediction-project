from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import utilities as ut
''' Finds the average classification error from a decision tree model using 
    validation

'''
def get_avg_error(clf, points, y):
    predictions = clf.predict(points)
    error = sum(y != predictions)
    return error / float(len(points))
    

''' Experiments with different parameters to minimize error on the voter data

    Tries two stops of early stopping at the range of ns specified.
    Type 0 refers to min sample leaves = n and type 1 is max_depth = n

    uses k fold cross verification
    Plots the results.

    returns the best n and what the error was for that n

    '''
def Trials(data, ns, trial_type=0, crit='gini', k=5):
    x, y = data
    val_errors = []
    train_errors = []
    length = len(x) / k
    bestn = None
    best_err = np.inf
    for n in ns:
        print n
        if trial_type == 0:
            clf = DecisionTreeClassifier(criterion=crit, min_samples_leaf=n)
        else:  
            clf = DecisionTreeClassifier(criterion=crit, max_depth=n)
        clf = clf.fit(train_x, train_y)

        train_errors.append(clf.score(x, y))
        val_err = np.mean(cross_val_score(clf, x, y_y, cv=5, scoring='accuracy'))
        val_errors.append(val_err)
       
        if val_err < best_err:
            bestn = n
            best_err = val_err

    print 100 * (1 - min(val_errors))
    plt.figure(trial_type)
    if type == 0:
        title = "Error vs n for Min Leaf Samples Stopping Criterion " + \
            " (impurity measure " + crit + ")"
    else:
        title = "Error vs n Max Depth Stopping Criterion " + \
            " (impurity measure " + crit + ")"
    plt.title(title)
    plt.xlabel("n")
    plt.ylabel("Error")    
    plt.plot(ns, train_errors, label="Training Error")
    plt.plot(ns, val_errors, label = "Validation Error")
    plt.legend(loc="best")
    plt.show()
    return bestn, best_err
   

''' try some ranges of parameters with both models and then train a model with
    whatever was best

    '''
if __name__ == '__main__':
    train_x, train_y = ut.import_train_data()
    test = ut.import_test_data()
   
    print 'starting trial 1'
    a, a_err = Trials((train_x, train_y), range(40, 300, 10))
    print 'starting trial 2'
    b, b_err = Trials((train_x, train_y), range(1, 19, 1), 1)
   

    if b_err < a_err:
        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=b))
        clf = clf.fit(train_x, train_y)
        result = clf.predict(test)
        print clf.score(train_x, train_y)
        ut.write_output_file(result)
    else:
        clf = AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf=a))
        clf = clf.fit(train_x, train_y)
        result = clf.predict(test)
        print clf.score(train_x, train_y)
        ut.write_output_file(result)
