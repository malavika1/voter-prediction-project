from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
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
        k_val_errors = []
        k_train_errors = []
        for i in range(k):
            bot = i * length
            top = (i + 1) * length
            
            val_x = x[bot:top]
            train_x = np.append(x[:bot], x[top:], axis=0)
            val_y = y[bot:top]
            train_y = np.append(y[:bot], y[top:], axis=0)

            if trial_type == 0:
                clf = AdaBoostClassifier(DecisionTreeClassifier(criterion=crit, min_samples_leaf=n))
            else:  
                clf = AdaBoostClassifier(DecisionTreeClassifier(criterion=crit, max_depth=n))
            clf = clf.fit(train_x, train_y)
            k_val_errors.append(get_avg_error(clf, val_x, val_y))
            k_train_errors.append(get_avg_error(clf, train_x, train_y))

        val_err = np.mean(k_val_errors)
        if val_err < best_err:
            bestn = n
            best_err = val_err
        val_errors.append(val_err)
        train_errors.append(np.mean(k_train_errors))
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
    train_x, train_y, test = ut.import_data()
   
    print 'starting trial'
    b, b_err = Trials((train_x, train_y), range(1, 5, 1), 1)
   
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=b))
    clf = clf.fit(train_x, train_y)
    result = clf.predict(test)
    print clf.score(train_x, train_y)
    ut.write_output_file(result)
