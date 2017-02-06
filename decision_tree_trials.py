from sklearn import tree
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
    

''' Experiments with different parameters for number of samples required to
    be in the leaf on the decision tree.
    Plots the results.

    '''
def Min_Leaf_Trials(data, crit='gini'):
    train_x, train_y = data
    cutoff = 4 * len(train_x) / 5
    val_x = train_x[:cutoff]
    val_y = train_y[:cutoff]

    train_x = train_x[cutoff:]
    train_y = train_y[cutoff:]

    val_errors = []
    train_errors = []
    ns = range(1, 40, 4) # can be changed later
    for n in ns:
        print n
        clf = tree.DecisionTreeClassifier(criterion=crit, min_samples_leaf=n)
        clf = clf.fit(val_x, val_y)
        val_errors.append(get_avg_error(clf, val_x, val_y))
        train_errors.append(get_avg_error(clf, train_x, train_y))
    print min(val_errors)
    plt.figure(1)
    title = "Error vs Min Leaf Samples Stopping Criteria " + " (impurity measure " + crit + ")"
    plt.title(title)
    plt.xlabel("n")
    plt.ylabel("Error")    
    plt.plot(ns, train_errors, label="Training Error")
    plt.plot(ns, val_errors, label = "Validation Error")
    plt.legend(loc="best")
    plt.show()

''' Experiments with different parameters for minimum depth required for the 
    decision tree.
    Plots the results.

    '''
def Min_Depth_trials(data, crit='gini'):
    train_x, train_y = data
    cutoff = 4 * len(train_x) / 5
    val_x = train_x[:cutoff]
    val_y = train_y[:cutoff]

    train_x = train_x[cutoff:]
    train_y = train_y[cutoff:]
    val_errors = []
    train_errors = []
    ns = range(3, 45, 3)
    for n in ns:
        print n
        clf = tree.DecisionTreeClassifier(criterion=crit, max_depth=n)
        clf = clf.fit(val_x, val_y)
        val_errors.append(get_avg_error(clf, val_x, val_y))
        train_errors.append(get_avg_error(clf, train_x, train_y))
    print min(val_errors)
    plt.figure(2)
    title = "Error vs Tree Depth Stopping Criteria" + " (impurity measure " + crit + ")"
    plt.title(title)
    plt.xlabel("n")
    plt.ylabel("Error")
    plt.plot(ns, train_errors, label="Training Error")
    plt.plot(ns, val_errors, label = "Validation Error")
    plt.legend(loc="best")
    plt.show()    
   
if __name__ == '__main__':
    data = ut.import_train_data()
    print 'starting trial 1'
    Min_Leaf_Trials(data)
    print 'starting trial 2'
    Min_Depth_trials(data)
    '''
    print 'starting trial 3'
    Min_Leaf_Trials(data, 'entropy')
    print 'starting trial 4'
    Min_Depth_trials(data, 'entropy')
    '''