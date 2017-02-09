import numpy as np
from sklearn.svm import LinearSVC
import utilities as ut
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

# Get the normalized data
X_train, y_train, X_test = ut.import_data()

clf_linsvc = LinearSVC(C=0.05)
clf_adatree = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2))
clf_forest = RandomForestClassifier(10, max_depth=15)
clf_logreg = LogisticRegression(C=0.15)
#clf_mlp = MLPClassifier(hidden_layer_sizes=(5), solver='adam',learning_rate_init=0.007,max_iter=500)
parameters = {
 'hidden_layer_sizes': [(3), (4), (5), (6), (5, 2), (4,2), (3,2), (6, 3), (5, 3), (4, 3)],
 'solver': ['adam', 'sgd']
 }
model = MLPClassifier(verbose=True, learning_rate_init=0.07,max_iter=500, solver='adam')
clf_mlp_grid = GridSearchCV(estimator=model,  param_grid=parameters)

eclf = VotingClassifier(estimators=[('LinearSVC', clf_linsvc), ('Ada Dec Tree', clf_adatree), ('Random Forest', clf_forest), ('Log Reg', clf_logreg), ('Neural Network MLP Grid', clf_mlp_grid)], voting='hard')

eclf.fit(X_train, y_train)
scores = cross_val_score(eclf, X_train, y_train, cv=5, scoring='accuracy')
print "Accuracy: %f" % scores.mean()

ut.write_output_file(eclf.predict(X_test), file_name='voting.csv')
