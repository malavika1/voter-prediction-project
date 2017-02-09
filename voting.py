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

# Get the normalized data
X_train, y_train, X_test = ut.import_data()

clf_linsvc = LinearSVC(C=0.05)
clf_adatree = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2))
clf_forest = RandomForestClassifier(88, max_depth=5)
clf_logreg = LogisticRegression(C=0.03)
clf_mlp = MLPClassifier(hidden_layer_sizes=(5), solver='adam',learning_rate_init=0.01,max_iter=500)

eclf = VotingClassifier(estimators=[('LinearSVC', clf_linsvc), ('Ada Dec Tree', clf_adatree), ('Random Forest', clf_forest), ('Log Reg', clf_logreg), ('Neural Network MLP', clf_mlp)], voting='hard')

eclf.fit(X_train, y_train)
scores = cross_val_score(eclf, X_train, y_train, cv=5, scoring='accuracy')
print "Accuracy: %f" % scores.mean()

ut.write_output_file(eclf.predict(X_test), file_name='voting.csv')
