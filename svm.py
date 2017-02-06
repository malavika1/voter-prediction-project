import numpy as np
from sklearn.svm import SVC
import utilities as ut

# Get the normalized data
X_train, y_train, X_test = ut.import_data()

# Fit the data
clf = SVC()
clf.fit(X_train, y_train)

print clf.score(X_train, y_train)

# Predictions
ut.write_output_file(clf.predict(X_test), file_name='svm.csv')