import numpy as np
from sklearn.linear_model import LogisticRegression
import utilities as ut
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# Get the normalized data
X_train, y_train, X_test = ut.import_data()

C_vals = np.arange(0.01, 0.2, 0.02)
scores = []
best_score = 0
best_C = 0
for C in C_vals:
    clf = LogisticRegression(C=C)
    clf.fit(X_train, y_train)

    score = np.mean(cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy'))
    scores.append(score)
    print C, score

    if score > best_score:
        best_score = score
        best_C = C

plt.plot(C_vals, scores)
plt.title('Logistic Regression Accuracy vs. Regularization')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.savefig('logreg.png')
plt.show()

# Predictions
clf = LogisticRegression(C=best_C)
clf.fit(X_train, y_train)
ut.write_output_file(clf.predict(X_test), file_name='logreg.csv')