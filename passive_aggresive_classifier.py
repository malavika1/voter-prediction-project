import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier
import utilities as ut
import matplotlib.pyplot as plt

# Get the normalized data
X_train, y_train, X_test = ut.import_data()

C_vals = np.arange(0.15, 7, 0.15)
loss_scores = []
squared_loss_scores = []
best_score = 0
best_C = 0
best_type = None
for C in C_vals:
    clf = PassiveAggressiveClassifier(C=C, loss='hinge')
    clf.fit(X_train, y_train)

    score = clf.score(X_train, y_train)
    loss_scores.append(score)

    if score > best_score:
        best_score = score
        best_C = C
        best_type = 'hinge'

    print C, score, "hinge loss"

    clf = PassiveAggressiveClassifier(C=C, loss= 'squared_hinge')
    clf.fit(X_train, y_train)

    score = clf.score(X_train, y_train)
    squared_loss_scores.append(score)

    print C, score, "squared_hinge loss"

    if score > best_score:
        best_score = score
        best_C = C
        best_type = 'squared_hinge'

plt.plot(C_vals, loss_scores, label="hinge")
plt.plot(C_vals, squared_loss_scores, label="squared")
plt.title('Passive Aggressive classifier Accuracy vs. Regularization')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.savefig('graphs/Kavya.png')

plt.show()
print best_score
# Predictions
clf = PassiveAggressiveClassifier(C=best_C, loss=best_type)
clf.fit(X_train, y_train)
ut.write_output_file(clf.predict(X_test), file_name='passive_aggressive.csv')