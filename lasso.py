import utilities as ut
from sklearn import linear_model

def main():
    '''Runs Lasso linear model, which estimates sparse coefficients'''
    X_train, y_train, X_test = ut.import_data()


    clf = linear_model.Lasso(alpha=0.4)
    clf.fit(X_train, y_train)
    score = clf.score(X_train, y_train)

    print('Score: ')
    print(score)

    ut.write_output_file(clf.predict(X_test), file_name='lasso.csv')

if __name__ == '__main__':
    main()
