import numpy as np
import time
import warnings
from sklearn import preprocessing
from sklearn import svm
from sklearn import ensemble
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import tree
from sklearn import discriminant_analysis
from sklearn import cross_validation
from sklearn import feature_selection
from sklearn import metrics
from sklearn import neural_network

warnings.filterwarnings("ignore")

TRAIN_FILE = 'data/train.csv'
TEST_FILE = 'data/test.csv'
OUTPUT_FILE = 'data/op.csv'
OUTPUT_FILE_HEADER = "ImageId,Label\n"
OUTPUT_FILE_CONFIG = 'w'
CLF_INDEX = 6

clfs = [
    (svm.SVC(), 'SVM'),  # 0
    (svm.LinearSVC(), 'Linear SVC'),  # 1
    (ensemble.AdaBoostClassifier(), 'AdaBoostClassifier'),  # 2
    (ensemble.BaggingClassifier(), 'BaggingClassifier'),  # 3
    (ensemble.ExtraTreesClassifier(), 'ExtraTreesClassifier'),  # 4
    (ensemble.GradientBoostingClassifier(), 'GradientBoostingClassifier'),  # 5
    (ensemble.RandomForestClassifier(), 'RandomForestClassifier'),  # 6
    (linear_model.LogisticRegression(), 'LogisticRegression'),  # 7
    (linear_model.RidgeClassifier(), 'RidgeClassifier'),  # 8
    (neighbors.KNeighborsClassifier(), 'KNeighborsClassifier'),  # 9
    (neighbors.NearestCentroid(), 'NearestCentroid'),  # 10
    (naive_bayes.GaussianNB(), 'GaussianNB'),  # 11
    (tree.DecisionTreeClassifier(), 'DecisionTreeClassifier'),  # 12
    (discriminant_analysis.LinearDiscriminantAnalysis(), 'LinearDiscriminantAnalysis'),  # 13
    (discriminant_analysis.QuadraticDiscriminantAnalysis(), 'QuadraticDiscriminantAnalysis'),  # 14
    (linear_model.PassiveAggressiveClassifier(), 'PassiveAggressiveClassifier'),  # 15
    (linear_model.SGDClassifier(), 'SGDClassifier'),  # 16
    (linear_model.Perceptron(), 'Perceptron'),  # 17
]


def write_result(results):
    f = open(OUTPUT_FILE, OUTPUT_FILE_CONFIG)
    f.write(OUTPUT_FILE_HEADER)
    count = 1
    for line in results:
        f.write(str(count) + "," + str(line) + "\n")
        count += 1
    f.close()


def evaluate_classifier(X_train, y_train):
    acc = 0
    index = -1
    for i in range(0, len(clfs + 1)):
        clf, clf_name = clfs[i]
        tic = time.time()
        scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=10)
        if scores.mean() > acc:
            acc = scores.mean()
            index = i
        print '%7.2fs - %20s - Accuracy: %.3f+/-%.3f - Classifier: %s' % (
        time.time() - tic, 'Training', scores.mean(), scores.std(), clf_name)
    return index


def main():
    # Load the training data
    tic = time.time()
    X = np.loadtxt(TRAIN_FILE, delimiter=",", skiprows=1)
    print '%7.2fs - %20s - Examples: %d' % (time.time() - tic, 'Training Set Loaded', len(X))

    # Scale data
    y = X[:, 0]
    X = X[:, 1:]
    X = preprocessing.scale(X)

    # # Select features
    # tic = time.time()
    # fs = ensemble.ExtraTreesClassifier().fit(X, y)
    # # clf = svm.LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
    # model = feature_selection.SelectFromModel(fs, prefit=True)
    # X = model.transform(X)
    # print time.time() - tic, 's\t\t\t- Feature Selection - ', X.shape

    # Training
    # index = evaluate_classifier(X, y)
    clf, clf_name = clfs[0]
    tic = time.time()
    model = clf.fit(X, y)
    print '%7.2fs - %20s - %s' % (time.time() - tic, 'Training Model', clf_name)

    # Testing Load
    tic = time.time()
    test_file = np.loadtxt(TEST_FILE, delimiter=",", skiprows=1)
    print '%7.2fs - %20s - Examples: %d' % (time.time() - tic, 'Testing Set Loaded', len(X))

    # Predicting
    tic = time.time()
    X_test = preprocessing.scale(test_file)
    results = model.predict(X_test)
    print '%7.2fs - %20s' % (time.time() - tic, 'Predicting')

    # Write output
    tic = time.time()
    write_result(results)
    print '%7.2fs - %20s' % (time.time() - tic, 'Finished')


if __name__ == "__main__":
    main()
