#!/usr/bin/env python

###########################################################################
# train_and_test.py: Training & validation for xgboost mini-project
# Author: Chris Hodapp (hodapp87@gmail.com)
# Date: 2017-11-09
###########################################################################

import pandas as pd
import sklearn.model_selection
import sklearn.preprocessing
import xgboost

def get_processed_data():
    """Returns training & testing data, preprocessed, standardized, and
    all categorical variables encoded.  Specifically, this returns
    (train_X, train_y, test_X, test_y), where 'train_X' is a DataFrame
    containing features for the training data, 'train_y' is a Series
    with the corresponding labels, and likewise for 'test_X' and
    'test_y' for testing data.
    """
    # Data source:
    # https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names
    train_raw = read_data("data/train_data.txt")
    test_raw = read_data("data/test_data.txt")
    train_X, train_y = feature_xform(train_raw)
    test_X,  test_y  = feature_xform(test_raw)
    standardize(train_X, test_X)
    
    return (train_X, train_y, test_X, test_y)

def read_data(filename):
    """Loads raw data for this project, given a filename; returns
    a Pandas DataFrame."""
    # Data dictionary:
    # https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names
    num_cols = ["radius", "texture", "perimeter", "area",
                "smoothness", "compactness", "concavity", "concave_points",
                "symmetry", "fractal_dim"]
    # Note from the data description that these 10 numerical columns
    # are given as: mean, standard error, and worst/largest.  Add a
    # _se suffix for standard error, and add a _w suffix for worst.
    columns = ["ID", "diag"] + \
              num_cols + \
              [c + "_std" for c in num_cols] + \
              [c + "_w" for c in num_cols]
    # Data is CSV with no header:
    return pd.read_csv(filename, names=columns, index_col=False)

def feature_xform(df):
    """Given raw input data, selects and transforms some features, and
    returns (X, y) where 'X' is a DataFrame with all numerical
    features and 'y' is a Series containing the corresponding labels
    (0 = benign, 1 = malignant).
    """
    # Ignore 'diag' (the label) and 'ID' (which tells us nothing):
    ignore = ('diag', 'ID')
    cols = [c for c in df.columns if c not in ignore]
    X = df[cols].copy()
    # Label is 1 for malignant, 0 for benign:
    y = (df.diag == "M") * 1
    return (X, y)

def standardize(train, test):
    """Given training and testing DataFrames ('train' and 'test'),
    standardize numerical columns of 'train' to mean 0 and standard
    deviation 1, and apply the same transform to 'test'.  This
    modifies both input DataFrames.
    """
    ss = sklearn.preprocessing.StandardScaler()
    train.iloc[:, :] = ss.fit_transform(train)
    # Use the same transform on test:
    test.iloc[:, :] = ss.transform(test)

def tune_xgboost(train_X, train_y, filename = None, verbose = 0):
    """Run a grid search in order to tune an xgboost model over 'train_X'
    and corresponding 'train_y'.  If 'filename' is given, then save
    the resultant GridSearchCV instance to this filename (for later
    loading with sklearn.externals.joblib.load).  Optional argument
    'verbose' is passed to GridSearchCV in case verbose output is
    desired.
    """
    params = {
        "max_depth": (3, 4, 5, 6),
        "learning_rate": (0.1, 0.15, 0.20),
        "gamma": (0.0, 0.05, 0.1),
        "min_child_weight": (1,),
        "subsample": (0.8, 0.85, 0.9, 0.95, 1.0),
        "reg_alpha": (0, 0.05, 0.1, 0.15, 0.2),
        "reg_lambda": (1.0, 1.1, 1.2, 1.3),
    }
    xgb = xgboost.XGBClassifier(nthread=-1, seed=1234, n_estimators=150)
    cv = sklearn.model_selection.GridSearchCV(xgb, params, cv=5, verbose=verbose)
    cv.fit(train_X, train_y)
    if filename:
        sklearn.externals.joblib.dump(cv, filename)
    if verbose:
        print("Optimal score:")
        print(cv.best_score_)
        print("Optimal parameters:")
        print(cv.best_params_)
    
if __name__ == '__main__':
    train_X, train_y, test_X, test_y = get_processed_data()
    # Train model over data, tune with grid search, and save to file:
    fname = "xgboost_gridsearch.pkl"
    tune_xgboost(train_X, train_y, fname, verbose=1)

    # Load model from file, and apply to test data:
    model = sklearn.externals.joblib.load(fname)
    test_y_pred = model.predict(test_X)
    test_acc = sklearn.metrics.accuracy_score(test_y, test_y_pred)
    print("Testing accuracy: {0}".format(test_acc))
