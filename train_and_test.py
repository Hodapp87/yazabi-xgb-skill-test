#!/usr/bin/env python

###########################################################################
# train_and_test.py: Training & validation for xgboost mini-project
# Author: Chris Hodapp (hodapp87@gmail.com)
# Date: 2017-11-09
###########################################################################

import pandas as pd
import sklearn.preprocessing

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
    X = df[cols]
    # Label is 1 for malignant, 0 for benign:
    y = (df.diag == "M") * 1
    return (X, y)
