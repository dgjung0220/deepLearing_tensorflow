# Day_04_04_preprocessing.py
import numpy as np
from sklearn import preprocessing


def add_dummy_feature():
    x = [[0, 1],
         [2, 3]]

    print(preprocessing.add_dummy_feature(x))


def Binarizer():
    x = [[1., -1., 2],
         [2., 0., 0.],
         [0., 1., -1.]]

    bin = preprocessing.Binarizer()
    print(bin)

    bin = bin.fit(x)
    print(bin)

    bin = preprocessing.Binarizer().fit(x)
    print(bin.transform(x))

    print(preprocessing.Binarizer(threshold=1.5).fit_transform(x))


def nan():
    import pandas as pd
    from io import StringIO

    text = '''a,b,c,d
    1,,3,4
    5,,7,8
    9,10,11,
    13,14,15,16'''

    df = pd.read_csv(StringIO(text))
    print(df)

    print(df.dropna())


def Imputer():
    # ["mean", "median", "most_frequent"]
    imp = preprocessing.Imputer()

    # 4 = (1 + 7) / 2
    # 5 = (2 + 4 + 9) / 3
    x = [[1, 2],
         [np.nan, 4],
         [7, 9]]

    imp.fit(x)
    print(imp.transform(x))

    print(imp.missing_values)
    print(imp.statistics_)
    print(imp.strategy)

    y = [[np.nan, 2],
         [6, np.nan],
         [7, 9]]

    print(imp.transform(y))


def LabelBinarizer():
    x = [1, 2, 6, 2, 4]

    lb = preprocessing.LabelBinarizer().fit(x)
    print(lb.transform(x))

    print(lb.classes_)

    y = ['yes', 'no', 'cancel', 'no']

    lb = preprocessing.LabelBinarizer().fit(y)
    print(lb.transform(y))

    print(lb.classes_)

    lb = preprocessing.LabelBinarizer(sparse_output=True).fit(y)
    print(lb.transform(y))


def LabelEncoder():
    x = [1, 2, 6, 2, 4]

    lb = preprocessing.LabelEncoder().fit(x)
    print(lb.transform(x))

    print(lb.classes_)

    y = ['yes', 'no', 'cancel', 'no']

    lb = preprocessing.LabelEncoder().fit(y)
    print(lb.transform(y))

    print(lb.classes_)


def MaxAbsScaler():
    x = [[1, -1, 5],
         [2, 0, -5],
         [0, 1, -10]]

    scaler = preprocessing.MaxAbsScaler()
    print(scaler.fit_transform(x))

    print(scaler.scale_)
    print(x / scaler.scale_)


# add_dummy_feature()
# Binarizer()
# nan()
# Imputer()
# LabelBinarizer()
# LabelEncoder()
# MaxAbsScaler()

