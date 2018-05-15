import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def norm_feature(df,feature_name):
    for name in feature_name:
        df[name] = df[name].map(lambda x:(x-df[name].min())/(df[name].max()-df[name].min()))


#
def split_train_test(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test





if __name__ == '__main__':
    pass
