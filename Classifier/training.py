# Import the needed libraries
from matplotlib import style
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix

np.random.seed(42)
style.use('fivethirtyeight')


def transformData(X, y):
    reps = [4 if val['1'] == 1 else 1 for val in y.iloc]
    X = X.loc[np.repeat(X.index.values, reps)]
    y = y.loc[np.repeat(y.index.values, reps)]
    return X, y


if __name__ == "__main__":
    type = 'AAC'  # AAC PSE GGAP
    file = ''
    if type == 'AAC':
        file = "../FeatureExtraction/AAC/AAC_data.csv"
    elif type == 'PSE':
        file = "../FeatureExtraction/PseAAC/PseAAC_data.csv"
    else:
        file = "../FeatureExtraction/GGAP/GGAP_data.csv"
    X = pd.read_csv(
        file, index_col=0, header=1)
    y = pd.read_csv("./target_data.csv", index_col=0, header=1)
    X_train, X_test, y_train, y_test = [], [], [], []
    skf = StratifiedKFold(n_splits=10)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        print(X_train.shape, X_test.shape,
              y_train.shape, y_test.values.ravel().shape)
        X_train, y_train = transformData(X_train, y_train)

        print(X_train.shape, X_test.shape,
              y_train.shape, y_test.values.ravel().shape)
        fit_rf = RandomForestClassifier(n_estimators=500)

        fit_rf.fit(X_train, y_train.values.ravel())
        joblib.dump(fit_rf, "./random_forest.joblib")
        y_pred = fit_rf.predict(X_test)

        # tính hiệu năng
        confu_m = confusion_matrix(y_test.values.ravel(), y_pred)
        total = sum(sum(confu_m))

        accuracy = (confu_m[0, 0] + confu_m[1, 1]) / total
        print('Accuracy : ', accuracy)

        sensitivity = confu_m[0, 0] / (confu_m[0, 0] + confu_m[0, 1])
        print('Sensitivity : ', sensitivity)

        specificity = confu_m[1, 1] / (confu_m[1, 0] + confu_m[1, 1])
        print('Specificity : ', specificity)
        print("Matthews correlation coefficient:",
              matthews_corrcoef(y_test.values.ravel(), y_pred))
