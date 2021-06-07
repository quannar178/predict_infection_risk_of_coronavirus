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


def transformData(X, Y):
    X1, Y1 = [], []
    idx = 0
    for i, row in Y.iterrows():
        try:
            X1.append(X.iloc[idx])
            Y1.append(row)
        except ValueError:
            print("error")
        finally:
            idx = idx + 1

    # for i in range(0, len(X1)):
    #     X.add(X1[i])
    #     Y.add(Y1[i])
    for x in X1:
        X.append(pd.DataFrame(x.values, columns=X.columns.values))
    for y in Y1:
        Y.append(pd.DataFrame(y.values, columns=Y.columns.values))
    return X, Y


if __name__ == "__main__":
    X = pd.read_csv(
        "../FeatureExtraction/GGAP/GGAP_data.csv", index_col=0, header=1)
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
