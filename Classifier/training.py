# Import the needed libraries
from matplotlib import style
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix

np.random.seed(42)
style.use('fivethirtyeight')


def transformData(X, y):
    reps = [4 if val['target'] == 1 else 1 for val in y.iloc]
    X = X.loc[np.repeat(X.index.values, reps)]
    y = y.loc[np.repeat(y.index.values, reps)]
    return X, y


def plot_feature_importance(fi):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(24, 8))
    ax1.plot(np.arange(0, len(fi.index)), fi['importance'])
    label_nrs = np.arange(0, len(fi.index), 5)
    ax1.set_xticks(label_nrs)
    ax1.set_xticklabels(fi['feature'][label_nrs], rotation=90)

    num_bar = min(len(fi.index), 30)
    ax2.barh(np.arange(0, num_bar, ), fi['importance'][:num_bar], align='center', alpha=0.5)
    ax2.set_yticks(np.arange(0, num_bar))
    ax2.set_yticklabels(fi['feature'][:num_bar])

    # fig.show()


def multidimensional_scaling_method(data):
    clf = MDS(n_components=2, random_state=np.random.RandomState(seed=7))
    x_mds = clf.fit_transform(data[['LN', 'SI']].to_numpy()[:507])
    # y = [
    #     [0.003427592116538132, 0.007712082262210797],
    #     [0.003427592116538132, 0.007712082262210797],
    #     [0.013245033112582781, 0.007358351729212656],
    #     [0.012555391432791729, 0.00812407680945347],
    #     [0.00815418828762046, 0.005930318754633061],
    #     [0.0015987210231814548, 0.004796163069544364]
    # ]
    colorize = dict(c=x_mds[:, 0], cmap=plt.cm.get_cmap('rainbow', 7))
    fig, ax = plt.subplots()
    ax.scatter(x_mds[:, 0], x_mds[:, 1], **colorize)
    ax.axis('equal');
    fig.show()


if __name__ == "__main__":
    type = 'GGAP'  # AAC PSE GGAP
    file = ''
    if type == 'AAC':
        file = "../FeatureExtraction/AAC/AAC_data.csv"
    elif type == 'PSE':
        file = "../FeatureExtraction/PseAAC/PseAAC_data.csv"
    else:
        file = "../FeatureExtraction/GGAP/GGAP_data.csv"
    X = pd.read_csv(
        file, index_col=0)
    y = pd.read_csv("./target_data.csv", index_col=0)
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

        fi = pd.DataFrame({'feature': X_train.columns, 'importance': fit_rf.feature_importances_}).sort_values(
            by='importance', ascending=False)
        fi = fi.reset_index()
        plot_feature_importance(fi)

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
    multidimensional_scaling_method(X)
