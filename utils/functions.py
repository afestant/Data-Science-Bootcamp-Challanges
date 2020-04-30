import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, plot_confusion_matrix, classification_report


def plot_numeric_distrib_sns(data, outcome, ncols, nrows, figsize):
    vars_ = data.columns
    c = 0
    plt.figure(figsize=figsize)
    for v in vars_:
        ax = plt.subplot(nrows, ncols, c + 1)
        sns.distplot(data[v], hue=outcome)
        plt.xlabel(v)
        plt.legend(loc='best')
        c = c + 1
    plt.show()
    #plt.savefig('plots/variables_distributions.png')

def plot_numeric_distrib(data, target, target_labels, ncols, nrows, figsize):
    color = ['b', 'r']
    vars_ = data.columns
    c = 0
    plt.figure(figsize=figsize)
    for v in vars_:
        ax = plt.subplot(nrows, ncols, c + 1)
        df = data[data[target] == target_labels[0]]
        _, nbins, _ = plt.hist(df[v], 20, color='g', alpha=0.6, label=target_labels[0])
        cc = 0
        for lbl in target_labels[1:]:
            df = data[data[target] == lbl]
            plt.hist(df[v], bins=nbins, color=color[cc], alpha=0.4, label=lbl)
            cc = cc+1
        plt.xlabel(v)
        plt.legend(loc='best')
        c = c + 1
    plt.show()


def plot_categorical_features(data, categorical_features):
    counter = 0
    for cat in categorical_features:
        # ax = plt.subplot(2, 1+len(categorical_features)/2, counter+1)
        #     df = data.groupby([cat, 'ATTACK SUCCESS']).size().groupby(
        #         level=0).apply(lambda x: 100 * x / x.sum()).unstack()
        df = data.groupby([cat]).size().unstack()
        df.plot(kind='bar')
        counter = counter + 1

def plot_categorical_features_classification(data, categorical_features, class_var, rel_perc):
    counter = 0
    for cat in categorical_features:
        # ax = plt.subplot(2, 1+len(categorical_features)/2, counter+1)
        df = data.groupby([cat, class_var]).size()
        if rel_perc==True:
            df = df.groupby(level=0).apply(lambda x: 100 * x / x.sum()).unstack()
        else:
            df = df.unstack()
        df.plot(kind='bar', stacked=True)
        counter = counter + 1


def correlation_heatmap(df):
    corr = df.corr()
    _, ax = plt.subplots(figsize=(30, 30))
    sns.heatmap(corr, annot=True, fmt='.1f', linewidths=.9, cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax)
    plt.show()


def plot_conf_matrix(classifier, X_test, y_test):
    disp = metrics.plot_confusion_matrix(classifier, X_test, y_test,
                                         cmap=plt.cm.Blues,
                                         normalize='true')
    plt.show()


def plot_precision_recall(classifier, X_test, y_test):
    disp = metrics.plot_precision_recall_curve(classifier, X_test, y_test)
    plt.show()


def plot_precision_recall_vs_thrs(classifier, X_test, y_test):
    predicted_prob = classifier.predict_proba(X_test)[:, 1]
    #     print((np.unique(predicted_prob)))
    precisions, recalls, thresholds = precision_recall_curve(y_test, predicted_prob)
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    #     print(thresholds)
    plt.legend(loc='best')
    plt.xlabel('Threshold')
    plt.ylim((0, 1.1))
    plt.show()


def plot_ROC(classifier, X_test, y_test):
    metrics.plot_roc_curve(classifier, X_test, y_test)  # doctest: +SKIP
    plt.show()


def print_scores(y_pred, y):
    print(f'f1 score = {metrics.f1_score(y, y_pred)}')
    print(f'accuracy = {metrics.accuracy_score(y, y_pred)}')
    print(f'precision = {metrics.precision_score(y, y_pred)}')
    print(f'recall = {metrics.recall_score(y, y_pred)}')
    # print(f'AUC = {metrics.roc_auc_score(y, y_pred, average=average, labels=labels)}')


def plot_feature_importance(classifier, X_train):
    importances = pd.DataFrame({'feature': X_train.columns, 'importance': np.round(classifier.feature_importances_, 4)})
    importances = importances.sort_values(by='importance', ascending=False).set_index('feature')
    importances.plot.barh()
    plt.title('Feature Importance')
    plt.xlabel('Feature Importance Score')
    plt.show()


def plot_feature_importance2(classifier, features):
    importances = pd.DataFrame({'feature': features, 'importance': np.round(classifier.feature_importances_, 4)})
    importances = importances.sort_values(by='importance', ascending=False).set_index('feature')
    importances.plot.barh()
    plt.title('Feature Importance')
    plt.xlabel('Feature Importance Score')
    plt.show()
