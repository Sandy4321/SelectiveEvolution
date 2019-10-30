import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9})

import errno
import os

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.utils.multiclass import unique_labels

def read_df(params=None):
    if params['file_type'] == 'csv':
        df = pd.read_csv(params['file_path'])
    if params['file_type'] == 'excel':
        df = pd.read_excel(params['file_path'])
    
    return df

def feature_importance_viz(importance_df=None, directory_name=None):
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    plt.figure(figsize=(10,6))
    plt.title("Feature importances")
    plt.bar(importance_df['variables'], importance_df['importance'],
        color="r")
    plt.xticks(fontsize=9, rotation=90)
    try:
        os.makedirs(directory_name)
    except OSError as exc: 
        if exc.errno == errno.EEXIST and os.path.isdir(directory_name):
            pass
    
    return plt.savefig(
        directory_name+'/feature_importance.png',
        bbox_inches='tight',pad_inches=0)


def confusion_matrix_viz(
    y_true,y_pred,normalize=False,title=None, cmap=plt.cm.Blues, classes=[1,0],
    directory_name=None, threshold=None):
    
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix, threshold : %d'%threshold
        else:
            title = 'Confusion matrix, no normalization, threshold : %d'%threshold
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="red" if cm[i, j] > thresh else "black")

    try:
        os.makedirs(directory_name)
    except OSError as exc: 
        if exc.errno == errno.EEXIST and os.path.isdir(directory_name):
            pass
    
    return fig.savefig(
        directory_name+'/confusion_matrix.png',
        bbox_inches='tight',pad_inches=0)

def round_with_thresh(df=None, column=None, threshold=None):
    series_rounded = np.where(df[column]>=threshold,1,0)
    return pd.Series(series_rounded)

def threshold_optimizer(df, y_true_col, y_pred_proba_col, directory_name=None):
    fpr, tpr, thresholds =roc_curve(df[y_true_col], df[y_pred_proba_col])
    #roc_auc = auc(fpr, tpr)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
    roc.ix[(roc.tf-0).abs().argsort()[:1]]
    optimal_threshold = roc.ix[(roc.tf-0).abs().argsort()[:1]]['thresholds'].values.tolist()

    fig, ax = plt.subplots()
    plt.plot(roc['tpr'])
    plt.plot(roc['1-fpr'], color = 'red')
    plt.xlabel('1-False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    ax.set_xticklabels([])    

    try:
        os.makedirs(directory_name)
    except OSError as exc: 
        if exc.errno == errno.EEXIST and os.path.isdir(directory_name):
            pass
    
    fig.savefig(
        directory_name+'/roc_curve.png',
        bbox_inches='tight',pad_inches=0)

    return optimal_threshold






