from selective_evolution import SelectiveEvolution

from utils import (
    read_df, feature_importance_viz, confusion_matrix_viz, round_with_thresh,
    threshold_optimizer
)

import argparse
import pandas as pd
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.utils.multiclass import unique_labels

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path")
    parser.add_argument("-encoding", default="utf_8")
    options = parser.parse_args()

    for _, value in parser.parse_args()._get_kwargs():
        if value is not None:
            if 'json' in value:
                config_file_path = value

    with open(config_file_path, "r") as f: config_dict = json.load(f)

    df = read_df(config_dict)
    predictive_algorithm = config_dict['predictive_algorithm']

    genetics = SelectiveEvolution()

    param_grid = config_dict['model_params']

    for key in param_grid.keys():
        for param in param_grid[key]:
            try:
                if "None" in param:
                    param_grid[key].remove(param)
                    param_grid[key].append(None)
            except TypeError:
                continue

    list_vars, importance_df, predicted_vs_true = genetics.genetic_algo(
        df=df, list_inputs=df.columns.tolist()[1:-1], 
        predictive_algorithm=predictive_algorithm,
        param_grid=param_grid,
        directory_name='./%s/best_params'%predictive_algorithm
    )

    feature_importance = feature_importance_viz(
        importance_df=importance_df,
        directory_name='./%s/feature_importance'%predictive_algorithm
    )

    best_threshold = threshold_optimizer(
        df=predicted_vs_true, y_true_col='true', y_pred_proba_col='predictions',
        directory_name='./%s/roc_curve'%predictive_algorithm
    )[0]

    confusion_matrix = confusion_matrix_viz(
        y_true=predicted_vs_true['true'], 
        y_pred=round_with_thresh(df=predicted_vs_true, column='predictions', threshold=best_threshold),
        directory_name='./%s/confusion_matrix'%predictive_algorithm, threshold=best_threshold
    )
