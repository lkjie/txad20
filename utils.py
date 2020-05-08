#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import numpy as np
import json
import csv
import os
import pickle
import re
import sys
import logging
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score


np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
# logging.getLogger().setLevel(logging.INFO)


def evaluation(y_true, y_pred_prob, threshold=0.5):
    # # eval
    # print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
    # lightgbm
    y_pred = np.where(y_pred_prob > threshold, 1, 0)

    res = precision_score(y_true, y_pred)
    print("precision_score : {}".format(res))
    res = recall_score(y_true, y_pred)
    print("recall_score : {}".format(res))
    res = roc_auc_score(y_true, y_pred_prob)
    print("roc_auc_score : {}".format(res))


def feature_importance(gbm):
    importance = gbm.feature_importance(importance_type='gain')
    names = gbm.feature_name()
    print("-" * 10 + 'feature_importance:')
    no_weight_cols = []
    for name, score in sorted(zip(names, importance), key=lambda x: x[1], reverse=True):
        if score <= 1e-8:
            no_weight_cols.append(name)
        else:
            print('{}: {}'.format(name, score))
    print("no weight columns: {}".format(no_weight_cols))


def base_train(x_train, y_train, x_test, y_test, cate_cols=None, job='classification'):
    # create dataset for lightgbm
    if not cate_cols:
        cate_cols = 'auto'
    lgb_train = lgb.Dataset(x_train, y_train, categorical_feature=cate_cols)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train, categorical_feature=cate_cols)

    if job == 'classification':
        params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 2,
        "use_missing": False,
        "boost_from_average": False,
        "n_jobs": -1
        }
    elif job == 'regression':
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2', 'l1'},
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 2,
            "n_jobs": -1
        }
    else:
        raise Exception("job error!")
    print('Starting training...')

    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=1000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=5)

    # print('Saving model...')
    # gbm.save_model(model_path)
    y_pred_prob = gbm.predict(x_test, num_iteration=gbm.best_iteration)
    if job == 'classification':
        res_auc = roc_auc_score(y_test, y_pred_prob)
        print("AUC: {}".format(res_auc))
        # if res_auc < 0.75:
        #     logging.error("auc too low, maybe some error, please recheck it. AUC过低，可能训练有误，已终止!")
        #     sys.exit(3)
        for i in np.arange(0.1, 1, 0.1):
            print("threshold is {}: ".format(i))
            evaluation(y_test, y_pred_prob, threshold=i)
    elif job == 'regression':
        pass
    feature_importance(gbm)
    return gbm
