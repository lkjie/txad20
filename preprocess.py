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
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from .utils import base_train


np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
logging.getLogger().setLevel(logging.INFO)


'''
（1）原始onehot特征，比如aid，age，gender等。
（2）向量特征，比如interest1，interest2，topic1，kw1等
（3）向量长度统计特征：interest1，interest2，interest5的长度统计。
（4）uid类的统计特征，uid的出现次数，uid的正样本次数，以及uid与ad特征的组合出现次数，组合正样本次数。
（5）uid的序列特征，比如uid=1时，总共出现了5次，序列为[-1,1,-1,-1,-1]，
则第一次出现时，特征为【】
第二次出现时，特征为【-1】
第三次出现时，特征为【-1，1】
第四次出现时，特征为【-1，1，-1】
第五次出现时，特征为【-1，1，-1，-1】
（6）组合特征：age与aid的组合，gender与aid的组合，interest1与aid的组合，interest2与aid的组合，topic1与topic2的组合，LBS与kw1的组合。


统计特征：
user纬度：点击广告总次数，creative_id个数，

'''


def agg_features(df_click_log, groupby_cols, stat_col, aggfunc):
    if type(groupby_cols) == str:
        groupby_cols = [groupby_cols]
    data = df_click_log[groupby_cols + [stat_col]]
    if aggfunc == "size":
        tmp = pd.DataFrame(data.groupby(groupby_cols).size()).reset_index()
    elif aggfunc == "count":
        tmp = pd.DataFrame(data.groupby(groupby_cols)[stat_col].count()).reset_index()
    elif aggfunc == "mean":
        tmp = pd.DataFrame(data.groupby(groupby_cols)[stat_col].mean()).reset_index()
    elif aggfunc == "unique":
        tmp = pd.DataFrame(data.groupby(groupby_cols)[stat_col].nunique()).reset_index()
    elif aggfunc == "max":
        tmp = pd.DataFrame(data.groupby(groupby_cols)[stat_col].max()).reset_index()
    elif aggfunc == "min":
        tmp = pd.DataFrame(data.groupby(groupby_cols)[stat_col].min()).reset_index()
    elif aggfunc == "sum":
        tmp = pd.DataFrame(data.groupby(groupby_cols)[stat_col].sum()).reset_index()
    elif aggfunc == "std":
        tmp = pd.DataFrame(data.groupby(groupby_cols)[stat_col].std()).reset_index()
    elif aggfunc == "median":
        tmp = pd.DataFrame(data.groupby(groupby_cols)[stat_col].median()).reset_index()
    elif aggfunc == "skew":
        tmp = pd.DataFrame(data.groupby(groupby_cols)[stat_col].skew()).reset_index()
    elif aggfunc == "unique_mean":
        group = data.groupby(groupby_cols)
        group = group.apply(lambda x: np.mean(list(Counter(list(x[stat_col])).values())))
        tmp = pd.DataFrame(group.reset_index())
    elif aggfunc == "unique_var":
        group = data.groupby(groupby_cols)
        group = group.apply(lambda x: np.var(list(Counter(list(x[stat_col])).values())))
        tmp = pd.DataFrame(group.reset_index())
    else:
        raise Exception("aggfunc error")
    feat_name = '_'.join(groupby_cols) + "_" + stat_col + "_" + aggfunc
    tmp.columns = groupby_cols + [feat_name]
    print(feat_name)
    return tmp
    # try:
    #     del df_click_log[feat_name]
    # except:
    #     pass
    # df_click_log = df_click_log.merge(tmp, how='left', on=groupby_cols)
    # return df_click_log


def get_features(df_click_log):
    paras = [
        (['user_id'], 'creative_id', 'unique'),
        (['user_id'], 'creative_id', 'count'),
        (['user_id'], 'ad_id', 'unique'),
        (['user_id'], 'ad_id', 'count'),
        (['user_id'], 'product_id', 'unique'),
        (['user_id'], 'product_id', 'count'),
        (['user_id'], 'product_category', 'unique'),
        (['user_id'], 'product_category', 'count'),
        (['user_id'], 'advertiser_id', 'unique'),
        (['user_id'], 'advertiser_id', 'count'),
        (['user_id'], 'industry', 'unique'),
        (['user_id'], 'industry', 'count')
    ]
    df_tmp = pd.DataFrame()
    for groupby_cols, stat_col, aggfunc in paras:
        tmp = agg_features(df_click_log, groupby_cols, stat_col, aggfunc)
        df_tmp = df_tmp.merge(tmp, how='left', on='user_id') if not df_tmp.empty else tmp
    return df_tmp


def main(args):
    txdir = "/Users/l/txad20/"
    df_ad = pd.read_csv(txdir + "train_preliminary/ad.csv")
    df_click_log = pd.read_csv(txdir + "train_preliminary/click_log.csv")
    df_user = pd.read_csv(txdir + "train_preliminary/user.csv")

    df_ad.loc[df_ad['product_id'] == '\\N', 'product_id'] = 0
    df_ad.loc[df_ad['industry'] == '\\N', 'industry'] = 0
    df_user['gender'] = df_user['gender'] - 1

    df_train, df_dev = train_test_split(df_user, test_size=0.2, random_state=2020)

    df_feat = get_features(df_click_log)

    df_train = df_train.merge(df_feat, how='left', on='user_id')
    df_dev = df_dev.merge(df_feat, how='left', on='user_id')
    y_train = df_train['gender']
    x_train = df_train.drop(['gender', 'age', 'user_id'], axis=1)
    y_dev = df_dev['gender']
    x_dev = df_dev.drop(['gender', 'age', 'user_id'], axis=1)

    gbm_gender = base_train(x_train, y_train, x_dev, y_dev, job='classification')

    y_train = df_train['age']
    x_train = df_train.drop(['gender', 'age', 'user_id'], axis=1)
    y_dev = df_dev['age']
    x_dev = df_dev.drop(['gender', 'age', 'user_id'], axis=1)
    gbm_age = base_train(x_train, y_train, x_dev, y_dev, job='regression')

    # 预测
    df_ad_test = pd.read_csv(txdir + "test/ad.csv")
    df_click_log_test = pd.read_csv(txdir + "test/click_log.csv")
    df_ad_test.loc[df_ad_test['product_id'] == '\\N', 'product_id'] = 0
    df_ad_test.loc[df_ad_test['industry'] == '\\N', 'industry'] = 0
    df_click_log_test = df_click_log_test.merge(df_ad_test, how='left')

    df_feat_test = get_features(df_click_log_test)
    df_res = df_feat_test[['user_id']]
    df_test = df_feat_test.drop(['user_id'], axis=1)
    df_res['predicted_age'] = gbm_age.predict(df_test)
    df_res['predicted_gender'] = gbm_gender.predict(df_test)
    df_res.loc[df_res['predicted_gender'] >= 0.5, 'predicted_gender'] = 2
    df_res.loc[df_res['predicted_gender'] < 0.5, 'predicted_gender'] = 1
    df_res.to_csv("data/submission.csv", index=False)


if __name__ == '__main__':
    args = None
    main(args)
