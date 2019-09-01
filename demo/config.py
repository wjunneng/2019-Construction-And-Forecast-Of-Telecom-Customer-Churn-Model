# -*- coding: utf-8 -*-
"""
    配置文件
"""
import os


class DefaultConfig(object):
    """
    参数配置
    """

    def __init__(self):
        pass

    # 项目路径
    project_path = '/'.join(os.path.abspath(__file__).split('/')[:-2])

    # test data path
    test_path = project_path + '/data/original/Test.csv'

    # train 15 p
    train_15p_path = project_path + '/data/original/Train15p.csv'

    # train 85 p
    train_85p_path = project_path + '/data/original/Train85p.csv'

    # cache
    test_cache_path = project_path + '/data/cache/test.h5'
    train_15p_cache_path = project_path + '/data/cache/train_15p.h5'
    train_85p_cache_path = project_path + '/data/cache/train_85p.h5'

    lgb_feature_cache_path = project_path + '/data/cache/lgb_feature.h5'
    xgb_feature_cache_path = project_path + '/data/cache/xgb_feature.h5'
    cat_feature_cache_path = project_path + '/data/cache/cat_feature.h5'

    # submit
    lgb_submit_path = project_path + '/data/submit/lgb_submit_0.5.csv'
    xgb_submit_path = project_path + '/data/submit/xgb_submit_0.5.csv'
    cat_submit_path = project_path + '/data/submit/cat_submit_0.5.csv'

    lgb_submits_mean_path = project_path + '/data/submit/lgb_submit_mean.csv'
    xgb_submit_mean_path = project_path + '/data/submit/xgb_submit_mean.csv'
    cat_submit_mean_path = project_path + '/data/submit/cat_submit_mean.csv'

    # no_replace
    no_replace = True

    # select_model
    select_model = 'lgb'

    # 类别特征
    label_column = ['id', 'account_length', 'area_code', 'state_2', 'state_3', 'state_4', 'state_5', 'state_6',
                    'state_8', 'state_9', 'state_10', 'state_11', 'state_12', 'phone_number_head', 'phone_number_tail',
                    'international_plan', 'voice_mail_plan', 'number_vmail_messages', 'total_intl_calls',
                    'total_night_calls', 'total_eve_calls', 'total_day_calls', ]

    # 连续特征
    outlier_columns = ['total_day_minutes', 'total_day_charge', 'total_eve_minutes', 'total_eve_charge',
                       'total_night_minutes', 'total_night_charge', 'total_intl_minutes', 'total_intl_charge',
                       'total_day_each_time', 'total_day_each_charge', 'total_eve_each_time', 'total_eve_each_charge',
                       'total_night_each_time', 'total_night_each_charge', 'total_intl_each_time',
                       'total_intl_each_charge']

    # semi_model
    semi_model = 'pseudo_labeler'
    # semi_model = 'label_propagation'
    # semi_model = 'label_spreading'

