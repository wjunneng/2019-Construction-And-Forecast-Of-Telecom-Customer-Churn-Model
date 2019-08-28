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

    merge_path = project_path + '/data/submit/submit_mean.csv'

    # no_replace
    no_replace = False

    # select_model
    select_model = 'lgb'

    # 单模型
    single_model = False

    # semi_model
    semi_model = 'pseudo_labeler'
    # semi_model = 'label_propagation'
    # semi_model = 'label_spreading'

