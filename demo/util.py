from config import DefaultConfig
from pseudoLabeler import PseudoLabeler


def get_test(**params):
    """
    返回test数据
    :param df:
    :param params:
    :return:
    """
    import pandas as pd

    return pd.read_csv(DefaultConfig.test_path)


def get_train_15p(**params):
    """
    返回train 15p 数据
    :param df:
    :param params:
    :return:
    """
    import pandas as pd

    return pd.read_csv(DefaultConfig.train_15p_path)


def get_train_85p(**params):
    """
    返回train 85p 数据
    :param df:
    :param params:
    :return:
    """

    import pandas as pd

    return pd.read_csv(DefaultConfig.train_85p_path)


def deal_id(df, **params):
    """
    处理id
    :param params:
    :return:
    """
    del df['id']

    return df


def deal_state(df, **params):
    """
    处理state
    """
    state_index = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5,
                   'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10,
                   'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15,
                   'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 20,
                   'U': 21, 'V': 22, 'W': 23, 'X': 24, 'Y': 25,
                   'Z': 26}

    df['state'] = df['state'].apply(
        lambda x: bin(state_index[x[0]])[2:].rjust(6, '0') + bin(state_index[x[1]])[2:].rjust(6, '0'))

    # df['state_1'] = df['state'].apply(lambda x: x[0])
    df['state_2'] = df['state'].apply(lambda x: x[1])
    df['state_3'] = df['state'].apply(lambda x: x[2])
    df['state_4'] = df['state'].apply(lambda x: x[3])
    df['state_5'] = df['state'].apply(lambda x: x[4])
    df['state_6'] = df['state'].apply(lambda x: x[5])
    # df['state_7'] = df['state'].apply(lambda x: x[6])
    df['state_8'] = df['state'].apply(lambda x: x[7])
    df['state_9'] = df['state'].apply(lambda x: x[8])
    df['state_10'] = df['state'].apply(lambda x: x[9])
    df['state_11'] = df['state'].apply(lambda x: x[10])
    df['state_12'] = df['state'].apply(lambda x: x[11])

    del df['state']

    return df


def deal_phone_number(df, **params):
    """
    处理phone_number
    :param df:
    :param params:
    :return:
    """
    from sklearn.preprocessing import LabelEncoder
    df['phone_number_head'] = df['phone_number'].apply(lambda x: x.split('-')[0])
    df['phone_number_tail'] = df['phone_number'].apply(lambda x: x.split('-')[-1])

    df['phone_number'] = df['phone_number'].apply(lambda x: str(x.split('-')[0]) + str(x.split('-')[-1]))
    df['phone_number'] = LabelEncoder().fit_transform(df['phone_number'])

    # del df['phone_number']

    return df


def deal_international_plan(df, **params):
    """
    处理international_plan
    :param df:
    :param params:
    :return:
    """
    from sklearn.preprocessing import LabelBinarizer

    df['international_plan'] = LabelBinarizer().fit_transform(df['international_plan'])

    return df


def deal_voice_mail_plan(df, **params):
    """
    处理voice_mail_plan
    :param df:
    :param params:
    :return:
    """
    from sklearn.preprocessing import LabelBinarizer

    df['voice_mail_plan'] = LabelBinarizer().fit_transform(df['voice_mail_plan'])

    return df


def add_feature_columns(df, **params):
    """
    添加特征
    :param df:
    :param params:
    :return:
    """
    # 白天每次通话时间
    df['total_day_each_time'] = df['total_day_minutes'] / df['total_day_calls']
    # 白天每次通话费用
    df['total_day_each_charge'] = df['total_day_charge'] / df['total_day_calls']

    # 中午每次通话时间
    df['total_eve_each_time'] = df['total_eve_minutes'] / df['total_eve_calls']
    # 中午每次通话费用
    df['total_eve_each_charge'] = df['total_eve_charge'] / df['total_eve_calls']

    # 夜间每次通话时间
    df['total_night_each_time'] = df['total_night_minutes'] / df['total_night_calls']
    # 夜间每次通话费用
    df['total_night_each_charge'] = df['total_night_charge'] / df['total_night_calls']

    # 国际通话每次通话时间
    df['total_intl_each_time'] = df['total_intl_minutes'] / df['total_intl_calls']
    # 国际通话每次通话费用
    df['total_intl_each_charge'] = df['total_intl_charge'] / df['total_intl_calls']

    # 中午通话时间占白天通话时间比例
    df['total_eve_day_ratio_time'] = df['total_eve_minutes'] / df['total_day_minutes']
    # 中午通话时间占白天通话费用
    df['total_eve_day_ratio_charge'] = df['total_eve_charge'] / df['total_day_charge']

    df = df.fillna(0)

    return df


def reduce_mem_usage(df, verbose=True):
    """
    减少内存消耗
    :param df:
    :param verbose:
    :return:
    """
    import numpy as np

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
            start_mem - end_mem) / start_mem))
    return df


def min_max_scaler(df, **params):
    """
    归一化数据
    :param df:
    :param params:
    :return:
    """
    import numpy as np
    columns = list(df.columns)

    for column in ['id', 'Churn', 'area_code', 'phone_number_head', 'phone_number_tail']:
        if column in columns:
            columns.remove(column)

    # 归一化函数
    # df[columns] = df[columns].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

    # log
    df[columns] = df[columns].apply(lambda x: np.log(x + 1))

    return df


def preprocessing(df, type, save=True, **params):
    """
    数据预处理
    :param df:
    :param params:
    :return:
    """
    import os
    import pandas as pd
    from collections import Counter

    if type is 'test' and os.path.exists(DefaultConfig.test_cache_path) and DefaultConfig.no_replace:
        df = reduce_mem_usage(pd.read_hdf(path_or_buf=DefaultConfig.test_cache_path, mode='r', key=type))

    elif type is 'train_15p' and os.path.exists(DefaultConfig.train_15p_cache_path) and DefaultConfig.no_replace:
        df = reduce_mem_usage(pd.read_hdf(path_or_buf=DefaultConfig.train_15p_cache_path, mode='r', key=type))

    elif type is 'train_85p' and os.path.exists(DefaultConfig.train_85p_cache_path) and DefaultConfig.no_replace:
        df = reduce_mem_usage(pd.read_hdf(path_or_buf=DefaultConfig.train_85p_cache_path, mode='r', key=type))

    else:
        # state
        df = deal_state(df)
        # phone_number
        df = deal_phone_number(df)
        # international_plan
        df = deal_international_plan(df)
        # voice_mail_plan
        df = deal_voice_mail_plan(df)
        # add_feature_columns
        add_feature_columns(df)

    if type is 'test' and save and not DefaultConfig.no_replace:
        del df['Churn']
        df = df.astype(float)
        # 归一化效果不好
        df = min_max_scaler(df)
        df.to_hdf(path_or_buf=DefaultConfig.test_cache_path, key=type, mode='w')

    elif type is 'train_15p' and save and not DefaultConfig.no_replace:
        df = df.astype(float)
        # 归一化效果不好
        df = min_max_scaler(df)

        # 剔除Churn属性
        columns = list(df.columns)
        columns.remove('Churn')

        # 填充nan
        df = df.fillna(1e-5)

        print('before: ', Counter(df['Churn']))
        # from imblearn.over_sampling import SMOTE
        # smo = SMOTE(ratio={0: 600, 1: 300}, random_state=42)
        # columns = list(df.columns)
        # columns.remove('Churn')
        # df_X, df_y = smo.fit_sample(df[columns], df['Churn'])

        # 过采样+欠采样
        # from imblearn.combine import SMOTEENN
        # smote_enn = SMOTEENN(ratio={0: 600, 1: 300}, random_state=42)
        # df_X, df_y = smote_enn.fit_sample(df[columns], df['Churn'])

        # 过采样+欠采样
        from imblearn.combine import SMOTETomek
        smote_tomek = SMOTETomek(ratio={0: 600, 1: 300}, random_state=42)
        df_X, df_y = smote_tomek.fit_sample(df[columns], df['Churn'])

        # 重新构造df
        df = pd.DataFrame(data=df_X, columns=columns)
        df['Churn'] = df_y

        print('after: ', Counter(df['Churn']))

        # 转化为int类型
        for column in ['id', 'area_code', 'phone_number_head', 'phone_number_tail']:
            df[column] = df[column].astype(int)

        df.to_hdf(path_or_buf=DefaultConfig.train_15p_cache_path, key=type, mode='w')

    elif type is 'train_85p' and save and not DefaultConfig.no_replace:
        df = df.astype(float)
        # 归一化效果不好
        df = min_max_scaler(df)
        df.to_hdf(path_or_buf=DefaultConfig.train_85p_cache_path, key=type, mode='w')

    return df


def get_85p_churn(train_15p, train_15p_churn, train_85p, **params):
    """
    半监督学习
    :param train_15p:
    :param train_15p_churn:
    :param train_85p:
    :param params:
    :return:
    """

    if DefaultConfig.semi_model == 'pseudo_labeler':
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from catboost import CatBoostClassifier

        model = None
        sample_rate = 0.3
        if DefaultConfig.select_model is 'xgb':
            model = XGBClassifier(nthread=10)
            sample_rate = 0.3

        elif DefaultConfig.select_model is 'lgb':
            model = LGBMClassifier(n_jobs=10)
            sample_rate = 0.3

        elif DefaultConfig.select_model is 'cat':
            model = CatBoostClassifier(thread_count=10)
            sample_rate = 0.3

        models = PseudoLabeler(
            model=model,
            unlabled_data=train_85p,
            features=train_85p.columns,
            target='Churn',
            sample_rate=sample_rate)

        models.fit(train_15p, train_15p_churn)
        train_85p['Churn'] = models.predict(train_85p)

    elif DefaultConfig.semi_model == 'label_spreading':
        from sklearn.semi_supervised import label_propagation

        label_spread = label_propagation.LabelSpreading(kernel='rbf', alpha=0.8, gamma=.25, max_iter=200, n_jobs=10)
        label_spread.fit(train_15p, train_15p_churn)
        train_85p['Churn'] = label_spread.predict(train_85p)

    elif DefaultConfig.semi_model == 'label_propagation':
        from sklearn.semi_supervised import LabelPropagation
        label_propagation = LabelPropagation(kernel='knn', gamma=.25, max_iter=200, n_jobs=10)
        label_propagation.fit(train_15p, train_15p_churn)
        train_85p['Churn'] = label_propagation.predict(train_85p)

    return train_85p


def lgb_model(X_train, y_train, X_test, columns, **params):
    """
    lgb 模型
    :param new_train:
    :param y:
    :param new_test:
    :param columns:
    :param params:
    :return:
    """

    if DefaultConfig.single_model is True:
        import lightgbm as lgb
        import numpy as np
        from sklearn.metrics import f1_score

        def lgb_f1_score(y_hat, data):
            y_true = data.get_label()
            y_bin = [1. if y_cont > y_hat.mean() else 0. for y_cont in y_hat]
            return 'f1', f1_score(y_true, y_bin), True

        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'num_leaves': 63,
            'learning_rate': 0.01,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bagging_seed': 0,
            'bagging_freq': 1,
            'verbose': -1,
            'reg_alpha': 1,
            'reg_lambda': 2,
            'lambda_l1': 0,
            'lambda_l2': 1,
            'num_threads': 10,
        }
        lgb_train = lgb.Dataset(X_train, y_train)
        lgbModel = lgb.train(params=params,
                             train_set=lgb_train,
                             num_boost_round=1000,
                             valid_sets=[lgb_train],
                             valid_names=['train'],
                             feval=lgb_f1_score
                             )

        prediction = lgbModel.predict(X_test, num_iteration=2000)

        return None, prediction, None

    else:
        import gc
        import numpy as np
        from sklearn.model_selection import StratifiedKFold
        import pandas as pd
        import lightgbm as lgb
        from sklearn.metrics import f1_score, log_loss, accuracy_score, roc_auc_score

        def lgb_f1_score(y_hat, data):
            y_true = data.get_label()
            y_hat = np.round(y_hat)
            return 'f1', f1_score(y_true, y_hat), True

        # 线下验证
        oof = np.zeros((X_train.shape[0]))
        # 线上结论
        prediction = np.zeros((X_test.shape[0]))
        seeds = [2255, 223344, 2019 * 2 + 1024, 332232111, 96, 48, 80247, 8, 254, 3434]
        num_model_seed = 10
        print('training')
        feature_importance_df = None
        for model_seed in range(num_model_seed):
            print('模型', model_seed + 1, '开始训练')
            oof_lgb = np.zeros((X_train.shape[0]))
            prediction_lgb = np.zeros((X_test.shape[0]))
            skf = StratifiedKFold(n_splits=5, random_state=seeds[model_seed], shuffle=True)

            for index, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
                print(index)
                train_x, test_x, train_y, test_y = X_train.iloc[train_index], X_train.iloc[test_index], y_train.iloc[
                    train_index], y_train.iloc[test_index]
                train_data = lgb.Dataset(train_x, label=train_y)
                validation_data = lgb.Dataset(test_x, label=test_y)
                gc.collect()
                params = {
                    'learning_rate': 0.01,
                    'boosting_type': 'gbdt',
                    'objective': 'binary',
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'num_leaves': 1000,
                    'verbose': -1,
                    'max_depth': -1,
                    'seed': 42,
                }
                bst = lgb.train(params, train_data, valid_sets=[validation_data], num_boost_round=10000,
                                verbose_eval=1000,
                                early_stopping_rounds=10000,
                                feval=lgb_f1_score)
                oof_lgb[test_index] += bst.predict(test_x)
                prediction_lgb += bst.predict(X_test) / 5
                gc.collect()

            oof += oof_lgb / num_model_seed
            prediction += prediction_lgb / num_model_seed
            print('logloss', log_loss(pd.get_dummies(y_train).values, oof_lgb))
            print('the roc_auc_score for train:', roc_auc_score(y_train, oof_lgb))  # 线下auc评分
        print('logloss', log_loss(pd.get_dummies(y_train).values, oof))
        print('ac', roc_auc_score(y_train, oof))
        return oof, prediction, feature_importance_df


def xgb_model(X_train, y_train, X_test, columns, **params):
    """
    xgb 模型
    :param new_train:
    :param y:
    :param new_test:
    :param columns:
    :param params:
    :return:
    """
    # import numpy as np
    # from sklearn.model_selection import StratifiedKFold
    # import xgboost as xgb
    # from sklearn.metrics import roc_auc_score
    # from sklearn.metrics import f1_score
    #
    # def xgb_f1_score(y_hat, data):
    #     y_true = data.get_label()
    #     y_hat = np.round(y_hat)  # scikits f1 doesn't like probabilities
    #     return 'f1', f1_score(y_true, y_hat), True
    #
    # new_test = new_test.values
    # new_train = new_train.values
    # y = y.values
    #
    # xgb_params = {'booster': 'gbtree',
    #               'eta': 0.01,
    #               'max_depth': 5,
    #               'subsample': 0.8,
    #               'colsample_bytree': 0.8,
    #               'obj': 'binary:logistic',
    #               'silent': True,
    #               }
    # n_splits = 10
    #
    # skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2019)
    # oof_xgb = np.zeros(new_train.shape[0])
    # prediction_xgb = np.zeros(new_test.shape[0])
    # cv_model = []
    # for i, (tr, va) in enumerate(skf.split(new_train, y)):
    #     print('fold:', i + 1, 'training')
    #     dtrain = xgb.DMatrix(new_train[tr], y[tr])
    #     dvalid = xgb.DMatrix(new_train[va], y[va])
    #     watchlist = [(dtrain, 'train'), (dvalid, 'valid_data')]
    #     bst = xgb.train(dtrain=dtrain, num_boost_round=30000, evals=watchlist, early_stopping_rounds=1000,
    #                     verbose_eval=50, params=xgb_params, feval=xgb_f1_score)
    #
    #     cv_model.append(bst)
    #
    #     oof_xgb[va] += bst.predict(xgb.DMatrix(new_train[va]), ntree_limit=bst.best_ntree_limit)
    #     prediction_xgb += bst.predict(xgb.DMatrix(new_test), ntree_limit=bst.best_ntree_limit)
    #
    # print('the roc_auc_score for train:', roc_auc_score(y, oof_xgb))
    # prediction_xgb /= n_splits
    # return oof_xgb, prediction_xgb, cv_model

    import gc
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import f1_score, log_loss, accuracy_score, roc_auc_score

    def xgb_f1_score(y, t):
        t = t.get_label()
        y_bin = [1. if y_cont > 0.5 else 0. for y_cont in y]
        return 'f1', f1_score(t, y_bin)

    # 线下验证
    oof = np.zeros((X_train.shape[0]))
    # 线上结论
    prediction = np.zeros((X_test.shape[0]))
    seeds = [2255, 223344, 2019 * 2 + 1024, 332232111, 96, 48, 80247, 8, 254, 3434]
    num_model_seed = 10
    print('training')
    feature_importance_df = None
    for model_seed in range(num_model_seed):
        print('模型', model_seed + 1, '开始训练')
        oof_xgb = np.zeros((X_train.shape[0]))
        prediction_xgb = np.zeros((X_test.shape[0]))
        skf = StratifiedKFold(n_splits=5, random_state=seeds[model_seed], shuffle=True)

        for index, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
            print(index)
            train_x, test_x, train_y, test_y = X_train.iloc[train_index], X_train.iloc[test_index], y_train.iloc[
                train_index], y_train.iloc[test_index]

            dtrain = xgb.DMatrix(train_x, train_y)
            dvalid = xgb.DMatrix(test_x, test_y)
            watchlist = [(dtrain, 'train'), (dvalid, 'valid_data')]
            gc.collect()
            xgb_params = {
                'booster': 'gbtree',
                'eta': 0.01,
                'max_depth': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'obj': 'binary:logistic',
                'silent': True,
            }
            bst = xgb.train(dtrain=dtrain, num_boost_round=30000, evals=watchlist, early_stopping_rounds=2000,
                            verbose_eval=50, params=xgb_params, feval=xgb_f1_score)
            oof_xgb[test_index] += bst.predict(xgb.DMatrix(test_x), ntree_limit=bst.best_ntree_limit)
            prediction_xgb += bst.predict(xgb.DMatrix(X_test), ntree_limit=bst.best_ntree_limit) / 5
            gc.collect()

        oof += oof_xgb / num_model_seed
        prediction += prediction_xgb / num_model_seed
        print('logloss', log_loss(pd.get_dummies(y_train).values, oof_xgb))
        print('the roc_auc_score for train:', roc_auc_score(y_train, oof_xgb))  # 线下auc评分
    print('logloss', log_loss(pd.get_dummies(y_train).values, oof))
    print('ac', roc_auc_score(y_train, oof))
    return oof, prediction, feature_importance_df


def cat_model(X_train, y_train, X_test, columns, **params):
    """
    catboost_model
    :param X_train:
    :param y_train:
    :param X_test:
    :param columns:
    :param params:
    :return:
    """
    import gc
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    import pandas as pd
    import catboost as cat
    from sklearn.metrics import log_loss, roc_auc_score

    oof = np.zeros((X_train.shape[0]))
    prediction = np.zeros((X_test.shape[0]))
    seeds = [2255, 223344, 2019 * 2 + 1024, 332232111, 96, 48, 80247, 8, 254, 3434]
    num_model_seed = 10
    for model_seed in range(num_model_seed):
        print(model_seed + 1)
        oof_cat = np.zeros((X_train.shape[0]))
        prediction_cat = np.zeros((X_test.shape[0]))
        skf = StratifiedKFold(n_splits=5, random_state=seeds[model_seed], shuffle=True)
        for index, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
            print(index)
            train_x, test_x, train_y, test_y = X_train.iloc[train_index], X_train.iloc[test_index], y_train.iloc[
                train_index], y_train.iloc[test_index]
            gc.collect()
            cbt_model = cat.CatBoostClassifier(iterations=2019, learning_rate=0.01, verbose=300, max_depth=7,
                                               eval_metric='Accuracy',
                                               early_stopping_rounds=2019, task_type='GPU',
                                               loss_function='Logloss')
            cbt_model.fit(train_x, train_y, eval_set=(test_x, test_y))
            oof_cat[test_index] += cbt_model.predict(test_x)
            prediction_cat += cbt_model.predict(X_test) / 5
            gc.collect()
        oof += oof_cat / num_model_seed
        prediction += prediction_cat / num_model_seed
        print('logloss', log_loss(pd.get_dummies(y_train).values, oof_cat))
        print('ac', roc_auc_score(y_train, oof_cat))
    print('logloss', log_loss(pd.get_dummies(y_train).values, oof))
    print('ac', roc_auc_score(y_train, oof))
    return oof, prediction, None


def save_result(model, testdata, prediction, **params):
    """
    保存结果
    :param model:
    :param testdata:
    :param prediction:
    :param params:
    :return:
    """
    import pandas as pd
    # 保存结果：
    sub = pd.DataFrame()
    sub['ID'] = testdata['id']
    sub['Predicted_Results'] = prediction
    # ∪概率大于0.5的置1，否则置0
    sub['Predicted_Results'] = sub['Predicted_Results'].apply(lambda x: x)
    # 模型预测测试集的标签分布
    print('test pre_churn distribution:\n', sub['Predicted_Results'].value_counts())

    if model is 'lgb':
        sub.to_csv(DefaultConfig.lgb_submit_path, index=None)

    elif model is 'xgb':
        sub.to_csv(DefaultConfig.xgb_submit_path, index=None)

    elif model is 'cat':
        sub.to_csv(DefaultConfig.cat_submit_path, index=None)


def model_predict(X_train, y_train, X_test, **params):
    """
    模型预测与结果保存
    :param traindata:
    :param testdata:
    :param label:
    :param params:
    :return:
    """
    import pandas as pd

    if DefaultConfig.select_model is 'lgb':
        print('model is :', DefaultConfig.select_model)
        # 模型训练预测：
        oof_lgb, prediction_lgb, feature_importance_df = lgb_model(X_train.iloc[:, 1:], y_train, X_test.iloc[:, 1:],
                                                                   X_train.iloc[:, 1:].columns)

        # 保存结果
        save_result(DefaultConfig.select_model, X_test, prediction_lgb)

        if feature_importance_df is not None:
            # 保存feature_importance_df
            feature_importance_df.to_hdf(path_or_buf=DefaultConfig.lgb_feature_cache_path, key='lgb')

    elif DefaultConfig.select_model is 'xgb':
        print('model is :', DefaultConfig.select_model)
        # 模型训练预测：
        oof_xgb, prediction_xgb, cv_model = xgb_model(X_train.iloc[:, 1:], y_train, X_test.iloc[:, 1:],
                                                      X_train.iloc[:, 1:].columns)

        # 保存结果
        save_result(DefaultConfig.select_model, X_test, prediction_xgb)

        if cv_model is not None:
            fi = []
            for i in cv_model:
                tmp = {
                    'name': X_train.columns,
                    'score': i.booster().get_fscore()
                }
                fi.append(pd.DataFrame(tmp))

            fi = pd.concat(fi)
            # 保存feature_importance_df
            fi.to_hdf(path_or_buf=DefaultConfig.xgb_feature_cache_path, key='xgb')

    elif DefaultConfig.select_model is 'cat':
        print('model is :', DefaultConfig.select_model)
        # 模型训练预测：
        oof_cat, prediction_cat, feature_importance_df = cat_model(X_train.iloc[:, 1:], y_train, X_test.iloc[:, 1:],
                                                                   X_train.iloc[:, 1:].columns)
        # 保存结果
        save_result(DefaultConfig.select_model, X_test, prediction_cat)


def draw_feature(**params):
    """
    绘制特征重要度
    :param model:
    :param params:
    :return:
    """
    import os
    import pandas as pd
    from matplotlib import pyplot as plt

    if os.path.exists(DefaultConfig.lgb_feature_cache_path) and DefaultConfig.select_model is 'lgb':
        # 读取feature_importance_df
        feature_importance_df = reduce_mem_usage(
            pd.read_hdf(path_or_buf=DefaultConfig.lgb_feature_cache_path, key=DefaultConfig.select_model, mode='r'))

        plt.figure(figsize=(8, 8))
        # 按照flod分组
        group = feature_importance_df.groupby(by=['fold'])

        result = []
        for key, value in group:
            value = value[['feature', 'importance']]

            result.append(value)

        result = pd.concat(result)
        # 5折数据取平均值
        result.groupby(['feature'])['importance'].agg('mean').sort_values(ascending=False).head(40).plot.barh()
        plt.show()

    if os.path.exists(DefaultConfig.xgb_feature_cache_path) and DefaultConfig.select_model is 'xgb':
        # 读取feature_importance_df
        feature_importance_df = reduce_mem_usage(
            pd.read_hdf(path_or_buf=DefaultConfig.xgb_feature_cache_path, key=DefaultConfig.select_model, mode='r'))

        plt.figure(figsize=(8, 8))
        feature_importance_df.groupby(['name'])['score'].agg('mean').sort_values(ascending=False).head(
            40).plot.barh()
        plt.show()


def get_result(**params):
    """

    :param params:
    :return:
    """
    import pandas as pd
    import numpy as np

    if DefaultConfig.select_model is 'lgb':
        lgb_data = pd.read_csv(DefaultConfig.lgb_submit_path)

        print('before: ', lgb_data[lgb_data['Predicted_Results'] >= 0.5].shape)

        lgb_data['Predicted_Results'] = lgb_data['Predicted_Results'].apply(
            lambda x: 1 if x >= np.mean(lgb_data['Predicted_Results'].values) + 0.25 * np.var(
                lgb_data['Predicted_Results'].values) else 0)

        print('after: ', lgb_data[lgb_data['Predicted_Results'] == 1].shape)

        lgb_data.to_csv(DefaultConfig.lgb_submits_mean_path, index=None)

    if DefaultConfig.select_model is 'xgb':
        xgb_data = pd.read_csv(DefaultConfig.xgb_submit_path)

        print('before: ', xgb_data[xgb_data['Predicted_Results'] >= 0.5].shape)

        xgb_data['Predicted_Results'] = xgb_data['Predicted_Results'].apply(
            lambda x: 1 if x >= np.mean(xgb_data['Predicted_Results'].values) + 0.25 * np.std(
                xgb_data['Predicted_Results'].values) else 0)

        print('after: ', xgb_data[xgb_data['Predicted_Results'] == 1].shape)

        xgb_data.to_csv(DefaultConfig.xgb_submit_mean_path, index=None)

    if DefaultConfig.select_model is 'cat':
        cat_data = pd.read_csv(DefaultConfig.cat_submit_path)

        print('before: ', cat_data[cat_data['Predicted_Results'] >= 0.5].shape)

        cat_data['Predicted_Results'] = cat_data['Predicted_Results'].apply(
            lambda x: 1 if x >= np.mean(cat_data['Predicted_Results'].values) + 0.0 * np.var(
                cat_data['Predicted_Results'].values) else 0)

        print('after: ', cat_data[cat_data['Predicted_Results'] == 1].shape)

        cat_data.to_csv(DefaultConfig.cat_submit_mean_path, index=None)

# def merge(**params):
#     """
#     合并
#     :param params:
#     :return:
#     """
#     import pandas as pd
#     lgb = pd.read_csv(DefaultConfig.lgb_submits_mean_path)
#     xgb = pd.read_csv(DefaultConfig.xgb_submit_mean_path)
#     # cat = pd.read_csv(DefaultConfig.cat_submit_mean_path)
#
#     result = []
#     for i in range(lgb.shape[0]):
#         # tmp = lgb.ix[i, 'Predicted_Results'] + xgb.ix[i, 'Predicted_Results'] + cat.ix[i, 'Predicted_Results']
#         tmp = lgb.ix[i, 'Predicted_Results'] + xgb.ix[i, 'Predicted_Results']
#         if tmp >= 1:
#             result.append(1)
#         else:
#             result.append(0)
#
#     lgb['Predicted_Results'] = result
#     lgb.to_csv(DefaultConfig.merge_path, index=None, encoding='utf-8')
#     print(lgb[lgb['Predicted_Results'] == 1].shape)
#
#
# if __name__ == '__main__':
#     merge()
