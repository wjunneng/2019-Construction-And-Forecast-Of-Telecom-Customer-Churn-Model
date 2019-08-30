import pandas as pd
from util import *
from config import *


def main():
    # """
    # 主函数
    # :return:
    # """
    # import time
    # from collections import Counter
    #
    # start = time.clock()
    #
    # # 加载数据
    # test = get_test()
    # train_15p = get_train_15p()
    # train_85p = get_train_85p()
    # print('\n加载数据 耗时： %s \n' % str(time.clock() - start))
    #
    # # 数据预处理
    # X_test = preprocessing(test, type='test')
    # train_15p = preprocessing(train_15p, type='train_15p')
    # train_85p = preprocessing(train_85p, type='train_85p')
    # print('\n数据预处理 耗时： %s \n' % str(time.clock() - start))
    #
    # # 剔除Churn属性
    # columns = list(train_15p.columns)
    # columns.remove('Churn')
    #
    # # boxcox变换
    # df = statistic_transform(pd.concat([train_15p[columns], train_85p, X_test], axis=0, ignore_index=True))
    # # 替换index
    # df.reset_index(drop=True, inplace=True)
    # # 替换nan
    # df = df.fillna(1e-5)
    #
    # train_15p_len = train_15p.shape[0]
    # train_85p_len = train_85p.shape[0]
    #
    # train_15p[columns] = df.loc[:train_15p_len, :]
    # train_85p = df.loc[train_15p_len:train_15p_len + train_85p_len - 1, :]
    # train_85p.reset_index(drop=True, inplace=True)
    # X_test = df.loc[train_15p_len + train_85p_len:, :]
    # X_test.reset_index(drop=True, inplace=True)
    #
    # print('before: ', Counter(train_15p['Churn']))
    # # from imblearn.over_sampling import SMOTE
    # # smo = SMOTE(ratio={0: 600, 1: 300}, random_state=42)
    # # columns = list(train_15p.columns)
    # # columns.remove('Churn')
    # # train_15p_X, train_15p_y = smo.fit_sample(train_15p[columns], train_15p['Churn'])
    #
    # # 过采样+欠采样
    # # from imblearn.combine import SMOTEENN
    # # smote_enn = SMOTEENN(ratio={0: 600, 1: 300}, random_state=42)
    # # train_15p_X, train_15p_y = smote_enn.fit_sample(train_15p[columns], train_15p['Churn'])
    #
    # # 过采样+欠采样  还未尝试
    # from imblearn.combine import SMOTETomek
    # smote_tomek = SMOTETomek(ratio={0: 600, 1: 300}, random_state=42)
    # train_15p_X, train_15p_y = smote_tomek.fit_sample(train_15p[columns], train_15p['Churn'])
    #
    # # 重新构造train_15p
    # train_15p = pd.DataFrame(data=train_15p_X, columns=columns)
    # train_15p['Churn'] = train_15p_y
    #
    # print('after: ', Counter(train_15p['Churn']))
    #
    # # 半监督学习标签列
    # train_85p = get_85p_churn(train_15p[columns], train_15p['Churn'], train_85p)
    # print('\n半监督学习 耗时： %s \n' % str(time.clock() - start))
    #
    # print('before: ', Counter(train_85p['Churn']))
    #
    # # 过采样+欠采样  还未尝试
    # from imblearn.combine import SMOTETomek
    # smote_tomek = SMOTETomek(ratio={0: 4000, 1: 2000}, random_state=42)
    # train_85p_X, train_85p_y = smote_tomek.fit_sample(train_85p[columns], train_85p['Churn'])
    #
    # # 重新构造train_85p
    # train_85p = pd.DataFrame(data=train_85p_X, columns=columns)
    # train_85p['Churn'] = train_85p_y
    #
    # print('after: ', Counter(train_85p['Churn']))
    #
    # # 合并训练集
    # train = pd.concat([train_15p, train_85p], axis=0, ignore_index=True)
    # # 加权重
    # # train = pd.concat([train_15p, train], axis=0, ignore_index=True)
    # # 划分x,y
    # X_train, y_train = train[columns], train['Churn'].astype(int)
    # print('\n合并训练集 耗时： %s \n' % str(time.clock() - start))
    #
    # for column in DefaultConfig.label_column:
    #     X_train[column] = X_train[column].astype(int)
    #     X_test[column] = X_test[column].astype(int)
    #
    # # 模型预测
    # model_predict(X_train, y_train, X_test)
    # print('\n模型预测 耗时： %s \n' % str(time.clock() - start))
    #
    # # 绘制特征重要图
    # draw_feature()
    # print('\n绘制特征重要图 耗时： %s \n' % str(time.clock() - start))

    # 使用mean 作为标准
    get_result()


if __name__ == '__main__':
    # 主函数
    main()
