import pandas as pd
from util import *


def main():
    # """
    # 主函数
    # :return:
    # """
    # import time
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
    # # 填充nan
    # train_15p = train_15p.fillna(0.00001)
    #
    # # 使用imlbearn库中上采样方法中的SMOTE接口
    # from imblearn.over_sampling import SMOTE
    # # 定义SMOTE模型，random_state相当于随机数种子的作用
    # smo = SMOTE(random_state=42)
    # columns = list(train_15p.columns)
    # columns.remove('Churn')
    # train_15p_X, train_15p_y = smo.fit_sample(train_15p[columns], train_15p['Churn'])
    #
    # # 重新构造train_15p
    # train_15p = pd.DataFrame(data=train_15p_X, columns=columns)
    # train_15p['Churn'] = train_15p_y
    #
    # # 半监督学习标签列
    # train_85p = get_85p_churn(train_15p[columns], train_15p['Churn'], train_85p)
    # print('\n半监督学习 耗时： %s \n' % str(time.clock() - start))
    #
    # # 合并训练集
    # train = pd.concat([train_15p, train_85p], axis=0, ignore_index=True)
    # # 加权重
    # # train = pd.concat([train_15p, train], axis=0, ignore_index=True)
    # # 划分x,y
    # X_train, y_train = train[columns], train['Churn'].astype(int)
    # print('\n合并训练集 耗时： %s \n' % str(time.clock() - start))
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
