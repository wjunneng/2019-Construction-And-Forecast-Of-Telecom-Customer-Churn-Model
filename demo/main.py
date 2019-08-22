import pandas as pd
from demo.util import *


def main():
    """
    主函数
    :return:
    """
    import time

    start = time.clock()

    # 加载数据
    test = get_test()
    train_15p = get_train_15p()
    train_85p = get_train_85p()
    print('\n加载数据 耗时： %s \n' % str(time.clock() - start))

    # 数据预处理
    X_test = preprocessing(test, type='test')
    train_15p = preprocessing(train_15p, type='train_15p')
    train_85p = preprocessing(train_85p, type='train_85p')
    print('\n数据预处理 耗时： %s \n' % str(time.clock() - start))

    # 半监督学习标签列
    train_85p = get_85p_churn(train_15p.iloc[:, :-1], train_15p.iloc[:, -1], train_85p)
    print('\n半监督学习 耗时： %s \n' % str(time.clock() - start))

    # 合并训练集
    train = pd.concat([train_15p, train_85p], axis=0, ignore_index=True)

    # 划分x,y
    X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
    print('\n合并训练集 耗时： %s \n' % str(time.clock() - start))

    # 模型预测
    model_predict(X_train, y_train, X_test.iloc[:, :-1])
    print('\n模型预测 耗时： %s \n' % str(time.clock() - start))

    # 绘制特征重要图
    draw_feature()
    print('\n绘制特征重要图 耗时： %s \n' % str(time.clock() - start))


if __name__ == '__main__':
    # 主函数
    main()
