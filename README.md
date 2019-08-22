# 2019-Construction-And-Forecast-Of-Telecom-Customer-Churn-Model
2019 电信客户流失模型之建置及预测


01/ 任务
某电信公司希望使用自有业务数据来构造电信客户流失模型，选手以训练数据为基础构建，运用所学习到的数据挖掘知识构建模型，并输出一个测试集预测结果。



02/ 数据
考生将取得训练数据及测试数据。

2.1 训练数据

训练数据包含4,000笔客户资料；每笔客户资料包含21个字段(1个客户ID字段、19个输入字段及1个目标字段-Churn是否流失(1代表流失，0代表未流失)。字段的定义可参考下文。



2.2 测试数据

测试数据包含1,000笔客户资料；字段个数与训练数据相同，唯目标字段的值全部填“Withheld”。



2.3 注意事项

特别要注意的是训练数据中只有600笔在目标字段上有值(Train15p.csv)，其余的3,400笔训练数据(Train85p.csv)并没有目标字段。如何善用这3,400笔没有目标字段的训练数据，提升分类模型的预测效能，将是此比赛的重点。



2.4 数据字段说明

id：客户ID

state：客户地区

account_length：客户往来期间

area_code：邮政编码

phone_number：电话号码

international_plan：开通国际通话

voice_mail_plan：开通语音邮箱

number_vmail_messages：语音邮件数量

total_day_minutes：白天总通话分钟数

total_day_calls：白天总通话通数

total_day_charge：白天总通话费用

total_eve_minutes：中午总通话分钟数

total_eve_calls：中午总通话通数

total_eve_charge：中午总通话费用

total_night_minutes：夜间总通话分钟数

total_night_calls：夜间总通话通数

total_night_charge：夜间总通话费用

total_intl_minutes：国际总通话分钟数

total_intl_calls：国际总通话通数

total_intl_charge： 国际总通话费用

Churn：目标变量(1代表流失用户，0代表未流失用户)



03/ 评分标准
评分方式是以违约户的F-Measure来评估预测结果的好坏。参赛选手的成绩以F-Measure的结果排序，F-Measure越大者越好。

混淆矩阵.jpg

我们通过混淆矩阵（Confusion matrix）得到的结果来计算F-mesure。

f1 下午3.23.18.jpg

其中，

响应率 = 模型预测响应准确的数目/总预测响应数目

查全率 = 模型预测响应准确的数目/总真实响应数目


【参考】

https://github.com/Weenkus/DataWhatNow-Codes/blob/master/pseudo_labeling_a_simple_semi_supervised_learning_method/pseudo_labeling_a_simple_semi_supervised_learning_method.ipynb