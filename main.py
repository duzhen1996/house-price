# coding:utf-8
# 这个文件我们的主要作用就是进行数据的检视，看看这个数据里面是什么东西
# 导入矩阵库用来处理矩阵
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
# 这两个库可以服务并整理数据

# 这里将数据读入
# 我们使用pandas读取csv文件，然后默认将第0列作为索引，一般情况下index是没有什么用的，加不加都可以
train_df = pd.read_csv('input/train.csv', index_col=0)
test_df = pd.read_csv('input/test.csv', index_col=0)

#我们可以使用getValues这个函数将所有的DataFrame数据转化为
#train_df作为DataFrame支持map的取值方式，取出的东西还是DataFrame
allPrice = train_df["SalePrice"]

#print allPrice

#然后我们将所有的房屋价格变成一个数组
allPriceArr = allPrice.get_values()
#print allPriceArr


#这里我们单独生成一个label集，并且将train_df的数据推出，单独存放
y_train = np.log1p(train_df.pop("SalePrice"))


#这里我们进行训练集和测试集的合并，因为训练集已经排除了房屋价格
all_df = pd.concat((train_df,test_df))

# print pd.get_dummies(all_df["MSSubClass"],prefix='MSSubClass').get_values()

#将数字转化为字符串
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)

#进行One-Hot操作
all_noRaw_df = pd.get_dummies(all_df)

# print all_dummpy_df

#计算每一个属性的平均值
mean_cols = all_noRaw_df.mean()

#将平均值作为缺省值插入
all_noRaw_df = all_noRaw_df.fillna(mean_cols)



#获取所有的数值属性
numeric_cols = all_df.columns[all_df.dtypes != 'object']

# print numeric_cols
#这里计算每一个属性的平均值
numeric_col_means = all_noRaw_df.loc[:,numeric_cols].mean()
#这里计算每一个属性的标准差
numeric_col_std = all_noRaw_df.loc[: , numeric_cols].std()

#这里将数据进行标准化
all_noRaw_df.loc[:,numeric_cols] = (all_noRaw_df.loc[:,numeric_cols] - numeric_col_means)/numeric_col_std

# 下面进行模型训练

# 将训练集和测试集重新分回
noRaw_train_df = all_noRaw_df.loc[train_df.index]
noRaw_test_df = all_noRaw_df.loc[test_df.index]


ridge = Ridge(15)

X_train = noRaw_train_df.get_values()
X_test = noRaw_test_df.get_values()
ridge.fit(X_train,y_train)

#这里是进行预测之后的结果，这个结果也是保存在一个dataframe中
y_ridge = np.expm1(ridge.predict(X_test))

#然后我们将我们的测试结果当道一个DataFrame中，这个DataFrame有两列，一个是数据集的索引，一个是我们测出的预测值
submission_df = pd.DataFrame(data = {'Id':test_df.index,'SalePrice':y_ridge})

submission_df.to_csv("input/result.csv")


# ridge.fit()
