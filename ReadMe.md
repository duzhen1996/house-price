# 房价回归预测

这是一档kaggle的习题，主要的内容我们得到了可以用来分析房价的80个特征，这80个特征和房价之间是有关系的。我们需要通过这80个特征来预测出房价。

## 数据查看

使用pandas获取数据的头部内容。编写了下面的代码。

```python
# coding:utf-8
# 这个文件我们的主要作用就是进行数据的检视，看看这个数据里面是什么东西
# 导入矩阵库用来处理矩阵
import numpy as np

import pandas as pd

# 这两个库可以服务并整理数据

# 这里将数据读入
# 我们使用pandas读取csv文件，然后默认将第0列作为索引，一般情况下index是没有什么用的，加不加都可以
train_df = pd.read_csv('input/train.csv', index_col=0)
test_df = pd.read_csv('input/test.csv', index_col=0)

#读进来之后看看数据是什么样
print train_df.head();
```

获得了如下的输出。

```shell
/System/Library/Frameworks/Python.framework/Versions/2.7/bin/python2.7 "/Users/zhendu/PycharmProjects/house price/lookData.py"
    MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape  \
Id                                                                    
1           60       RL         65.0     8450   Pave   NaN      Reg   
2           20       RL         80.0     9600   Pave   NaN      Reg   
3           60       RL         68.0    11250   Pave   NaN      IR1   
4           70       RL         60.0     9550   Pave   NaN      IR1   
5           60       RL         84.0    14260   Pave   NaN      IR1   

   LandContour Utilities LotConfig    ...     PoolArea PoolQC Fence  \
Id                                    ...                             
1          Lvl    AllPub    Inside    ...            0    NaN   NaN   
2          Lvl    AllPub       FR2    ...            0    NaN   NaN   
3          Lvl    AllPub    Inside    ...            0    NaN   NaN   
4          Lvl    AllPub    Corner    ...            0    NaN   NaN   
5          Lvl    AllPub       FR2    ...            0    NaN   NaN   

   MiscFeature MiscVal MoSold  YrSold  SaleType  SaleCondition  SalePrice  
Id                                                                         
1          NaN       0      2    2008        WD         Normal     208500  
2          NaN       0      5    2007        WD         Normal     181500  
3          NaN       0      9    2008        WD         Normal     223500  
4          NaN       0      2    2006        WD        Abnorml     140000  
5          NaN       0     12    2008        WD         Normal     250000  

[5 rows x 80 columns]

Process finished with exit code 0

```

head函数会默认获得5行80列的输出主要包含很多方面，这些东西实际上是不能直接被使用的。我们需要对这些数据进行预处理，主要包含下面的方面：

1. NaN的数据是不能被使用的，这个是缺省值，我们需要想办法处理一下。
2. 这里面有很多的英文属性，我们需要进行处理，将这些英文属性进行转化和建模。
3. id这一列是样本的索引编号，没有什么用，如果将这些值加入考虑，极可能产生过拟合，我们需要将这一列当做噪音处理。



## 数据的初步统计

我们初步统计了不同类型住宅的占比，因为样本的选取是随机的，所以我们可以看出低密度住宅的占比和数量是最多的。

![](https://ws1.sinaimg.cn/large/006tKfTcly1fgygvefr3ij30rm0iq74m.jpg)

我们看一下住房面积的分布，主要是集中在10000平方英尺，也就是900平方米。

![](https://ws1.sinaimg.cn/large/006tKfTcly1fgyh5zlv4fj30xk0w43zj.jpg)





## 合并与预处理

这里我们需要进行一下测试集和训练集的合并，我们的这个合并不是为了将测试集和训练集合起来进行训练，而是为了帮助二者进行预处理。此外有一个非常重要的属性—**SalePrice**，这个属性就是我们要进行预测的东西，这个属性不会出现在测试集中，我们需要将这个属性拿出来，防止训练集和测试集不等长。

我们将训练集中的price数据拿出来，并且单独构成一个新的训练集合，这个集合我们可以称作“Label集”。

我们上文使用的数据叫做所使用的是一种叫做DataFrame的数据类型，我们现在想做的就是将所有的price拿出来。做了做这个过程，我们需要将我们的这个DataFrame转化为一个数组，然后我们需要看看这个price的分布情况，看看是不是均匀，需要进一步处理。

我们首先先把price里面的东西取出来。

```python
#我们可以使用getValues这个函数将所有的DataFrame数据转化为
#train_df作为DataFrame支持map的取值方式，取出的东西还是DataFrame
allPrice = train_df["SalePrice"]
print allPrice

#然后我们将所有的房屋价格变成一个数组
allPriceArr = allPrice.get_values()
print allPriceArr
```

最后所有price存的数组就是在allPriceArr中。

```shell
/System/Library/Frameworks/Python.framework/Versions/2.7/bin/python2.7 "/Users/zhendu/PycharmProjects/house price/lookData.py"
Id
1       208500
2       181500
3       223500
4       140000
5       250000
6       143000
7       307000
8       200000
9       129900
10      118000
11      129500
12      345000
13      144000
14      279500
15      157000
16      132000
17      149000
18       90000
19      159000
20      139000
21      325300
22      139400
23      230000
24      129900
25      154000
26      256300
27      134800
28      306000
29      207500
30       68500
         ...  
1431    192140
1432    143750
1433     64500
1434    186500
1435    160000
1436    174000
1437    120500
1438    394617
1439    149700
1440    197000
1441    191000
1442    149300
1443    310000
1444    121000
1445    179600
1446    129000
1447    157900
1448    240000
1449    112000
1450     92000
1451    136000
1452    287090
1453    145000
1454     84500
1455    185000
1456    175000
1457    210000
1458    266500
1459    142125
1460    147500
Name: SalePrice, Length: 1460, dtype: int64

#这里就是所有价格组成的数组
[208500 181500 223500 ..., 266500 142125 147500]

Process finished with exit code 0

```

借此我们看了一个price的大致分布。

![](https://ws2.sinaimg.cn/large/006tNbRwgy1fghbsc89idj31a80rmmzd.jpg)

这是一张散点图，我们看到了房价的大致分布。主要分布在100000-200000之间，这并不是一种很好的分布。房价的偏移比较大，所以我们需要做一步“平滑化”操作，将数据变成更加均匀和稳定，有利于提高机器学习的准确率。

而解决平滑化的最简单的问题方式就是使用log(x+1)的处理。我们进行了处理，并且获得了一个新的分布。

![](https://ws3.sinaimg.cn/large/006tNbRwgy1fghc3r4hpyj311k0mmtak.jpg)

这样子整个分布就变得非常对称和均匀。这就是我们在price方面做的预处理。当然在我们在预测之后要把这个经过平均化处理的price再换回去。

所以现在我们需要将训练集中的房屋价格的部分从训练集中推出，组成一个label集，也就是机器学习的y值。

```python
y_train = np.log1p(train_df.pop["SalePrice"])
```

`y_train`就是所有label集的集合，我们打印当中的值。虽然使用的是numpy的接口，但是经过处理和导出的y_train还是一个DataFrame的结构。

```shell
/System/Library/Frameworks/Python.framework/Versions/2.7/bin/python2.7 "/Users/zhendu/PycharmProjects/house price/lookData.py"
Id
1       12.247699
2       12.109016
3       12.317171
4       11.849405
5       12.429220
6       11.870607
7       12.634606
8       12.206078
9       11.774528
10      11.678448
11      11.771444
12      12.751303
13      11.877576
14      12.540761
15      11.964007
16      11.790565
17      11.911708
18      11.407576
19      11.976666
20      11.842236
21      12.692506
22      11.845110
23      12.345839
24      11.774528
25      11.944714
26      12.454108
27      11.811555
28      12.631344
29      12.242891
30      11.134604
          ...    
1431    12.165985
1432    11.875838
1433    11.074436
1434    12.136192
1435    11.982935
1436    12.066816
1437    11.699413
1438    12.885673
1439    11.916395
1440    12.190964
1441    12.160034
1442    11.913720
1443    12.644331
1444    11.703554
1445    12.098493
1446    11.767575
1447    11.969724
1448    12.388398
1449    11.626263
1450    11.429555
1451    11.820418
1452    12.567555
1453    11.884496
1454    11.344519
1455    12.128117
1456    12.072547
1457    12.254868
1458    12.493133
1459    11.864469
1460    11.901590
Name: SalePrice, Length: 1460, dtype: float64

Process finished with exit code 0

```

然后我们就要进行合并工作，因为现在训练集和测试集全部变成79维了，所以合并就是单纯的内容叠加，将测试集接在训练集下面。

```python
#这里我们进行训练集和测试集的合并，因为训练集已经排除了房屋价格
all_df = pd.concat((train_df,test_df))

print all_df.shape
```



## 数据转化与特征工程

特征工程主要解决英文属性向数字化的转化，并且我们需要去解决属性缺省的问题。

为了方便我们进行特征工程，staggle提供了一个数据描述文件，里面阐明了每一个属性的意思。这里面有一个属性是非常要值得注意的。就是MSSubClass。这里实际上表述了一个住所的类型，可能是为了方面存储，这里面每一个住所的类型都是用一个int类型来进行替代，这在机器学习中是一种非常危险的噪音。因为实际上这些数字只是代表一个编号，这个编号实际上没有大小的差别，只是单纯的编号而已，所以我们需要对这个编号转化一下，把这个编号变为一个string，消除其大小比较的潜在特性。

```
MSSubClass: Identifies the type of dwelling involved in the sale.	

        20	1-STORY 1946 & NEWER ALL STYLES
        30	1-STORY 1945 & OLDER
        40	1-STORY W/FINISHED ATTIC ALL AGES
        45	1-1/2 STORY - UNFINISHED ALL AGES
        50	1-1/2 STORY FINISHED ALL AGES
        60	2-STORY 1946 & NEWER
        70	2-STORY 1945 & OLDER
        75	2-1/2 STORY ALL AGES
        80	SPLIT OR MULTI-LEVEL
        85	SPLIT FOYER
        90	DUPLEX - ALL STYLES AND AGES
       120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
       150	1-1/2 STORY PUD - ALL AGES
       160	2-STORY PUD - 1946 & NEWER
       180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
       190	2 FAMILY CONVERSION - ALL STYLES AND AGES
```

我们使用onehot的方式来进行房屋类别的特征值处理，onehot实际上就是将不同的类别属性值转化为一个线性数组。pandas有一个叫做get_dummies的方法，可以非常方便地做到One-Hot。实际上这个过程就是将一个属性拆分成了多个属性。

```python
print pd.get_dummies(all_df["MSSubClass"],prefix='MSSubClass').head()
```

我们这里使用了这个函数，它接收的第一个形参的就是规定了我们要进行OneHot操作的那一列，第二列的形参规定了拆分出的新属性的属性名前缀。我们看一下打印的结果。

```shell
/System/Library/Frameworks/Python.framework/Versions/2.7/bin/python2.7 "/Users/zhendu/PycharmProjects/house price/lookData.py"
    MSSubClass_20  MSSubClass_30  MSSubClass_40  MSSubClass_45  MSSubClass_50  \
Id                                                                              
1               0              0              0              0              0   
2               1              0              0              0              0   
3               0              0              0              0              0   
4               0              0              0              0              0   
5               0              0              0              0              0   

    MSSubClass_60  MSSubClass_70  MSSubClass_75  MSSubClass_80  MSSubClass_85  \
Id                                                                              
1               1              0              0              0              0   
2               0              0              0              0              0   
3               1              0              0              0              0   
4               0              1              0              0              0   
5               1              0              0              0              0   

    MSSubClass_90  MSSubClass_120  MSSubClass_150  MSSubClass_160  \
Id                                                                  
1               0               0               0               0   
2               0               0               0               0   
3               0               0               0               0   
4               0               0               0               0   
5               0               0               0               0   

    MSSubClass_180  MSSubClass_190  
Id                                  
1                0               0  
2                0               0  
3                0               0  
4                0               0  
5                0               0  

Process finished with exit code 0

```

我们可以看到下面这些属性被拆分成了很多的属性，并且属性名的前缀就是我们规定的"MSSubClass"。

实际上MSSubClass并不是唯一一个表达了“分类“的属性，pandas可以将反复重复出现的字符串属性值智能地判断为”类别“属性，所以我们首先需要把这个非常特殊的、使用数字作为分类的属性变为字符串属性，然后我们可以一口气将所有的类似属性全部进行One-Hot操作（因为其他的”类别“属性已经使用字符串来进行属性的划分了）。

```python
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)
all_dummpy_df = pd.get_dummies(all_df)
print all_dummpy_df.head()
```

然后我们进行输出，我们发现现在已经变成303维的数据了，而这一步实际上就已经把所有的字符串变为数字了。

```python
/System/Library/Frameworks/Python.framework/Versions/2.7/bin/python2.7 "/Users/zhendu/PycharmProjects/house price/lookData.py"
    MSSubClass  LotFrontage  LotArea  OverallQual  OverallCond  YearBuilt  \
Id                                                                          
1           60         65.0     8450            7            5       2003   
2           20         80.0     9600            6            8       1976   
3           60         68.0    11250            7            5       2001   
4           70         60.0     9550            7            5       1915   
5           60         84.0    14260            8            5       2000   

    YearRemodAdd  MasVnrArea  BsmtFinSF1  BsmtFinSF2          ...            \
Id                                                            ...             
1           2003       196.0       706.0         0.0          ...             
2           1976         0.0       978.0         0.0          ...             
3           2002       162.0       486.0         0.0          ...             
4           1970         0.0       216.0         0.0          ...             
5           2000       350.0       655.0         0.0          ...             

    SaleType_ConLw  SaleType_New  SaleType_Oth  SaleType_WD  \
Id                                                            
1                0             0             0            1   
2                0             0             0            1   
3                0             0             0            1   
4                0             0             0            1   
5                0             0             0            1   

    SaleCondition_Abnorml  SaleCondition_AdjLand  SaleCondition_Alloca  \
Id                                                                       
1                       0                      0                     0   
2                       0                      0                     0   
3                       0                      0                     0   
4                       1                      0                     0   
5                       0                      0                     0   

    SaleCondition_Family  SaleCondition_Normal  SaleCondition_Partial  
Id                                                                     
1                      0                     1                      0  
2                      0                     1                      0  
3                      0                     1                      0  
4                      0                     0                      0  
5                      0                     1                      0  

[5 rows x 288 columns]

Process finished with exit code 0

```

下面我们来处理NaN的问题，我们采用一种比较简单粗暴的方式来处理，那就是使用平均值去填充。我们将平均值计算出来，实际上可以看到，这个平均值还是存放在一个DataFrame的数据类型中。

```python
mean_cols = all_dummpy_df.mean()

print mean_cols
```

我们打印一下求平均值的结果：

```python
/System/Library/Frameworks/Python.framework/Versions/2.7/bin/python2.7 "/Users/zhendu/PycharmProjects/house price/lookData.py"
LotFrontage                 69.305795
LotArea                  10168.114080
OverallQual                  6.089072
OverallCond                  5.564577
YearBuilt                 1971.312778
YearRemodAdd              1984.264474
MasVnrArea                 102.201312
BsmtFinSF1                 441.423235
BsmtFinSF2                  49.582248
BsmtUnfSF                  560.772104
TotalBsmtSF               1051.777587
1stFlrSF                  1159.581706
2ndFlrSF                   336.483727
LowQualFinSF                 4.694416
GrLivArea                 1500.759849
BsmtFullBath                 0.429894
BsmtHalfBath                 0.061364
FullBath                     1.568003
HalfBath                     0.380267
BedroomAbvGr                 2.860226
KitchenAbvGr                 1.044536
TotRmsAbvGrd                 6.451524
Fireplaces                   0.597122
GarageYrBlt               1978.113406
GarageCars                   1.766621
GarageArea                 472.874572
WoodDeckSF                  93.709832
OpenPorchSF                 47.486811
EnclosedPorch               23.098321
3SsnPorch                    2.602261
                             ...     
GarageCond_TA                0.909215
PavedDrive_N                 0.073998
PavedDrive_P                 0.021240
PavedDrive_Y                 0.904762
PoolQC_Ex                    0.001370
PoolQC_Fa                    0.000685
PoolQC_Gd                    0.001370
Fence_GdPrv                  0.040425
Fence_GdWo                   0.038369
Fence_MnPrv                  0.112710
Fence_MnWw                   0.004111
MiscFeature_Gar2             0.001713
MiscFeature_Othr             0.001370
MiscFeature_Shed             0.032545
MiscFeature_TenC             0.000343
SaleType_COD                 0.029805
SaleType_CWD                 0.004111
SaleType_Con                 0.001713
SaleType_ConLD               0.008907
SaleType_ConLI               0.003083
SaleType_ConLw               0.002741
SaleType_New                 0.081877
SaleType_Oth                 0.002398
SaleType_WD                  0.865022
SaleCondition_Abnorml        0.065091
SaleCondition_AdjLand        0.004111
SaleCondition_Alloca         0.008222
SaleCondition_Family         0.015759
SaleCondition_Normal         0.822885
SaleCondition_Partial        0.083933
Length: 303, dtype: float64

Process finished with exit code 0

```

然后我们使用平均值来弥补缺失。

```python
all_dummpy_df = all_dummpy_df.fillna(mean_cols)
```

然后我们将数值属性进行标准化，这个过程和之前将标签”平滑化“的思路是一样的，主要做的工作就是将一个不太均匀的分布变成一个”类正态分布“，我们可以继续使用log的方法，这里我们再秀一种另外一种方法（这个方法在《概率论》这门课中反复出现过）：

（每一个数据点的当前值-平均值）/标准差

所以我们需要将标准差和平均值单独算出来，然后进行计算和填充就好了。这个方法比log的方式复杂，但是因为是《概率论》这个课程中谈到的方法，所以这种平滑化的方式可信度是有保证的。

首先我们先取出所有数值属性的属性名。我们使用DataFrame的columns属性（这个属性存储了列的基本信息）来获取数值属性名。

```python
numeric_cols = all_df.columns[all_df.dtypes != 'object']
print numeric_cols
```

我们这里的代码获得了所有的数值属性。我们可以看到使用OneHot被拆分的属性并没有在这里面出现。这个说明那个十分方便的One-Hot操作实际并没有真正拆分出那些属性，虽然我们在打印的时候看到他们是不同的属性，但是实际上他们在”逻辑上“，也就是在columns这个属性中还是保持着原本属性的名称和类型（也就是String类型，这就是为什么那么表示类别的属性没有被选出来）。

```shell
/System/Library/Frameworks/Python.framework/Versions/2.7/bin/python2.7 "/Users/zhendu/PycharmProjects/house price/lookData.py"
Index([u'LotFrontage', u'LotArea', u'OverallQual', u'OverallCond',
       u'YearBuilt', u'YearRemodAdd', u'MasVnrArea', u'BsmtFinSF1',
       u'BsmtFinSF2', u'BsmtUnfSF', u'TotalBsmtSF', u'1stFlrSF', u'2ndFlrSF',
       u'LowQualFinSF', u'GrLivArea', u'BsmtFullBath', u'BsmtHalfBath',
       u'FullBath', u'HalfBath', u'BedroomAbvGr', u'KitchenAbvGr',
       u'TotRmsAbvGrd', u'Fireplaces', u'GarageYrBlt', u'GarageCars',
       u'GarageArea', u'WoodDeckSF', u'OpenPorchSF', u'EnclosedPorch',
       u'3SsnPorch', u'ScreenPorch', u'PoolArea', u'MiscVal', u'MoSold',
       u'YrSold'],
      dtype='object')

Process finished with exit code 0
```

实际上我们将代码改动一下就可以看出端倪。

```python
numeric_cols = all_df.columns[all_df.dtypes == 'object']

print numeric_cols
```

我们将所有”非数值属性“取出，并且打印。

```shell
/System/Library/Frameworks/Python.framework/Versions/2.7/bin/python2.7 "/Users/zhendu/PycharmProjects/house price/lookData.py"
Index([u'MSSubClass', u'MSZoning', u'Street', u'Alley', u'LotShape',
       u'LandContour', u'Utilities', u'LotConfig', u'LandSlope',
       u'Neighborhood', u'Condition1', u'Condition2', u'BldgType',
       u'HouseStyle', u'RoofStyle', u'RoofMatl', u'Exterior1st',
       u'Exterior2nd', u'MasVnrType', u'ExterQual', u'ExterCond',
       u'Foundation', u'BsmtQual', u'BsmtCond', u'BsmtExposure',
       u'BsmtFinType1', u'BsmtFinType2', u'Heating', u'HeatingQC',
       u'CentralAir', u'Electrical', u'KitchenQual', u'Functional',
       u'FireplaceQu', u'GarageType', u'GarageFinish', u'GarageQual',
       u'GarageCond', u'PavedDrive', u'PoolQC', u'Fence', u'MiscFeature',
       u'SaleType', u'SaleCondition'],
      dtype='object')

Process finished with exit code 0
```

我们可以看到”类别属性“并没有像被One-Hot处理掉那样被拆分。他们在”逻辑上“还是保持着原有的样子。

关于这个编程上的细节就阐述到这里，下一步我们根据这个求出标准差和均值，并且进行进一步计算。

```python
#这里计算每一个属性的平均值
numeric_col_means = all_dummpy_df.loc[:,numeric_cols].mean()
#这里计算每一个属性的标准差
numeric_col_std = all_dummpy_df.loc[: , numeric_cols].std()

#这里将数据进行标准化
all_dummpy_df.loc[:,numeric_cols] = (all_dummpy_df.loc[:,numeric_cols] - numeric_col_means)/numeric_col_std

print all_dummpy_df
```

这里标准化就进行完了。

以上就是特征工程的全部内容。



## 建立模型

首先我们将之前合并的总数据集重新划分为训练集和测试集，我们之前已经把第0列设置为了索引，现在我们使用index属性将测试机和训练集重新分回。

我们使用ridge来进行回归预测。

```python
# 下面进行模型训练

# 将训练集和测试集重新分回
noRaw_train_df = all_noRaw_df.loc[train_df.index]
noRaw_test_df = all_noRaw_df.loc[test_df.index]


ridge = Ridge(14)

X_train = noRaw_train_df.get_values()
X_test = noRaw_test_df.get_values()
ridge.fit(X_train,y_train)

#这里是进行预测之后的结果，这个结果也是保存在一个dataframe中
y_ridge = np.expm1(ridge.predict(X_test))

#然后我们将我们的测试结果当道一个DataFrame中，这个DataFrame有两列，一个是数据集的索引，一个是我们测出的预测值
submission_df = pd.DataFrame(data = {'Id':test_df.index,'SalePrice':y_ridge})

submission_df.to_csv("input/result.csv")

```

Ridge这个回归算法需要一个参数，我们从0.1开始不断进行测试，0.1、1、10、100、1000，结果发现在10-100之间排名比较高，然后我们又开始尝试10、20、30、40，这个时候我们发现在10到20之间分数比较高，最后我们提交了15，并获得了在这套方案下的尽可能最好的成绩。

![](https://ws2.sinaimg.cn/large/006tNbRwgy1fghylry0b7j31kw11wtcm.jpg)

最终的成绩排进了前三分之一。











