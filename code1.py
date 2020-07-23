import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import xgboost as xgb
import operator
from sklearn.preprocessing import PolynomialFeatures,scale
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score,log_loss,average_precision_score,make_scorer
import time



#第一步：数据清洗（处理离群点，处理缺失值，特征的基于方差的选择，处理字符串的数据格式）
def pre_process_info(master_info):
    # 每行缺失值大于36的认为是离群点，
    null_num = []#记录缺失值的数量
    drop = []
    count = 0
    for row in master_info.values:
        num = pd.Series(row).isnull().sum()
        null_num.append(num)
        if num > 36:
            drop.append(count) #异常值的index
            null_num.remove(num) #删除
        count += 1
    for i in drop:
        master_info.drop(labels=[i], axis=0, inplace=True)
    master_info = master_info.reset_index(drop=True)  # 很强大，可以还原索引，从新变为默认的整型索引

    #处理缺失值
    # drop掉WeblogInfo_1   WeblogInfo_3
    master_info.drop(labels=["WeblogInfo_3", "WeblogInfo_1"], axis=1, inplace=True)
    # UserInfo_13  UserInfo_12  UserInfo_11  因为用户的信息有可能有些不愿意填，可以重新归为一类，填充-1即可
    master_info.fillna({"UserInfo_13": -1, "UserInfo_12": -1, "UserInfo_11": -1}, inplace=True)
    # 剩下缺失值少的，选择用众数来填充类别型，均值填充数值型
    nul_cols = [col for col in master_info.columns if (master_info[col].isnull().sum() != 0)]
    for col in nul_cols:
        if master_info[col].dtype != "object":
            val = master_info[col].mean()
        else:
            val = master_info[col].value_counts().index[0]
        master_info.fillna({col: val}, axis=0,inplace=True)

    # 剔除标准差变化很小的特征，特征太弱了，都聚在一堆分不开
    col = ["WeblogInfo_10", "WeblogInfo_49", "WeblogInfo_44", "WeblogInfo_41", "WeblogInfo_46",
           "WeblogInfo_55", "SocialNetwork_1", "WeblogInfo_43", "WeblogInfo_47", "WeblogInfo_52",
           "SocialNetwork_11", "WeblogInfo_58", "WeblogInfo_40", "WeblogInfo_32", "WeblogInfo_31", "WeblogInfo_23"]
    master_info.drop(labels=col, axis=1, inplace=True)

    # 处理文本的数据
    # 查看字符串的异常
    # master_info 的 UserInfo_8的重庆市和重庆，  UserInfo_9的 '中国联通 ', '中国联通'，
    # update_info的 UserupdateInfo1  '_Age',和 '_age'等
    master_info.loc[:, "UserInfo_8"] = master_info["UserInfo_8"].apply(
        lambda x: x if ("市" in x) or ("州" in x and len(x) > 5) else x + "市")
    master_info.loc[:, "UserInfo_9"] = master_info["UserInfo_9"].apply(lambda x: x.strip())

    # 根据缺失值，来衡量用户完善信息的程度，构建一个连续值特征
    master_info = pd.concat([master_info, pd.DataFrame(null_num, columns=["null_name"])], axis=1)
    return master_info

#第二步：特征工程（处理类别特征（几个，几十个，几百个取值的处理），在对类别型特征的多维度提取，时间序列的处理，连续性数据标准化，特征组合生成新特征在进行特征xgboost的选取）
def feature_engineering(master_info):
    # 一般的类别型信息处理
    # one-hot编码  UserInfo_22 ,UserInfo_23，Education_Info2，Education_Info3，Education_Info4,Education_Info6,Education_Info7,Education_Info8
    # WeblogInfo_19 WeblogInfo_20 WeblogInfo_21
    cols = ["UserInfo_22", "UserInfo_23", "Education_Info2", "Education_Info3", "Education_Info4", "Education_Info6",
            "Education_Info7", "Education_Info8", "WeblogInfo_19", "WeblogInfo_20", "WeblogInfo_21",
            "UserInfo_9"]
    master_info = pd.get_dummies(data=master_info, columns=cols)

    #类别型特征有几十个的取值
    #对类别型的省份信息有32个类别，进行绘图得出了每个省份违约概率的强度，以此来表明特征值的强度
    # 对于master_info的UserInfo_7，32个省份通过筛选出来区分度强，违约率高的6个省份，然后自行进行独热编码，
    """广东 浙江 山东 江苏 福建 河南"""
    master_info["UserInfo_7广东"] = 0
    master_info.loc[master_info["UserInfo_7"] == "广东", "UserInfo_7广东"] = 1
    master_info["UserInfo_7浙江"] = 0
    master_info.loc[master_info["UserInfo_7"] == "浙江", "UserInfo_7浙江"] = 1
    master_info["UserInfo_7山东"] = 0
    master_info.loc[master_info["UserInfo_7"] == "山东", "UserInfo_7山东"] = 1
    master_info["UserInfo_7江苏"] = 0
    master_info.loc[master_info["UserInfo_7"] == "江苏", "UserInfo_7江苏"] = 1
    master_info["UserInfo_7福建"] = 0
    master_info.loc[master_info["UserInfo_7"] == "福建", "UserInfo_7福建"] = 1
    master_info["UserInfo_7河南"] = 0
    master_info.loc[master_info["UserInfo_7"] == "河南", "UserInfo_7河南"] = 1
    master_info.drop(labels="UserInfo_7", inplace=True, axis=1)

    #对类别型特征的省份信息进行不同维度的挖掘
    # UserInfo_19  类似 UserInfo_7，但是特征比较强还得挖掘，  可以用一线，二线，三线城市来区分！
    one = ['湖南省', '江苏省', '浙江省', '广东省', '安徽省', '上海市', '北京市']
    two = ['四川省', '福建省', '湖北省', '江西省', '山东省', '天津市', '甘肃省', '贵州省', '陕西省', '重庆市', '河北省', '青海省'
        , '黑龙江省', '吉林省', '河南省', '海南省']
    three = ['辽宁省', '内蒙古自治区', '山西省', '云南省', '广西壮族自治区', '宁夏回族自治区', '新疆维吾尔自治区', '西藏自治区']
    master_info.loc[:, "UserInfo_19"] = master_info["UserInfo_19"].apply(lambda x: "one" if x in one else x)
    master_info.loc[:, "UserInfo_19"] = master_info["UserInfo_19"].apply(lambda x: "two" if x in two else x)
    master_info.loc[:, "UserInfo_19"] = master_info["UserInfo_19"].apply(lambda x: "three" if x in three else x)
    master_info = pd.get_dummies(data=master_info, columns=["UserInfo_19"], drop_first=True)

    #类别特征特别多 有300多个值！
    # 利用Xgboost来分别对UserInfo_20  UserInfo_8  UserInfo_4  UserInfo_2进行特征选择，前top40就行！
    tmp_UserInfo_2 = pd.get_dummies(data=master_info, columns=["UserInfo_2"])
    tmp_UserInfo_4 = pd.get_dummies(data=master_info, columns=["UserInfo_4"])  # 注意：编码后直接就删除UserInfo_4列
    tmp_UserInfo_8 = pd.get_dummies(data=master_info, columns=["UserInfo_8"])
    tmp_UserInfo_20 = pd.get_dummies(data=master_info, columns=["UserInfo_20"])
    top40_UserInfo_2 = feature_xgboost_model_40(tmp_UserInfo_2,
                                             tmp_UserInfo_2.filter(regex="UserInfo_2_.*?").columns.tolist())  # 注意格式！！！
    top40_UserInfo_4 = feature_xgboost_model_40(tmp_UserInfo_4,
                                             tmp_UserInfo_4.filter(regex="UserInfo_4_.*?").columns.tolist())
    top40_UserInfo_8 = feature_xgboost_model_40(tmp_UserInfo_8,
                                             tmp_UserInfo_8.filter(regex="UserInfo_8_.*?").columns.tolist())
    top40_UserInfo_20 = feature_xgboost_model_40(tmp_UserInfo_20,
                                              tmp_UserInfo_20.filter(regex="UserInfo_20_.*?").columns.tolist())
    master_info = pd.concat([master_info, tmp_UserInfo_2[top40_UserInfo_2], tmp_UserInfo_4[top40_UserInfo_4],
                             tmp_UserInfo_8[top40_UserInfo_8], tmp_UserInfo_20[top40_UserInfo_20]], axis=1)

    #在进行挖掘，类别型特征特别多，进行计数在分成6段在进行离散化
    # 对UserInfo_8 各个城市进行计数，进行绘图，分6段，组成一个6维向量，可生成6个维度！
    y = master_info.UserInfo_8.value_counts().values
    y = np.log(y + 1)
    x = master_info.UserInfo_8.value_counts().index
    city1 = x[y > 5.5].tolist()
    city2 = x[(5.5 >= y) & (y > 4.5)].tolist()
    city3 = x[(4.5 >= y) & (y > 3.5)].tolist()
    city4 = x[(3.5 >= y) & (y > 2.5)].tolist()
    city5 = x[(2.5 >= y) & (y > 1.5)].tolist()
    city6 = x[1.5 >= y].tolist()
    master_info["UserInfo_8_city1"] = 0
    master_info["UserInfo_8_city2"] = 0
    master_info["UserInfo_8_city3"] = 0
    master_info["UserInfo_8_city4"] = 0
    master_info["UserInfo_8_city5"] = 0
    master_info["UserInfo_8_city6"] = 0
    for i in range(len(master_info)):
        city = master_info.loc[i, "UserInfo_8"]
        if city in city1:
            master_info.loc[i, "UserInfo_8_city1"] = 1
        elif city in city2:
            master_info.loc[i, "UserInfo_8_city2"] = 1
        elif city in city3:
            master_info.loc[i, "UserInfo_8_city3"] = 1
        elif city in city4:
            master_info.loc[i, "UserInfo_8_city4"] = 1
        elif city in city5:
            master_info.loc[i, "UserInfo_8_city5"] = 1
        elif city in city6:
            master_info.loc[i, "UserInfo_8_city6"] = 1

    # 位置特征的在提取！
    # 对 2，4，8，20进行地理位置差异特征计算！进行组合diff_24 diff_28 ...有差异就为1，没有就为0！
    master_info["diff_24"] = 0
    master_info.loc[master_info["UserInfo_2"] == master_info["UserInfo_4"], "diff_24"] = 1
    master_info["diff_28"] = 0
    master_info.loc[master_info["UserInfo_2"] == master_info["UserInfo_8"], "diff_28"] = 1
    master_info["diff_220"] = 0
    master_info.loc[master_info["UserInfo_2"] == master_info["UserInfo_20"], "diff_220"] = 1
    master_info["diff_48"] = 0
    master_info.loc[master_info["UserInfo_4"] == master_info["UserInfo_8"], "diff_48"] = 1
    master_info["diff_420"] = 0
    master_info.loc[master_info["UserInfo_4"] == master_info["UserInfo_20"], "diff_420"] = 1
    master_info["diff_820"] = 0
    master_info.loc[master_info["UserInfo_8"] == master_info["UserInfo_20"], "diff_820"] = 1

    # 对时间序列 ListingInfo的处理
    # 统计每天的借贷人数，绘制逾期，和未逾期的曲线
    # 可以看出，时间特征，能力比较强因为可以很好的区别开逾期和未逾期的！一个无关，一个基本是线性关系！
    # 对时间序列进行了连续化处理，也可连续化处理每50天一个类别标记！
    base = datetime.datetime.strptime("2013111", "%Y%m%d")
    for i in range(len(master_info)):
        data_str = master_info.loc[i, "ListingInfo"].replace("/", "")
        days = datetime.datetime.strptime(data_str, "%Y%m%d")
        day = (days - base).days
        block = day
        master_info.loc[i, "ListingInfo"] = block

    # drop掉UserInfo_24 UserInfo_2 UserInfo_4  UserInfo_8  UserInfo_20
    master_info.drop(labels=["UserInfo_24", "UserInfo_2", "UserInfo_4", "UserInfo_8", "UserInfo_20"], axis=1,
                     inplace=True)

    # 归一化数值型数据
    col = master_info.columns[master_info.dtypes != "uint8"].tolist()
    col.remove("Idx"), col.remove("target")
    processed_col = ["UserInfo_7广东", "UserInfo_7浙江", "UserInfo_7山东", "UserInfo_7江苏", "UserInfo_7福建", "UserInfo_7河南",
                     "UserInfo_8_city1", 'UserInfo_8_city1', 'UserInfo_8_city2', 'UserInfo_8_city3', 'UserInfo_8_city4',
                     'UserInfo_8_city5',
                     'UserInfo_8_city6', 'diff_24', 'diff_28', 'diff_220', 'diff_48', 'diff_420', 'diff_820']
    col = list(set(col) - set(processed_col))
    tmp = scale(master_info[col],axis=1)
    master_info.drop(labels=col, axis=1, inplace=True)
    master_info = pd.concat([pd.DataFrame(tmp, columns=col), master_info], axis=1)  # 注意列名

    # 组合特征，利用xgboost找到已有的top20，进行两两组合，生成7000多维，在利用xgboost筛选出来前500，加入到原始特征
    # 可以尝试加减乘除，取log(x*y)
    columns = master_info.columns.tolist()
    columns.remove("Idx")  # 一定要注意标签
    top40_feature = feature_xgboost_model_40(master_info, columns)
    poly = PolynomialFeatures(degree=2)
    top40_feature = master_info.loc[:, top40_feature].copy()
    new_feature = poly.fit_transform(top40_feature)
    # 从new_feature中选择出，top500加入原始特征中
    build_top500_feature = topn_xgboost_model(new_feature, master_info["target"],n=500)
    build_top500_feature = [int(i.replace("f", "")) for i in build_top500_feature]
    cols = ["build_top500_feature" + str(i) for i in build_top500_feature]
    master_info = pd.concat([master_info, pd.DataFrame(data=new_feature[:, build_top500_feature], columns=cols)], axis=1)
    return master_info

#第三步：利用xgboost筛选出前300的特征
def feature_selection(master_info,n=300):
    columns = master_info.columns.tolist()
    columns.remove("Idx"),columns.remove("target")
    x = master_info[columns]
    y = master_info["target"]
    depth = 7
    eta = 0.001
    ntrees = 500
    params = {"objective": "binary:logistic",
              "booster": "gbtree",
              "eta": eta,
              "max_depth": depth,
              "subsample": 0.5,
              "colsample_bytree": 0.7,
              "silent": 1,
              "n_jobs": 1,
              }
    dtrain = xgb.DMatrix(x, y)
    model = xgb.train(params, dtrain, ntrees)
    implace = model.get_fscore()
    print("implace:",implace)
    print("implace len:",len(implace))
    f_topn= sorted(implace.items(), key=operator.itemgetter(1), reverse=True)[:n]
    f_topn = [i[0] for i in f_topn]
    return f_topn

#筛选出top40的特征
def feature_xgboost_model_40(data_info,columns):
    x = data_info[columns]
    y = data_info["target"]
    depth = 10
    eta = 0.01
    ntrees = 500
    params = {"objective": "binary:logistic",
              "booster": "gbtree",
              "eta": eta,
              "max_depth": depth,
              "subsample": 0.5,
              "colsample_bytree": 0.7,
              "silent": 1,
              "n_jobs":1,
              }
    dtrain = xgb.DMatrix(x,y)
    model = xgb.train(params, dtrain, ntrees)
    implace = model.get_fscore()
    f_top40 = sorted(implace.items(),key=operator.itemgetter(1),reverse=True)[:40]
    f_top40 = [ i[0] for i in f_top40 ]
    print("类别型特征挑选出了最好的40个特征")
    return  f_top40

#筛选出topn的特征
def topn_xgboost_model(x,y,n=500):
    depth = 8
    eta = 0.01
    ntrees = 500
    params = {"objective": "binary:logistic",
              "booster": "gbtree",
              "eta": eta,
              "max_depth": depth,
              "subsample": 0.5,
              "colsample_bytree": 0.7,
              "silent": 1,
              "n_jobs":1,
              }
    dtrain = xgb.DMatrix(x,y)
    model = xgb.train(params, dtrain, ntrees)
    implace = model.get_fscore()
    print("多项式组合特征新生成出了：{}个特征".format(len(x)))
    f_topn = sorted(implace.items(),key=operator.itemgetter(1),reverse=True)[:n]
    f_topn = [ i[0] for i in f_topn ]
    return  f_topn

#第四步：模型融合（处理样本不均衡，对SVM,XGBOOST,LR进行网格搜素参数，对产出的预测概率在送入linear回归学习得到特征的系数权重，利用权重和调好的参数进行最终的预测）
def merge_model(master_info,good_feature,random_state=100):
    #处理样本不均衡问题
    x_train,x_test,y_train,y_test = train_test_split(master_info[good_feature].copy(),master_info["target"].copy(),random_state=random_state,test_size=0.3)
    over_smaple = SMOTE(random_state=random_state)
    x_train,y_train= over_smaple.fit_sample(x_train,y_train)
    x_test = np.array(x_test)  # 训练集和测试集的类型必须一样 ，要注意！！！！
    y_test = np.array(y_test)
    #建立LR模型
    lr_model = logistic_regrestion(x_train,y_train,random_state)
    lr_model.fit(x_train,y_train)
    lr_pred = lr_model.predict_proba(x_test)
    score = roc_auc_score(y_test,lr_pred[:,1])
    log_score = log_loss(y_test,lr_pred[:,1])
    print("lr_pred:",lr_pred[:,1])
    print("AUC:",score)
    print("log:",log_score)
    #建立Xgboost模型（注意：早停）
    xgb_model = xgboost(x_train,y_train,random_state)
    xgb_model.fit(x_train, y_train, eval_metric="logloss", early_stopping_rounds=10,eval_set=[(x_test, y_test)], verbose = True)
    print(xgb_model.best_ntree_limit,xgb_model.best_iteration,xgb_model.best_score)
    xgb_pred = xgb_model.predict_proba(x_test)
    score = roc_auc_score(y_test,xgb_pred[:,1])
    log_score = log_loss(y_test, xgb_pred[:,1])
    print("xgb_pred:", xgb_pred[:,1])
    print("AUC:", score)
    print("log:", log_score)
    #建立svm模型
    svm_model = svm(x_train,y_train,random_state)
    svm_model.fit(x_train,y_train)
    svm_pred = svm_model.predict_proba(x_test)
    score = roc_auc_score(y_test,svm_pred[:,1])
    log_score = log_loss(y_test, svm_pred[:, 1])
    print("svm_pred:", svm_pred[:,1])
    print("AUC:", score)
    print("log:", log_score)
    #对模型进行学习权重
    feature = pd.concat([pd.DataFrame(lr_pred[:,1],columns=["lr1"]),
                         pd.DataFrame(xgb_pred[:,1],columns=["xgb1"]),
                         pd.DataFrame(svm_pred[:,1],columns=["svm1"]) ],axis=1)
    weight_ = get_weights(feature,y_test)
    #进行融合
    lr_pred = lr_model.predict_proba(x_test)*weight_[0]
    xgb_pred = xgb_model.predict_proba(x_test)*weight_[1]
    svm_pred = svm_model.predict_proba(x_test)*weight_[2]
    pred = (lr_pred+xgb_pred+svm_pred)
    log_score = log_loss(y_test, pred[:, 1])
    print("最终的预测：",pred[:,1])
    #进行验证auc
    auc = roc_auc_score(y_test,pred[:,1])
    print("AUC值为：",auc)
    print("log:", log_score)

def get_weights(feature,y_train):
    from sklearn.linear_model import LinearRegression
    """
    fit_intercept：是否存在截距，默认存在
    normalize：标准化开关，默认关闭
    方程分为两个部分存放，coef_存放回归系数，intercept_则存放截距，因此要查看方程，就是查看这两个变量的取值。
    """
    feature,y_train = np.array(feature),np.array(y_train)
    model = LinearRegression()
    model.fit(feature,y_train)
    print("系数分别为：",model.coef_)
    coef = [abs(i) for i in model.coef_]
    # 归一化coef
    sum_num = sum(coef)
    coef = [i / sum_num for i in coef]
    print("归一化后系数分别为：", coef)
    return coef

def svm(x_train,y_train,random_state=100):
    from sklearn.svm import SVC
    """
    C：是惩罚系数，即对误差的宽容度。c越高，说明越不能容忍出现误差,容易过拟合。C越小，容易欠拟合。C过大或过小，泛化能力变差
    gamma：是选择RBF函数作为kernel后，该函数自带的一个参数。隐含地决定了数据映射到新的特征空间后的分布，
            gamma越大，方差越小，高斯分布瘦高，可能会过拟合  gamma值越小，方差会大，可能欠拟合，训练误差太小
    degree：多项式poly核函数的维度，默认为3
    """
    model = SVC(C=1,random_state=random_state,kernel="rbf",gamma="auto",cache_size=7000,probability=True)
    """
    人家的参数：
    clf = SVC(C=C,kernel='rbf',gamma=gamma,probability=True,cache_size=7000,class_weight='balanced',verbose=True,random_state=random_seed)
    """
    params = {"kernel": ['linear', 'poly', 'rbf', 'sigmoid']}
    best_kernel = search_params(x_train,y_train,params,model)["kernel"]

    params = {"C":range(14,30,5)}  #我晕了SVC的C参数真的是慢，我用的线性核竟然慢到这种程度！
    best_c = search_params(x_train,y_train,params,model)["C"]

    # params = {"gamma":range(1,17,4)}
    # best_gama = search_params(x_train,y_train,params,model)["gamma"]

    return SVC(C=best_c,kernel=best_kernel,random_state=random_state,probability=True,cache_size=7000,gamma="auto")#在这里要这样设置才可以！！！！因为会拖慢速度，所以是可选项

def xgboost(x_train,y_train,random_state=100):
    from xgboost import XGBClassifier
    """
    通用参数：
        booster:gbtree基于树模型，gbliner基于线模型
        silent:静默模式开启不会输出信息，一般需要开启
        nthread：2，线程数量，不设置就是所有
    booster参数：
        只介绍tree booster，因为它的表现远远胜过linear booster，所以linear booster很少用到
        eta: 和 GBM中的 learning rate 参数类似。 通过减少每一步的权重，可以提高模型的鲁棒性。 典型值为0.01-0.2。
        min_child_weight：XGBoost的这个参数是最小样本权重的和，这个参数用于避免过拟合。当它的值较大时，可以避免模型学习到局部的特殊样本。                                      但是如果这个值过高，会导致欠拟合
        max_depth：树的最大深度。 这个值也是用来避免过拟合的。max_depth越大，模型会学到更具体的样本，
        gamma：在节点分裂时，只有分裂后损失函数的值下降了，才会分裂这个节点。Gamma指定了节点分裂所需的最小损失函数下降值。 
                这个参数的值越大，算法越保守。
        alpha[默认1]权重的L1正则化项。  lambda[默认1]权重的L2正则化项
    学习目标参数：
        objective[默认reg:linear]这个参数定义需要被最小化的损失函数。binary:logistic 二分类的逻辑回归，返回预测的概率(不是类别)
                multi:softmax 使用softmax的多分类器，返回预测的类别(不是概率)在这种情况下，你还需要多设一个参数：num_class(类别数目)
                multi:softprob 和multi:softmax参数一样，但是返回的是每个数据属于各个类别的概率。
        eval_metric[默认值取决于objective参数的值]  rmse 均方根误差 mae 平均绝对误差  ，
        
    """
    # skl包：XGBClassifier（....）.booster().get_fscore()  xgb包：model.get_fscore()/model.feature_implance
    model = XGBClassifier(max_depth=10,learning_rate=0.1,n_estimators=100,objective="binary:logistic",min_child_weight=5,
                          gamma=1,random_state=random_state,reg_alpha=1,reg_lambda=1,base_score=0.5,silent=False,nthread=1,n_jobs=3)
    params = {"n_estimators":range(100,1001,100)}
    best_n = search_params(x_train,y_train,params,model)["n_estimators"]

    params = {"learning_rate":[0.01,0.05,0.5,1.]}
    best_rate = search_params(x_train,y_train,params,model)["learning_rate"]

    params = {"max_depth":range(1,13,3)}
    best_depth = search_params(x_train,y_train,params,model)["max_depth"]

    params = {"min_child_weight":range(1,13,2)}
    best_weight = search_params(x_train,y_train,params,model)["min_child_weight"]

    params = {"gamma":range(0,10,2)}
    best_gain = search_params(x_train,y_train,params,model)["gamma"]

    params = {"base_score":[0.1,0.3,0.6,0.8]}
    best_base_score = search_params(x_train,y_train,params,model)["base_score"]

    return XGBClassifier(max_depth=best_depth,learning_rate=best_rate,n_estimators=best_n,objective="binary:logistic",min_child_weight=best_weight,
                          gamma=best_gain,random_state=1,reg_alpha=1,reg_lambda=random_state,base_score=best_base_score,silent=False,nthread=1,n_jobs=3)

def logistic_regrestion(x_train,y_train,random_state=100):
    from sklearn.linear_model import LogisticRegression
    #分类方式选择参数：multi_class : ovr , mvm(n*(n-1)/2次选择), ovo
    #类型权重参数（误分代价高）： class_weight ：class_weight={0:0.9, 1:0.1}，这样类型0的权重为90%，而类型1的权重为10%。 代价敏感学习 / balanced
    #样本权重参数： sample_weight ：调节样本权重的方法有两种：
    # 第一种是在class_weight使用balanced。第二种是在调用fit函数时，通过sample_weight来自己调节每个样本权重。
    #solver="saga" 随机梯度下降的一种变形支持L1， sag随机梯度下降法不支持L1
    model = LogisticRegression(penalty="l1",C=1,random_state=random_state,solver="saga",max_iter=30000)
    params = {"penalty":["l2","l1"]}
    best_penalty = search_params(x_train,y_train,params,model)["penalty"]

    params = {"C":[0.01,0.1,0.5,1.5,2,3]}
    best_C = search_params(x_train,y_train,params,model)["C"]
    return LogisticRegression(penalty=best_penalty,C=best_C,random_state=random_state,solver="saga",max_iter=30000)

def search_params(x_train,y_train,params,model):
    #负正类率(False Postive Rate)FPR: FP/(FP+TN)，代表分类器预测的正类中实际负实例占所有负实例的比例
    #真正类率(True Postive Rate)TPR: TP/(TP+FN),代表分类器预测的正类中实际正实例占所有正实例的比例
    from sklearn.model_selection import GridSearchCV
    print("model:{}".format(model.__class__))
    "average_precision_score则会预测值的平均准确率,该分值对应于precision-recall曲线下的面积。"
    sco = make_scorer(average_precision_score, greater_is_better=True)
    search = GridSearchCV(estimator=model,param_grid=params,cv=6,n_jobs=5,scoring="neg_log_loss")
    search.fit(x_train,y_train)
    print("best_params:",search.best_params_)
    print("best_neg-log:",search.best_score_)
    return search.best_params_



if __name__ == '__main__':
    import warnings
    s = time.time()
    warnings.filterwarnings("ignore")
    master_info = pd.read_csv("PPD_Training_Master_GBK_3_1_Training_Set.csv",encoding="gbk")

    # 第一步：数据清洗
    print("数据清洗>>>>")
    master_info = pre_process_info(master_info.loc[:200])#.loc[:2000]



    t = str((time.time() - s) // 60) + "分钟" + str(int((time.time() - s) % 60)) + "秒"
    print("花费的时间：", t)
    # 第二步特征工程
    print("特征工程>>>>>>>")
    master_info = feature_engineering(master_info)
    t = str((time.time() - s) // 60) + "分钟" + str(int((time.time() - s) % 60)) + "秒"
    print("特征处理花费的时间：", t)

    #第三步特征选择
    print("特征选择>>>>>>>>>")
    good_feature = feature_selection(master_info,n=40)
    print("选择特征完成>>>>>>")
    print("goog_feature：特征个数{}".format(len(good_feature)),good_feature)
    t = str((time.time() - s) // 60) + "分钟" + str(int((time.time() - s) % 60)) + "秒"
    print("特征处理花费的时间：", t)
    print(np.array(master_info.loc[:2,good_feature]))

    #第四步建立模型 LR，Xgboost，svm，并且融合
    merge_model(master_info,good_feature,random_state=100)
    t = str((time.time() - s) // 60) + "分钟" + str(int((time.time() - s) % 60)) + "秒"
    print("特征处理花费的时间：", t)
