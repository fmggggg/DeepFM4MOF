'''
# Time   : 2023/08/02
# Author : Minggao Feng
'''
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

def sparseFeature(feat, feat_onehot_dim, embed_dim):
    return {'feat': feat, 'feat_onehot_dim': feat_onehot_dim, 'embed_dim': embed_dim}

def denseFeature(feat):
    return {'feat': feat}

def create_hmof_dataset(file_path,ifcoldstart=True,propertyID=1,prop_test_frac=0.9, embed_dim=8, test_size=0.2):
    data = pd.read_csv(file_path)

    dense_features = ['surface_area [m^2/g]','void_fraction','largest_free_sphere_diameter [A]',\
        'largest_included_sphere_diameter [A]','metal_linker','organic_linker1','organic_linker2','clique_density [mmol/cm^3]']
    sparse_features = ['MOFID','targetID','functional_groups','topology']

    #缺失值填充
    data[dense_features] = data[dense_features].fillna(0)
    data[sparse_features] = data[sparse_features].fillna('-1')

    #归一化
    data[dense_features] = MinMaxScaler().fit_transform(data[dense_features])
    #LabelEncoding编码
    for col in sparse_features:
        data[col] = LabelEncoder().fit_transform(data[col])
    # feature_columns = [[denseFeature(feat) for feat in dense_features]] + \
    #        [[sparseFeature(feat, 9000, embed_dim) for feat in sparse_features]]
    feature_columns = [[denseFeature(feat) for feat in dense_features]] + \
           [[sparseFeature(feat, data[feat].nunique(), embed_dim) for feat in sparse_features]]
    # print("feature_columns:",feature_columns)
    #数据集划分
    if ifcoldstart:
        test_target=data[data['targetID']==(propertyID-1)]#目标特征
        train_target=data[data['targetID']!=(propertyID-1)]#非目标特征
        test_test_target=test_target.sample(frac=prop_test_frac)#90%的目标特征，用于测试最终模型
        train_test_target=test_target[~test_target.index.isin(test_test_target.index)]#剩余10%的18号特征，用于训练和验证
        # data=pd.concat([train_target,train_test_target])
        X_10 = train_test_target.drop(['label'], axis=1).values
        y_10 = train_test_target['label'].values
        X_train_10, X_test, y_train_10, y_test = train_test_split(X_10, y_10, test_size=test_size)#把10%中的一部分用作训练，一部分验证
        X_not18 = train_target.drop(['label'], axis=1).values
        y_not18 = train_target['label'].values
        X_train = np.concatenate((X_train_10,X_not18), axis=0)
        y_train = np.concatenate((y_train_10,y_not18), axis=0)
    else:
        X = data.drop(['label'], axis=1).values
        y = data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        test_test_target=0
    return feature_columns, (X_train, y_train), (X_test, y_test),test_test_target
