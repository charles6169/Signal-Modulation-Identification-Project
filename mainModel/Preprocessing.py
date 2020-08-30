# -*- coding: utf-8 -*-
# @Time    : 2020/1/3 23:54
# @Author  : 张秦
# @File    : Preprocessing.py
# @Software: PyCharm
# @文件描述:

from sklearn.preprocessing import MinMaxScaler

def Preprocessing(trainData,trainLabel,testData,testLabel,ifShow):
    min_max_scaler  = MinMaxScaler()
    trainData= min_max_scaler .fit_transform(trainData)
    testData= min_max_scaler .fit_transform(testData)
    if ifShow=="true":
        print(trainData[0])
        print(testData[0])