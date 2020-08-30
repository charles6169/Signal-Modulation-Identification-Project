# -*- coding: utf-8 -*-
# @Time    : 2020/1/3 23:48
# @Author  : 张秦
# @File    : main.py
# @Software: PyCharm
# @文件描述:

import xlrd
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

from readfile import readfile,readfile_2
from Preprocessing import Preprocessing
from Algorithm import algorithm
from displayFunction import draw_pac,confusionMatrix,confusionMatrix2
from Model import *




if __name__=="__main__":

    # print("hello")
    print("读取数据...")
    trainData,trainLabel,testData,testLabel=readfile_2("true",5)
    print("预处理...")
    Preprocessing(trainData, trainLabel, testData, testLabel,"true")
    # print("PCA...")
    # dimensions=4
    # draw_pac(trainData,dimensions)
    print("常见机器学习识别...")
    pre=algorithm(trainData, trainLabel, testData, testLabel)




    lebels1=["2ASK","4ASK","2FSK","4FSK","2PSK","4PSK"]
    lebels2= ["2ASK", "4ASK", "MFSK", "2PSK", "4PSK"]
    lebels3=["2ASK","4ASK","2FSK","4FSK","8FSK","2PSK","4PSK","16QAM","32QAM","OFDM"]
    # print("绘制混淆矩阵...")
    # confusionMatrix(testLabel, pre, lebels2)

    print("定制混淆矩阵...")
    confusionMatrix2(lebels3)



    # learning_rate = 1e-5
    # model = models()
    # model.add(Dense(input_size=None, hidden_layer_num=5, activation=activations.relu))
    # model.add(Dense(input_size=None, hidden_layer_num=5, activation=activations.relu))
    # model.add(Dense(input_size=None, hidden_layer_num=5, activation=activations.softmax))
    # model.fit(train_data=trainData, train_labels=trainLabel, shuffle=False, epochs=6, batch_size=50,
    #           learning_rate=learning_rate, measure=measure.cross_entropy)
    # model.plot()
