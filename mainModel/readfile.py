# -*- coding: utf-8 -*-
# @Time    : 2020/1/3 23:54
# @Author  : 张秦
# @File    : readfile.py
# @Software: PyCharm
# @文件描述:
import xlrd
import numpy as np

rootPath = "D:/1_PythonDev/WorkSpace/ModRecogWorkSpace/data/HighOrderCumData/experiment3"
trainName = rootPath + "/train20.xls"
testName = rootPath + "/test20.xls"


def readfile(ifshow,featureNum):
    trainData = []
    trainLabel=[]
    testData=[]
    testLabel=[]

    trainWb = xlrd.open_workbook(trainName)
    testWb = xlrd.open_workbook(testName)
    trainTable = trainWb.sheets()[0]  # open the first sheet
    testTable = testWb.sheets()[0]

    # print(table)  #<xlrd.sheet.Sheet object at 0x0000020A68601320>
    trainRow = trainTable.nrows
    testRow = testTable.nrows
    for i in range(0, trainRow):
        trainData.append(trainTable.row_values(i)[0:featureNum-1])
        trainLabel.append(trainTable.row_values(i)[featureNum])
    for i in range(0, testRow):
        testData.append(testTable.row_values(i)[0:featureNum-1])
        testLabel.append(testTable.row_values(i)[featureNum])

    trainData=np.array(trainData)
    testData=np.array(testData)
    if ifshow=="true":
        print(trainData[0])
        print(trainLabel[0])
        print(testData[0])
        print(testLabel[0])

        print(trainData.shape)
        print(testData.shape)

    return trainData,trainLabel,testData,testLabel

# 将2FSK和4FSK合并
def readfile_2(ifshow, featureNum):
    trainData = []
    trainLabel = []
    testData = []
    testLabel = []

    trainWb = xlrd.open_workbook(trainName)
    testWb = xlrd.open_workbook(testName)
    trainTable = trainWb.sheets()[0]  # open the first sheet
    testTable = testWb.sheets()[0]

    # print(table)  #<xlrd.sheet.Sheet object at 0x0000020A68601320>
    trainRow = trainTable.nrows
    testRow = testTable.nrows
    for i in range(0, trainRow):
        if trainTable.row_values(i)[featureNum] == 3 or trainTable.row_values(i)[featureNum] == 4:
            trainData.append(trainTable.row_values(i)[0:featureNum - 1])
            trainLabel.append(3)
        else:
            trainData.append(trainTable.row_values(i)[0:featureNum - 1])
            trainLabel.append(trainTable.row_values(i)[featureNum])
    for i in range(0, testRow):
        if testTable.row_values(i)[featureNum] == 3 or testTable.row_values(i)[featureNum] == 4:
            testData.append(testTable.row_values(i)[0:featureNum - 1])
            testLabel.append(3)
        else:
            testData.append(testTable.row_values(i)[0:featureNum - 1])
            testLabel.append(testTable.row_values(i)[featureNum])

    trainData = np.array(trainData)
    testData = np.array(testData)
    if ifshow == "true":
        print(trainData[0])
        print(trainLabel[0])
        print(testData[0])
        print(testLabel[0])

        print(trainData.shape)
        print(testData.shape)

    return trainData, trainLabel, testData, testLabel
