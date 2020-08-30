# -*- coding: utf-8 -*-
# @Time    : 2020/1/3 23:55
# @Author  : 张秦
# @File    : displayFunction.py
# @Software: PyCharm
# @文件描述:

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


def draw_pac(X_conversion,dimensions):
    # pca = PCA(n_components=29)
    pca = PCA(n_components=dimensions)
    pca.fit(X_conversion)

    plt.figure(num=1, figsize=(8, 6))
    # 画图
    x = np.linspace(1, dimensions, dimensions)
    plt.plot(x, pca.explained_variance_ratio_, color='red')
    # 设置坐标轴名称
    plt.xlabel('features')
    plt.ylabel('explained_variance_ratio')
    # 设置坐标轴刻度
    my_x_ticks = np.arange(1, 12, 1)
    plt.xticks(my_x_ticks)
    plt.savefig(r'D:\0-信号调制识别\2-实验报告与图集\小波变换+高阶累积量+深度学习\实验图集\特征PCA图\test\explained_variance_ratio.jpg')

    plt.figure(num=2, figsize=(8, 6))
    # 画图
    x = np.linspace(1, dimensions, dimensions)
    plt.plot(x, pca.explained_variance_, color='blue')

    # 设置坐标轴名称
    plt.xlabel('features')
    plt.ylabel('explained_variance')
    # 设置坐标轴刻度
    my_x_ticks = np.arange(1, 12, 1)
    plt.xticks(my_x_ticks)

    plt.savefig(r'D:\0-信号调制识别\2-实验报告与图集\小波变换+高阶累积量+深度学习\实验图集\特征PCA图\test\explained_variance.jpg')


from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm,labels,title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.tick_params(labelsize=10)  #
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=6)
    plt.yticks(xlocations, labels)

    plt.xlabel('Predicted label')
    plt.tick_params(labelsize=13)  #
    plt.ylabel('True label')
    plt.tick_params(labelsize=13)  #
    # plt.colorbar()

    # 显示数据
    for first_index in range(len(cm)):  # 第几行
        for second_index in range(len(cm[first_index])):  # 第几列
            plt.text(first_index, second_index, cm[first_index][second_index])
    # plt.figure(figsize=(8, 8))
    plt.show()

    plt.savefig(r"D:\0-信号调制识别\2-实验报告与图集\小波变换+高阶累积量+深度学习\实验图集\混淆矩阵\ConfusionMatrix2.jpg")

def confusionMatrix(y_true, y_pred,labels):
    cm=confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, labels,title='Confusion Matrix', cmap=plt.cm.binary)





