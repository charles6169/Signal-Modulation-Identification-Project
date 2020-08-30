

# 导入公共包
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# from RUSBoost_LiBin import RUSBoostClassifier
# from TVSVM import *
# from TVSVM import TwinSVMClassifier

from sklearn import manifold
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, mean_squared_error, accuracy_score
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig

from imblearn.over_sampling import SMOTE
import cmath
import time
import numpy as np
import xgboost
import warnings

warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)
import sys

# def dichotomy(y_true, y_pred,y_score, n_class):
#
#     if n_class == 2:
#         print("验证集准确率: ", accuracy_score(y_pred, y_true) * 100, "%")
#         print("小数类 精确率: ", metrics.precision_score(y_true, y_pred, pos_label=1)* 100, "%")
#         print("多数类 精确率: ", metrics.precision_score(y_true, y_pred, pos_label=0)* 100, "%")
#         print("小数类 召回率: ", metrics.recall_score(y_true, y_pred, pos_label=1)* 100, "%")
#         print("多数类 召回率: ", metrics.recall_score(y_true, y_pred, pos_label=0)* 100, "%")
#         print("F1 分数: ", metrics.f1_score(y_true, y_pred, pos_label=1))
#         # print("f0.5_score:", metrics.fbeta_score(y_true, y_pred, beta=0.5))
#         # print("f2_score:", metrics.fbeta_score(y_true, y_pred, beta=2.0))
#         Recall_minority = metrics.recall_score(y_true, y_pred, pos_label=1)
#         Recall_majority = metrics.recall_score(y_true, y_pred, pos_label=0)
#         precision_scoce_minority = metrics.precision_score(y_true, y_pred, pos_label=1)
#         print("G_mean: ", cmath.sqrt(Recall_minority * precision_scoce_minority))
#         print("Balanced Accuracy: ", ((Recall_minority + Recall_majority) / 2)* 100, "%")
#
#         auc=False
#         if auc==True:
#             fpr, tpr, threshold = roc_curve(y_true, y_score)  ###计算真正率和假正率
#             roc_auc = auc(fpr, tpr)  ###计算auc的值
#             # plt.figure()
#             lw = 2
#
#             plt.figure(num=1, figsize=(7, 7))
#             plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
#             plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#             plt.xlim([0.0, 1.0])
#             plt.ylim([0.0, 1.05])
#             plt.xlabel('False Positive Rate')
#             plt.ylabel('True Positive Rate')
#             plt.title('Receiver operating characteristic example')
#             plt.legend(loc="lower right")
#             plt.show()
#     elif n_class == 10:
#         print('验证集准确率：', accuracy_score(y_pred, y_true) * 100, "%")

y_score = None

def algorithm(trainData, trainLabel, testData, testLabel):
    # MR_RandomForestClassifier(trainData, trainLabel, testData, testLabel)
    # MR_DecisionTree(trainData, trainLabel, testData, testLabel)
    # MR_KNN(trainData, trainLabel, testData, testLabel)
    #
    # MR_GBDT(trainData, trainLabel, testData, testLabel)
    # MR_LogisticRegression(trainData, trainLabel, testData, testLabel)
    # MR_LinearDiscriminantAnalysis(trainData, trainLabel, testData, testLabel)
    # MR_SVM(trainData, trainLabel, testData, testLabel)
    # MR_AdaBoost(trainData, trainLabel, testData, testLabel)
    # MR_GaussianNB(trainData, trainLabel, testData, testLabel)
    pre=MR_XGBoost(trainData, trainLabel, testData, testLabel)
    return pre


def MR_RandomForestClassifier(trainData, trainLabel, testData, testLabel):
    ### Random Forest Classifier
    t1 = time.time()
    clf = RandomForestClassifier(n_estimators=8)  # 初始化分类器
    clf.fit(trainData, trainLabel)  # 使用训练集对测试集进行训练
    pre = clf.predict(testData)  # 使用逻辑回归函数对测试集进行预测
    t2 = time.time()
    ############### 保存模型 ##############

    print("RandomForestClassifier 耗时：", (t2 - t1) * 1000, "ms")
    print('RandomForestClassifier 识别率：', np.mean(pre == testLabel) * 100, "%")
    model_weight_File = r'D:\1_PythonDev\WorkSpace\ModRecogWorkSpace\model\RandomForestClassifier.model'
    joblib.dump(clf, model_weight_File)


def MR_DecisionTree(trainData, trainLabel, testData, testLabel):
    ### Random Forest Classifier
    t1 = time.time()
    clf = tree.DecisionTreeClassifier()  # 初始化分类器
    clf.fit(trainData, trainLabel)  # 使用训练集对测试集进行训练
    pre = clf.predict(testData)  # 使用逻辑回归函数对测试集进行预测
    t2 = time.time()
    ############### 保存模型 ##############

    print("DecisionTree 耗时：", (t2 - t1) * 1000, "ms")
    print('DecisionTree 识别率：', np.mean(pre == testLabel) * 100, "%")
    model_weight_File = r'D:\1_PythonDev\WorkSpace\ModRecogWorkSpace\model\DecisionTree.model'
    joblib.dump(clf, model_weight_File)


def MR_KNN(trainData, trainLabel, testData, testLabel):
    ### Random Forest Classifier
    t1 = time.time()
    clf = KNeighborsClassifier() # 初始化分类器
    clf.fit(trainData, trainLabel)  # 使用训练集对测试集进行训练
    pre = clf.predict(testData)  # 使用逻辑回归函数对测试集进行预测
    t2 = time.time()
    ############### 保存模型 ##############

    print("KNN 耗时：", (t2 - t1) * 1000, "ms")
    print('KNN 识别率：', np.mean(pre == testLabel) * 100, "%")
    model_weight_File = r'D:\1_PythonDev\WorkSpace\ModRecogWorkSpace\model\KNN.model'
    joblib.dump(clf, model_weight_File)


def MR_XGBoost(trainData, trainLabel, testData, testLabel):
    ### Random Forest Classifier
    t1 = time.time()
    clf = xgboost.XGBClassifier()  # 初始化分类器
    clf.fit(trainData, trainLabel)  # 使用训练集对测试集进行训练
    pre = clf.predict(testData)  # 使用逻辑回归函数对测试集进行预测
    t2 = time.time()
    ############### 保存模型 ##############

    print("xgboost 耗时：", (t2 - t1) * 1000, "ms")
    print('xgboost 识别率：', np.mean(pre == testLabel) * 100, "%")
    model_weight_File = r'D:\1_PythonDev\WorkSpace\ModRecogWorkSpace\model\xgboost.model'
    joblib.dump(clf, model_weight_File)

    return pre



def MR_GBDT(trainData, trainLabel, testData, testLabel):
    ### Random Forest Classifier
    t1 = time.time()
    clf = GradientBoostingClassifier(n_estimators=200)  # 初始化分类器
    clf.fit(trainData, trainLabel)  # 使用训练集对测试集进行训练
    pre = clf.predict(testData)  # 使用逻辑回归函数对测试集进行预测
    t2 = time.time()
    ############### 保存模型 ##############

    print("GBDT 耗时：", (t2 - t1) * 1000, "ms")
    print('GBDT 识别率：', np.mean(pre == testLabel) * 100, "%")
    model_weight_File = r'D:\1_PythonDev\WorkSpace\ModRecogWorkSpace\model\GBDT.model'
    joblib.dump(clf, model_weight_File)


def MR_LogisticRegression(trainData, trainLabel, testData, testLabel):
    ### Random Forest Classifier
    t1 = time.time()
    clf = LogisticRegression(penalty='l2')  # 初始化分类器
    clf.fit(trainData, trainLabel)  # 使用训练集对测试集进行训练
    pre = clf.predict(testData)  # 使用逻辑回归函数对测试集进行预测
    t2 = time.time()
    ############### 保存模型 ##############

    print("LogisticRegression 耗时：", (t2 - t1) * 1000, "ms")
    print('LogisticRegression 识别率：', np.mean(pre == testLabel) * 100, "%")
    model_weight_File = r'D:\1_PythonDev\WorkSpace\ModRecogWorkSpace\model\LogisticRegression.model'
    joblib.dump(clf, model_weight_File)



def MR_LinearDiscriminantAnalysis(trainData, trainLabel, testData, testLabel):
    ### Random Forest Classifier
    t1 = time.time()
    clf = LinearDiscriminantAnalysis()  # 初始化分类器
    clf.fit(trainData, trainLabel)  # 使用训练集对测试集进行训练
    pre = clf.predict(testData)  # 使用逻辑回归函数对测试集进行预测
    t2 = time.time()
    ############### 保存模型 ##############

    print("LinearDiscriminantAnalysis 耗时：", (t2 - t1) * 1000, "ms")
    print('LinearDiscriminantAnalysis 识别率：', np.mean(pre == testLabel) * 100, "%")
    model_weight_File = r'D:\1_PythonDev\WorkSpace\ModRecogWorkSpace\model\LinearDiscriminantAnalysis.model'
    joblib.dump(clf, model_weight_File)


def MR_SVM(trainData, trainLabel, testData, testLabel):
    ### Random Forest Classifier
    t1 = time.time()
    clf = SVC(kernel='rbf', probability=True)   # 初始化分类器
    clf.fit(trainData, trainLabel)  # 使用训练集对测试集进行训练
    pre = clf.predict(testData)  # 使用逻辑回归函数对测试集进行预测
    t2 = time.time()
    ############### 保存模型 ##############

    print("SVM 耗时：", (t2 - t1) * 1000, "ms")
    print('SVM 识别率：', np.mean(pre == testLabel) * 100, "%")
    # model_weight_File = r'D:\1_PythonDev\WorkSpace\ModRecogWorkSpace\model\SVM.model'
    # joblib.dump(clf, model_weight_File)

def MR_AdaBoost(trainData, trainLabel, testData, testLabel):
    ### Random Forest Classifier
    t1 = time.time()
    clf = AdaBoostClassifier()  # 初始化分类器
    clf.fit(trainData, trainLabel)  # 使用训练集对测试集进行训练
    pre = clf.predict(testData)  # 使用逻辑回归函数对测试集进行预测
    t2 = time.time()
    ############### 保存模型 ##############

    print("AdaBoost 耗时：", (t2 - t1) * 1000, "ms")
    print('AdaBoost 识别率：', np.mean(pre == testLabel) * 100, "%")
    model_weight_File = r'D:\1_PythonDev\WorkSpace\ModRecogWorkSpace\model\AdaBoost.model'
    joblib.dump(clf, model_weight_File)

def MR_GaussianNB(trainData, trainLabel, testData, testLabel):
    ### Random Forest Classifier
    t1 = time.time()
    clf =GaussianNB()  # 初始化分类器
    clf.fit(trainData, trainLabel)  # 使用训练集对测试集进行训练
    pre = clf.predict(testData)  # 使用逻辑回归函数对测试集进行预测
    t2 = time.time()
    ############### 保存模型 ##############

    print("GaussianNB 耗时：", (t2 - t1) * 1000, "ms")
    print('GaussianNB 识别率：', np.mean(pre == testLabel) * 100, "%")
    model_weight_File = r'D:\1_PythonDev\WorkSpace\ModRecogWorkSpace\model\GaussianNB.model'
    joblib.dump(clf, model_weight_File)


# def MR_TwinSVMClassifier(train_X, test_X, train_Y, test_Y, mode, n_class, csv_Test):
#     """
#     函数描述：
#     输入参数：
#     返回值：
#     """
#     t1 = time.time()
#     ### GaussianNB
#     clf = TwinSVMClassifier(**params2)
#     clf.fit(train_X, train_Y)  # 使用训练集对测试集进行训练
#     pre = clf.predict(test_X)  # 使用逻辑回归函数对测试集进行预测
#
#
#     ############### 保存模型 ##############
#     model_weight_File = r'D:\1_PythonCode\ModRecog\Zhang\Feature实验\models_weights\RUSBoostClassifier1.model'
#     joblib.dump(clf, model_weight_File)
#
#     # # # ############## 盲测 ##############
#     t2 = time.time()
#     print("TwinSVMClassifier 训练耗时：", (t2 - t1) * 1000, "ms")
#
#     dichotomy(test_Y, pre,y_score, n_class)
#     Testdata, Testlabel = load_feature_Test(csv_Test, mode)
#     Testdata = DelFeature(Testdata)
#     # print(Testdata.shape)
#     # print(Testlabel.shape)
#     # print(Testlabel)
#     # # ############## 数据处理 ##############
#     # 1.特征归一化处理
#     # Testdata = minmax.fit_transform(Testdata)
#     # 2.降维处理
#     # draw_pac()
#     # Testdata=two_components(Testdata,Testlabel,pic=False)
#     t1 = time.time()
#     Testpre = clf.predict(Testdata)
#     # print(Testpre)
#     # # ############## 输出结果 ##############
#     print('TwinSVMClassifier 盲测识别率：', np.mean(Testpre == Testlabel) * 100, "%")
#     t2 = time.time()
#     print('TwinSVMClassifier 盲测耗时：', (t2 - t1) * 1000, 'ms')
#     print(" ")
#     print("TwinSVMClassifier 预测结果", judge(Testpre, n_class))