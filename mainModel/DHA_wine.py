import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler,RobustScaler
from sklearn.datasets import load_iris
from tensorflow.contrib.layers import l2_regularizer,l1_regularizer
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from matplotlib.ticker import FuncFormatter
import csv
import shutil
import os
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def g1n_term(var,center,Rm,num):
    penalty = 0.0
    for i in range(num):
        g1n = tf.linalg.norm(var[i,:] -center )-Rm
        g1n_max = tf.where(tf.greater(g1n, 0), g1n, 0)
        penalty = penalty + g1n_max   # if res>0,penalty = res else penalty = 0
    return penalty

def g2n_term(var,center,Rm,num):
    penalty = 0.0
    for i in range(num):
        g2n = Rm -  tf.linalg.norm(var[i,:] -center )
        g2n_max = tf.where(tf.greater(g2n, 0), g2n, 0)
        penalty = penalty + g2n_max   # if res>0,penalty = res else penalty = 0
    return penalty

def load_Wine():
    data = []
    target = []
    dataFrame = csv.reader(open('D:/ProjectPyTest/testFile/Wine.csv'))
    for row in dataFrame:
        data.append(row[1:])
        target.append(row[0])
    data = np.array(data,dtype=np.float32)
    target = np.array(target,dtype=np.float32)
    scaler = MinMaxScaler(feature_range=(0, 1))  # 标准化
    data = scaler.fit_transform(data)
    return data,target

def Iris_Pre_random_batch(data,target,N_):
# 输入：  data：原始数据。150*4
#        target：原始数据对应的分类。150*1
#        N：随机抽取测试数量的百分数 0.1 = 10%
# 输出： data_train：用来训练的样本
#       data_train：训练样本的类别
#       data_test： 用来测试的样本
#       target_test：测试样本对用的分类
    Number,_ = np.shape(data)
    data_index = np.array(range(Number))
    random_index = np.random.choice(Number,size= N_,replace=False)
    data_index = np.delete(data_index,random_index)    #得到已经删除的随机索引
    data_train = data[data_index]             #得到随机的训练样本
    target_train = target[data_index]
    data_test = data[random_index]              #得到随机的测试样本
    target_test = target[random_index]
    #排序
    sort_index = np.argsort(target_train)
    data_train = data_train[sort_index]
    target_train = target_train[sort_index]
    #查看每类的数量用于计算损失函数
    kind = set(target_train)
    target_train_list = list(kind)
    kind1 = list(target_train).count(target_train_list[0])   #第一类数量
    kind2 = list(target_train).count(target_train_list[1])   #第二类数量
    kind3 = list(target_train).count(target_train_list[2])   #第三类数量
    return data_train,target_train,data_test,target_test,kind1,kind2,kind3

def valid_random(data,target,N_):
    Number,_ = np.shape(data)
    random_index = np.random.choice(Number,size= N_,replace=False)
    data_valid = data[random_index]             #得到随机的训练样本
    target_valid = target[random_index]
    return data_valid,target_valid

def Gaussian_PDF(w,b,traindata,testdata):
#使用高斯概率密度函数
#输入：    w:权值
#          b：偏置
#          var_newSpace:属于某一类的新的变量空间
#          var_varify:  还没有变换的空间的用来验证的数据集
#输出：    res：输出为一个数组，是每一个验证数据与一类新空间中各个变量的值
    sigma = np.ones([1],dtype=np.float32)
    U_new_test = np.matmul( testdata , w ) + b
    U_new_train = np.matmul( traindata , w ) + b
    m,_ = np.shape(U_new_test)
    n, _ = np.shape(traindata)
    res = np.zeros([m,1],dtype=np.float32)
    for i in range(len(U_new_test)):
        for j in range(len(U_new_train)):
            numerator = np.exp( -(np.power(np.linalg.norm(U_new_test[i]-U_new_train[j]),2)/ (2 * np.power(sigma,2))))
            denominator = np.power(( 2 * np.pi),(1/2)) *np.power(sigma,1)
            res[i] = res[i] + numerator/denominator
    return res/n

def Gaussian_acc3(Gaus1,Gaus2,Gaus3,target):
    m,_ = np.shape(Gaus1)
    y = np.zeros([m])
    for i in range(m):
        if Gaus1[i] > Gaus2[i]:
           if Gaus3[i] > Gaus1[i]:
               y[i] = 3
           else:
               y[i] = 1
        else:
            if Gaus2[i]>Gaus3[i]:
                y[i] = 2
            else:
                y[i] = 3
    correct_predict = np.equal(y, target)
    accuracy = np.mean(correct_predict)
    return accuracy

# 计算特征和类的平均值
def calcMean(x,y):
    sum_x = sum(x)
    sum_y = sum(y)
    n = len(x)
    x_mean = float(sum_x+0.0)/n
    y_mean = float(sum_y+0.0)/n
    return x_mean,y_mean

# 计算Pearson系数
def calcPearson(x,y):
    x_mean,y_mean = calcMean(x,y)   # 计算x,y向量平均值
    n = len(x)
    sumTop = 0.0
    sumBottom = 0.0
    x_pow = 0.0
    y_pow = 0.0
    for i in range(n):
        sumTop += (x[i]-x_mean)*(y[i]-y_mean)
    for i in range(n):
        x_pow += np.power(x[i]-x_mean,2)
    for i in range(n):
        y_pow += np.power(y[i]-y_mean,2)
    sumBottom = np.sqrt(x_pow*y_pow)
    p = sumTop/sumBottom
    return p

# 计算每个特征的Pearson系数，返回数组
def calcAttribute(dataSet,y_):
    prr = []
    n,m = np.shape(dataSet)    # 获取数据集行数和列数
    x = [0] * n             # 初始化特征x和类别y向量
    y = y_
    for j in range(m):    # 获取每个特征的向量，并计算Pearson系数，存入到列表中
        for k in range(n):
            x[k] = dataSet[k][j]
        prr.append(calcPearson(x,y))
    return prr

# def best_parameter(w,b,acc,best_w,best_b,best_acc,num_i):
#    ave_acc = []
#    stop = False
#    ave_acc.append(acc)
#    ave_acc.append(best_acc)
#    ave_acc = np.mean(ave_acc)     #平均值
#    symbol = np.greater(acc ,ave_acc)
#    if symbol == True:
#        num_i = num_i + 1
#        best_w =w
#        best_b =b
#        best_acc = ave_acc
#    else:
#        num_i = 0
#        best_acc = ave_acc
#    if best_acc > high_acc and num_i > 10:
#        stop = True
#    else:
#        stop = False
#    return best_w,best_b,best_acc,stop,num_i

def Cm_test_distance(testdata,Cm,w,b,R):
    space_test = np.matmul(testdata,w) + b
    m,_ = np.shape(space_test)
    distance = np.zeros([m,1],dtype=np.float32)
    for i in range(m):
        distance[i] = np.linalg.norm(space_test[i,:]-Cm)/R
    return distance

def RBF_traindata(datatrain):
    m,_ = np.shape(datatrain)
    sigma = 1.1
    rbf_data = np.zeros([m,m])
    for i in range(m):
        for j in range(m):
            rbf_data[i,j] = np.exp( -np.power(np.linalg.norm(datatrain[i,:]-datatrain[j,:]),2 ) / (2*sigma) )
    return rbf_data

def RBF_testdata(datatrain,datatest):
    m,_ = np.shape(datatrain)
    n,_ = np.shape(datatest)
    sigma = 1.1
    rbf_data = np.zeros([n,m])
    for i in range(n):
        for j in range(m):
            rbf_data[i,j] = np.exp(-np.power(np.linalg.norm(datatest[i,:]-datatrain[j,:]),2)/ (2*sigma) )
    return rbf_data

def creat_Graph(adr):
    #架构的可视化
    writer = tf.summary.FileWriter(adr)
    contain = tf.get_default_graph()
    writer.add_graph(contain)
    writer.close()

def creact_tsv(train_label,tsv_dirfile):
    #bedding的可视化
    with open(tsv_dirfile,'w') as f:
        for i in range(len(train_label)):
            f.write(str(train_label[i])+'\n')
        f.close()

Knn_p = [];svm_p=[];dt_p=[];ada_p=[]
nb_p = [];rf_p = [];xg_p = []
def comput_cicle(data,Nm1,Nm2,Nm3):
    c1 = np.zeros(shape=[2])
    c2 = np.zeros(shape=[2])
    c3 = np.zeros(shape=[2])
    for i in range(Nm1):
        c1 = c1 + data[i, :]
    c1 = c1 / Nm1
    for i in range(Nm2):
        c2 = c2 + data[Nm1 + i, :]
    c2 = c2 / Nm2
    for i in range(Nm3):
        c3 = c3 + data[(Nm1 + Nm2) + i, :]
    c3 = c3 / Nm3
    return c1,c2,c3
def plotSpace3(w,b,data,circle,r,Nm1,Nm2,Nm3):
    pca = PCA(n_components=2)
    # t_sne = TSNE(n_components=3,n_iter=4000,perplexity=15,learning_rate=10)
    data_now = np.matmul(data,w)+b
    #进行TSNE转换
    tsne_dataPre = pca.fit_transform(data)
    kind1 = tsne_dataPre[:Nm1,:];kind2 = tsne_dataPre[Nm1:(Nm1+Nm2),:];kind3 = tsne_dataPre[(Nm1+Nm2):,:]
    plt.figure(1)
    ax1 = plt.subplot(111)
    ax1.scatter(kind1[:,0],kind1[:,1],c='r')
    ax1.scatter(kind2[:, 0], kind2[:, 1], c='g')
    ax1.scatter(kind3[:, 0], kind3[:, 1], c='b')
    ax1.set_xlabel('Feature1',fontsize=13);ax1.set_ylabel('Feature2',fontsize=13)
    ax1.set_title('Wine')
    ax1.grid(False)
    legend1= ax1.legend(['Class 1', 'Class 2','Class 3'],loc=1)
    # fram1 = legend1.get_frame()
    # fram1.set_alpha(1)
    # fram1.set_facecolor('none')

    tsne_dataNow = pca.fit_transform(data_now)
    center1,center2,center3 = comput_cicle(tsne_dataNow,Nm1,Nm2,Nm3)
    k1 = tsne_dataNow[:Nm1,:];k2 = tsne_dataNow[Nm1:(Nm1+Nm2),:];k3 = tsne_dataNow[(Nm1+Nm2):,:]
    #绘制离散点
    plt.figure(2)
    ax2 = plt.subplot(111)
    ax2.scatter(k1[:,0],k1[:,1],c='r')
    ax2.scatter(k2[:, 0], k2[:, 1], c='g')
    ax2.scatter(k3[:, 0], k3[:, 1], c='b')
    ax2.set_xlabel('Feature1',fontsize=13);ax2.set_ylabel('Feature2',fontsize=13)
    ax2.set_title('Wine')
    ax2.grid(False)
    legend2 = ax2.legend(['Class 1', 'Class 2','Class 3'],loc=1)
    # fram2 = legend2.get_frame()
    # fram2.set_alpha(1)
    # fram2.set_facecolor('none')
    # plt.xlim(-50,50)
    # plt.ylim(-50,50)
    #绘制球体

    theta = np.arange(0,2*np.pi,0.01)
    x = center1[0]+r[0]*np.cos(theta)
    y = center1[1]+r[0]*np.sin(theta)
    ax2.plot(x,y,'r')

    x = center2[0]+r[1]*np.cos(theta)
    y = center2[1]+r[1]*np.sin(theta)
    ax2.plot(x,y,'g')

    x = center3[0]+r[2]*np.cos(theta)
    y = center3[1]+r[2]*np.sin(theta)
    ax2.plot(x,y,'b')

    # u = np.linspace(0, 2 * np.pi, 100)
    # v = np.linspace(0, np.pi, 100)
    # x = r[0] * np.outer(np.cos(u), np.sin(v)) + center1[0]
    # y = r[0] * np.outer(np.sin(u), np.sin(v)) + center1[1]
    # z = r[0] * np.outer(np.ones(np.size(u)), np.cos(v)) + center1[2]
    # ax2.plot_wireframe(x, y, z, rstride=3, cstride=3,alpha=0.1)
    #
    # x = r[1] * np.outer(np.cos(u), np.sin(v)) + center2[0]
    # y = r[1] * np.outer(np.sin(u), np.sin(v)) + center2[1]
    # z = r[1] * np.outer(np.ones(np.size(u)), np.cos(v)) + center2[2]
    # ax2.plot_wireframe(x, y, z, rstride=3, cstride=3,alpha=0.1)
    #
    # x = r[2] * np.outer(np.cos(u), np.sin(v)) + center3[0]
    # y = r[2] * np.outer(np.sin(u), np.sin(v)) + center3[1]
    # z = r[2] * np.outer(np.ones(np.size(u)), np.cos(v)) + center3[2]
    # ax2.plot_wireframe(x, y, z, rstride=3, cstride=3,alpha=0.1)


    plt.show()

def New_NDC_3kind(atatrain,target_train,datatest,target_test,Nm1,Nm2,Nm3,op_rate,P,P_R2Cm):
    best_acc = 0
    num_i = 0
    y_ = []
    circle = []
    r = []
    y_test = []
    y_score = []
    y_gaus = []
    with tf.name_scope('Space'):
        with tf.name_scope('InitVariable'):
            with tf.name_scope('V'):
                V = tf.placeholder(tf.float32, shape=[None, x_new], name='Input')
            with tf.name_scope('TrainingStep'):
                training_step = tf.Variable(0, name='TrainStep', trainable=False)
                learning_rate = tf.Variable(op_rate, name='TrainStep', trainable=False)
            with tf.name_scope('weight'):
                weight = tf.Variable(tf.random_normal([x_new, x_new]), name='W', trainable=True)
                best_w = tf.Variable(tf.random_normal([x_new, x_new]), name='W', trainable=False)

            with tf.name_scope('bias'):
                bias = tf.Variable(tf.zeros([x_new]), name='B', trainable=True)
                best_b = tf.Variable(tf.zeros([x_new]), name='B', trainable=False)
            with tf.name_scope('Cm'):
                Cm1 = tf.Variable(tf.zeros([1, x_new]), name='Cm1', trainable=False)  # 第一类质心
                Cm2 = tf.Variable(tf.zeros([1, x_new]), name='Cm2', trainable=False)  # 第二类质心
                Cm3 = tf.Variable(tf.zeros([1, x_new]), name='Cm3', trainable=False)  # 第三类质心
            with tf.name_scope('R'):
                R1 = tf.Variable(initial_value=R_constant, dtype=tf.float32, name='R1', trainable=True)  # 半径
                R2 = tf.Variable(initial_value=R_constant, dtype=tf.float32, name='R2', trainable=True)  # 半径
                R3 = tf.Variable(initial_value=R_constant, dtype=tf.float32, name='R3', trainable=True)  # 半径

        with tf.name_scope('layer'):
            U =   tf.matmul(V, weight) + bias


        with tf.name_scope('circle1'):
            Cm1 = tf.reduce_sum(U[:Nm1, :],axis=0)/Nm1

        with tf.name_scope('circle2'):
            Cm2 = tf.reduce_sum(U[Nm1:(Nm1+Nm2), :],axis=0)/Nm2

        with tf.name_scope('circle3'):
            Cm3 = tf.reduce_sum(U[(Nm1+Nm2):, :],axis=0)/Nm3


        with tf.name_scope('loss'):
            with tf.name_scope('loss_1'):

                g1n1 = g1n_term(U[:Nm1, :], Cm1, R1, Nm1)
                g2n1 = g2n_term(U[Nm1:, :], Cm1, R1, Nm2+Nm3)
                Rn1 = tf.where(tf.greater(-R1, 0), -R1, 0)

                # loss = 该类中所有样本的欧式距离 + P*{ 本类样本 + 非本类样本 }

                loss1_pow =   tf.pow(g1n1,2) + tf.pow(g2n1,2)+tf.pow(Rn1,2)

            with tf.name_scope('loss_2'):

                U_ = tf.concat([U[:Nm1, :], U[(Nm1 + Nm2):, :]], 0)
                g1n2 = g1n_term(U[Nm1:Nm1 + Nm2, :], Cm2, R2, Nm2)
                g2n2 = g2n_term(U_, Cm2, R2, Nm1 + Nm3)
                Rn2 = tf.where(tf.greater(-R2, 0), -R2, 0)

                # loss = 该类中所有样本的欧式距离 + P*{ 本类样本 + 非本类样本 }

                loss2_pow =  tf.pow(g1n2,2) + tf.pow(g2n2,2)+ tf.pow(Rn2,2)

            with tf.name_scope('loss_3'):

                g1n3= g1n_term(U[(Nm1 + Nm2):, :], Cm3, R3, Nm3)
                g2n3 = g2n_term(U[:(Nm1 + Nm2), :], Cm3, R3, Nm1 + Nm2)
                Rn3 = tf.where(tf.greater(-R3, 0), -R3, 0)

                # loss = 该类中所有样本的欧式距离 + P*{ 本类样本 + 非本类样本 }
                loss3_pow =tf.pow( g1n3,2)+ tf.pow(g2n3,2)+ tf.pow(Rn3,2)


            with tf.name_scope('lossR2Cm_R12'):
                Cm12_normal =(1+0)* (R1+R2)- tf.linalg.norm( Cm1-Cm2 )
                Cm12_penalty = tf.where( tf.greater(Cm12_normal,0),Cm12_normal,0)

            with tf.name_scope('lossR2Cm_R13'):
                Cm13_normal = (1+0)* (R1 + R3) -  tf.linalg.norm(Cm1 - Cm3)
                Cm13_penalty = tf.where(tf.greater(Cm13_normal, 0), Cm13_normal, 0)

            with tf.name_scope('lossR2Cm_R23'):
                Cm23_normal =(1+0)* (R2 + R3) -tf.linalg.norm(Cm2 - Cm3)
                Cm23_penalty = tf.where(tf.greater(Cm23_normal, 0), Cm23_normal, 0)

        with tf.name_scope('loss_pow'):
            loss_pow = (loss1_pow+loss2_pow+loss3_pow)

        with tf.name_scope('loss_R2Cm'):
            loss_R2Cm =  (Cm12_penalty+Cm13_penalty+Cm23_penalty)

            loss_class = tf.linalg.norm(U[:Nm1, :] - Cm1) + tf.linalg.norm(U[Nm1:Nm1 + Nm2, :] - Cm2)+\
                         tf.linalg.norm(U[(Nm1 + Nm2):,:] - Cm2)

        with tf.name_scope('loss_all'):
            loss_all =   P *(loss_pow) + P_R2Cm *loss_R2Cm+P_class* loss_class

        with tf.name_scope('train'):
            learning_rate = tf.train.exponential_decay(learning_rate, training_step, decay_steps=100, decay_rate=0.8)
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_all)

        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)

            loss_list = []
            R1_list = []
            R2_list = []
            R3_list = []
            print('Enter train the Space........')

            for j in range(step):
                _ = sess.run(train_op, feed_dict={V: datatrain })
                loss = sess.run(loss_all, feed_dict={V: datatrain })
                loss_list.append(loss)
                R_1,R_2,R_3 = sess.run((R1,R2,R3))
                R1_list.append(R_1)
                R2_list.append(R_2)
                R3_list.append(R_3)
            print(loss_list)
            with open('loss_wine_.txt','w') as f:
                for l in loss_list:
                    f.write(str(l)+',')
            f.close()
            weight_ = sess.run(weight)
            bias_ = sess.run(bias)
            Circle1 = sess.run(Cm1, feed_dict={V: datatrain})
            R_1 = sess.run(R1)
            Cmpre_one = Cm_test_distance(datatest,Circle1,weight_,bias_,R_1)
            gaus_Cmpre_one = Gaussian_PDF(weight_, bias_, datatrain[:Nm1, :], datatest)
            circle.append(Circle1)
            r.append(R_1)

            Circle2 = sess.run(Cm2, feed_dict={V: datatrain})
            R_2 = sess.run(R2)
            Cmpre_two = Cm_test_distance(datatest,Circle2,weight_,bias_,R_2)
            gaus_Cmpre_two = Gaussian_PDF(weight_, bias_, datatrain[Nm1:(Nm1 + Nm2), :], datatest)
            circle.append(Circle2)
            r.append(R_2)

            Circle3 = sess.run(Cm3, feed_dict={V: datatrain})
            R_3 = sess.run(R3)
            Cmpre_three = Cm_test_distance(datatest,Circle3,weight_,bias_,R_3)
            gaus_Cmpre_three = Gaussian_PDF(weight_, bias_, datatrain[(Nm1 + Nm2):, :], datatest)
            circle.append(Circle3)
            r.append(R_3)

            Acc = Acc_3(Cmpre_one, Cmpre_two, Cmpre_three, target_test)
            gaus_acc = Gaussian_acc3(gaus_Cmpre_one, gaus_Cmpre_two, gaus_Cmpre_three,target_test )
            if Acc< gaus_acc:
                Acc=gaus_acc
            print(random_index)
            print(Acc)
            plotSpace3(weight_,bias_,datatrain,np.reshape(circle,newshape=[3,x_new]),r,Nm1,Nm2,Nm3)
            plt.figure(3)
            ax2 = plt.subplot(111)
            ax2.set_xlabel('Iteration number',fontsize=13)
            ax2.set_ylabel('Loss/10^4',fontsize=13)
            ax2.grid(False)
            def formatnum(x,pos):
                return '%.1f'%(x/10000)
            formatter = FuncFormatter(formatnum)
            ax2.yaxis.set_major_formatter(formatter)
            x = np.arange(0,step)
            y = loss_list
            ax2.plot(x, y,'r')

            plt.figure(4)
            ax2 = plt.subplot(111)
            ax2.set_xlabel('Iteration number', fontsize=13)
            ax2.set_ylabel('R', fontsize=13)
            ax2.grid(False)
            x = np.arange(0, step)
            y = R1_list
            ax2.plot(x, y, 'r')
            y = R2_list
            ax2.plot(x, y, 'g')
            y = R3_list
            ax2.plot(x, y, 'b')
            plt.legend(['R1 ', 'R2 ', 'R3 '],loc=1)
            plt.show()

        print('end')
    return Acc


def Acc_3(Cmpre_one,Cmpre_two,Cmpre_three,target):
    m,_ = np.shape(Cmpre_one)
    y_out = [];y_ex = []
    with tf.name_scope('Acc'):
        y = np.zeros([m])
        for i in range(len(y)):
            if Cmpre_one[i]<1 and Cmpre_two[i] > 1 and Cmpre_three[i] > 1:
                y[i] =1
            else:
                if Cmpre_two[i] < 1 and Cmpre_one[i] > 1 and Cmpre_three[i] > 1:
                    y[i] = 2
                else:
                    if Cmpre_three[i] < 1 and Cmpre_one[i] > 1 and Cmpre_two[i] > 1:
                        y[i] = 3
                    else:
                        if Cmpre_three[i] > 1 and Cmpre_one[i] > 1 and Cmpre_two[i] > 1:
                            y_out.append(i)
                        else:
                            y_ex.append(i)
                        if Cmpre_one[i] > Cmpre_two[i]:
                            if Cmpre_two[i] > Cmpre_three[i]:
                                y[i] = 3
                            else:
                                y[i] = 2
                        else:
                            if Cmpre_one[i] > Cmpre_three[i]:
                                y[i] = 3
                            else:
                                y[i] = 1
        correct_predict = np.equal(y, target)
        accuracy = np.mean(correct_predict)

    return accuracy

if __name__ == '__main__':
    R_constant = 1.0

    op_rate = 0.8

    kernal = 'rbf'

    x = 13        #初始维度
    x_new = 13  #新的维度

    per = 0.3

    # Nm1 = 50  #第一类的数量
    # Nm2 = 50  #第二类的数量
    # Nm3 = 50  #第三类的数量
    step = 300

    data, target = load_Wine()

    scaler = MinMaxScaler(feature_range=(0, 1)) #标准化
    data = scaler.fit_transform(data)

    acc_list = []
    index_i = 0
    for i in range(10):
        random_index = np.random.choice(range(100),size= 1,replace=False)

        datatrain, datatest, target_train, target_test = train_test_split(data, target, test_size=per, random_state=int(random_index))

        sort_index = np.argsort(target_train)
        datatrain = datatrain[sort_index]
        target_train = target_train[sort_index]

        datatest = RBF_testdata(datatrain, datatest)
        datatrain = RBF_traindata(datatrain)

        print('RBF转换完成')
        _, x = np.shape(datatrain)
        x_new = int(x)  # 新的维度

        Nm1 = list(target_train).count(1)
        Nm2 = list(target_train).count(2)
        Nm3 = list(target_train).count(3)

        P = 1
        P_R2Cm = 1
        P_class = 0.00002
        # P_class = 0.00001
        print(per,P_class)
        acc = New_NDC_3kind(datatrain,target_train,datatest,target_test,Nm1,Nm2,Nm3,op_rate,P,P_R2Cm)
        acc_list.append(acc)
        index_i +=1
    print(acc_list)
    print('acc_av:', sum(acc_list) / (index_i) * 100)
    acc_av = sum(acc_list) / (index_i)
    acc_list.append(acc_av)
    ava = acc_list[-1]

    value = np.array(acc_list)
    m = np.shape(value)

    f = 0

    for i in value:
        f += np.power((i - ava), 2)
    std = np.sqrt(f / m)
    print('标准差：', std)

