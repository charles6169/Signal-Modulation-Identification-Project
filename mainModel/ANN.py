import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

class Data:
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.l = x.shape[1]
        self.batch_size = batch_size
        self.pos = 0

    def forward(self):
        # Mini-batch
        pos = self.pos
        bat = self.batch_size
        l = self.l
        if pos + bat >= l:
            ret = (self.x[:, pos:l], self.y[pos:l])
            self.pos = 0
            index = range(l)
            np.random.shuffle(list(index))
            self.x = self.x[:, index]
            self.y = self.y[index]
        else:
            ret = (self.x[:, pos:pos + bat], self.y[pos:pos + bat])
            self.pos += self.batch_size

        return ret, self.pos

    def backward(self, d):
        pass


class FullyConnect:
    def __init__(self, l_x, l_y, L2=0, keep_prob=1, methods='Grad',
                 k1=0.9, k2=0.999, batch_normal=False, predict=False):
        np.random.seed(42)
        self.l_x = l_x
        self.weights = np.random.randn(l_y, l_x) * np.sqrt(2 / l_x)
        self.bias = np.random.randn(l_y, 1)
        self.lr = 0
        self.L2 = L2
        self.keep_prob = keep_prob
        self.methods = methods
        # Monmentum
        self.vdw = 0
        self.vdb = 0
        # RmSprop
        self.sdw = 0
        self.sdb = 0
        # Adam

        # batch-Normalization
        self.batch_normal = batch_normal
        self.gram = np.random.randn(l_y, 1)
        self.beta = np.random.randn(l_y, 1)
        self.mean = np.zeros((l_y, 1))
        self.std = np.zeros((l_y, 1))
        self.predict = predict

    def forward(self, x):
        # drop-out
        iskeep = np.random.rand(1, self.l_x) < self.keep_prob
        self.keep_weights = self.weights * iskeep / self.keep_prob

        self.x = x

        self.y = np.dot(self.keep_weights, self.x) + self.bias

        # batch-Normalization
        if self.batch_normal:
            self._batch_normalization()

        return self.y

    def _batch_normalization(self):
        if self.predict:
            predict_y_norm = (self.y - self.mean) / (self.std + 1e-8)
            self.y = self.gram * predict_y_norm + self.beta
        else:
            n = self.y.shape[1]
            self.tmean = np.mean(self.y, axis=1, keepdims=True)
            self.tstd = np.std(self.y, axis=1, keepdims=True)
            self.y_norm = (self.y - self.tmean) / (self.tstd + 1e-8)
            self.y = self.gram * self.y_norm + self.beta

            self.mean = 0.9 * self.mean + 0.1 * self.tmean
            self.std = 0.9 * self.std + 0.1 * self.tstd

    def backward(self, d):
        if self.batch_normal:
            d = d * self.gram / self.tstd
            self.dgram = d * self.y_norm
            self.gram -= self.lr * np.sum(self.dgram, axis=1, keepdims=True) / self.y.shape[1]
            self.beta -= self.lr * np.sum(d, axis=1, keepdims=True) / self.y.shape[1]

        self.dw = np.dot(d, self.x.T) / self.x.shape[1] + self.L2 * self.keep_weights / (2 * self.x.shape[1])
        self.db = np.sum(d, axis=1, keepdims=True) / self.x.shape[1]
        self.dx = np.dot(self.keep_weights.T, d)

        # 优化
        self._optimize(self.methods)
        return self.dx

    def _optimize(self, methods='Grad', k1=0.9, k2=0.999):
        if methods == 'Grad':
            self.weights -= self.lr * self.dw
            self.bias -= self.lr * self.db
        elif methods == 'Monmentum':
            # 未修正
            self.vdw = k1 * self.vdw + (1 - k1) * self.dw
            self.vdb = k1 * self.vdb + (1 - k1) * self.db
            self.weights -= self.lr * self.vdw
            self.bias -= self.lr * self.vdb
        elif methods == 'RMSprop':
            # 未修正
            self.sdw = k2 * self.sdw + (1 - k2) * self.dw ** 2
            self.sdb = k2 * self.sdb + (1 - k2) * self.db ** 2
            self.weights -= self.lr * self.dw / (np.sqrt(self.sdw) + 1e-8)
            self.bias -= self.lr * self.db / (np.sqrt(self.sdb) + 1e-8)
        elif methods == 'Adam':
            self.vdw = k1 * self.vdw + (1 - k1) * self.dw
            self.vdb = k1 * self.vdb + (1 - k1) * self.db
            self.sdw = k2 * self.sdw + (1 - k2) * self.dw ** 2
            self.sdb = k2 * self.sdb + (1 - k2) * self.db ** 2

            self.weights -= self.lr * self.vdw / (np.sqrt(self.sdw) + 1e-8)
            self.bias -= self.lr * self.vdb / (np.sqrt(self.sdb) + 1e-8)

class Sigmoid:
    def __init__(self):
        pass
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    def forward(self, x):
        self.x = x
        self.y = self.sigmoid(x)
        return self.y
    def backward(self, d):
        sig = self.sigmoid(self.x)
        self.dx = d*sig*(1-sig)
        return self.dx

class Relu1:
    def __init__(self):
        pass
    def relu1(self, x):
        s = np.ones_like(x)/10
        s[x > 0] = 1
        return x*s
    def forward(self, x):
        self.x = x
        self.y = self.relu1(x)
        return self.y
    def backward(self, d):
        s = np.ones_like(self.x)/10
        s[self.x > 0] = 1
        return d*s

class Relu:
    def __init__(self):
        pass
    def relu(self, x):
        return x*(x>0)
    def forward(self, x):
        self.x = x
        self.y = self.relu(x)
        return self.y
    def backward(self, d):
        r = self.x > 0
        return d*r

class Relu:
    def __init__(self):
        pass
    def relu(self, x):
        return x*(x>0)
    def forward(self, x):
        self.x = x
        self.y = self.relu(x)
        return self.y
    def backward(self, d):
        r = self.x > 0
        return d*r

class QuadraticLoss:
    def __init__(self, L2):
        self.L2 = L2
    def forward(self, x, label):
        self.x = x
        self.label = np.zeros_like(x)
        for i in range(len(label)):
            self.label[label[i], i] = 1
        self.loss = np.sum(np.square(self.x - self.label))/self.x.shape[1]/2
        return self.loss
    def backward(self):
        self.dx = (self.x - self.label)
        return self.dx

class Accuracy:
    def __init__(self):
        pass
    def forward(self, x, label):
        self.accuracy = 0
        for i in range(len(label)):
            xx = np.argmax(x[:, i])
            if xx == label[i]:
                self.accuracy += 1
        self.accuracy = 1.0*self.accuracy/x.shape[1]
        return self.accuracy


class ANN:
    def __init__(self, layer_sizes, epochs=20, batch_size=1, learning_rate=0.01, L2=0, keep_probs=None,
                 methods='Grad', k1=0.9, k2=0.999, batch_normal=False):
        self.ls = layer_sizes
        self.bs = batch_size
        self.lr = learning_rate
        self.epochs = epochs

        # 正则化
        self.L2 = L2
        self.keeｐ_probs = keep_probs

        # 优化算法
        self.methods = methods
        self.k1 = k1
        self.k2 = k2

        # batch-normal
        self.batch_normal = batch_normal

    def fit(self, X, y):
        data_layer = Data(X, y, self.bs)
        input_size = X.shape[0]
        out_size = len(set(y))
        inner_layers = []

        losslayer = QuadraticLoss(0)

        if self.ls == []:
            inner_layers.append(FullyConnect(input_size, out_size, self.L2))
            inner_layers.append(Sigmoid())
        elif self.ls != None:
            inner_layers.append(FullyConnect(input_size, self.ls[0], self.L2, batch_normal=self.batch_normal))
            inner_layers.append(Relu1())

            for i in range(0, len(self.ls) - 1):
                full_layer = FullyConnect(self.ls[i], self.ls[i + 1], self.L2, batch_normal=self.batch_normal)
                if self.keep_probs != None:
                    full_layer.keep_prob = self.keep_probs[i]
                inner_layers.append(full_layer)
                inner_layers.append(Relu1())

            inner_layers.append(FullyConnect(self.ls[-1], out_size, self.L2))
            inner_layers.append(Sigmoid())

        for layer in inner_layers:
            layer.lr = self.lr  # 为所有中间层设置学习速率

            layer.methods = self.methods
            layer.k1 = self.k1
            layer.k2 = self.k2

        print(len(inner_layers))
        # 学习
        for i in range(self.epochs):
            losssum = 0
            iters = 0
            print('epochs:', i)
            while True:
                data, pos = data_layer.forward()  # 从数据层取出数据
                x, label = data
                for layer in inner_layers:  # 前向计算
                    x = layer.forward(x)

                loss = losslayer.forward(x, label)  # 调用损失层forward函数计算损失函数值
                losssum += loss
                iters += 1
                d = losslayer.backward()  # 调用损失层backward函数层计算将要反向传播的梯度
                for layer in inner_layers[::-1]:  # 反向传播
                    d = layer.backward(d)
                if pos == 0:
                    print('loss:', losssum / iters)
                    break
        self.inner_layers = inner_layers

    def predict(self, X):
        for layer in self.inner_layers:
            layer.predict = True
            X = layer.forward(X)
        return X

    def accuracy(self, y2, y):
        accuracy = Accuracy()
        accu = accuracy.forward(y2, y)  # 调用准确率层forward()函数求出准确率
        print('accuracy:', accu)
        return accu


from readfile import readfile
if __name__ == '__main__':
    # digits = load_digits()
    # data = digits.data
    # target = digits.target
    # scaler = StandardScaler()
    # data = scaler.fit_transform(data)
    #
    # train_x = data[:1500].T
    # train_y = target[:1500]
    # test_x = data[1500:].T
    # test_y = target[1500:]

    trainData, trainLabel, testData, testLabel = readfile("true")
    ann = ANN([10, 5], batch_size=30, epochs=50, learning_rate=1, batch_normal=False)
    ann.fit(trainData, trainLabel)

    pre = ann.predict(testData)
    ann.accuracy(pre, testLabel)