import numpy as np

def cross_entropy(y_hat, y):
    return np.nan_to_num(-np.mean(np.log(y_hat) * y))


def mse(y_hat, y):
    # print("MSE")
    # print(y)
    # print(y_hat)
    # print(np.sum(np.abs(y - y_hat)) / (y.shape[0] * y.shape[1]))
    return np.sum(np.abs(y - y_hat)) / (y.shape[0] * y.shape[1])


def sigmoid_forward(x):
    return np.nan_to_num(1 / (1 + np.exp(-x)))


def sigmoid_backward(x):
    return x * (1 - x)  # 返回求导结果


def tanh_forward(x):
    return np.nan_to_num((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))


def tanh_backward(x):
    return np.nan_to_num(1 - x * x)


def relu_forward(x):
    return np.nan_to_num(np.where(x > 0, x, 0))


def relu_backward(x):
    return np.nan_to_num(np.where(x > 0, 1, 0))


def accuracy(output, label):
    return np.mean(np.equal(output.argmax(axis=1), label.argmax(axis=1)))


def softmax(output):
    data = np.exp(output)
    sum = np.sum(data, axis=1, keepdims=True)
    return data / sum


class models:
    model_list = []
    accuracy_list = []
    loss_list = []
    predict_list = []

    batch_input = None
    batch_label = None
    learning_rate = 1e-3
    measure = 'cross_entropy'

    def __init__(self):
        pass

    def add(self, dense):
        self.model_list.append(dense)

    def summary(self):
        self.init()
        for model in self.model_list:
            model.print()

    def init(self):
        for i, model in enumerate(self.model_list):
            '第一层输入'
            if i == 0:
                '修正首层的输入'
                model.input_size = self.batch_input.shape[1]
                model.init()
                model.input = self.batch_input
                model.output = np.dot(self.batch_input, model.w) + model.b

            else:
                '推导每层的形状'
                last_model = self.model_list[i - 1]
                model.input_size = last_model.output.shape[1]
                model.init()
                model.input = last_model.output
                model.output = np.dot(last_model.output, model.w) + model.b

    def forward(self):
        for i, model in enumerate(self.model_list):
            '第一层输入'
            if i == 0:
                model.input = self.batch_input
                model.output = np.dot(self.batch_input, model.w) + model.b
            # '其他层'
            else:
                last_model = self.model_list[i - 1]
                model.input = last_model.output
                model.output = np.dot(last_model.output, model.w) + model.b

            '激活函数'
            model.unactivated_output = model.output
            if model.activation == 'sigmoid':
                model.output = sigmoid_forward(model.output)
            elif model.activation == 'relu':
                model.output = relu_forward(model.output)
            elif model.activation == 'tanh':
                model.output = tanh_forward(model.output)
            elif model.activation == 'softmax':
                model.output = softmax(model.output)

        if self.measure == 'mse':
            self.loss_list.append(mse(self.model_list[len(self.model_list) - 1].output, self.batch_label))
        elif self.measure == 'cross_entropy':
            self.loss_list.append(cross_entropy(self.model_list[len(self.model_list) - 1].output, self.batch_label))
            self.accuracy_list.append(accuracy(self.model_list[len(self.model_list) - 1].output, self.batch_label))
        return model.output

    def backward(self):
        loss = self.model_list[len(self.model_list) - 1].output - self.batch_label
        for i in range(0, len(self.model_list)).__reversed__():
            model = self.model_list[i]
            if model.activation == 'relu':
                loss = loss * relu_backward(model.unactivated_output)
            elif model.activation == 'sigmoid':
                loss = loss * sigmoid_backward(model.output)
            elif model.activation == 'tanh':
                loss = loss * tanh_backward(model.output)

            model.b -= self.learning_rate * loss
            model.w -= self.learning_rate * model.input.T.dot(loss)
            loss = loss.dot(model.w.T)

    def plot(self):
        import matplotlib.pyplot as plt
        epochs = range(1, len(self.loss_list) + 1)
        if len(self.accuracy_list) == len(self.loss_list):
            plt.plot(epochs, self.accuracy_list, 'b', label='accuracy')  # blue plot
        plt.plot(epochs, self.loss_list, 'r', label='loss')
        plt.title("Training and validation loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def fit(self, train_data, train_labels, shuffle, epochs, batch_size, learning_rate,
            measure):
        self.learning_rate = learning_rate
        if measure is not None:
            self.measure = measure
        for i in range(epochs):
            for j in range(train_data.shape[0] // batch_size):
                x = train_data[j:j + batch_size]
                y = train_labels[j:j + batch_size]
                if shuffle:
                    x, y = self.shuffle(x, y)
                self.batch_input = x
                # 检查维度
                if len(y.shape) == 1:
                    y = np.expand_dims(y, axis=1)
                self.batch_label = y
                # if i == 0 and j == 0:
                #     print("epochs for %s steps " % (train_data.shape[0] // batch_size))
                #     self.summary()
                if len(self.loss_list) == 0:
                    print("epochs for %s steps " % (train_data.shape[0] // batch_size))
                    self.summary()
                # if len(self.loss_list) != 0 and len(self.loss_list) % 1000 == 0:
                #     self.learning_rate = self.learning_rate * 0.5
                self.forward()
                self.backward()
            if measure == 'cross_entropy':
                print("accuracy: ", self.accuracy_list[-1])
            print("loss: ", self.loss_list[-1])

    # 这里需要保证传入的数据是批量的整数倍
    def predict(self, test_data, batch_size, clear_history=False):
        assert (batch_size == self.batch_label.shape[0])
        if clear_history:
            self.predict_list = []
        for i in range(test_data.shape[0] // batch_size):
            self.batch_input = test_data[i:i + batch_size]
            for predict in self.forward():
                self.predict_list.append(predict)
        return self.predict_list

    def verify(self, test_data, test_labels, batch_size, clear_history=False):
        assert (batch_size == self.batch_label.shape[0])
        if clear_history:
            self.accuracy_list = []
            self.loss_list = []
        for i in range(test_data.shape[0] // batch_size):
            self.batch_input = test_data[i:i + batch_size]
            self.batch_label = test_labels[i:i + batch_size]
            self.forward()

    def shuffle(self, x, y):
        import random
        seed = random.randint(1, 1000)
        np.random.seed(seed)
        np.random.shuffle(x)
        np.random.seed(seed)
        np.random.shuffle(y)
        return x, y


class activations:
    relu = 'relu'
    sigmoid = 'sigmoid'
    tanh = 'tanh'
    softmax = 'softmax'


class measure:
    cross_entropy = 'cross_entropy'
    mse = 'mse'


class Dense:
    '描述的是神经网络单元'
    input_size = 0
    hidden_layer_num = 0
    activation = None

    '需要训练的参数'
    w = None
    b = None
    input = None
    unactivated_output = None
    output = None
    bias = None

    '这里的input_size 是输入的向量长度非batch_size,或上一层神经元数量,是可以通过推断得出的，输出向量的高度就是batchsize'

    def __init__(self, input_size, hidden_layer_num, activation):
        self.input_size = input_size
        self.hidden_layer_num = hidden_layer_num
        self.activation = activation

    def init(self):
        self.w = 2 * np.random.random((self.input_size, self.hidden_layer_num)) - 1
        self.b = 0

    def print(self):
        print("(%3s,%3s) activation = %5s" % (self.input_size, self.hidden_layer_num, self.activation))
        print("-" * 50)
