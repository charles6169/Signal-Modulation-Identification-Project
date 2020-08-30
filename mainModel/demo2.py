from Model import *

# # 从这里开始编写网络
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 60000
mnist = input_data.read_data_sets('C:/Users/JDUSER/Documents/我的坚果云/代码/tftyd/data/', one_hot=True)
data = mnist.train.next_batch(batch_size)
x = data[0]
y = data[1]

learning_rate = 1e-5
model = models()
model.add(Dense(input_size=None, hidden_layer_num=32, activation=activations.relu))
model.add(Dense(input_size=None, hidden_layer_num=64, activation=activations.relu))
model.add(Dense(input_size=None, hidden_layer_num=10, activation=activations.softmax))
model.fit(train_data=x, train_labels=y, shuffle=False, epochs=6, batch_size=50,
          learning_rate=learning_rate, measure=measure.cross_entropy)
model.plot()
