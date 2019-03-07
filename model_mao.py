from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout,LSTM
from keras.layers import Conv2D,MaxPool2D,Conv1D,MaxPooling1D,GlobalAveragePooling1D
from keras.optimizers import SGD
import numpy as np


def model_1(train_input,train_labels,test_input,test_labels):
    model=Sequential([
        Flatten(input_shape=(512, 1)),
        Dense(128),
        Activation('relu'),
        Dropout(0.25),
        Dense(128),
        Activation('relu'),
        Dropout(0.25),
        Dense(8),
        Activation('softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

    model.fit(train_input, train_labels, epochs=100, batch_size=32)
    loss, acc = model.evaluate(test_input, test_labels, batch_size=32)
    print(loss)
    print(acc)
    return model


def model_2(train_input,train_labels,test_input,test_labels):
    #基于一维特征向量的卷积网络
    model = Sequential()
    model.add(Conv1D(32, 5, activation='relu', input_shape=(512,1)))
    model.add(Conv1D(32, 5, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.25))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.25))
    model.add(Dense(8, activation='softmax'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(train_input, train_labels, epochs=10, batch_size=32,validation_split=0.2)
    loss, acc = model.evaluate(test_input, test_labels, batch_size=32)
    model.save("mao_nmf_512.model")
    print(loss)
    print(acc)
    return model


def model_3(train_input,train_labels,test_input,test_labels):
    #典型的CNN卷积网络
    model = Sequential()
    # 第一个卷积层，32个卷积核，大小５x5，卷积模式SAME,激活函数relu,输入张量的大小
    model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(16, 16, 2)))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='Same', activation='relu'))
    # 池化层,池化核大小２x2
    model.add(MaxPool2D(pool_size=(2, 2)))
    # 随机丢弃四分之一的网络连接，防止过拟合
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    # 全连接层,展开操作，
    model.add(Flatten())
    # 添加隐藏层神经元的数量和激活函数
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    # 输出层
    model.add(Dense(8, activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

    model.fit(train_input, train_labels, epochs=20, batch_size=32, validation_split=0.2)

    loss, acc = model.evaluate(test_input, test_labels, batch_size=32)
    model.save("mao_mfcc_32_16.model")
    print(loss)
    print(acc)
    return model

def model_4(train_input,train_labels,test_input,test_labels):
    data_dim = 32
    timesteps = 16
    num_classes = 8
    model = Sequential()
    model.add(LSTM(32, return_sequences=True,
                   input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32))  # return a single vector of dimension 32
    model.add(Dense(8, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])



    model.fit(train_input, train_labels,
              batch_size=64, epochs=5,
              validation_split=0.1)
    model.save("lstm.model")

    loss, acc = model.evaluate(test_input, test_labels, batch_size=32)
    print(loss)
    print(acc)
    return model