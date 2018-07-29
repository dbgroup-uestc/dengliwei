#coding=UTF-8

import cPickle
import numpy as np

import sys

reload(sys)

sys.setdefaultencoding('utf8')

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras import backend as K

def triple2vec(Atriple, w2v): #传入一个三元组，将该三元组<e,p,o>转换成向量，其中p中的单词数为8，如果不够则用0向量补齐，如果多了，则截断
    vec = []
    vec.append(w2v[Atriple[0]].astype(float))
    vec.append(w2v[Atriple[2]].astype(float))
    if len(Atriple[1]) >= 8:
        i = 0
        for word in Atriple[1]:
            if i < 8:
                vec.append(w2v[word].astype(float))
                i = i + 1
    elif len(Atriple[1]) < 8 and len(Atriple[1]) >= 1:
        i = 0
        for word in Atriple[1]:
			vec.append(w2v[word].astype(float))
			i = i + 1
		#生成一个300维的0向量
        zeros = [0 for j in range(300)]
        zeros = np.array(zeros, dtype=float)
        while i < 8:
            vec.append(zeros)
            i = i + 1
    return vec

def splitdata(revs, cv, w2v):
    """
    Transforms sentences into a 2-d matrix.
    """
    x_train, x_test, x_cv = [], [], []		#训练集、测试集、cv集的数据
    y_train, y_test, y_cv = [], [], []	#训练集、测试集、cv集的标签
    for rev in revs:  #对所有的text进行操作
        if rev["split_cv"]==cv:            #如果正好split==cv，那么就将三元组作为test集
            x_test.append(triple2vec(rev['Triple'],w2v))
            y_test.append(rev['y'])
        elif rev["split_test"]==cv:
			x_cv.append(triple2vec(rev['Triple'],w2v))
			y_cv.append(rev['y'])
        else:  
            x_train.append(triple2vec(rev['Triple'],w2v))     #否则，就将三元组作为训练集
            y_train.append(rev['y'])
    return x_train, x_test, y_train, y_test, x_cv, y_cv    #返回训练集与测试集组成的列表

if __name__=="__main__":
    print "loading data..."
    x = cPickle.load(open("mr.p","rb"))		#加载之前处理好的数据
    w2v, revs = x[0], x[1]					#数据还原\
    print "data loaded!"
    #设置训练参数
    batch_size = 128    
    epochs = 12   #设置训练次数
    pool_size = 2
    kernel_size = 3
    strides = 1
    
    
    results = []
    
    
    # dimension of input data
    img_rows, img_cols = 10, 300   #(8+2)*300
    input_shape = (img_rows, img_cols)
    
    for cv in range(0,10):		#将数据组成10种组合，然后训练
        x_train, x_test, y_train, y_test, x_cv, y_cv = splitdata(revs, cv, w2v)  #将整个三元组集合分成训练集和测试集
        #x_train sample*10*300     y sample*1
        
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        x_cv = np.array(x_cv)

        print 'x_train shape:', x_train.shape
        print x_train.shape[0], 'train samples'
        print x_cv.shape[0], 'cv samples'
        print x_test.shape[0], 'test samples'

        #构建CNN
        # model = Sequential()
        # model.add(Conv2D(32, kernel_size=(3, 3),
                         # activation='relu',
                         # input_shape=input_shape))
        # model.add(Conv2D(64, (3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        # model.add(Flatten())
        # model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(num_classes, activation='softmax'))
        
        #conv+Max_pool+conv+Max_pool+Flatten+Merge+Full_Connected
        model = Sequential()
        model.add(Conv1D(filters=64,kernel_size=kernel_size,activation='relu', input_shape=input_shape))  #构建一个卷积层 Convolution Region Size=3,这里的filters的大小应该为多少？
        model.add(MaxPooling1D(pool_size=pool_size,strides=strides))  #构建一个池化层 Max Pooling Units=2
        model.add(Dropout(0.2))  #Dropout Rate=0.2

        model.add(Conv1D(filters=32,kernel_size=kernel_size,activation='relu'))  #构建一个卷积层 Convolution Region Size=3
        model.add(MaxPooling1D(pool_size=pool_size,strides=strides))  #构建一个池化层 Max Pooling Units=2
        model.add(Dropout(0.2))  #Dropout Rate=0.2

        model.add(Flatten())
        
        model.add(Dense(300,activation='relu'))   #全连接层
        model.add(Dense(1,activation='relu'))   #全连接层，最后输出为1个概率,设置全连接层的激活函数为Relu

        #模型编译，设置相应的训练参数
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer='sgd',
                      metrics=['accuracy'],
                      momentum=0.6)  #Optimizer=sgd , momentum=0.6
#
        model.fit(x_train, y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1,
                 validation_data=(x_cv, y_cv))   #verbose：日志显示，0表示不在标准输出流输出日志信息，1为输出进度条记录
       score = model.evaluate(x_test, y_test, verbose=0)
       print('Test loss:', score[0])
       print('Test accuracy:', score[1])
       results.append(score)
